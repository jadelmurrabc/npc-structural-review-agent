"""NPC Structural Review Tool — Two-tool architecture.

Tool 1: extract_document() — finds PDF in user message, extracts text, stores to GCS
Tool 2: structural_review() — reads text from GCS, runs full checklist review

PDF extraction uses the proven Enterprise pattern:
  tool_context.user_content → parts (dict or object) → base64 inlineData → PyMuPDF

Arabic documents: Gemini 2.5 Pro reads Arabic natively. No translation needed.
We use SYSTEM_INSTRUCTION_ARABIC to tell Gemini to read Arabic and respond in English.
"""
import base64
import json
import logging
import os
import random
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from google.adk.tools import ToolContext
from base_agent.logic.checklist_loader import get_applicable_sub_components, load_applicability, load_checklist
from base_agent.logic.scoring import validate_score, calculate_overall_score
from base_agent.logic.report_generator import generate_report

logger = logging.getLogger(__name__)

MAX_WORKERS = 5          # 5 parallel Gemini calls avoids 429 rate limits (8 caused RESOURCE_EXHAUSTED)
MAX_RETRIES = 4          # One extra retry for transient 429s
RETRY_BASE_DELAY = 6     # Give Gemini breathing room between retries
RETRY_MAX_DELAY = 30     # Handle sustained rate limiting gracefully

_DOCUMENT_CACHE = {}
_GCS_EXTRACTED_BUCKET = "npc-extracted-docs"


# ═══════════════════════════════════════════════════════════════
#  PDF Extraction — Enterprise-proven pattern
# ═══════════════════════════════════════════════════════════════

def _part_mime(part) -> str:
    try:
        if isinstance(part, dict) and "inlineData" in part:
            return (part.get("inlineData") or {}).get("mimeType", "") or ""
        inline = getattr(part, "inline_data", None)
        if inline is not None:
            return getattr(inline, "mime_type", "") or ""
        return getattr(part, "mime_type", "") or ""
    except Exception:
        return ""


def _extract_bytes(part) -> bytes | None:
    if part is None:
        return None
    if isinstance(part, dict) and "inlineData" in part:
        try:
            b64_data = (part.get("inlineData") or {}).get("data")
            if b64_data:
                return base64.b64decode(b64_data)
        except Exception as e:
            logger.warning("Failed to decode inlineData base64: %s", e)
    try:
        inline = getattr(part, "inline_data", None)
        if inline is not None:
            data = getattr(inline, "data", None)
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
    except Exception:
        pass
    try:
        data = getattr(part, "data", None)
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
    except Exception:
        pass
    return None


def _text_quality_score(text: str) -> float:
    """Score extracted text quality 0.0 (garbage) to 1.0 (good).
    Detects Arabic PDFs with hidden Latin OCR layer (garbled short sequences)."""
    if not text or len(text.strip()) < 10:
        return 0.0
    stripped = text.strip()
    # Arabic / RTL content → always valid
    arabic_chars = sum(
        1 for c in stripped
        if ('\u0600' <= c <= '\u06FF') or ('\u0750' <= c <= '\u077F')
        or ('\uFB50' <= c <= '\uFDFF') or ('\uFE70' <= c <= '\uFEFF')
    )
    if arabic_chars > 20:
        return 1.0
    words = stripped.split()
    if len(words) < 4:
        return 0.1
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 2.2:
        return 0.05
    real_words = sum(1 for w in words if re.match(r'^[A-Za-z]{3,}$', w))
    real_ratio = real_words / len(words)
    if real_ratio < 0.20:
        return 0.15
    weird = sum(1 for c in stripped if ord(c) > 127 and not (
        '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F'
    ))
    weird_ratio = weird / max(len(stripped), 1)
    if weird_ratio > 0.30:
        return 0.20
    score = (real_ratio * 0.65) + (min(avg_word_len, 7.0) / 7.0 * 0.35)
    return round(min(score, 1.0), 3)


def _render_page_png(page, dpi: int = 150) -> bytes:
    """Render a fitz.Page to PNG bytes."""
    import fitz
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    return pix.tobytes("png")


def _vision_ocr_page(png_bytes: bytes, page_num: int) -> str:
    """Send rendered page to Gemini Vision → extract text (Arabic/English/mixed).
    Uses gemini-2.0-flash for speed — OCR doesn't need the full pro model."""
    try:
        client = _get_client()
        # Use flash model for OCR — much faster, cheaper, and just as accurate for text extraction
        ocr_model = "gemini-2.0-flash"
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        contents = {
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": b64}},
                {"text": (
                    "Extract ALL text visible on this document page exactly as written.\n"
                    "Preserve the original language (Arabic, English, or mixed).\n"
                    "Keep ALL headings, bullet points, tables, numbers, dates.\n"
                    "Preserve natural reading order. Output ONLY extracted text."
                )},
            ],
        }
        resp = client.models.generate_content(
            model=ocr_model, contents=contents,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        extracted = (resp.text or "").strip()
        logger.info("Vision OCR page %d: %d chars", page_num, len(extracted))
        return extracted
    except Exception as e:
        logger.error("Vision OCR failed page %d: %s", page_num, e)
        return ""


_QUALITY_THRESHOLD = 0.45
_MIN_CHARS_PAGE = 60
_OCR_BATCH_THRESHOLD = 0.40


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Smart extraction: PyMuPDF → quality check → Vision OCR fallback for bad pages.
    Handles Arabic PDFs with hidden garbled Latin OCR layer.
    Uses parallel Vision OCR for speed on large documents."""
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    # Phase 1: Fast PyMuPDF pass + quality scoring
    page_data = []  # (text, quality)
    for i in range(total_pages):
        raw = doc[i].get_text("text")
        quality = _text_quality_score(raw)
        page_data.append((raw, quality))

    bad_pages = [
        i for i, (text, q) in enumerate(page_data)
        if q < _QUALITY_THRESHOLD or len((text or "").strip()) < _MIN_CHARS_PAGE
    ]
    bad_ratio = len(bad_pages) / max(total_pages, 1)

    print(f"PDF quality scan: {len(bad_pages)}/{total_pages} pages below threshold ({bad_ratio*100:.0f}%)", flush=True)
    logger.info("PDF quality: %d/%d bad pages (%.0f%%)", len(bad_pages), total_pages, bad_ratio * 100)

    # Phase 2: Vision OCR for bad pages (or all pages if mostly bad)
    if bad_ratio >= _OCR_BATCH_THRESHOLD:
        target_pages = list(range(total_pages))
        print(f"Full Vision OCR pass ({total_pages} pages) — document has hidden/garbled text layer", flush=True)
    else:
        target_pages = bad_pages

    if target_pages:
        # Pre-render all target pages to PNG (fast, CPU-only)
        page_pngs = {}
        for idx in target_pages:
            try:
                page_pngs[idx] = _render_page_png(doc[idx])
            except Exception as e:
                logger.error("Render page %d failed: %s", idx + 1, e)

        doc.close()  # Close doc early — we have the PNGs now

        # Parallel Vision OCR (4 workers for flash model)
        _OCR_WORKERS = 4

        def _ocr_one(idx):
            png = page_pngs.get(idx)
            if not png:
                return idx, "", False
            txt = _vision_ocr_page(png, idx + 1)
            ok = bool(txt and len(txt.strip()) > 15)
            return idx, txt, ok

        ocr_count = 0
        with ThreadPoolExecutor(max_workers=_OCR_WORKERS) as ex:
            futs = {ex.submit(_ocr_one, idx): idx for idx in target_pages}
            for fut in as_completed(futs):
                try:
                    idx, txt, ok = fut.result()
                    if ok:
                        page_data[idx] = (txt, 1.0)
                        ocr_count += 1
                    print(f"  Vision OCR page {idx+1}/{total_pages}: {'OK' if ok else 'fallback'}", flush=True)
                except Exception as e:
                    idx = futs[fut]
                    logger.error("Vision OCR page %d failed: %s", idx + 1, e)
    else:
        doc.close()
        ocr_count = 0

    # Build final output with [PAGE X] markers
    pages = []
    for i, (text, quality) in enumerate(page_data):
        stripped = (text or "").strip()
        if stripped:
            pages.append(f"[PAGE {i + 1}]\n{stripped}")
        else:
            pages.append(f"[PAGE {i + 1}]\n[No text extracted]")

    print(f"Extraction complete: {total_pages} pages, {ocr_count} via Vision OCR", flush=True)
    return "\n\n".join(pages)


def _get_user_message_parts(tool_context) -> list:
    user_content = None
    for attr in ("user_content", "userContent"):
        if hasattr(tool_context, attr):
            val = getattr(tool_context, attr)
            user_content = val() if callable(val) else val
            break
    if user_content is None:
        return []
    parts = getattr(user_content, "parts", None)
    if parts is None:
        return []
    parts = parts() if callable(parts) else parts
    return list(parts) if parts else []


def _extract_pdf_from_user_message(tool_context) -> str | None:
    try:
        parts = _get_user_message_parts(tool_context)
        for idx, part in enumerate(parts):
            mime = _part_mime(part)
            if "pdf" not in mime.lower():
                continue
            pdf_bytes = _extract_bytes(part)
            if not pdf_bytes:
                continue
            text = _extract_text_from_pdf_bytes(pdf_bytes)
            if text and len(text) > 200:
                logger.info("Extracted %d chars from user message PDF part[%d]", len(text), idx)
                return text
        return None
    except Exception as e:
        logger.warning("User message PDF extraction failed: %s", e)
        return None


def _resolve_maybe_coroutine(val):
    import inspect
    if inspect.iscoroutine(val):
        import asyncio
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, val).result()
        except RuntimeError:
            return asyncio.run(val)
    return val


def _extract_pdf_from_artifacts(tool_context) -> str | None:
    try:
        if not hasattr(tool_context, 'list_artifacts'):
            return None
        keys = _resolve_maybe_coroutine(tool_context.list_artifacts())
        if not keys:
            return None
        for key in keys:
            try:
                part = _resolve_maybe_coroutine(tool_context.load_artifact(key))
                if part is None:
                    continue
                pdf_bytes = _extract_bytes(part)
                if pdf_bytes:
                    mime = _part_mime(part)
                    if "pdf" in (mime or "").lower() or "pdf" in key.lower():
                        text = _extract_text_from_pdf_bytes(pdf_bytes)
                        if text and len(text) > 200:
                            return text
            except Exception as e:
                logger.warning("Artifact '%s' failed: %s", key, e)
        return None
    except Exception as e:
        logger.error("Artifact extraction failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════
#  GCS Cache Helpers
# ═══════════════════════════════════════════════════════════════

def _get_session_id(tool_context) -> str:
    if tool_context is None:
        return f"fallback_{int(time.time())}"
    try:
        inv_ctx = getattr(tool_context, '_invocation_context', None) or getattr(tool_context, 'invocation_context', None)
        if inv_ctx:
            session = getattr(inv_ctx, 'session', None)
            if session:
                sid = getattr(session, 'id', None) or getattr(session, 'session_id', None)
                if sid:
                    return str(sid)
    except Exception:
        pass
    return f"session_{int(time.time())}"


def _upload_text_to_gcs(session_id: str, text: str) -> str | None:
    try:
        from google.cloud import storage as gcs_storage
        client = gcs_storage.Client()
        blob = client.bucket(_GCS_EXTRACTED_BUCKET).blob(f"extracted/{session_id}.txt")
        blob.upload_from_string(text, content_type="text/plain; charset=utf-8")
        logger.info("Uploaded %d chars to gs://%s/extracted/%s.txt", len(text), _GCS_EXTRACTED_BUCKET, session_id)
        return f"gs://{_GCS_EXTRACTED_BUCKET}/extracted/{session_id}.txt"
    except Exception as e:
        logger.error("GCS upload failed: %s", e)
        return None


def _download_text_from_gcs(session_id: str) -> str | None:
    try:
        from google.cloud import storage as gcs_storage
        blob = gcs_storage.Client().bucket(_GCS_EXTRACTED_BUCKET).blob(f"extracted/{session_id}.txt")
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")
    except Exception as e:
        logger.error("GCS download failed: %s", e)
        return None


# Report persistence — survives agent-framework timeouts
_REPORT_CACHE: dict = {}
_GCS_REPORTS_PREFIX = "reports"


def _save_report(session_id: str, report: str, criteria_summary: list, overall_score: float) -> None:
    """Save report to in-memory cache and GCS for retrieval if the main call times out."""
    _REPORT_CACHE["report"] = report
    _REPORT_CACHE["criteria_summary"] = criteria_summary
    _REPORT_CACHE["overall_score"] = overall_score
    _REPORT_CACHE["session_id"] = session_id
    try:
        from google.cloud import storage as gcs_storage
        payload = json.dumps({
            "report": report,
            "criteria_summary": criteria_summary,
            "overall_score": overall_score,
        }, ensure_ascii=False)
        client = gcs_storage.Client()
        blob = client.bucket(_GCS_EXTRACTED_BUCKET).blob(f"{_GCS_REPORTS_PREFIX}/{session_id}.json")
        blob.upload_from_string(payload, content_type="application/json; charset=utf-8")
        print(f"Report saved to GCS ({len(report):,} chars)", flush=True)
    except Exception as e:
        logger.error("Report GCS save failed: %s", e)


def _load_report(session_id: str) -> dict | None:
    """Load a saved report from cache or GCS."""
    if _REPORT_CACHE.get("session_id") == session_id and _REPORT_CACHE.get("report"):
        return _REPORT_CACHE
    try:
        from google.cloud import storage as gcs_storage
        blob = gcs_storage.Client().bucket(_GCS_EXTRACTED_BUCKET).blob(f"{_GCS_REPORTS_PREFIX}/{session_id}.json")
        if not blob.exists():
            return None
        data = json.loads(blob.download_as_text(encoding="utf-8"))
        return data
    except Exception as e:
        logger.error("Report GCS load failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════
#  Arabic Detection (Gemini reads Arabic natively — no translation)
# ═══════════════════════════════════════════════════════════════

def _is_arabic(text: str, threshold: float = 0.2) -> bool:
    if not text:
        return False
    sample = text[:15000]
    arabic = sum(1 for c in sample if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F'
                 or '\uFB50' <= c <= '\uFDFF' or '\uFE70' <= c <= '\uFEFF')
    alpha = sum(1 for c in sample if c.isalpha())
    return alpha > 50 and (arabic / alpha) > threshold


# ═══════════════════════════════════════════════════════════════
#  TOOL 1: extract_document
# ═══════════════════════════════════════════════════════════════

def extract_document(tool_context: ToolContext = None) -> dict:
    """Extract full text from uploaded PDF and cache it.
    Call FIRST before structural_review.
    """
    global _DOCUMENT_CACHE
    print("=== extract_document CALLED ===", flush=True)

    # FIX: Clear stale cache (module globals persist between sessions on Agent Engine)
    _DOCUMENT_CACHE.clear()

    if tool_context is None:
        return {"status": "error", "message": "No tool context.", "pages": 0, "characters": 0}

    extracted_text = None
    method = "none"

    # Method 0: Enterprise pre-extracted text
    try:
        parts = _get_user_message_parts(tool_context)
        all_text = []
        for part in parts:
            pt = getattr(part, 'text', '') or (part.get('text', '') if isinstance(part, dict) else '')
            if pt and '<start_of_user_uploaded_file' not in pt and '<end_of_user_uploaded_file' not in pt:
                if len(pt.strip()) > 500:
                    all_text.append(pt.strip())
        if all_text:
            extracted_text = '\n'.join(all_text)
            if len(extracted_text) > 500:
                method = "enterprise_text"
                print(f"SUCCESS enterprise text: {len(extracted_text)} chars", flush=True)
    except Exception as e:
        print(f"Enterprise text extraction failed: {e}", flush=True)

    # Method 1: Binary PDF from user_content (local ADK)
    if not extracted_text:
        extracted_text = _extract_pdf_from_user_message(tool_context)
        if extracted_text and len(extracted_text) > 200:
            method = "user_content_pdf"
            print(f"SUCCESS user_content PDF: {len(extracted_text)} chars", flush=True)

    # Method 2: Artifacts (ADK Web)
    if not extracted_text:
        extracted_text = _extract_pdf_from_artifacts(tool_context)
        if extracted_text and len(extracted_text) > 200:
            method = "artifacts"
            print(f"SUCCESS artifacts: {len(extracted_text)} chars", flush=True)

    if not extracted_text or len(extracted_text) <= 200:
        print("No binary PDF found", flush=True)
        return {"status": "no_file_found", "message": "Could not extract binary PDF. The document text will be passed directly to the review tool.", "pages": 0, "characters": 0}

    page_nums = re.findall(r'\[PAGE\s+(\d+)\]', extracted_text)
    max_page = max(int(p) for p in page_nums) if page_nums else 0

    session_id = _get_session_id(tool_context)
    _upload_text_to_gcs(session_id, extracted_text)
    _DOCUMENT_CACHE.update({"text": extracted_text, "source": method, "pages": max_page, "session_id": session_id})

    try:
        state = getattr(tool_context, 'state', None)
        if state is not None and hasattr(state, '__setitem__'):
            state["_extracted_session_id"] = session_id
    except Exception:
        pass

    logger.info("extract_document SUCCESS: %d pages, %d chars, method=%s", max_page, len(extracted_text), method)
    return {"status": "success", "message": f"Extracted {max_page} pages ({len(extracted_text):,} characters).", "pages": max_page, "characters": len(extracted_text), "method": method}


# ═══════════════════════════════════════════════════════════════
#  Gemini Client — always returns str
# ═══════════════════════════════════════════════════════════════

def _get_client():
    return genai.Client(vertexai=True, project=os.getenv("GOOGLE_CLOUD_PROJECT", ""), location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))


def _get_max_page(text: str) -> int:
    nums = re.findall(r'\[PAGE\s+(\d+)\]', text)
    return max(int(p) for p in nums) if nums else 0


def _extract_entity_name(text: str) -> str | None:
    m = re.search(r'(?:Ministry|Authority|Committee|Council|Bureau|Institute|Department)\s+(?:of|for)\s+[\w\s\-&]+', text[:5000], re.IGNORECASE)
    if m and len(m.group(0)) > 10:
        return m.group(0).strip()[:120]
    return None


def _extract_strategy_title(text: str) -> str | None:
    m = re.search(r'(?:National|Qatar)\s+[\w\s\-]+(?:Strategy|Plan|Framework|Roadmap|Policy|Vision)\s*(?:\d{4}[\s\-]*\d{0,4})?', text[:5000], re.IGNORECASE)
    if m and len(m.group(0)) > 10:
        return m.group(0).strip()[:200]
    return None


def _call_gemini(prompt, system_instruction=""):
    """Call Gemini. ALWAYS returns str — never None or list."""
    client = _get_client()
    model = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-pro")
    config = types.GenerateContentConfig(
        temperature=0.0,
        system_instruction=system_instruction if system_instruction else None,
    )
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            text = resp.text
            if text is None:
                return ""
            if isinstance(text, list):
                return "\n".join(str(t) for t in text)
            return str(text)
        except Exception as e:
            last_exc = e
            if any(kw in str(e).lower() for kw in ["429", "resource_exhausted", "500", "503", "unavailable", "timeout"]):
                delay = min(RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 2), RETRY_MAX_DELAY)
                logger.warning("Gemini retry %d/%d in %.1fs: %s", attempt + 1, MAX_RETRIES, delay, str(e)[:200])
                time.sleep(delay)
            else:
                raise
    raise last_exc


# ═══════════════════════════════════════════════════════════════
#  Evaluation Prompts — English + Arabic
# ═══════════════════════════════════════════════════════════════

SYSTEM_INSTRUCTION = (
    "You are an expert strategy document reviewer for Qatar's National Planning Council. "
    "You evaluate strategy documents against a structural checklist with precise scoring rubrics. "
    "Be thorough, accurate, and evidence-based. Cite page numbers using [PAGE X] markers.\n\n"
    "FORMATTING RULES FOR ALL OUTPUT FIELDS:\n"
    "- ALL output fields (evidence, reasoning, gap_to_next, recommendation, evidence_summary) "
    "must be PLAIN TEXT strings. Never wrap them in JSON objects or arrays.\n"
    "- Never output curly braces {} or square brackets [] inside any text field "
    "(except for [PAGE X] citations).\n"
    "- For the 'evidence' field: present each piece of evidence as a separate quoted line. "
    "Use this exact format for each piece:\n"
    '  "Quoted text from document" [PAGE X]\n'
    "- If the document contains a TABLE, reproduce the table content as structured lines, "
    "with each row on its own line using pipe separators: Column1 | Column2 | Column3. "
    "Do NOT collapse table rows into a single run-on paragraph.\n"
    "- For the 'evidence_summary' field: write a concise paragraph summarizing what was found. "
    "Never use JSON format.\n\n"
    "TERMINOLOGY ACCURACY:\n"
    "- When describing where evidence appears in the document, use PRECISE terminology:\n"
    "  * 'Objective' = a stated strategic objective or goal of the strategy\n"
    "  * 'Outcome' or 'Key outcome' = a result or deliverable listed under a project\n"
    "  * 'Initiative' or 'Project' = a named program of work\n"
    "  * 'Risk' = an identified risk in the risk section\n"
    "  * 'Mitigation' = an action to address a risk\n"
    "- NEVER call a project outcome or deliverable a 'goal'. Use 'outcome', 'deliverable', or 'result'.\n"
    "- NEVER call a risk mitigation action an 'initiative'. Use 'mitigation measure' or 'mitigation action'.\n\n"
    "TABLE AND STRUCTURED CONTENT HANDLING:\n"
    "- Pay special attention to tables, especially on the LAST pages of the document. "
    "Tables near the end often contain roadmaps, risk matrices, KPI summaries, or budget tables.\n"
    "- When a table row contains multiple columns, read EACH column carefully and attribute "
    "content to the correct column header. Do not conflate column values.\n"
    "- In risk tables: distinguish between 'Risk Description' column and 'Mitigation' column.\n"
    "- In project tables: distinguish between 'Project Name', 'Owner', 'Timeline', and 'Budget' columns.\n"
    "- EVIDENCE FROM TABLES: When quoting evidence from a table, extract ONLY the meaningful "
    "cell values. NEVER include column headers (e.g., 'Supporting Agency', 'Program', 'Project', "
    "'Action', 'Goal', 'Conclusion', 'Output') in the evidence text. Instead, summarize the row "
    "content as a readable sentence. For example, instead of 'n — Supporting Agency — Program — "
    "Project — Action — Goal — Ministry of X — Law Enforcement — Speed Control — ...', write: "
    "'Ministry of X: Law Enforcement — Speed Control — ...' \n\n"
    "Match evidence to the rubric level it genuinely fits. "
    "GROUNDING RULE: ONLY cite text that ACTUALLY appears verbatim in the document. "
    "NEVER fabricate content. If evidence is absent, say so. "
    "Respond with valid JSON only."
)

SYSTEM_INSTRUCTION_ARABIC = (
    "You are an expert strategy document reviewer for Qatar's National Planning Council. "
    "The document you will evaluate is written in ARABIC. You can read Arabic natively. "
    "Evaluate the Arabic text directly — do NOT say you cannot read it.\n\n"
    "CRITICAL LANGUAGE RULES:\n"
    "- The 'evidence' field: quotes in ORIGINAL ARABIC text exactly as written in the document.\n"
    "- ALL OTHER FIELDS MUST BE IN ENGLISH. This is mandatory and non-negotiable:\n"
    "  * 'reasoning' → ENGLISH\n"
    "  * 'recommendation' → ENGLISH\n"
    "  * 'gap_to_next' → ENGLISH\n"
    "  * 'evidence_summary' → ENGLISH\n"
    "  * 'rubric_walkthrough' → ENGLISH\n"
    "  * 'selected_score_justification' → ENGLISH\n"
    "- Even though the source document is Arabic, your analysis MUST be written in English.\n"
    "- If you find yourself writing Arabic outside the 'evidence' field, STOP and rewrite in English.\n"
    "- Cite page numbers using [PAGE X] markers.\n\n"
    "FORMATTING RULES FOR ALL OUTPUT FIELDS:\n"
    "- ALL output fields must be PLAIN TEXT strings. Never wrap them in JSON objects or arrays.\n"
    "- Never output curly braces {} or square brackets [] inside any text field "
    "(except for [PAGE X] citations).\n"
    "- For the 'evidence' field: present each piece of evidence as a separate quoted line. "
    "Use this exact format for each piece of Arabic evidence:\n"
    '  "Arabic quoted text from document" [PAGE X]\n'
    "  Separate each quote with a blank line.\n"
    "- If the document contains a TABLE, reproduce the table content as structured lines. "
    "Use pipe separators for columns: Column1 | Column2 | Column3. "
    "Each row goes on its own line. Do NOT collapse table rows into a single paragraph.\n"
    "- For the 'evidence_summary' field: write a concise paragraph IN ENGLISH summarizing "
    "what was found. Never use JSON format.\n\n"
    "TERMINOLOGY ACCURACY:\n"
    "- When describing document content in English fields, use PRECISE terminology:\n"
    "  * 'Objective' = a stated strategic objective (هدف استراتيجي)\n"
    "  * 'Outcome' or 'Key outcome' = a result or deliverable (نتيجة / مخرج)\n"
    "  * 'Initiative' or 'Project' = a named program (مبادرة / مشروع)\n"
    "  * 'Risk' = an identified risk (مخاطر)\n"
    "  * 'Mitigation measure' = a risk mitigation action (إجراء التخفيف)\n"
    "- NEVER call a project outcome a 'goal'. Use 'outcome', 'deliverable', or 'result'.\n"
    "- NEVER call a risk mitigation an 'initiative'. Use 'mitigation measure'.\n\n"
    "TABLE AND STRUCTURED CONTENT HANDLING:\n"
    "- Pay special attention to tables, especially on the LAST pages of the document. "
    "Arabic documents often place roadmaps, risk matrices, and KPI summaries at the end.\n"
    "- When reading Arabic tables, respect the right-to-left column order.\n"
    "- Read EACH column carefully and attribute content to the correct column header.\n"
    "- In risk tables: distinguish between the 'Risk' column and 'Mitigation' column.\n"
    "- In project tables: distinguish between 'Project Name', 'Owner', 'Timeline' columns.\n"
    "- EVIDENCE FROM TABLES: When quoting evidence from a table, extract ONLY the meaningful "
    "cell values. NEVER include column headers (e.g., 'الجهة الداعمة', 'البرنامج', 'المشروع', "
    "'الإجراء', 'الهدف', 'الخلاصة', 'المخرج', or their English equivalents 'Supporting Agency', "
    "'Program', 'Project', 'Action', 'Goal') in the evidence text. Summarize the row content "
    "as a readable sentence using only the cell values.\n\n"
    "Match evidence to the rubric level it genuinely fits. "
    "GROUNDING RULE: ONLY cite text that ACTUALLY appears verbatim in the document. "
    "NEVER fabricate content. If evidence is absent, say so. "
    "Respond with valid JSON only. ALL JSON field values except 'evidence' MUST be in ENGLISH."
)

SCOPE_BOUNDARIES = {
    "1.1": "SCOPE: DEDICATED diagnostic/situational analysis (SWOT, gap analysis). Different from Criteria 7 (Risks).",
    "1.2": "SCOPE: EXPLICIT data-driven comparisons against NAMED peer countries. Generic references = 0.25 max.",
    "1.3": "SCOPE: FORWARD-LOOKING trends (megatrends, tech, demographics). Historical = 1.1. Mentioned=0.25; connected to interventions=0.5.",
    "2.1": "SCOPE: VISION STATEMENT — clarity, ambition, specificity, national alignment. Vague=0.25; stated but unspecific=0.5.",
    "2.2": "SCOPE: MISSION STATEMENT — Purpose, Scope, Beneficiaries. All three with minor gaps=0.75. One missing=0.5 max.",
    "2.3": "SCOPE: DEFINED VALUES/principles and operational relevance.",
    "3.1": "SCOPE: STRATEGIC OBJECTIVES/PILLARS (not KPIs). Measurability and vision linkage.",
    "3.2": "SCOPE: KPI DEFINITIONS — methodology, cadence, source, calculation. Just names=0.25; some methodology=0.5.",
    "3.3": "SCOPE: QUANTIFIED BASELINES and TIME-BOUND TARGETS. Partial=0.5.",
    "3.4": "SCOPE: SMART KPIs. Names without methodology NOT SMART. Deliverables are NOT KPIs.",
    "4.1": "SCOPE: INTERNAL governance — committees, roles, reporting within implementing bodies.",
    "4.2": "SCOPE: EXTERNAL stakeholders. Named roles per action = mapping (0.75+). Needs engagement mechanisms for 1.0.",
    "4.3": "SCOPE: COORDINATION MECHANISMS — cadence, reporting, escalation, accountability.",
    "5.1": "SCOPE: TRANSFORMATIVE initiatives linked to objectives. Routine ops don't qualify.",
    "5.2": "SCOPE: Hierarchy: initiatives → projects with traceable linkage.",
    "6.1": "SCOPE: DESIGNATED OWNERS per project. Clear single owner=1.0; ambiguous=0.75 max.",
    "6.2": "SCOPE: DETAILED project-level timelines. Strategy-level only=0.25.",
    "6.3": "SCOPE: QUANTIFIED budgets. 'Will seek funding'=NOT a budget. No figures=0.0.",
    "7.1": "SCOPE: DEDICATED risk section for IMPLEMENTATION risks. Challenges=diagnostic (Criteria 1). No risk section=0.0.",
    "7.2": "SCOPE: EXPLICIT mitigation for risks. If 7.1=0.0, this MUST=0.0. Projects are NOT mitigations.",
}


def _build_sub_component_prompt(sub_info, classification, document_text, max_page=0, text_source="pymupdf", is_arabic=False):
    sub_id = sub_info["sub_component_id"]
    sub_name = sub_info["sub_component_name"]
    rubric_text = "\n".join(f"  {k}: {v}" for k, v in sub_info["rubric"].items())
    scope = SCOPE_BOUNDARIES.get(sub_id, "")
    scope_block = f"\n\nSCOPE BOUNDARY:\n{scope}" if scope else ""
    cond = "\n\nCONDITIONAL: If no relevant content, respond with applicable=false." if sub_info.get("is_conditional") else ""
    warn = "*** WARNING: Agent-provided text may be incomplete. Score CONSERVATIVELY. ***\n\n" if text_source == "agent" else ""

    arabic_note = ""
    if is_arabic:
        arabic_note = (
            "ARABIC DOCUMENT — LANGUAGE REQUIREMENTS:\n"
            "This document is in ARABIC. Read it directly in Arabic — do NOT say you cannot read it.\n"
            "- 'evidence' field: Quote in ORIGINAL ARABIC exactly as it appears in the document.\n"
            "- ALL OTHER FIELDS MUST BE IN ENGLISH — this is mandatory:\n"
            "  reasoning, recommendation, gap_to_next, evidence_summary, rubric_walkthrough\n"
            "- Do NOT write Arabic in any field other than 'evidence'.\n"
            "- Present each Arabic quote on its own line with [PAGE X] citation.\n"
            "- When quoting from Arabic tables, use pipe separators for row structure.\n\n"
        )

    return (
        f"Evaluating {sub_id}: {sub_name}\nCriteria: {sub_info['component_name']}\nClassification: {classification}\n\n"
        f"QUESTION:\n{sub_info.get('question', '')}\n\nRUBRIC:\n{rubric_text}{cond}{scope_block}\n\n"
        f"{arabic_note}"
        f"{warn}"
        "INSTRUCTIONS:\n"
        "1. Read the ENTIRE document carefully, including tables on the last pages.\n"
        "2. Extract evidence with [PAGE X] citations.\n"
        "3. Walk through each rubric level from 0.0 to 1.0 and assess which level the evidence supports.\n"
        "4. Score = the highest level that is FULLY supported by evidence.\n"
        "5. Scores should vary across sub-criteria — do NOT default to one score.\n"
        "6. Score guide: 0.0=Absent, 0.25=Vague/mentioned, 0.5=Partial/incomplete, 0.75=Good with minor gaps, 1.0=Complete.\n"
        "7. Explain clearly: what was found, what rubric level it matches, what is needed for higher scores.\n"
        "8. gap_to_next: describe ALL gaps remaining to reach 1.0 (not just the next level).\n"
        f"9. Valid page range: 1 to {max_page}. Never reference PAGE 0.\n"
        "10. Every quote in evidence MUST appear verbatim in the document. Never fabricate.\n\n"
        "EVIDENCE FORMAT RULES:\n"
        "- Present evidence as a plain text string with quoted passages.\n"
        "- Each quote on its own line: \"quoted text\" [PAGE X]\n"
        "- For tables, use pipe-separated format:\n"
        "  Header1 | Header2 | Header3 [PAGE X]\n"
        "  Value1 | Value2 | Value3\n"
        "- NEVER use JSON objects or arrays inside the evidence field.\n"
        "- NEVER use curly braces {} in any text field.\n\n"
        "TERMINOLOGY RULES:\n"
        "- Project outcomes/deliverables → call them 'outcome' or 'deliverable', NEVER 'goal'\n"
        "- Risk mitigation actions → call them 'mitigation measure', NEVER 'initiative'\n"
        "- Strategic objectives → call them 'objective' or 'strategic objective'\n"
        "- Use precise document structure terms: 'section', 'table', 'chapter', 'annex'\n\n"
        f'Respond ONLY with valid JSON in this exact structure:\n'
        f'{{"sub_component_id":"{sub_id}","sub_component_name":"{sub_name}",'
        '"applicable":true,'
        '"rubric_walkthrough":{"0.0_assessment":"...","0.25_assessment":"...","0.5_assessment":"...",'
        '"0.75_assessment":"...","1.0_assessment":"...","selected_score_justification":"..."},'
        '"score":0.0,'
        '"pages":[1],'
        '"evidence":"Quote1 [PAGE X]\\n\\nQuote2 [PAGE Y]",'
        '"evidence_summary":"Concise English paragraph summarizing findings.",'
        '"reasoning":"Detailed English explanation of score rationale.",'
        '"gap_to_next":"English description of all gaps to reach 1.0.",'
        '"recommendation":"English recommendation for improvement."}\n\n'
        f"DOCUMENT:\n{document_text}"
    )


# ═══════════════════════════════════════════════════════════════
#  JSON Extraction & Parsing — crash-proof
# ═══════════════════════════════════════════════════════════════

def _extract_json(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()
    if text.startswith("`"):
        lines = text.split("\n")
        if lines[0].startswith("`"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("`"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    idx = text.find("{")
    if idx == -1:
        return None
    text = text[idx:]
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"' and not esc:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[:i + 1]
    return None


def _sanitize_text_field(value: str) -> str:
    """Clean up a text field by removing residual JSON artifacts, stray braces,
    and ensuring consistent formatting."""
    if not value or not isinstance(value, str):
        return value or ""

    s = value.strip()

    # Remove wrapping quotes if the entire string is double-quoted
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()

    # Remove fenced code blocks that leaked through
    if s.startswith("```"):
        s = re.sub(r'```\w*\n?', '', s).strip()

    # Remove stray JSON-like wrappers: {"text": "actual content"}
    if s.startswith('{') and s.endswith('}'):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                # Extract the best text value
                for key in ("text", "content", "value", "summary", "assessment",
                            "reasoning", "recommendation", "evidence", "comment"):
                    if key in obj and isinstance(obj[key], str):
                        s = obj[key].strip()
                        break
                else:
                    # Take first string value
                    for v in obj.values():
                        if isinstance(v, str) and v.strip():
                            s = v.strip()
                            break
        except json.JSONDecodeError:
            pass

    # Remove orphaned curly braces that aren't part of [PAGE X] citations
    # First protect [PAGE X] markers
    s = re.sub(r'\{([^}]*)\}', lambda m: m.group(1) if '[PAGE' not in m.group(0) else m.group(0), s)

    # Clean up any remaining naked braces not inside PAGE markers
    # Remove lone { or } that don't form valid structures
    result = []
    i = 0
    while i < len(s):
        if s[i] == '{' and not (i + 1 < len(s) and s[i + 1:].lstrip().startswith('"')):
            # Lone opening brace — skip it
            i += 1
            continue
        elif s[i] == '}':
            # Lone closing brace — skip it
            i += 1
            continue
        else:
            result.append(s[i])
            i += 1
    s = ''.join(result)

    # Normalize multiple blank lines to max 2
    s = re.sub(r'\n{3,}', '\n\n', s)

    return s.strip()


def _parse_sub_response(resp, sub_info, max_page=0):
    """Outer crash-proof wrapper."""
    try:
        return _parse_sub_response_inner(resp, sub_info, max_page)
    except Exception as e:
        logger.error("Parse crash %s: %s\n%s", sub_info.get("sub_component_id", "?"), e, traceback.format_exc())
        return _fallback_sub(sub_info)


def _parse_sub_response_inner(resp, sub_info, max_page=0):
    if not isinstance(resp, str):
        resp = str(resp) if resp is not None else ""
    j = _extract_json(resp)
    if not j:
        return _fallback_sub(sub_info)
    try:
        r = json.loads(j)
    except json.JSONDecodeError:
        return _fallback_sub(sub_info)

    if r.get("applicable", True) and r.get("score") is not None:
        try:
            r["score"] = validate_score(float(r["score"]))
        except (ValueError, TypeError):
            r["score"] = 0.0

    # Normalize ALL text fields to clean strings
    text_fields = ("evidence", "reasoning", "gap_to_next", "recommendation", "evidence_summary")
    for f in text_fields:
        val = r.get(f)
        if val is None:
            r[f] = ""
        elif isinstance(val, list):
            # Convert list to properly formatted text
            parts = []
            for item in val:
                if isinstance(item, dict):
                    # Handle {"quote": "...", "page": X} format
                    quote = item.get("quote", "")
                    page = item.get("page")
                    text_val = item.get("text", "")
                    if quote and page is not None:
                        parts.append(f'"{quote}" [PAGE {page}]')
                    elif quote:
                        parts.append(f'"{quote}"')
                    elif text_val:
                        parts.append(str(text_val))
                    else:
                        # Extract any string value
                        for v in item.values():
                            if isinstance(v, str) and v.strip():
                                parts.append(v.strip())
                                break
                elif isinstance(item, str):
                    parts.append(item.strip())
                else:
                    parts.append(str(item))
            r[f] = "\n\n".join(p for p in parts if p)
        elif isinstance(val, dict):
            # Handle dict returned instead of string
            for key in ("text", "content", "value", "summary", "assessment"):
                if key in val and isinstance(val[key], str):
                    r[f] = val[key].strip()
                    break
            else:
                # Fallback: join all string values
                parts = [str(v).strip() for v in val.values() if v]
                r[f] = "\n".join(parts)
        elif not isinstance(val, str):
            r[f] = str(val)

    # Apply sanitization to all text fields
    for f in text_fields:
        r[f] = _sanitize_text_field(r[f])

    # Safety net for any other unexpected list/dict fields
    for k, v in list(r.items()):
        if k in text_fields:
            continue  # Already handled above
        if isinstance(v, list) and k != "pages":
            r[k] = "\n".join(str(e) for e in v)
        elif isinstance(v, dict) and k not in ("rubric_walkthrough",):
            r[k] = str(v)

    # Normalize pages
    if isinstance(r.get("pages"), list):
        r["pages"] = [max(1, min(int(p), max_page if max_page > 0 else 9999))
                       for p in r["pages"] if isinstance(p, (int, float)) and int(p) >= 0]
    elif r.get("pages") is not None:
        v = r["pages"]
        r["pages"] = [max(1, int(v))] if isinstance(v, (int, float)) and v > 0 else []
    else:
        r["pages"] = []

    # Clamp page references in text fields
    for f in text_fields:
        val = r.get(f, "")
        if not isinstance(val, str):
            val = str(val)
        val = re.sub(r'\[PAGE\s+0\]', '[PAGE 1]', val)
        if max_page > 0:
            val = re.sub(
                r'\[PAGE\s+(\d+)\]',
                lambda m: "[PAGE 1]" if int(m.group(1)) < 1
                          else f"[PAGE {max_page}]" if int(m.group(1)) > max_page
                          else m.group(0),
                val)
        r[f] = val

    r.pop("rubric_walkthrough", None)

    # --- Arabic language enforcement ---
    # If non-evidence fields came back mostly in Arabic, flag as parse failure
    # so _evaluate_sub retries with reinforced English instructions.
    english_fields = ("reasoning", "recommendation", "gap_to_next", "evidence_summary")
    arabic_char_count = 0
    total_char_count = 0
    for f in english_fields:
        val = r.get(f, "")
        if isinstance(val, str):
            total_char_count += len(val)
            arabic_char_count += sum(1 for c in val if '\u0600' <= c <= '\u06ff')
    if total_char_count > 50 and arabic_char_count / total_char_count > 0.3:
        logger.warning("Non-evidence fields are >30%% Arabic (%d/%d chars) — flagging for retry",
                        arabic_char_count, total_char_count)
        r["_evaluation_error"] = True

    return r


def _is_fallback_result(result: dict) -> bool:
    """Check if a result is a fallback (evaluation error) rather than a real evaluation."""
    return result.get("_evaluation_error", False)


def _fallback_sub(si):
    return {"sub_component_id": si["sub_component_id"], "sub_component_name": si["sub_component_name"],
            "applicable": not si.get("is_conditional", False), "score": 0.0, "pages": [],
            "evidence": "", "evidence_summary": "",
            "reasoning": "Automatic evaluation could not be completed for this sub-component. The model's response could not be parsed into a valid assessment. A manual review is required to properly score this criterion.",
            "gap_to_next": "", "recommendation": "A manual review of this sub-component is required. The automated evaluation was unable to generate a valid assessment. Please review the source document against the rubric criteria and assign an appropriate score.",
            "_evaluation_error": True}


def _evaluate_sub(si, cls, doc, mp=0, ts="pymupdf", is_arabic=False):
    """Evaluate one sub-criteria. NEVER raises — returns fallback on any error.
    Retries once with simplified prompt if first attempt fails to parse."""
    sid = si["sub_component_id"]
    logger.info("Evaluating %s (arabic=%s)", sid, is_arabic)

    for attempt in range(2):
        try:
            sys_instr = SYSTEM_INSTRUCTION_ARABIC if is_arabic else SYSTEM_INSTRUCTION
            prompt = _build_sub_component_prompt(si, cls, doc, mp, ts, is_arabic=is_arabic)

            if attempt == 1:
                # Retry with extra emphasis on valid JSON output and English
                logger.info("Retrying %s with reinforced instructions", sid)
                prompt += (
                    "\n\nIMPORTANT RETRY NOTE: Your previous response could not be parsed. "
                    "You MUST respond with ONLY a single valid JSON object. "
                    "Do not include any text before or after the JSON. "
                    "Do not use markdown code fences. Just output the raw JSON object. "
                    "ALL fields except 'evidence' MUST be written in ENGLISH. "
                    "Do NOT write Arabic in reasoning, recommendation, gap_to_next, or evidence_summary."
                )

            raw = _call_gemini(prompt, system_instruction=sys_instr)
            r = _parse_sub_response(raw, si, mp)

            # Check if parse produced a fallback result
            if _is_fallback_result(r) and attempt == 0:
                logger.warning("Parse fallback for %s on attempt 1, retrying...", sid)
                time.sleep(2)
                continue  # Retry

            r["sub_component_id"] = sid
            r["sub_component_name"] = si["sub_component_name"]
            logger.info("Done %s: score=%s (attempt %d)", sid, r.get("score"), attempt + 1)
            return r
        except Exception as e:
            logger.error("Failed %s (attempt %d): %s\n%s", sid, attempt + 1, e, traceback.format_exc())
            if attempt == 0:
                time.sleep(2)
                continue  # Retry on exception too

    # All attempts failed
    logger.error("All attempts failed for %s, returning fallback", sid)
    fb = _fallback_sub(si)
    fb["sub_component_id"] = sid
    fb["sub_component_name"] = si["sub_component_name"]
    return fb


def _generate_local_comment(comp_name: str, subs: list[dict]) -> str:
    """Generate a component-level comment from sub-component scores. No LLM needed."""
    scored = [(s["sub_component_name"], s.get("score", 0) or 0)
              for s in subs if s.get("applicable", True) and s.get("score") is not None and not s.get("_evaluation_error")]
    has_errors = any(s.get("_evaluation_error") for s in subs)

    if not scored:
        return f"Provides absent coverage of {comp_name.lower()}." if not has_errors else (
            f"Coverage of {comp_name.lower()} could not be determined due to evaluation errors.")

    avg = sum(sc for _, sc in scored) / len(scored)
    lvl = "complete" if avg >= 0.875 else "high" if avg >= 0.625 else "moderate" if avg >= 0.375 else "low" if avg >= 0.125 else "absent"

    strengths = [name for name, sc in scored if sc >= 0.75]
    gaps = [name for name, sc in scored if sc < 0.75]

    parts = [f"Provides {lvl} coverage"]
    if strengths and gaps:
        parts.append(f", with strong performance in {', '.join(strengths[:2]).lower()}")
        parts.append(f" but gaps in {', '.join(gaps[:2]).lower()}")
    elif strengths:
        parts.append(f" across all assessed sub-components")
    elif gaps:
        parts.append(f", with gaps across {', '.join(gaps[:2]).lower()}")
    parts.append(".")

    if has_errors:
        parts.append(" Note: one or more sub-components could not be scored and require manual review.")

    return "".join(parts)


def _add_na_subs(results, cls, comp_subs):
    na_ids = set(load_applicability(cls)["not_applicable"])
    existing = {s["sub_component_id"] for s in results}
    names = {s["id"]: s["name"] for s in comp_subs}
    for sd in comp_subs:
        sid = sd["id"]
        if sid in na_ids and sid not in existing:
            results.append({"sub_component_id": sid, "sub_component_name": names.get(sid, sid),
                           "applicable": False, "score": None, "pages": [], "evidence": "", "evidence_summary": "",
                           "reasoning": "", "gap_to_next": "", "recommendation": "",
                           "na_reason": f"Not required for {cls.capitalize()} strategies"})
    results.sort(key=lambda x: x["sub_component_id"])
    return results


# ═══════════════════════════════════════════════════════════════
#  Criteria Summary Builder (defensive imports)
# ═══════════════════════════════════════════════════════════════

def _build_criteria_summary(comp_results: list, classification: str) -> list:
    """Build criteria-level summary with inline scoring (no runtime imports)."""

    def _calc_band(avg):
        if avg >= 0.875:
            return "100%"
        if avg >= 0.625:
            return "75%"
        if avg >= 0.375:
            return "50%"
        if avg >= 0.125:
            return "25%"
        return "0%"

    def _band_label(band):
        return {"100%": "Complete", "75%": "High", "50%": "Moderate", "25%": "Low", "0%": "Absent"}.get(band, "Absent")

    summary = []
    for comp in comp_results:
        sub_results = comp.get("sub_results", [])
        applicable_scores = [s["score"] for s in sub_results if s.get("applicable", True) and s.get("score") is not None]
        if applicable_scores:
            raw_avg = sum(applicable_scores) / len(applicable_scores)
            band = _calc_band(raw_avg)
            label = _band_label(band)
        else:
            band = "0%"
            label = "Absent"

        observation = comp.get("comment", "")
        recs = []
        for sub in sub_results:
            if (sub.get("applicable", True) and sub.get("score") is not None
                    and sub["score"] < 1.0 and sub.get("recommendation")):
                rec_text = sub["recommendation"].strip()
                if rec_text:
                    recs.append(f"[{sub['sub_component_id']}] {rec_text}")

        summary.append({
            "id": comp["id"], "name": comp["name"], "score": band, "label": label,
            "observation": observation,
            "recommendation": " | ".join(recs) if recs else "No recommendations — criteria fully met.",
        })
    return summary


# ═══════════════════════════════════════════════════════════════
#  TOOL 2: structural_review
# ═══════════════════════════════════════════════════════════════

def structural_review(
    document_text: str = "",
    classification: str = "",
    strategy_title: str = "Untitled Strategy",
    entity_name: str = "Unknown Entity",
    tool_context: ToolContext = None,
) -> dict:
    """Full structural checklist review. Call extract_document FIRST."""
    try:
        if not classification or not classification.strip():
            return {"error": True, "message": "Missing classification. Provide: entity, sectoral, or thematic."}
        classification = classification.lower().strip()
        if classification not in ("entity", "sectoral", "thematic"):
            return {"error": True, "message": f"Invalid classification: {classification}."}

        logger.info("structural_review: cls=%s, agent_text=%d", classification, len(document_text or ""))
        text_source = "agent"

        # Priority 1: In-memory cache (set by extract_document in same session)
        cached = _DOCUMENT_CACHE.get("text", "")
        if cached and len(cached) > 500:
            document_text, text_source = cached, f"cache_{_DOCUMENT_CACHE.get('source', '?')}"

        # Priority 2: GCS by session_id
        if text_source == "agent":
            gcs = _download_text_from_gcs(_get_session_id(tool_context))
            if gcs and len(gcs) > 500:
                document_text, text_source = gcs, "gcs_cache"

        # Priority 3: GCS via session state
        if text_source == "agent" and tool_context:
            try:
                ssid = getattr(tool_context, 'state', {}).get("_extracted_session_id", "")
                if ssid:
                    st = _download_text_from_gcs(ssid)
                    if st and len(st) > 500:
                        document_text, text_source = st, "gcs_state"
            except Exception:
                pass

        # Priority 4: Direct extraction (last resort)
        if text_source == "agent" and tool_context:
            logger.info("Cache miss — direct extraction...")
            ext = _extract_pdf_from_user_message(tool_context) or _extract_pdf_from_artifacts(tool_context)
            if ext and len(ext) > 500:
                document_text, text_source = ext, "direct"
                _upload_text_to_gcs(_get_session_id(tool_context), ext)
                _DOCUMENT_CACHE["text"] = ext

        logger.info("Source: %s (%d chars)", text_source, len(document_text or ""))

        if not document_text or len(document_text.strip()) < 100:
            return {"error": True, "message": "No document text. Upload PDF and call extract_document first."}

        # === Arabic Detection — NO TRANSLATION (Gemini reads Arabic natively) ===
        is_arabic_doc = _is_arabic(document_text)
        if is_arabic_doc:
            logger.info("Arabic document detected (%d chars) — will use Arabic-aware prompts", len(document_text))
            print("Arabic document detected — using native Arabic evaluation (no translation needed)", flush=True)

        # Extract metadata
        if entity_name in ("Unknown Entity", "", None):
            entity_name = _extract_entity_name(document_text) or "Unknown Entity"
        if strategy_title in ("Untitled Strategy", "", None):
            strategy_title = _extract_strategy_title(document_text) or "Untitled Strategy"

        # Inject page markers if missing
        max_page = _get_max_page(document_text)
        if max_page == 0:
            lines, new, cc, pn = document_text.split('\n'), ["[PAGE 1]"], 0, 1
            for line in lines:
                cc += len(line) + 1
                new.append(line)
                if cc > 3000:
                    pn += 1
                    new.append(f"\n[PAGE {pn}]")
                    cc = 0
            document_text = '\n'.join(new)
            max_page = _get_max_page(document_text)

        subs = get_applicable_sub_components(classification)
        logger.info("Evaluating %d sub-criteria (%d chars, arabic=%s)", len(subs), len(document_text), is_arabic_doc)
        print(f"Starting evaluation: {len(subs)} sub-criteria, {len(document_text):,} chars, arabic={is_arabic_doc}", flush=True)
        t_eval_start = time.time()

        # === Parallel evaluation with is_arabic flag ===
        # Stagger submissions slightly to avoid burst rate-limiting
        results = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {}
            for i, si in enumerate(subs):
                futs[ex.submit(_evaluate_sub, si, classification, document_text, max_page, text_source, is_arabic_doc)] = si
                if i < len(subs) - 1 and (i + 1) % MAX_WORKERS == 0:
                    time.sleep(1.0)  # Brief pause between batches
            for fut in as_completed(futs):
                si = futs[fut]
                try:
                    results[si["sub_component_id"]] = fut.result()
                except Exception as e:
                    logger.error("Failed %s: %s", si["sub_component_id"], e)
                    results[si["sub_component_id"]] = _fallback_sub(si)

        print(f"Sub-criteria evaluated in {time.time() - t_eval_start:.1f}s", flush=True)

        # === Build component results (local comments — no LLM needed) ===
        t_comp = time.time()
        checklist_comps = load_checklist(classification)["components"]
        comp_results = []
        for comp in checklist_comps:
            csubs = [results[s["id"]] for s in comp["sub_components"] if s["id"] in results]
            csubs = _add_na_subs(csubs, classification, comp["sub_components"])
            comment = _generate_local_comment(comp["name"], csubs)
            comp_results.append({"id": comp["id"], "name": comp["name"], "comment": comment, "sub_results": csubs})

        print(f"Component comments generated in {time.time() - t_comp:.1f}s", flush=True)
        try:
            t_report = time.time()
            report = generate_report(strategy_title=strategy_title, entity_name=entity_name, classification=classification, component_results=comp_results)
            print(f"Report generated in {time.time() - t_report:.1f}s ({len(report):,} chars)", flush=True)
        except Exception as e:
            logger.exception("generate_report failed")
            report = f"# Report Generation Error\n\nCriteria were evaluated but report formatting failed: {e}"

        all_s = []
        for c in comp_results:
            for s in c.get("sub_results", []):
                if s.get("applicable", True) and s.get("score") is not None and not s.get("_evaluation_error"):
                    all_s.append(s["score"])

        criteria_summary = _build_criteria_summary(comp_results, classification)
        print(f"\n=== CRITERIA SUMMARY ===\n{json.dumps(criteria_summary, indent=2)}\n{'=' * 40}", flush=True)

        # Count evaluation errors for reporting
        eval_errors = sum(1 for c in comp_results for s in c.get("sub_results", []) if s.get("_evaluation_error"))
        if eval_errors:
            print(f"WARNING: {eval_errors} sub-component(s) could not be evaluated automatically.", flush=True)

        # Persist report to GCS and cache so it survives agent-framework timeouts
        overall = calculate_overall_score(all_s)
        session_id = _get_session_id(tool_context)
        _save_report(session_id, report, criteria_summary, overall)

        total_elapsed = time.time() - t_eval_start
        print(f"Total review pipeline: {total_elapsed:.1f}s", flush=True)

        return {
            "error": False,
            "overall_score": overall,
            "criteria_summary": criteria_summary,
            "report": report,
        }

    except Exception as e:
        logger.exception("Review failed")
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg.upper():
            return {"error": True, "message": "Rate limit exceeded. Wait 2-3 minutes."}
        return {"error": True, "message": f"Error: {msg}", "technical_detail": traceback.format_exc()[-500:]}


# ═══════════════════════════════════════════════════════════════
#  TOOL 3: get_review_report — retrieve a completed report
# ═══════════════════════════════════════════════════════════════

def get_review_report(
    tool_context: ToolContext = None,
) -> dict:
    """Retrieve the most recently completed review report.

    Use this tool if the structural_review call timed out but the evaluation
    had already completed. The report is persisted to GCS and can be
    recovered with this tool.
    """
    # Try in-memory cache first (fastest)
    if _REPORT_CACHE.get("report"):
        return {
            "error": False,
            "overall_score": _REPORT_CACHE.get("overall_score", 0),
            "criteria_summary": _REPORT_CACHE.get("criteria_summary", []),
            "report": _REPORT_CACHE["report"],
        }

    # Try GCS by session_id
    session_id = _get_session_id(tool_context)
    data = _load_report(session_id)
    if data and data.get("report"):
        return {
            "error": False,
            "overall_score": data.get("overall_score", 0),
            "criteria_summary": data.get("criteria_summary", []),
            "report": data["report"],
        }

    return {"error": True, "message": "No completed report found. Run structural_review first."}