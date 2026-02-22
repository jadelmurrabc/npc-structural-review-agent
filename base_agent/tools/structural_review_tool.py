"""NPC Structural Review Tool — Two-tool architecture.

Tool 1: extract_document() — finds PDF in user message, extracts text, stores to GCS
Tool 2: structural_review() — reads text from GCS, runs full checklist review

PDF extraction uses the proven Enterprise pattern:
  tool_context.user_content → parts (dict or object) → base64 inlineData → PyMuPDF
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

MAX_WORKERS = 4
MAX_RETRIES = 5
RETRY_BASE_DELAY = 8

_DOCUMENT_CACHE = {}
_GCS_EXTRACTED_BUCKET = "npc-extracted-docs"


# ═══════════════════════════════════════════════════════════════
#  PDF Extraction — Enterprise-proven pattern
# ═══════════════════════════════════════════════════════════════

def _part_mime(part) -> str:
    """Get mime type from both dict-style (Enterprise) and object-style (local ADK) parts."""
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
    """Extract raw bytes from dict-style (inlineData) or object-style (inline_data) parts."""
    if part is None:
        return None

    # 1) Enterprise / AgentSpace: dict with inlineData (base64-encoded)
    if isinstance(part, dict) and "inlineData" in part:
        try:
            inline = part.get("inlineData") or {}
            b64_data = inline.get("data")
            if b64_data:
                return base64.b64decode(b64_data)
        except Exception as e:
            logger.warning("Failed to decode inlineData base64: %s", e)

    # 2) Local ADK: object with inline_data.data (raw bytes)
    try:
        inline = getattr(part, "inline_data", None)
        if inline is not None:
            data = getattr(inline, "data", None)
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
    except Exception as e:
        logger.warning("Failed to read inline_data bytes: %s", e)

    # 3) Direct data attribute
    try:
        data = getattr(part, "data", None)
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
    except Exception:
        pass

    # 4) file_data with file_uri — not available on Enterprise
    # (kept as stub for future GCS-based extraction if needed)

    return None


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF, with [PAGE X] markers."""
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append(f"[PAGE {page_num + 1}]\n{text}")
    doc.close()
    return "\n\n".join(pages)


def _get_user_message_parts(tool_context) -> list:
    """Get current user message parts — works across local ADK and Enterprise."""
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
    """PRIMARY: read PDF from current user message parts (Enterprise pattern)."""
    try:
        parts = _get_user_message_parts(tool_context)
        print(f"METHOD 1 user_content: {len(parts)} parts", flush=True)

        for idx, part in enumerate(parts):
            mime = _part_mime(part)
            print(f"  Part[{idx}]: mime={mime} type={type(part).__name__} repr={repr(part)[:200]}", flush=True)

            if "pdf" not in mime.lower():
                continue

            pdf_bytes = _extract_bytes(part)
            if not pdf_bytes:
                logger.warning("Part[%d] is PDF but no bytes extracted", idx)
                continue

            logger.info("Part[%d]: %d bytes, running PyMuPDF...", idx, len(pdf_bytes))
            text = _extract_text_from_pdf_bytes(pdf_bytes)
            if text and len(text) > 200:
                logger.info("SUCCESS: %d chars from user message PDF", len(text))
                return text

        print("  No PDF in user message parts", flush=True)
        return None
    except Exception as e:
        print(f"  User message extraction FAILED: {e}", flush=True)
        return None






def _resolve_maybe_coroutine(val):
    """If val is a coroutine, run it synchronously. Otherwise return as-is."""
    import inspect
    if inspect.iscoroutine(val):
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # Already in async context — create task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, val).result()
        except RuntimeError:
            # No running loop — safe to use asyncio.run
            return asyncio.run(val)
    return val


def _extract_pdf_from_artifacts(tool_context) -> str | None:
    """FALLBACK: load PDF from saved artifacts (ADK Web)."""
    try:
        if not hasattr(tool_context, 'list_artifacts'):
            logger.info("tool_context has no list_artifacts")
            return None
        raw_keys = tool_context.list_artifacts()
        keys = _resolve_maybe_coroutine(raw_keys)
        if not keys:
            logger.info("No artifacts found")
            return None
        logger.info("Found %d artifacts: %s", len(keys), keys[:5])
        for key in keys:
            try:
                raw_part = tool_context.load_artifact(key)
                part = _resolve_maybe_coroutine(raw_part)
                if part is None:
                    continue
                pdf_bytes = _extract_bytes(part)
                if pdf_bytes:
                    mime = _part_mime(part)
                    if "pdf" in (mime or "").lower() or "pdf" in key.lower():
                        text = _extract_text_from_pdf_bytes(pdf_bytes)
                        if text and len(text) > 200:
                            logger.info("Extracted %d chars from artifact '%s'", len(text), key)
                            return text
            except Exception as e:
                logger.warning("Artifact '%s' failed: %s", key, e)
        return None
    except Exception as e:
        logger.error("Artifact extraction failed: %s", e, exc_info=True)
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
        bucket = client.bucket(_GCS_EXTRACTED_BUCKET)
        blob_name = f"extracted/{session_id}.txt"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(text, content_type="text/plain; charset=utf-8")
        logger.info("Uploaded %d chars to gs://%s/%s", len(text), _GCS_EXTRACTED_BUCKET, blob_name)
        return f"gs://{_GCS_EXTRACTED_BUCKET}/{blob_name}"
    except Exception as e:
        logger.error("GCS upload failed: %s", e, exc_info=True)
        return None


def _download_text_from_gcs(session_id: str) -> str | None:
    try:
        from google.cloud import storage as gcs_storage
        client = gcs_storage.Client()
        blob = client.bucket(_GCS_EXTRACTED_BUCKET).blob(f"extracted/{session_id}.txt")
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")
    except Exception as e:
        logger.error("GCS download failed: %s", e, exc_info=True)
        return None


def _find_latest_gcs_text() -> str | None:
    try:
        from google.cloud import storage as gcs_storage
        blobs = list(gcs_storage.Client().bucket(_GCS_EXTRACTED_BUCKET).list_blobs(prefix="extracted/", max_results=20))
        if not blobs:
            return None
        blobs.sort(key=lambda b: b.time_created or 0, reverse=True)
        return blobs[0].download_as_text(encoding="utf-8")
    except Exception as e:
        logger.error("GCS latest scan failed: %s", e, exc_info=True)
        return None






# ═══════════════════════════════════════════════════════════════
#  Arabic / Non-English Translation Support
# ═══════════════════════════════════════════════════════════════

def _is_arabic(text: str, threshold: float = 0.2) -> bool:
    """Detect if text has significant Arabic content."""
    if not text:
        return False
    sample = text[:15000]
    arabic = sum(1 for c in sample if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F'
                 or '\uFB50' <= c <= '\uFDFF' or '\uFE70' <= c <= '\uFEFF')
    alpha = sum(1 for c in sample if c.isalpha())
    return alpha > 50 and (arabic / alpha) > threshold


def _translate_arabic(text: str) -> str:
    """Translate Arabic document to English, preserving structure and data."""
    CHUNK = 80000
    instruction = (
        "You are a professional Arabic-to-English translator for government strategy documents. "
        "Translate with 100% accuracy. Never skip, summarize, or add content."
    )
    rules = (
        "RULES:\n"
        "1. Preserve ALL [PAGE X] markers exactly as-is.\n"
        "2. Preserve ALL numbers, dates, percentages, statistics, names exactly.\n"
        "3. Preserve document structure (headings, lists, tables, bullet points).\n"
        "4. Translate EVERYTHING — do not skip any section.\n"
        "5. Output ONLY the English translation, no commentary.\n\n"
    )

    if len(text) <= CHUNK:
        return _call_gemini(f"{rules}DOCUMENT:\n{text}", system_instruction=instruction)

    # Chunk for large documents
    lines = text.split('\n')
    chunks, current, size = [], [], 0
    for line in lines:
        current.append(line)
        size += len(line) + 1
        if size >= CHUNK:
            chunks.append('\n'.join(current))
            current, size = [], 0
    if current:
        chunks.append('\n'.join(current))

    parts = []
    for i, chunk in enumerate(chunks):
        logger.info("Translating chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
        print(f"Translating chunk {i+1}/{len(chunks)}...", flush=True)
        prompt = f"Translate part {i+1} of {len(chunks)}.\n{rules}TEXT:\n{chunk}"
        parts.append(_call_gemini(prompt, system_instruction=instruction))
        if i < len(chunks) - 1:
            time.sleep(3)
    return '\n'.join(parts)


# ═══════════════════════════════════════════════════════════════
#  TOOL 1: extract_document
# ═══════════════════════════════════════════════════════════════

def extract_document(tool_context: ToolContext = None) -> dict:
    """Extracts full text from uploaded PDF and stores it in GCS.

    Call this FIRST before structural_review.

    Args:
        tool_context: Automatically injected by ADK.

    Returns:
        Dict with status, page count, character count.
    """
    global _DOCUMENT_CACHE
    print("=== extract_document CALLED ===", flush=True)

    if tool_context is None:
        return {"status": "error", "message": "No tool context.", "pages": 0, "characters": 0}

    extracted_text = None
    method = "none"

    # Method 0: Enterprise pre-extracted text (tags in user_content)
    try:
        parts = _get_user_message_parts(tool_context)
        all_text = []
        for part in parts:
            pt = getattr(part, 'text', '') or (part.get('text', '') if isinstance(part, dict) else '')
            if pt and '<start_of_user_uploaded_file' not in pt and '<end_of_user_uploaded_file' not in pt:
                # Skip the user's typed message (short text like "sectoral")
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
        print("No binary PDF found — agent will pass text via document_text parameter", flush=True)
        return {"status": "no_file_found", "message": "Could not extract binary PDF. The document text will be passed directly to the review tool.", "pages": 0, "characters": 0}

    page_nums = re.findall(r'\[PAGE\s+(\d+)\]', extracted_text)
    max_page = max(int(p) for p in page_nums) if page_nums else 0

    # Store to GCS + cache
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
#  Gemini Client & Helpers
# ═══════════════════════════════════════════════════════════════

def _get_client():
    return genai.Client(vertexai=True, project=os.getenv("GOOGLE_CLOUD_PROJECT", ""), location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))


def _get_max_page(text: str) -> int:
    nums = re.findall(r'\[PAGE\s+(\d+)\]', text)
    return max(int(p) for p in nums) if nums else 0


def _extract_entity_name(text: str) -> str | None:
    for pat in [r'(?:Ministry|Authority|Committee|Council|Bureau|Institute|Department)\s+(?:of|for)\s+[\w\s\-&]+']:
        m = re.search(pat, text[:5000], re.IGNORECASE)
        if m and len(m.group(0)) > 10:
            return m.group(0).strip()[:120]
    return None


def _extract_strategy_title(text: str) -> str | None:
    for pat in [r'(?:National|Qatar)\s+[\w\s\-]+(?:Strategy|Plan|Framework|Roadmap|Policy|Vision)\s*(?:\d{4}[\s\-]*\d{0,4})?']:
        m = re.search(pat, text[:5000], re.IGNORECASE)
        if m and len(m.group(0)) > 10:
            return m.group(0).strip()[:200]
    return None


def _call_gemini(prompt, system_instruction=""):
    client = _get_client()
    model = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-pro")
    config = types.GenerateContentConfig(temperature=0.0, system_instruction=system_instruction if system_instruction else None)
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(model=model, contents=prompt, config=config).text
        except Exception as e:
            last_exc = e
            if any(kw in str(e).lower() for kw in ["429", "resource_exhausted", "500", "503", "unavailable", "timeout"]):
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, RETRY_BASE_DELAY * 0.3)
                logger.warning("Gemini retry %d/%d in %.1fs: %s", attempt + 1, MAX_RETRIES, delay, str(e)[:200])
                time.sleep(delay)
            else:
                raise
    raise last_exc


# ═══════════════════════════════════════════════════════════════
#  Evaluation Prompts & Logic
# ═══════════════════════════════════════════════════════════════

SYSTEM_INSTRUCTION = (
    "You are an expert strategy document reviewer for Qatar's National Planning Council. "
    "You evaluate strategy documents against a structural checklist with precise scoring rubrics. "
    "Be thorough, accurate, and evidence-based. Cite page numbers using [PAGE X] markers. "
    "Match evidence to the rubric level it genuinely fits. "
    "GROUNDING RULE: ONLY cite text that ACTUALLY appears verbatim in the document. "
    "NEVER fabricate content. If evidence is absent, say so. "
    "Respond with valid JSON only."
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


def _build_sub_component_prompt(sub_info, classification, document_text, max_page=0, text_source="pymupdf"):
    sub_id = sub_info["sub_component_id"]
    sub_name = sub_info["sub_component_name"]
    rubric_text = "\n".join(f"  {k}: {v}" for k, v in sub_info["rubric"].items())
    scope = SCOPE_BOUNDARIES.get(sub_id, "")
    scope_block = f"\n\nSCOPE BOUNDARY:\n{scope}" if scope else ""
    cond = "\n\nCONDITIONAL: If no relevant content, respond with applicable=false." if sub_info.get("is_conditional") else ""
    warn = "*** WARNING: Agent-provided text may be fabricated. Score CONSERVATIVELY. ***\n\n" if text_source == "agent" else ""

    return (
        f"Evaluating {sub_id}: {sub_name}\nCriteria: {sub_info['component_name']}\nClassification: {classification}\n\n"
        f"QUESTION:\n{sub_info.get('question', '')}\n\nRUBRIC:\n{rubric_text}{cond}{scope_block}\n\n"
        "INSTRUCTIONS:\n"
        "1. Read ENTIRE document. 2. Extract evidence with [PAGE X]. 3. Walk through each rubric level 0.0→1.0.\n"
        "4. Score = highest level fully supported. 5. Scores vary — don't default.\n"
        "6. 0.0=Absent, 0.25=Vague, 0.5=Incomplete, 0.75=Minor gaps, 1.0=Complete.\n"
        "7. Explain match, next level needs, why not met. 8. gap_to_next: all gaps to 1.0.\n"
        f"9. Pages 1-{max_page}. Never PAGE 0. 10. Every quote must appear verbatim.\n\n"
        f"{warn}"
        f'Respond ONLY JSON:\n{{"sub_component_id":"{sub_id}","sub_component_name":"{sub_name}",'
        '"applicable":true,"rubric_walkthrough":{"0.0_assessment":"...","0.25_assessment":"...","0.5_assessment":"...",'
        '"0.75_assessment":"...","1.0_assessment":"...","selected_score_justification":"..."},'
        '"score":0.0,"pages":[1],"evidence":"...","evidence_summary":"...","reasoning":"...","gap_to_next":"...","recommendation":"..."}\n\n'
        f"DOCUMENT:\n{document_text}"
    )


def _extract_json(text):
    text = text.strip()
    if text.startswith("`"):
        lines = text.split("\n")
        if lines[0].startswith("`"): lines = lines[1:]
        if lines and lines[-1].strip().startswith("`"): lines = lines[:-1]
        text = "\n".join(lines).strip()
    idx = text.find("{")
    if idx == -1: return None
    text = text[idx:]
    depth = 0; in_str = False; esc = False
    for i, ch in enumerate(text):
        if esc: esc = False; continue
        if ch == "\\" and in_str: esc = True; continue
        if ch == '"' and not esc: in_str = not in_str; continue
        if in_str: continue
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: return text[:i+1]
    return None


def _parse_sub_response(resp, sub_info, max_page=0):
    j = _extract_json(resp)
    if not j: return _fallback_sub(sub_info)
    try:
        r = json.loads(j)
    except json.JSONDecodeError:
        return _fallback_sub(sub_info)

    if r.get("applicable", True) and r.get("score") is not None:
        try:
            r["score"] = validate_score(float(r["score"]))
        except (ValueError, TypeError):
            r["score"] = 0.0

    # === STEP 1: Normalize ALL text fields to strings FIRST (prevents regex crash) ===
    for f in ("evidence", "reasoning", "gap_to_next", "recommendation", "evidence_summary"):
        val = r.get(f)
        if val is None:
            r[f] = ""
        elif isinstance(val, list):
            r[f] = "\n".join(str(e) for e in val)
        elif not isinstance(val, str):
            r[f] = str(val)

    # === STEP 2: Normalize pages ===
    if isinstance(r.get("pages"), list):
        r["pages"] = [max(1, min(int(p), max_page if max_page > 0 else 9999))
                       for p in r["pages"] if isinstance(p, (int, float)) and int(p) >= 0]
    elif r.get("pages") is not None:
        v = r["pages"]
        r["pages"] = [max(1, int(v))] if isinstance(v, (int, float)) and v > 0 else []
    else:
        r["pages"] = []

    # === STEP 3: Clamp page references in text fields (safe — all strings now) ===
    for f in ("evidence", "reasoning", "gap_to_next", "recommendation", "evidence_summary"):
        val = r.get(f, "")
        if not isinstance(val, str):
            val = str(val)
        # Clamp PAGE 0 → PAGE 1
        val = re.sub(r'\[PAGE\s+0\]', '[PAGE 1]', val)
        # Clamp out-of-range pages
        if max_page > 0:
            val = re.sub(
                r'\[PAGE\s+(\d+)\]',
                lambda m: "[PAGE 1]" if int(m.group(1)) < 1
                          else f"[PAGE {max_page}]" if int(m.group(1)) > max_page
                          else m.group(0),
                val)
        r[f] = val

    r.pop("rubric_walkthrough", None)
    return r


def _fallback_sub(si):
    return {"sub_component_id": si["sub_component_id"], "sub_component_name": si["sub_component_name"],
            "applicable": not si.get("is_conditional", False), "score": 0.0, "pages": [],
            "evidence": "Parse error.", "evidence_summary": "Parse error",
            "reasoning": "Manual review required.", "gap_to_next": "", "recommendation": "Manual review required."}


def _evaluate_sub(si, cls, doc, mp=0, ts="pymupdf"):
    sid = si["sub_component_id"]
    logger.info("Evaluating %s", sid)
    raw = _call_gemini(_build_sub_component_prompt(si, cls, doc, mp, ts), system_instruction=SYSTEM_INSTRUCTION)
    r = _parse_sub_response(raw, si, mp)
    r["sub_component_id"] = sid
    r["sub_component_name"] = si["sub_component_name"]
    logger.info("Done %s: score=%s", sid, r.get("score"))
    return r


def _build_comment_prompt(name, subs):
    lines = []
    for s in subs:
        if s.get("applicable", True):
            pct = int((s.get("score", 0) or 0) * 100)
            lines.append(f"  {s['sub_component_id']} {s['sub_component_name']}: {pct}%")
        else:
            lines.append(f"  {s['sub_component_id']} {s['sub_component_name']}: N/A")
    scores = [s.get("score", 0) for s in subs if s.get("applicable", True) and s.get("score") is not None]
    avg = sum(scores) / len(scores) if scores else 0
    lvl = "complete" if avg >= 1 else "high" if avg >= .75 else "moderate" if avg >= .5 else "low" if avg >= .25 else "absent"
    return (f"Write 1-2 sentence assessment for: {name}\n\n" + "\n".join(lines) +
            f"\n\nBand: {lvl.upper()}.\nStart with 'Provides {lvl} coverage'.\n"
            "Mention key strengths, then key gaps. Keep it concise.\n"
            "CRITICAL: Output ONLY a plain text sentence. NO JSON, NO code blocks, NO markdown, NO quotes.")


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
#  Criteria Summary Builder (for main agent consumption)
# ═══════════════════════════════════════════════════════════════

def _build_criteria_summary(comp_results: list, classification: str) -> list:
    """Build a criteria-level summary for the main agent.

    Returns a list of dicts, one per criteria, with:
      - id, name, score (band %), label
      - observation (component comment)
      - recommendation (consolidated from sub-criteria)
    """
    from base_agent.logic.scoring import calculate_display_band, band_label

    summary = []
    for comp in comp_results:
        sub_results = comp.get("sub_results", [])

        # Calculate criteria-level score band
        applicable_scores = [
            s["score"] for s in sub_results
            if s.get("applicable", True) and s.get("score") is not None
        ]
        if applicable_scores:
            raw_avg = sum(applicable_scores) / len(applicable_scores)
            band = calculate_display_band(raw_avg)
            label = band_label(band)
        else:
            band = "0%"
            label = "Absent"

        # Observation = component comment (already generated by LLM)
        observation = comp.get("comment", "")

        # Recommendations = consolidated from sub-criteria with score < 1.0
        recs = []
        for sub in sub_results:
            if (sub.get("applicable", True)
                    and sub.get("score") is not None
                    and sub["score"] < 1.0
                    and sub.get("recommendation")):
                rec_text = sub["recommendation"].strip()
                if rec_text:
                    recs.append(f"[{sub['sub_component_id']}] {rec_text}")

        summary.append({
            "id": comp["id"],
            "name": comp["name"],
            "score": band,
            "label": label,
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
    """Full structural checklist review. Call extract_document FIRST.

    Args:
        document_text: Fallback text from agent.
        classification: entity, sectoral, or thematic.
        strategy_title: Title of the strategy.
        entity_name: Entity name.
        tool_context: ADK tool context (auto-injected).
    """
    try:
        if not classification or not classification.strip():
            return {"error": True, "message": "Missing classification. Provide: entity, sectoral, or thematic."}
        classification = classification.lower().strip()
        if classification not in ("entity", "sectoral", "thematic"):
            return {"error": True, "message": f"Invalid classification: {classification}."}

        logger.info("structural_review: cls=%s, agent_text=%d", classification, len(document_text or ""))
        text_source = "agent"

        # Check 1: In-memory cache
        cached = _DOCUMENT_CACHE.get("text", "")
        if cached and len(cached) > 500:
            document_text, text_source = cached, f"cache_{_DOCUMENT_CACHE.get('source', '?')}"

        # Check 2: GCS by session_id
        if text_source == "agent":
            gcs = _download_text_from_gcs(_get_session_id(tool_context))
            if gcs and len(gcs) > 500:
                document_text, text_source = gcs, "gcs_cache"

        # Check 3: GCS latest
        if text_source == "agent":
            lat = _find_latest_gcs_text()
            if lat and len(lat) > 500:
                document_text, text_source = lat, "gcs_latest"

        # Check 4: Session state
        if text_source == "agent" and tool_context:
            try:
                ssid = getattr(tool_context, 'state', {}).get("_extracted_session_id", "")
                if ssid:
                    st = _download_text_from_gcs(ssid)
                    if st and len(st) > 500:
                        document_text, text_source = st, "gcs_state"
            except Exception:
                pass

        # Check 5: Direct extraction
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

        # === Arabic / Non-English Translation ===
        if _is_arabic(document_text):
            logger.info("Arabic document detected (%d chars) — translating...", len(document_text))
            print(f"Arabic document detected — translating to English...", flush=True)
            try:
                translated = _translate_arabic(document_text)
                if translated and len(translated) > 200:
                    document_text = translated
                    logger.info("Translation complete: %d chars", len(document_text))
                    print(f"Translation complete: {len(document_text)} chars", flush=True)
                else:
                    logger.warning("Translation returned insufficient text, using original")
            except Exception as e:
                logger.error("Translation failed: %s — using original Arabic text", e)
                print(f"Translation failed: {e} — using original", flush=True)

        if entity_name in ("Unknown Entity", "", None):
            entity_name = _extract_entity_name(document_text) or "Unknown Entity"
        if strategy_title in ("Untitled Strategy", "", None):
            strategy_title = _extract_strategy_title(document_text) or "Untitled Strategy"

        max_page = _get_max_page(document_text)
        if max_page == 0:
            lines, new, cc, pn = document_text.split('\n'), ["[PAGE 1]"], 0, 1
            for line in lines:
                cc += len(line) + 1; new.append(line)
                if cc > 3000: pn += 1; new.append(f"\n[PAGE {pn}]"); cc = 0
            document_text = '\n'.join(new)
            max_page = _get_max_page(document_text)

        subs = get_applicable_sub_components(classification)
        logger.info("Evaluating %d sub-criteria", len(subs))

        results = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {}
            for i, si in enumerate(subs):
                futs[ex.submit(_evaluate_sub, si, classification, document_text, max_page, text_source)] = si
                if i > 0 and i % MAX_WORKERS == 0: time.sleep(2)
            for fut in as_completed(futs):
                si = futs[fut]
                try:
                    results[si["sub_component_id"]] = fut.result()
                except Exception as e:
                    logger.error("Failed %s: %s", si["sub_component_id"], e)
                    results[si["sub_component_id"]] = _fallback_sub(si)

        comp_results = []
        for comp in load_checklist(classification)["components"]:
            csubs = [results[s["id"]] for s in comp["sub_components"] if s["id"] in results]
            csubs = _add_na_subs(csubs, classification, comp["sub_components"])
            try:
                comment = _call_gemini(_build_comment_prompt(comp["name"], csubs), system_instruction=SYSTEM_INSTRUCTION).strip()
                # Strip JSON bleed-through (Gemini sometimes returns JSON instead of plain text)
                if comment.startswith('{') or comment.startswith('['):
                    try:
                        parsed = json.loads(comment)
                        if isinstance(parsed, dict):
                            comment = str(next(iter(parsed.values()), comment))
                        elif isinstance(parsed, list):
                            comment = " ".join(str(x) for x in parsed)
                    except json.JSONDecodeError:
                        pass
                if comment.startswith('```'):
                    comment = re.sub(r'```\w*\n?', '', comment).strip()
                if comment.startswith('"') and comment.endswith('"'):
                    comment = comment[1:-1]
            except Exception:
                scores = [s.get("score", 0) for s in csubs if s.get("applicable", True) and s.get("score") is not None]
                avg = sum(scores) / len(scores) if scores else 0
                lvl = "high" if avg >= .75 else "moderate" if avg >= .5 else "low" if avg >= .25 else "absent"
                comment = f"Provides {lvl} coverage."
            comp_results.append({"id": comp["id"], "name": comp["name"], "comment": comment, "sub_results": csubs})

        report = generate_report(strategy_title=strategy_title, entity_name=entity_name, classification=classification, component_results=comp_results)
        all_s, na = [], 0
        for c in comp_results:
            for s in c.get("sub_results", []):
                if s.get("applicable", True) and s.get("score") is not None: all_s.append(s["score"])
                elif not s.get("applicable", True): na += 1

        # Build criteria-level summary for the main agent
        criteria_summary = _build_criteria_summary(comp_results, classification)
        print(f"\n=== CRITERIA SUMMARY FOR MAIN AGENT ===\n{json.dumps(criteria_summary, indent=2)}\n{'='*40}", flush=True)

        return {
            "error": False,
            "overall_score": calculate_overall_score(all_s, na_count=na),
            "criteria_summary": criteria_summary,
            "report": report,
        }

    except Exception as e:
        logger.exception("Review failed")
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg.upper():
            return {"error": True, "message": "Rate limit exceeded. Wait 2-3 minutes."}
        return {"error": True, "message": f"Error: {msg}", "technical_detail": traceback.format_exc()[-500:]}