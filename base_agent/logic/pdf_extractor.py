"""Extract text from PDF files with page boundary markers.

Smart extraction pipeline (per page):
  1. PyMuPDF text layer  →  quality score
  2. If quality < threshold  →  render page as PNG  →  Gemini Vision OCR

This correctly handles Arabic PDFs that contain a hidden garbled Latin OCR
layer (invisible to the eye) which causes PyMuPDF to return garbage instead
of the actual Arabic content embedded in the page's image layer.
"""
import base64
import logging
import os
import re
import time

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ─── Tunable constants ────────────────────────────────────────────────────────
_MIN_CHARS_PAGE = 60       # chars below this always triggers Vision OCR
_QUALITY_THRESHOLD = 0.45  # quality score below this triggers Vision OCR
_OCR_DPI = 150             # render resolution for Vision OCR (balance: quality vs speed/cost)
_OCR_RATE_DELAY = 1.5      # seconds between Vision OCR calls (rate-limit buffer)
_OCR_BATCH_THRESHOLD = 0.40  # if >40% of pages fail quality, do a full Vision pass upfront


# ─── Quality detection ────────────────────────────────────────────────────────

def _text_quality_score(text: str) -> float:
    """
    Score extracted text quality from 0.0 (garbage) to 1.0 (good real content).

    Detects the specific failure mode of Arabic PDFs with a hidden Latin OCR layer:
    the extracted text looks like short random Latin sequences with low word density.
    Real Arabic content scores 1.0; real English scores ~0.7-1.0; garbage scores <0.3.
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    stripped = text.strip()

    # Arabic / RTL content → always valid
    arabic_chars = sum(
        1 for c in stripped
        if ('\u0600' <= c <= '\u06FF')
        or ('\u0750' <= c <= '\u077F')
        or ('\uFB50' <= c <= '\uFDFF')
        or ('\uFE70' <= c <= '\uFEFF')
    )
    if arabic_chars > 20:
        return 1.0

    words = stripped.split()
    if len(words) < 4:
        return 0.1

    avg_word_len = sum(len(w) for w in words) / len(words)

    # Garbage OCR text: average word length << real text (avg ~4-6 chars for English)
    if avg_word_len < 2.2:
        return 0.05

    # Ratio of words that look like real Latin/English words (3+ letters)
    real_words = sum(1 for w in words if re.match(r'^[A-Za-z]{3,}$', w))
    real_ratio = real_words / len(words)

    if real_ratio < 0.20:
        return 0.15   # Very few recognizable words → OCR garbage

    # Penalize high proportion of encoding-artifact characters
    weird = sum(1 for c in stripped if ord(c) > 127 and not (
        '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F'
    ))
    weird_ratio = weird / max(len(stripped), 1)
    if weird_ratio > 0.30:
        return 0.20

    # Score based on real-word ratio + average word length
    score = (real_ratio * 0.65) + (min(avg_word_len, 7.0) / 7.0 * 0.35)
    return round(min(score, 1.0), 3)


def _page_has_images(page: fitz.Page) -> bool:
    """True if the page contains raster images (indicator of a scanned page)."""
    return len(page.get_images(full=False)) > 0


# ─── Vision OCR via Gemini ────────────────────────────────────────────────────

def _render_page_as_png(page: fitz.Page, dpi: int = _OCR_DPI) -> bytes:
    """Render a PDF page to PNG bytes at the given DPI."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    return pix.tobytes("png")


def _ocr_page_with_gemini(png_bytes: bytes, page_num: int) -> str:
    """
    Send a rendered page image to Gemini Vision and return the extracted text.

    Works for Arabic, English, and mixed-language pages.
    """
    from google import genai
    from google.genai import types

    try:
        client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
        model = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-pro")

        b64 = base64.b64encode(png_bytes).decode("utf-8")

        contents = {
            "role": "user",
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": b64,
                    }
                },
                {
                    "text": (
                        "Extract ALL text visible on this document page exactly as written.\n"
                        "• Preserve the original language — Arabic, English, or mixed.\n"
                        "• Keep ALL headings, bullet points, tables, numbers, dates, percentages.\n"
                        "• Preserve the natural reading order and document structure.\n"
                        "• Output ONLY the extracted text. No commentary, no explanations."
                    )
                },
            ],
        }

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        extracted = (response.text or "").strip()
        logger.info("Vision OCR page %d: %d chars", page_num, len(extracted))
        return extracted

    except Exception as e:
        logger.error("Vision OCR failed for page %d: %s", page_num, e)
        return ""


# ─── Per-page smart extraction ────────────────────────────────────────────────

def _extract_page_smart(
    page: fitz.Page,
    page_num: int,
    force_vision: bool = False,
) -> tuple[str, str]:
    """
    Extract text from one page using the best available method.

    Returns:
        (text, method) — method is 'pymupdf', 'vision_ocr', or 'pymupdf_fallback'
    """
    if not force_vision:
        raw = page.get_text("text")
        quality = _text_quality_score(raw)
        char_count = len((raw or "").strip())

        logger.debug(
            "Page %d: pymupdf=%d chars, quality=%.2f, has_images=%s",
            page_num, char_count, quality, _page_has_images(page),
        )

        # Accept if quality is good enough and has meaningful content
        if quality >= _QUALITY_THRESHOLD and char_count >= _MIN_CHARS_PAGE:
            return raw, "pymupdf"

        # Text-only page with some content but low char count → probably a title/header page; keep it
        if not _page_has_images(page) and char_count >= 20 and quality >= 0.35:
            return raw, "pymupdf"

        logger.info(
            "Page %d: quality too low (%.2f score, %d chars) → Vision OCR",
            page_num, quality, char_count,
        )

    # ── Vision OCR ──
    try:
        png = _render_page_as_png(page)
        vision_text = _ocr_page_with_gemini(png, page_num)
        if vision_text and len(vision_text.strip()) > 15:
            return vision_text, "vision_ocr"
    except Exception as e:
        logger.error("Vision OCR failed page %d: %s", page_num, e)

    # Last resort fallback
    fallback = page.get_text("text")
    return fallback, "pymupdf_fallback"


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_text_from_bytes(pdf_bytes: bytes, smart_ocr: bool = True) -> str:
    """
    Extract full text from PDF bytes with [PAGE X] markers.

    Pipeline:
      1. Fast pass — extract all pages with PyMuPDF and score quality.
      2. If document-wide quality is low (>_OCR_BATCH_THRESHOLD of pages fail),
         do a full Vision OCR pass to avoid partial results.
      3. Otherwise selectively re-extract only the bad pages with Vision OCR.

    Args:
        pdf_bytes:  Raw bytes of the PDF file.
        smart_ocr:  Enable Gemini Vision OCR fallback (default: True).

    Returns:
        Full document text with [PAGE X] markers.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    if not smart_ocr:
        # Simple fast path — no Vision OCR
        pages = []
        for i in range(total_pages):
            text = doc[i].get_text("text").strip()
            pages.append(f"[PAGE {i + 1}]\n{text}" if text else f"[PAGE {i + 1}]\n[No text]")
        doc.close()
        return "\n\n".join(pages)

    # ── Phase 1: Fast PyMuPDF pass + quality scoring ──────────────────────────
    page_results: list[tuple[str, float, str]] = []  # (text, quality, method)

    for i in range(total_pages):
        page = doc[i]
        raw = page.get_text("text")
        quality = _text_quality_score(raw)
        page_results.append((raw, quality, "pymupdf"))

    # Decide strategy
    bad_pages = [
        i for i, (text, q, _) in enumerate(page_results)
        if q < _QUALITY_THRESHOLD or len((text or "").strip()) < _MIN_CHARS_PAGE
    ]
    bad_ratio = len(bad_pages) / max(total_pages, 1)

    logger.info(
        "PDF quality scan: %d/%d pages below threshold (%.0f%%) → %s",
        len(bad_pages), total_pages, bad_ratio * 100,
        "FULL Vision OCR pass" if bad_ratio >= _OCR_BATCH_THRESHOLD else "selective Vision OCR",
    )
    print(
        f"PDF quality: {len(bad_pages)}/{total_pages} pages need Vision OCR "
        f"({'full pass' if bad_ratio >= _OCR_BATCH_THRESHOLD else 'selective'})",
        flush=True,
    )

    # ── Phase 2: Vision OCR for bad pages ─────────────────────────────────────
    if bad_ratio >= _OCR_BATCH_THRESHOLD:
        # Full Vision OCR pass (most pages are scanned/hidden-text)
        target_pages = list(range(total_pages))
    else:
        target_pages = bad_pages

    ocr_count = 0
    for idx in target_pages:
        page = doc[idx]
        page_num = idx + 1
        print(f"  Vision OCR: page {page_num}/{total_pages}...", flush=True)

        try:
            png = _render_page_as_png(page)
            vision_text = _ocr_page_with_gemini(png, page_num)
            if vision_text and len(vision_text.strip()) > 15:
                page_results[idx] = (vision_text, 1.0, "vision_ocr")
                ocr_count += 1
        except Exception as e:
            logger.error("Vision OCR page %d failed: %s", page_num, e)

        # Rate-limit buffer between Gemini Vision calls
        if ocr_count > 0 and idx < target_pages[-1]:
            time.sleep(_OCR_RATE_DELAY)

    doc.close()

    # ── Build final output ────────────────────────────────────────────────────
    methods = {"pymupdf": 0, "vision_ocr": 0, "pymupdf_fallback": 0}
    output_pages = []

    for i, (text, quality, method) in enumerate(page_results):
        methods[method] = methods.get(method, 0) + 1
        stripped = (text or "").strip()
        entry = f"[PAGE {i + 1}]\n{stripped}" if stripped else f"[PAGE {i + 1}]\n[No text extracted]"
        output_pages.append(entry)

    logger.info(
        "Extraction done: %d pages | pymupdf=%d, vision_ocr=%d, fallback=%d",
        total_pages, methods["pymupdf"], methods["vision_ocr"], methods["pymupdf_fallback"],
    )
    print(
        f"Extraction complete: {total_pages} pages | "
        f"pymupdf={methods['pymupdf']}, vision_ocr={methods['vision_ocr']}",
        flush=True,
    )

    return "\n\n".join(output_pages)


def extract_text_with_pages(pdf_path: str, smart_ocr: bool = True) -> str:
    """Extract full text from a PDF file path with [PAGE X] markers."""
    with open(pdf_path, "rb") as f:
        return extract_text_from_bytes(f.read(), smart_ocr=smart_ocr)


def get_document_stats(text: str) -> dict:
    """Return basic stats about the extracted document text."""
    page_count = text.count("[PAGE ")
    char_count = len(text)
    word_count = len(text.split())
    estimated_tokens = word_count * 1.3
    return {
        "page_count": page_count,
        "char_count": char_count,
        "word_count": word_count,
        "estimated_tokens": int(estimated_tokens),
    }