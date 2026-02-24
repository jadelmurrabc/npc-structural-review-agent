"""Generate the markdown report from scored results."""
from __future__ import annotations

from datetime import datetime
import json
import re
from typing import Any, Optional, Tuple

from base_agent.logic.scoring import (
    calculate_display_band, band_label, calculate_overall_score,
    score_label,
)


def _safe_score(sub: dict) -> float:
    """Return a sub-criteria's score as a float, treating None as 0.0."""
    val = sub.get("score")
    return val if val is not None else 0.0


def _strip_mandatory_prefix(text: str) -> str:
    """Remove leading [MANDATORY] tag(s) from recommendation text."""
    if not text:
        return text
    cleaned = text.strip()
    while cleaned.upper().startswith("[MANDATORY]"):
        cleaned = cleaned[len("[MANDATORY]"):].strip()
    return cleaned


def _is_evaluation_error(sub: dict) -> bool:
    """Detect if a sub-result is an evaluation error (fallback).
    Checks for the explicit flag first, then for legacy sentinel text patterns."""
    if sub.get("_evaluation_error"):
        return True
    # Legacy detection for results without the flag
    ev = sub.get("evidence", "")
    reason = sub.get("reasoning", "")
    if ev in ("Evaluation error.", "Evaluation error", ""):
        if "manual review" in reason.lower() or "could not be completed" in reason.lower():
            return True
    return False


def _component_aggregates(sub_results: list[dict]) -> tuple[str, str]:
    """Return (band, label) for a list of sub-results.
    Evaluation errors are excluded from the calculation."""
    applicable_scores = [
        s["score"] for s in sub_results
        if s.get("applicable", True) and s.get("score") is not None
        and not _is_evaluation_error(s)
    ]
    if applicable_scores:
        raw_avg = sum(applicable_scores) / len(applicable_scores)
        band = calculate_display_band(raw_avg)
        label = band_label(band)
    else:
        band = "0%"
        label = "Absent"
    return band, label


def _component_included(sub_results: list[dict], band: str) -> bool:
    """Determine if a criteria's content is 'Included' in the document.

    Returns False if:
    - The criteria band is Absent (0%), OR
    - All sub-criteria scores are 0.0 or None (excluding evaluation errors)
    Returns True otherwise.
    """
    if band == "0%":
        return False
    applicable_scores = [
        s["score"] for s in sub_results
        if s.get("applicable", True) and s.get("score") is not None
        and not _is_evaluation_error(s)
    ]
    return any(score > 0.0 for score in applicable_scores)


def _score_bar(score_val: float) -> str:
    """Return a simple text-based visual indicator for a score."""
    filled = round(score_val * 4)  # 0-4 blocks
    return "[" + "█" * filled + "░" * (4 - filled) + "]"


def _append_blockquote(lines: list[str], text: str) -> None:
    """
    Append a multi-line string as a proper markdown blockquote where *every*
    line is prefixed with '> '. This prevents JSON / code fences / markdown
    from "bleeding" out of the quote when text contains newlines.
    """
    if text is None:
        return
    txt = str(text).rstrip("\n")
    if not txt.strip():
        return
    for ln in txt.splitlines():
        # Keep empty lines inside blockquote for readability
        lines.append("> " + ln if ln.strip() else ">")


# ═══════════════════════════════════════════════════════════════
#  Robust JSON span detection
# ═══════════════════════════════════════════════════════════════

def _find_balanced_json_span(s: str, start: int) -> Optional[Tuple[int, int]]:
    """
    Find a balanced JSON object/array span in string s starting at or after `start`.
    Returns (json_start, json_end_exclusive) if found, else None.
    """
    i = start
    n = len(s)

    while i < n and s[i].isspace():
        i += 1
    if i >= n or s[i] not in "{[":
        return None

    json_start = i
    stack = []
    stack.append("}" if s[i] == "{" else "]")

    i += 1
    in_str = False
    esc = False

    while i < n and stack:
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "}]":
                if not stack or ch != stack[-1]:
                    return None
                stack.pop()
        i += 1

    if stack:
        return None
    return (json_start, i)


# ═══════════════════════════════════════════════════════════════
#  Text extraction from structured objects
# ═══════════════════════════════════════════════════════════════

def _extract_text_from_obj(obj: Any) -> str:
    """
    Extract the best human-readable text from a JSON-like object.
    """
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj.strip()

    if isinstance(obj, dict):
        preferred_keys = [
            "initiatives_and_projects_assessment",
            "assessment_comment",
            "comment",
            "summary",
            "assessment",
            "reasoning",
            "text",
            "message",
            "content",
            "value",
        ]
        for k in preferred_keys:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # Handle evidence item: {"quote": "...", "page": X}
        quote = obj.get("quote", "")
        page = obj.get("page")
        if quote and page is not None:
            return f'"{quote}" [PAGE {page}]'
        elif quote:
            return f'"{quote}"'

        # fallback: first non-empty string value
        for v in obj.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    if isinstance(obj, list):
        # join extracted strings from items
        parts = []
        for it in obj:
            t = _extract_text_from_obj(it)
            if t:
                parts.append(t)
        return "\n".join(parts)

    return str(obj).strip()


def _format_evidence_list(evidence_list: list[Any]) -> str:
    """
    Format evidence_list JSON array into clean evidence lines.
    Expected item shape: {"quote": "...", "page": 11} but supports strings too.
    """
    out_lines: list[str] = []
    for item in evidence_list:
        if isinstance(item, dict):
            quote = (item.get("quote") or "").strip()
            page = item.get("page")
            if quote and page is not None:
                out_lines.append(f'"{quote}" [PAGE {page}]')
            elif quote:
                out_lines.append(f'"{quote}"')
            else:
                # fallback dict to string (rare)
                extracted = _extract_text_from_obj(item)
                if extracted:
                    out_lines.append(extracted)
        elif isinstance(item, str):
            out_lines.append(item.strip())
        else:
            s = str(item).strip()
            if s:
                out_lines.append(s)
    return "\n".join([ln for ln in out_lines if ln.strip()])


# ═══════════════════════════════════════════════════════════════
#  Clean residual braces from text
# ═══════════════════════════════════════════════════════════════

def _clean_residual_braces(text: str) -> str:
    """Remove stray curly braces {} that leaked from JSON into rendered text.
    Preserves [PAGE X] markers and legitimate brace usage."""
    if not text:
        return text

    # Step 1: Try to detect and extract content from JSON wrappers
    stripped = text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                extracted = _extract_text_from_obj(obj)
                if extracted:
                    return extracted
        except json.JSONDecodeError:
            pass

    # Step 2: Remove inline JSON objects like {"key": "value"}
    # Pattern: { followed by quoted key, colon, quoted value, }
    text = re.sub(
        r'\{["\'][\w_]+["\']\s*:\s*["\'][^"\']*["\']\s*\}',
        lambda m: _extract_text_from_obj(json.loads(m.group(0).replace("'", '"')))
        if _try_json_parse(m.group(0).replace("'", '"')) else m.group(0),
        text
    )

    # Step 3: Remove lone orphan braces that don't form valid JSON
    # Count braces — if unbalanced, strip the orphans
    open_count = text.count('{')
    close_count = text.count('}')
    if open_count != close_count:
        # Remove orphan braces carefully
        result = []
        depth = 0
        for ch in text:
            if ch == '{':
                depth += 1
                # Only keep if it looks like start of a structure
                # For plain text output, just skip lone braces
                continue
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                continue
            else:
                result.append(ch)
        text = ''.join(result)
    elif open_count > 0 and open_count == close_count:
        # Balanced braces — check if they form valid JSON
        if stripped.startswith('{') and stripped.endswith('}') and open_count == 1:
            # Single wrapper — try to parse as JSON, else just strip braces
            try:
                obj = json.loads(stripped)
                extracted = _extract_text_from_obj(obj)
                if extracted:
                    text = extracted
            except json.JSONDecodeError:
                # Not valid JSON — remove the wrapping braces
                text = stripped[1:-1].strip()
        else:
            # Multiple balanced brace pairs inline — remove them all
            # e.g. "text with {random} braces {here}" → "text with random braces here"
            text = re.sub(r'\{([^}]*)\}', r'\1', text)

    return text


def _try_json_parse(s: str) -> bool:
    """Return True if string is valid JSON."""
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


# ═══════════════════════════════════════════════════════════════
#  Master text normalizer
# ═══════════════════════════════════════════════════════════════

def re_find_json_marker(s: str) -> Optional[int]:
    """
    Find the start index of a JSON object/array that is likely introduced by a 'JSON' label.
    Returns index of '{' or '[' if found, else None.
    """
    # Prefer a literal 'JSON' line marker if present
    pos = s.find("JSON")
    if pos != -1:
        brace = s.find("{", pos)
        bracket = s.find("[", pos)
        candidates = [c for c in [brace, bracket] if c != -1]
        if candidates:
            return min(candidates)

    # fallback: if the whole string is a JSON object/array
    stripped = s.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return s.find(stripped[0])
    return None


def _normalize_text(value: Any) -> str:
    """
    Normalize a value to clean text.
    Handles:
    - dict/list values (extract human string)
    - strings containing fenced ```json blocks
    - strings containing 'JSON' + inline JSON
    - strings containing 'evidence_list: [ ... ]'
    - residual curly braces from JSON artifacts
    """
    if value is None:
        return ""

    # If already structured, extract immediately
    if isinstance(value, (dict, list)):
        return _clean_residual_braces(_extract_text_from_obj(value))

    if not isinstance(value, str):
        return str(value).strip()

    s = value.strip()
    if not s:
        return ""

    # 1) Handle fenced JSON blocks: ```json ... ```
    if "```json" in s:
        start = s.find("```json")
        after = start + len("```json")
        end = s.find("```", after)
        if end != -1:
            payload = s[after:end].strip()
            try:
                obj = json.loads(payload)
                extracted = _extract_text_from_obj(obj)
                if extracted:
                    before = s[:start].strip()
                    after_txt = s[end + 3:].strip()
                    parts = [p for p in [before, extracted, after_txt] if p]
                    return _clean_residual_braces("\n".join(parts).strip())
            except Exception:
                # If JSON parse fails, remove the fence markers
                before = s[:start].strip()
                inner = payload
                after_txt = s[end + 3:].strip()
                parts = [p for p in [before, inner, after_txt] if p]
                return _clean_residual_braces("\n".join(parts).strip())

    # 2) Handle any remaining ``` fences (non-json)
    if "```" in s:
        s = re.sub(r'```\w*\n?', '', s).strip()

    # 3) Handle lines like: JSON\n{ ... }
    json_marker = re_find_json_marker(s)
    if json_marker is not None:
        jstart = json_marker
        span = _find_balanced_json_span(s, jstart)
        if span:
            js, je = span
            blob = s[js:je]
            try:
                obj = json.loads(blob)
                extracted = _extract_text_from_obj(obj)
                if extracted:
                    before = s[:js].replace("JSON", "").strip()
                    after_txt = s[je:].strip()
                    parts = [p for p in [before, extracted, after_txt] if p]
                    return _clean_residual_braces("\n".join(parts).strip())
            except Exception:
                pass

    # 4) Handle "evidence_list: [ ... ]"
    if "evidence_list" in s and "[" in s:
        idx = s.find("evidence_list")
        bracket = s.find("[", idx)
        if bracket != -1:
            span = _find_balanced_json_span(s, bracket)
            if span:
                js, je = span
                blob = s[js:je]
                try:
                    arr = json.loads(blob)
                    if isinstance(arr, list):
                        formatted = _format_evidence_list(arr)
                        before = s[:idx].strip()
                        after_txt = s[je:].strip()
                        parts = [p for p in [before, formatted, after_txt] if p]
                        return _clean_residual_braces("\n".join(parts).strip())
                except Exception:
                    pass

    # 5) Handle string that is itself a JSON object/array
    if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
        try:
            obj = json.loads(s)
            extracted = _extract_text_from_obj(obj)
            if extracted:
                return _clean_residual_braces(extracted)
        except json.JSONDecodeError:
            pass

    # 6) Final cleanup: remove residual braces
    result = _clean_residual_braces(s)

    # 7) Remove wrapping double-quotes if present
    if result.startswith('"') and result.endswith('"') and result.count('"') == 2:
        result = result[1:-1].strip()

    return result


# ═══════════════════════════════════════════════════════════════
#  Evidence formatting — numbered reference table
# ═══════════════════════════════════════════════════════════════

MAX_EVIDENCE_ROWS = 8


def _parse_evidence_items(evidence_txt: str) -> list[tuple[str, str]]:
    """Parse evidence text into (quote_text, page_number) pairs.

    Splits on [PAGE X] markers. Consecutive items from the same page
    are merged into a single evidence item (joined with ' — ') so the
    table shows coherent citations rather than fragmented lines.
    """
    if not evidence_txt:
        return []

    segments = re.split(r'\[PAGE\s+(\d+)\]', evidence_txt)

    raw: list[tuple[str, str]] = []
    for i in range(0, len(segments) - 1, 2):
        text = segments[i].strip()
        page = segments[i + 1].strip()

        text = text.strip('"\'«»\u201c\u201d\u2018\u2019 \n')
        text = text.lstrip('\n•❖-–—*,.;: ')
        # Strip leading conjunctions left over from list splits
        text = re.sub(r'^(?:and|or|,)\s+', '', text, flags=re.IGNORECASE)

        if text:
            raw.append((text, page))

    if not raw:
        return []

    # Merge consecutive items that share the same page number
    merged: list[tuple[str, str]] = [raw[0]]
    for text, page in raw[1:]:
        prev_text, prev_page = merged[-1]
        if page == prev_page:
            merged[-1] = (prev_text + " — " + text, page)
        else:
            merged.append((text, page))

    return merged


def _extract_evidence_pages(evidence_txt: str) -> list[int]:
    """Extract sorted unique page numbers from evidence text [PAGE X] markers."""
    pages = sorted(set(int(m) for m in re.findall(r'\[PAGE\s+(\d+)\]', evidence_txt or "")))
    return pages


def _format_evidence_for_display(evidence_txt: str) -> list[str]:
    """Render evidence as a markdown table: Source Text | Pg.

    Caps output at MAX_EVIDENCE_ROWS to keep the report scannable.
    Page column on the right gives the source text maximum width.
    """
    items = _parse_evidence_items(evidence_txt)

    if not items:
        if evidence_txt.strip():
            return [f"> {evidence_txt.strip()[:500]}"]
        return []

    # Cap and note if truncated
    truncated = len(items) > MAX_EVIDENCE_ROWS
    display_items = items[:MAX_EVIDENCE_ROWS]

    lines = [
        "| Source Text | Pg. |",
        "|:-----------|:---:|",
    ]
    for text, page in display_items:
        cell = _clean_for_table_cell(text, max_length=300)
        lines.append(f"| {cell} | {page} |")

    if truncated:
        lines.append("")
        lines.append(f"*… and {len(items) - MAX_EVIDENCE_ROWS} additional references across the cited pages.*")

    return lines


# ═══════════════════════════════════════════════════════════════
#  Arabic PDF extraction cleanup
# ═══════════════════════════════════════════════════════════════

# Unicode ranges used in cleanup patterns
_AR = '\u0600-\u06ff'          # Arabic block (letters, marks, digits)
_AR_DIAC = '\u064b-\u0652'    # Arabic diacritical marks (tashkeel)

def _clean_arabic_extraction(text: str) -> str:
    """Fix common Arabic PDF text-extraction artifacts.

    Only applies safe, conservative fixes:
    1. Remove tatweel (kashida stretching character)
    2. Remove zero-width / bidi-override characters that break rendering
    3. Reattach diacritical marks that got separated from their base letter
    4. Rejoin tanwin-fatha + alef that got split ("عً ا" → "عًا")
    5. Rejoin a single Arabic letter that was split off the end of a word
       ("مشاري ع" → "مشاريع") — preserves standalone "و" (and)
    6. Collapse leftover multiple spaces
    """
    if not text:
        return text

    # 1  Tatweel
    text = text.replace('\u0640', '')

    # 2  Zero-width joiners, bidi marks, BOM
    text = re.sub(r'[\u200b-\u200f\u2028-\u202e\u2066-\u2069\ufeff]', '', text)

    # 3  Reattach diacritics: "حكومة ً" → "حكومةً"
    text = re.sub(rf'([{_AR}])\s+([{_AR_DIAC}])', r'\1\2', text)

    # 4  Tanwin-fatha + space + alef variants: "مشروعً ا" → "مشروعًا"
    text = re.sub(r'(\u064b)\s+([\u0627\u0623\u0625\u0622])', r'\1\2', text)

    # 5  Single trailing Arabic letter rejoined to its word
    #    "مشاري ع" → "مشاريع" but NOT "اختصاصات و حالة" (و = "and")
    def _rejoin_letter(m):
        letter_with_diac = m.group(2)
        base = letter_with_diac.translate(_STRIP_DIACRITICS_TABLE)
        if base in _STANDALONE_AR_LETTERS:
            return m.group(0)   # preserve standalone word
        return m.group(1) + m.group(2)

    text = re.sub(
        rf'([{_AR}][{_AR}{_AR_DIAC}]+)'       # 2+ Arabic chars (the word body)
        rf'\s'                                  # single space
        rf'([{_AR}][{_AR_DIAC}]*)'             # single Arabic char + optional diacritics
        rf'(?=\s|$|[^{_AR}{_AR_DIAC}])',       # followed by space / end / non-Arabic
        _rejoin_letter, text
    )

    # 6  Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


# Pre-computed translation table to strip Arabic diacritics from a character
_STRIP_DIACRITICS_TABLE = str.maketrans('', '', ''.join(
    chr(c) for c in range(0x064b, 0x0653)
))

# Single Arabic letters that are legitimate standalone words — never rejoin these
_STANDALONE_AR_LETTERS = frozenset('و')


# ═══════════════════════════════════════════════════════════════
#  PDF table-header noise cleaner (fully generic)
# ═══════════════════════════════════════════════════════════════

# Common column header terms found across government strategy PDFs.
# When ≥3 consecutive dash-separated segments match these, they are
# column headers that leaked into evidence text and should be stripped.
_KNOWN_TABLE_HEADERS = frozenset({
    # English — common across any government strategy table
    "n", "no", "no.", "#",
    "program", "programme", "project", "action", "goal", "goals",
    "output", "outputs", "outcome", "outcomes", "conclusion",
    "indicator", "indicators", "description", "status",
    "supporting agency", "responsible entity", "owner",
    "key stakeholders", "stakeholders",
    "timeline", "budget", "risk", "mitigation",
    # Arabic equivalents
    "الجهة الداعمة", "البرنامج", "المشروع", "الإجراء", "الهدف",
    "الخلاصة", "المخرج", "المؤشر", "الوصف", "الحالة",
    "الجدول الزمني", "الميزانية", "الجهة المسؤولة", "المخاطر", "التخفيف",
})

# TOC dot-leader pattern: "word:.......number" or "word.........number"
_TOC_DOT_LEADER_RE = re.compile(r'[.:]\s*[.…]{3,}\s*\d')


def _clean_table_header_noise(text: str) -> str:
    """Strip PDF table column-header boilerplate from evidence text.

    Strategy PDFs contain tables where Gemini concatenates column headers
    and cell values using ' — ' separators. This function finds the longest
    run of ≥3 consecutive known column-header terms and strips them,
    keeping only the prefix description and the actual data values.

    Works on ANY strategy document — no file-specific patterns.
    """
    if not text:
        return text

    # --- Clean TOC dot-leader artifacts (universal) ---
    if _TOC_DOT_LEADER_RE.search(text):
        text = re.sub(r'([^.]):?\s*[.…]{3,}\s*(\d+)', r'\1 (p.\2)', text)
        text = re.sub(r' {2,}', ' ', text)

    if " — " not in text:
        return text

    # --- Find and strip the longest run of ≥3 known headers ---
    segments = text.split(" — ")

    best_start, best_len = -1, 0
    i = 0
    while i < len(segments):
        seg_lower = segments[i].strip().lower()
        if seg_lower in _KNOWN_TABLE_HEADERS:
            run_start = i
            j = i
            while j < len(segments) and segments[j].strip().lower() in _KNOWN_TABLE_HEADERS:
                j += 1
            run_len = j - run_start
            if run_len >= 3 and run_len > best_len:
                best_start = run_start
                best_len = run_len
            i = j
        else:
            i += 1

    if best_start < 0:
        return text  # no header run found

    # Split into: prefix (before headers) + data (after headers)
    prefix_segs = segments[:best_start]
    data_segs = segments[best_start + best_len:]

    # Clean data segments: remove empty cells and bare row numbers
    data_segs = [s for s in data_segs if s.strip() not in ("--", "-", "—", "")]
    if data_segs and re.match(r'^\d+$', data_segs[0].strip()):
        data_segs = data_segs[1:]

    # Clean prefix: strip trailing known header that got attached to description
    prefix_text = " — ".join(prefix_segs).rstrip(" :—-,") if prefix_segs else ""
    if prefix_text:
        colon_match = re.search(r':\s*(\w[\w\s]*)$', prefix_text)
        if colon_match and colon_match.group(1).strip().lower() in _KNOWN_TABLE_HEADERS:
            prefix_text = prefix_text[:colon_match.start()].rstrip()

    data_text = " — ".join(data_segs)

    if prefix_text and data_text:
        return f"{prefix_text}: {data_text}"
    elif data_text:
        return data_text
    elif prefix_text:
        return prefix_text
    return text


# ═══════════════════════════════════════════════════════════════
#  Table cell sanitizer (shared by evidence table + index table)
# ═══════════════════════════════════════════════════════════════

def _clean_for_table_cell(text: str, max_length: int = 500) -> str:
    """Clean text for safe use inside a markdown table cell."""
    if not text:
        return ""
    cleaned = text.replace('\n', ' ').replace('\r', ' ')
    cleaned = cleaned.replace('|', ' — ')
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    cleaned = cleaned.replace('{', '').replace('}', '')
    cleaned = _clean_table_header_noise(cleaned)
    cleaned = _clean_arabic_extraction(cleaned)
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length - 1] + "…"
    return cleaned.strip()


# ═══════════════════════════════════════════════════════════════
#  Report generation
# ═══════════════════════════════════════════════════════════════

def generate_report(
    strategy_title: str,
    entity_name: str,
    classification: str,
    component_results: list[dict],
) -> str:
    """
    Build the full markdown structural review report.

    Args:
        strategy_title: Name of the strategy document.
        entity_name: Name of the submitting entity.
        classification: entity, sectoral, or thematic.
        component_results: List of criteria dicts, each containing sub_results.

    Returns:
        Complete markdown report as a string.
    """
    lines: list[str] = []
    review_date = datetime.now().strftime("%Y-%m-%d")

    # Collect applicable scores for overall calculation.
    # N/A sub-criteria and evaluation errors are excluded entirely.
    all_scores: list[float] = []
    eval_error_count = 0
    for comp in component_results:
        for sub in comp.get("sub_results", []):
            if _is_evaluation_error(sub):
                eval_error_count += 1
                continue
            if sub.get("applicable", True) and sub.get("score") is not None:
                all_scores.append(sub["score"])

    overall_score = calculate_overall_score(all_scores)

    # ─────────────────────────────────────────────
    # HEADER & SCORE OVERVIEW
    # ─────────────────────────────────────────────
    lines.append("# Structural Checklist Review")
    lines.append("")
    lines.append("| Field | Detail |")
    lines.append("|-------|--------|")
    lines.append(f"| **Strategy** | {strategy_title} |")
    lines.append(f"| **Entity** | {entity_name} |")
    lines.append(f"| **Classification** | {classification.capitalize()} |")
    lines.append(f"| **Review Date** | {review_date} |")
    lines.append(f"| **Overall Score** | **{overall_score}%** |")
    lines.append("")
    if eval_error_count > 0:
        lines.append(f"> ⚠️ **Note:** {eval_error_count} sub-component{'s' if eval_error_count != 1 else ''} could not be automatically evaluated and {'are' if eval_error_count != 1 else 'is'} excluded from the overall score. Manual review is required for {'these items' if eval_error_count != 1 else 'this item'}.")
        lines.append("")
    lines.append("")

    # ─────────────────────────────────────────────
    # SUMMARY SCORECARD
    # ─────────────────────────────────────────────
    lines.append("## Summary Scorecard")
    lines.append("")
    lines.append("| # | Criteria | Included | Assessment | Score |")
    lines.append("|:---:|-----------|:--------:|:----------:|:-----:|")

    for comp in component_results:
        comp_id = comp["id"]
        comp_name = comp["name"]
        sub_results = comp.get("sub_results", [])
        band, label = _component_aggregates(sub_results)
        included = "Yes" if _component_included(sub_results, band) else "No"
        lines.append(f"| {comp_id} | {comp_name} | {included} | {label} | {band} |")

    lines.append("")
    lines.append("> **Scale:** Absent (0%) · Low (25%) · Mid (50%) · High (75%) · Complete (100%)")
    lines.append("")
    lines.append("")

    # ─────────────────────────────────────────────
    # PER-CRITERIA DETAIL
    # ─────────────────────────────────────────────
    for comp in component_results:
        comp_id = comp["id"]
        comp_name = comp["name"]
        sub_results = comp.get("sub_results", [])
        band, label = _component_aggregates(sub_results)
        has_content = _component_included(sub_results, band)

        lines.append("---")
        lines.append("")
        lines.append(f"## Criteria {comp_id}: {comp_name}")
        lines.append("")
        lines.append("| Assessment | Included in Document |")
        lines.append("|:----------:|:--------------------:|")
        lines.append(f"| **{label} ({band})** | {'Yes' if has_content else 'No'} |")
        lines.append("")

        # Consolidated comment (must support multi-line safely)
        comp_comment = _normalize_text(comp.get("comment", ""))
        if comp_comment:
            _append_blockquote(lines, comp_comment)
            lines.append("")

        # Sub-criteria details
        for sub in sub_results:
            sub_id = sub["sub_component_id"]
            sub_name = sub["sub_component_name"]

            lines.append("")

            if not sub.get("applicable", True):
                reason = sub.get("na_reason", f"Not applicable for {classification.capitalize()} classification")
                lines.append(f"### {sub_id} — {sub_name}")
                lines.append("")
                lines.append(f"**Status:** Not Applicable — *{reason}*")
                lines.append("")
                continue

            # Check for evaluation error
            is_error = _is_evaluation_error(sub)

            score_val = _safe_score(sub)

            if is_error:
                # Render evaluation error section
                lines.append(f"### {sub_id} — {sub_name}")
                lines.append("")
                lines.append("**Status:** ⚠️ Evaluation Error — *Automatic evaluation could not be completed*")
                lines.append("")
                lines.append("**Note:**")
                lines.append("")
                lines.append("> This sub-component could not be automatically evaluated. The model's response")
                lines.append("> could not be parsed into a valid assessment. A manual review is required to")
                lines.append("> properly score this criterion against the rubric.")
                lines.append("")
                lines.append("**Recommendation [MANUAL REVIEW REQUIRED]:**")
                lines.append("")
                lines.append("> Please review the source document against the rubric criteria for this")
                lines.append("> sub-component and assign an appropriate score manually. This evaluation")
                lines.append("> error does not indicate that the content is absent from the document.")
                lines.append("")
                continue

            label_str = score_label(score_val)
            pct = int(score_val * 100)
            bar = _score_bar(score_val)

            # Normalize evidence once — used for both pages and display
            evidence_txt = _normalize_text(sub.get("evidence", ""))

            # Derive pages from actual [PAGE X] markers in evidence text
            # This prevents hallucinated page lists in the header
            evidence_pages = _extract_evidence_pages(evidence_txt)
            pages_str = ", ".join(str(p) for p in evidence_pages) if evidence_pages else "Not cited"

            lines.append(f"### {sub_id} — {sub_name}")
            lines.append("")
            lines.append(f"**Score:** {bar} **{label_str} ({pct}%)** &nbsp;&nbsp;|&nbsp;&nbsp; **Pages:** {pages_str}")
            lines.append("")

            # Evidence section
            if evidence_txt:
                lines.append("**Evidence:**")
                lines.append("")
                lines.extend(_format_evidence_for_display(evidence_txt))
                lines.append("")

            # Reasoning section
            reasoning_txt = _normalize_text(sub.get("reasoning", ""))
            if reasoning_txt:
                lines.append("**Reasoning:**")
                lines.append("")
                lines.append(reasoning_txt)
                lines.append("")

            # Gap to next section
            gap_txt = _normalize_text(sub.get("gap_to_next", ""))
            if gap_txt and score_val < 1.0:
                lines.append("**Gap to Maximum Score (Complete — 100%):**")
                lines.append("")
                _append_blockquote(lines, gap_txt)
                lines.append("")

            # Recommendation section
            rec_txt = _normalize_text(sub.get("recommendation", ""))
            if rec_txt and score_val < 1.0:
                rec_txt = _strip_mandatory_prefix(rec_txt)
                lines.append("**Recommendation [MANDATORY]:**")
                lines.append("")
                _append_blockquote(lines, rec_txt)
                lines.append("")

    # ─────────────────────────────────────────────
    # EVIDENCE INDEX
    # ─────────────────────────────────────────────
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Evidence Index")
    lines.append("")
    lines.append("| Sub-Criteria | Score | Pages | Summary |")
    lines.append("|---------------|:-----:|-------|---------|")

    for comp in component_results:
        for sub in comp.get("sub_results", []):
            sub_id = sub["sub_component_id"]
            sub_name = sub["sub_component_name"]
            if not sub.get("applicable", True):
                lines.append(f"| {sub_id} {sub_name} | N/A | — | Not applicable for {classification.capitalize()} |")
                continue
            if _is_evaluation_error(sub):
                lines.append(f"| {sub_id} {sub_name} | ⚠️ | — | Evaluation could not be completed — manual review required |")
                continue
            score_val = _safe_score(sub)
            pct = int(score_val * 100)
            ev_pages = _extract_evidence_pages(_normalize_text(sub.get("evidence", "")))
            pages_str = ", ".join(str(p) for p in ev_pages) if ev_pages else "—"
            summary = _clean_for_table_cell(_normalize_text(sub.get("evidence_summary", "")))
            lines.append(f"| {sub_id} {sub_name} | {pct}% | {pages_str} | {summary} |")

    lines.append("")
    lines.append("")

    # ─────────────────────────────────────────────
    # MANDATORY RECOMMENDATIONS
    # ─────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Mandatory Recommendations")
    lines.append("")

    rec_num = 0
    error_num = 0
    for comp in component_results:
        for sub in comp.get("sub_results", []):
            if _is_evaluation_error(sub):
                error_num += 1
                sub_id = sub["sub_component_id"]
                sub_name = sub["sub_component_name"]
                lines.append(f"**E{error_num}. [{sub_id} {sub_name}]** — ⚠️ Evaluation Error")
                lines.append("")
                lines.append("> This sub-component could not be automatically evaluated. Please review the source document against the rubric criteria and assign an appropriate score manually. This does not indicate the content is absent from the document.")
                lines.append("")
                continue
            score_val = _safe_score(sub)
            rec_txt = _normalize_text(sub.get("recommendation", ""))
            if sub.get("applicable", True) and score_val < 1.0 and rec_txt:
                rec_num += 1
                sub_id = sub["sub_component_id"]
                sub_name = sub["sub_component_name"]
                pct = int(score_val * 100)
                # Clean for recommendation section — keep as flowing text
                rec_text = _strip_mandatory_prefix(rec_txt).replace("\n", " ")
                # Ensure no stray braces in recommendation text
                rec_text = rec_text.replace('{', '').replace('}', '')
                lines.append(f"**{rec_num}. [{sub_id} {sub_name}]** — Current: {pct}%")
                lines.append("")
                lines.append(f"> {rec_text}")
                lines.append("")

    total_items = rec_num + error_num
    summary_parts = []
    if rec_num:
        summary_parts.append(f"{rec_num} mandatory recommendation{'s' if rec_num != 1 else ''}")
    if error_num:
        summary_parts.append(f"{error_num} requiring manual review")
    lines.append(f"**Total: {' + '.join(summary_parts) if summary_parts else '0 recommendations'}**")
    lines.append("")

    return "\n".join(lines)