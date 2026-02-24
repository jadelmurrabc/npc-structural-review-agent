"""Generate the markdown report from scored results."""
from __future__ import annotations

from datetime import datetime
import json
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


def _component_aggregates(sub_results: list[dict]) -> tuple[str, str]:
    """Return (band, label) for a list of sub-results."""
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
    return band, label


def _component_included(sub_results: list[dict], band: str) -> bool:
    """Determine if a criteria's content is 'Included' in the document.

    Returns False if:
    - The criteria band is Absent (0%), OR
    - All sub-criteria scores are 0.0 or None
    Returns True otherwise.
    """
    if band == "0%":
        return False
    applicable_scores = [
        s["score"] for s in sub_results
        if s.get("applicable", True) and s.get("score") is not None
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
        ]
        for k in preferred_keys:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

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
                out_lines.append(f'Quote: "{quote}" [PAGE {page}]')
            elif quote:
                out_lines.append(f'Quote: "{quote}"')
            else:
                # fallback dict to string (rare)
                out_lines.append(_extract_text_from_obj(item) or str(item))
        elif isinstance(item, str):
            out_lines.append(item.strip())
        else:
            out_lines.append(str(item))
    return "\n".join([ln for ln in out_lines if ln.strip()])


def _normalize_text(value: Any) -> str:
    """
    Normalize a value to clean text.
    Handles:
    - dict/list values (extract human string)
    - strings containing fenced ```json blocks
    - strings containing 'JSON' + inline JSON
    - strings containing 'evidence_list: [ ... ]'
    """
    if value is None:
        return ""

    # If already structured, extract immediately
    if isinstance(value, (dict, list)):
        return _extract_text_from_obj(value)

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
                    # keep any surrounding non-json text too
                    before = s[:start].strip()
                    after_txt = s[end + 3 :].strip()
                    parts = [p for p in [before, extracted, after_txt] if p]
                    return "\n".join(parts).strip()
            except Exception:
                # If JSON parse fails, just remove the fence markers (still keep content)
                before = s[:start].strip()
                inner = payload
                after_txt = s[end + 3 :].strip()
                parts = [p for p in [before, inner, after_txt] if p]
                return "\n".join(parts).strip()

    # 2) Handle lines like:
    # JSON
    # { ... }
    # Try to parse the first JSON object/array that appears after a 'JSON' marker
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
                    return "\n".join(parts).strip()
            except Exception:
                pass

    # 3) Handle "evidence_list: [ ... ]"
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
                        # keep any text before evidence_list label if exists
                        before = s[:idx].strip()
                        after_txt = s[je:].strip()
                        parts = [p for p in [before, formatted, after_txt] if p]
                        return "\n".join(parts).strip()
                except Exception:
                    pass

    return s


def re_find_json_marker(s: str) -> Optional[int]:
    """
    Find the start index of a JSON object/array that is likely introduced by a 'JSON' label.
    Returns index of '{' or '[' if found, else None.
    """
    # Prefer a literal 'JSON' line marker if present
    # e.g. "JSON\n{...}"
    pos = s.find("JSON")
    if pos != -1:
        # look for first { or [ after 'JSON'
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
    # N/A sub-criteria are excluded entirely (not counted in numerator or denominator).
    all_scores: list[float] = []
    for comp in component_results:
        for sub in comp.get("sub_results", []):
            if sub.get("applicable", True) and sub.get("score") is not None:
                all_scores.append(sub["score"])

    overall_score = calculate_overall_score(all_scores)

    # ─────────────────────────────────────────────
    # HEADER & SCORE OVERVIEW
    # ─────────────────────────────────────────────
    lines.append("# Structural Checklist Review")
    lines.append("")
    lines.append(f"| Field | Detail |")
    lines.append(f"|-------|--------|")
    lines.append(f"| **Strategy** | {strategy_title} |")
    lines.append(f"| **Entity** | {entity_name} |")
    lines.append(f"| **Classification** | {classification.capitalize()} |")
    lines.append(f"| **Review Date** | {review_date} |")
    lines.append(f"| **Overall Score** | **{overall_score}%** |")
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

        lines.append(f"---")
        lines.append("")
        lines.append(f"## Criteria {comp_id}: {comp_name}")
        lines.append("")
        lines.append(f"| Assessment | Included in Document |")
        lines.append(f"|:----------:|:--------------------:|")
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

            score_val = _safe_score(sub)
            label_str = score_label(score_val)
            pct = int(score_val * 100)
            pages = sub.get("pages", [])
            pages_str = ", ".join(str(p) for p in pages) if pages else "Not found"
            bar = _score_bar(score_val)

            lines.append(f"### {sub_id} — {sub_name}")
            lines.append("")
            lines.append(f"**Score:** {bar} **{label_str} ({pct}%)** &nbsp;&nbsp;|&nbsp;&nbsp; **Pages:** {pages_str}")
            lines.append("")

            evidence_txt = _normalize_text(sub.get("evidence", ""))
            if evidence_txt:
                lines.append("**Evidence:**")
                lines.append("")
                for ev_line in evidence_txt.splitlines():
                    lines.append(f"> {ev_line}" if ev_line.strip() else ">")
                lines.append("")

            reasoning_txt = _normalize_text(sub.get("reasoning", ""))
            if reasoning_txt:
                lines.append("**Reasoning:**")
                lines.append("")
                lines.append(reasoning_txt)
                lines.append("")

            gap_txt = _normalize_text(sub.get("gap_to_next", ""))
            if gap_txt and score_val < 1.0:
                lines.append(f"**Gap to Maximum Score (Complete — 100%):**")
                lines.append("")
                _append_blockquote(lines, gap_txt)
                lines.append("")

            rec_txt = _normalize_text(sub.get("recommendation", ""))
            if rec_txt and score_val < 1.0:
                rec_txt = _strip_mandatory_prefix(rec_txt)
                lines.append(f"**Recommendation [MANDATORY]:**")
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
            score_val = _safe_score(sub)
            pct = int(score_val * 100)
            pages = sub.get("pages", [])
            pages_str = ", ".join(str(p) for p in pages) if pages else "—"
            summary = _normalize_text(sub.get("evidence_summary", ""))
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
    for comp in component_results:
        for sub in comp.get("sub_results", []):
            score_val = _safe_score(sub)
            rec_txt = _normalize_text(sub.get("recommendation", ""))
            if sub.get("applicable", True) and score_val < 1.0 and rec_txt:
                rec_num += 1
                sub_id = sub["sub_component_id"]
                sub_name = sub["sub_component_name"]
                pct = int(score_val * 100)
                rec_text = _strip_mandatory_prefix(rec_txt).replace("\n", " ")
                lines.append(f"**{rec_num}. [{sub_id} {sub_name}]** — Current: {pct}%")
                lines.append("")
                lines.append(f"> {rec_text}")
                lines.append("")

    lines.append(f"**Total: {rec_num} mandatory recommendations**")
    lines.append("")

    return "\n".join(lines)