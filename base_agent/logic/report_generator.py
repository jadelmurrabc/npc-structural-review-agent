"""Generate the markdown report from scored results."""
from datetime import datetime
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

    # Collect applicable scores and count N/A items for overall calculation.
    all_scores: list[float] = []
    na_count: int = 0
    for comp in component_results:
        for sub in comp.get("sub_results", []):
            if sub.get("applicable", True) and sub.get("score") is not None:
                all_scores.append(sub["score"])
            elif not sub.get("applicable", True):
                na_count += 1

    overall_score = calculate_overall_score(all_scores, na_count=na_count)

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

        # Consolidated comment
        if comp.get("comment"):
            lines.append(f"> {comp['comment']}")
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

            # Guard against explicit None scores
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

            if sub.get("evidence"):
                lines.append("**Evidence:**")
                lines.append("")
                # Put evidence in a blockquote for visual separation
                for ev_line in sub["evidence"].split("\n"):
                    lines.append(f"> {ev_line}")
                lines.append("")

            if sub.get("reasoning"):
                lines.append("**Reasoning:**")
                lines.append("")
                lines.append(sub["reasoning"])
                lines.append("")

            if sub.get("gap_to_next") and score_val < 1.0:
                lines.append(f"**Gap to Maximum Score (Complete — 100%):**")
                lines.append("")
                lines.append(f"> {sub['gap_to_next']}")
                lines.append("")

            if sub.get("recommendation") and score_val < 1.0:
                rec_text = _strip_mandatory_prefix(sub["recommendation"])
                lines.append(f"**Recommendation [MANDATORY]:**")
                lines.append("")
                lines.append(f"> {rec_text}")
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
            summary = sub.get("evidence_summary", "")
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
            if sub.get("applicable", True) and score_val < 1.0 and sub.get("recommendation"):
                rec_num += 1
                sub_id = sub["sub_component_id"]
                sub_name = sub["sub_component_name"]
                pct = int(score_val * 100)
                rec_text = _strip_mandatory_prefix(sub["recommendation"]).replace("\n", " ")
                lines.append(f"**{rec_num}. [{sub_id} {sub_name}]** — Current: {pct}%")
                lines.append("")
                lines.append(f"> {rec_text}")
                lines.append("")

    lines.append(f"**Total: {rec_num} mandatory recommendations**")
    lines.append("")

    return "\n".join(lines)