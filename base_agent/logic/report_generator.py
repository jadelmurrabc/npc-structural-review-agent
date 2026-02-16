"""Generate the markdown report from scored results."""
from datetime import datetime
from base_agent.logic.scoring import (
    calculate_display_band, band_label, calculate_overall_score,
    component_has_evidence, score_label,
)


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
        component_results: List of component dicts, each containing sub_results.

    Returns:
        Complete markdown report as a string.
    """
    lines = []
    review_date = datetime.now().strftime("%Y-%m-%d")

    # Collect all scores for overall calculation
    all_scores = []
    for comp in component_results:
        for sub in comp.get("sub_results", []):
            if sub.get("applicable", True) and sub.get("score") is not None:
                all_scores.append(sub["score"])

    overall_score = calculate_overall_score(all_scores)

    # --- Header ---
    lines.append("# Structural Checklist Review")
    lines.append("")
    lines.append(f"**Strategy:** {strategy_title}")
    lines.append(f"**Entity:** {entity_name}")
    lines.append(f"**Classification:** {classification.capitalize()}")
    lines.append(f"**Review Date:** {review_date}")
    lines.append(f"**Overall Structural Score:** {overall_score}%")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Summary Scorecard ---
    lines.append("## Summary Scorecard")
    lines.append("")
    lines.append("| # | Component | Included | Assessment | Score |")
    lines.append("|---|-----------|----------|------------|-------|")

    for comp in component_results:
        comp_id = comp["id"]
        comp_name = comp["name"]
        sub_results = comp.get("sub_results", [])
        applicable_scores = [s["score"] for s in sub_results if s.get("applicable", True) and s.get("score") is not None]
        has_content = component_has_evidence(sub_results)
        included = "Yes" if has_content else "No"

        if applicable_scores:
            raw_avg = sum(applicable_scores) / len(applicable_scores)
            band = calculate_display_band(raw_avg)
            label = band_label(band)
        else:
            band = "0%"
            label = "Absent"

        lines.append(f"| {comp_id} | {comp_name} | {included} | {label} | {band} |")

    lines.append("")
    lines.append("Assessment Scale: Absent (0%) - Low (25%) - Moderate (50%) - High (75%) - Complete (100%)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Per-Component Detail ---
    for comp in component_results:
        comp_id = comp["id"]
        comp_name = comp["name"]
        sub_results = comp.get("sub_results", [])
        applicable_scores = [s["score"] for s in sub_results if s.get("applicable", True) and s.get("score") is not None]
        has_content = component_has_evidence(sub_results)

        if applicable_scores:
            raw_avg = sum(applicable_scores) / len(applicable_scores)
            band = calculate_display_band(raw_avg)
            label = band_label(band)
        else:
            band = "0%"
            label = "Absent"

        lines.append(f"## Component {comp_id}: {comp_name}")
        lines.append("")
        lines.append(f"**Assessment:** {label} ({band})")
        lines.append(f"**Included in Document:** {'Yes' if has_content else 'No'}")
        lines.append("")

        # Consolidated comment
        if comp.get("comment"):
            lines.append(comp["comment"])
            lines.append("")

        # Sub-component details
        for sub in sub_results:
            sub_id = sub["sub_component_id"]
            sub_name = sub["sub_component_name"]

            lines.append(f"### Sub-Component {sub_id} -- {sub_name}")
            lines.append("")

            if not sub.get("applicable", True):
                reason = sub.get("na_reason", f"Not applicable for {classification.capitalize()} classification")
                lines.append(f"**Status:** Not Applicable ({reason})")
                lines.append("")
                lines.append("---")
                lines.append("")
                continue

            score_val = sub.get("score", 0.0)
            label_str = score_label(score_val)
            pct = int(score_val * 100)
            pages = sub.get("pages", [])
            pages_str = ", ".join(str(p) for p in pages) if pages else "Not found"

            lines.append(f"**Score:** {label_str} ({pct}%)")
            lines.append(f"**Pages:** {pages_str}")
            lines.append("")

            if sub.get("evidence"):
                lines.append("**Evidence:**")
                lines.append(sub["evidence"])
                lines.append("")

            if sub.get("reasoning"):
                lines.append("**Reasoning:**")
                lines.append(sub["reasoning"])
                lines.append("")

            if sub.get("gap_to_next"):
                next_score = min(score_val + 0.25, 1.0)
                next_label = score_label(next_score)
                next_pct = int(next_score * 100)
                lines.append(f"**To reach the next level ({next_label} -- {next_pct}%):**")
                lines.append(sub["gap_to_next"])
                lines.append("")

            if sub.get("recommendation"):
                lines.append("**Recommendation:** [MANDATORY]")
                lines.append(sub["recommendation"])
                lines.append("")

            lines.append("---")
            lines.append("")

    # --- Evidence Index ---
    lines.append("## Evidence Index")
    lines.append("")
    lines.append("| Sub-Component | Score | Pages | Summary |")
    lines.append("|---------------|-------|-------|---------|")

    for comp in component_results:
        for sub in comp.get("sub_results", []):
            sub_id = sub["sub_component_id"]
            if not sub.get("applicable", True):
                lines.append(f"| {sub_id} {sub['sub_component_name']} | N/A | -- | Not applicable for {classification.capitalize()} |")
                continue
            score_val = sub.get("score", 0.0)
            pct = int(score_val * 100)
            pages = sub.get("pages", [])
            pages_str = ", ".join(str(p) for p in pages) if pages else "--"
            summary = sub.get("evidence_summary", "")
            lines.append(f"| {sub_id} {sub['sub_component_name']} | {pct}% | {pages_str} | {summary} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Mandatory Recommendations ---
    lines.append("## Mandatory Recommendations")
    lines.append("")
    lines.append("| # | Sub-Component | Current | Recommendation |")
    lines.append("|---|---------------|---------|----------------|")

    rec_num = 0
    for comp in component_results:
        for sub in comp.get("sub_results", []):
            if sub.get("applicable", True) and sub.get("score", 0.0) < 1.0 and sub.get("recommendation"):
                rec_num += 1
                sub_id = sub["sub_component_id"]
                pct = int(sub["score"] * 100)
                rec_text = sub["recommendation"].replace("\n", " ")
                lines.append(f"| {rec_num} | {sub_id} {sub['sub_component_name']} | {pct}% | {rec_text} |")

    lines.append("")
    lines.append(f"Total: {rec_num} mandatory recommendations")

    return "\n".join(lines)