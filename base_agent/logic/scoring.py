"""Deterministic scoring engine. Pure math, no LLM calls."""


VALID_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}

# Total sub-criteria in the structural checklist (methodology page 15).
# The overall score formula is: (sum of grades / 20 questions) × 100.
# This denominator is FIXED at 20 regardless of classification.
TOTAL_SUB_COMPONENTS = 20

SCORE_LABELS = {
    0.0: "Absent",
    0.25: "Low",
    0.5: "Mid",
    0.75: "High",
    1.0: "Complete",
}

BAND_LABELS = {
    "0%": "Absent",
    "25%": "Low",
    "50%": "Moderate",
    "75%": "High",
    "100%": "Complete",
}


def validate_score(score: float) -> float:
    if score not in VALID_SCORES:
        closest = min(VALID_SCORES, key=lambda x: abs(x - score))
        return closest
    return score


def score_label(score: float) -> str:
    return SCORE_LABELS.get(score, "Unknown")


def calculate_display_band(raw_average: float) -> str:
    if raw_average >= 1.0:
        return "100%"
    elif raw_average >= 0.75:
        return "75%"
    elif raw_average >= 0.50:
        return "50%"
    elif raw_average >= 0.25:
        return "25%"
    else:
        return "0%"


def band_label(band: str) -> str:
    return BAND_LABELS.get(band, "Unknown")


def aggregate_component(sub_scores: list[float]) -> dict:
    if not sub_scores:
        return {
            "raw_average": 0.0,
            "display_band": "0%",
            "display_label": "Absent",
            "sub_count": 0,
        }
    raw_avg = sum(sub_scores) / len(sub_scores)
    band = calculate_display_band(raw_avg)
    return {
        "raw_average": round(raw_avg, 4),
        "display_band": band,
        "display_label": band_label(band),
        "sub_count": len(sub_scores),
    }


def calculate_overall_score(applicable_scores: list[float], na_count: int = 0) -> float:
    """Calculate the overall structural checklist score as a percentage.

    Per NPC methodology (page 15):
        Overall Score = (sum of grades per question / 20 questions) × 100

    The denominator is ALWAYS 20 (total sub-criteria in the framework).
    N/A sub-criteria are treated as fully compliant (1.0 each) so that
    classifications with fewer applicable items (e.g., Entity with 17)
    are not structurally capped below 100%.

    Args:
        applicable_scores: List of scores (0.0-1.0) for sub-criteria
            that were evaluated (applicable=True, score is not None).
        na_count: Number of Not Applicable sub-criteria (including
            conditional items that were excluded). Each contributes 1.0
            to the numerator.

    Returns:
        Overall score as a percentage (0.0-100.0).
    """
    if not applicable_scores and na_count == 0:
        return 0.0
    numerator = sum(applicable_scores) + (na_count * 1.0)
    return round(numerator / TOTAL_SUB_COMPONENTS * 100, 2)


def has_evidence(sub_result: dict) -> bool:
    score = sub_result.get("score")
    if score is None:
        return False
    return score > 0.0


def component_has_evidence(sub_results: list[dict]) -> bool:
    return any(has_evidence(sr) for sr in sub_results)