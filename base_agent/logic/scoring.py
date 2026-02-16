"""Deterministic scoring engine. Pure math, no LLM calls."""


VALID_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}

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


def calculate_overall_score(all_sub_scores: list[float]) -> float:
    if not all_sub_scores:
        return 0.0
    raw_avg = sum(all_sub_scores) / len(all_sub_scores)
    return round(raw_avg * 100, 2)


def has_evidence(sub_result: dict) -> bool:
    score = sub_result.get("score")
    if score is None:
        return False
    return score > 0.0


def component_has_evidence(sub_results: list[dict]) -> bool:
    return any(has_evidence(sr) for sr in sub_results)