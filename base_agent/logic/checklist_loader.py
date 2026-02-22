"""Load evaluation criteria from the client-editable questions.json and applicability rules."""
import json
from base_agent.config import CONFIG_DIR, load_config_json


def load_applicability(classification: str) -> dict:
    """Return applicability rules for a given classification."""
    data = load_config_json("applicability.json")
    classification = classification.lower().strip()
    if classification not in data:
        raise ValueError(f"Unknown classification: {classification}. Must be entity, sectoral, or thematic.")
    return data[classification]


def load_questions() -> dict:
    """Load the unified questions file (client-editable)."""
    return load_config_json("questions.json")


def _resolve_rubric(scoring: dict, classification: str) -> dict:
    """Resolve a scoring dict to the classification-specific rubric.

    Supports two formats:
      - Nested (new):  {"entity": {"0.0": "...", ...}, "sectoral": {...}, ...}
      - Flat (legacy):  {"0.0": "...", "0.25": "...", ...}

    Returns the flat rubric dict for the given classification.
    """
    classification = classification.lower().strip()
    if classification in scoring:
        return scoring[classification]
    return scoring


def load_checklist(classification: str) -> dict:
    """Return the full checklist from questions.json (backward-compatible interface)."""
    questions = load_questions()
    return {
        "classification": classification.lower().strip(),
        "components": questions["components"],
    }


def get_applicable_sub_components(classification: str) -> list[dict]:
    """Return a flat list of applicable sub-components with their rubrics and questions.

    Items in 'applicable' are always scored.
    Items in 'conditional' are included but marked so the LLM can exclude
    them if no relevant content is found in the document.
    Items in 'not_applicable' are excluded entirely.
    """
    questions = load_questions()
    applicability = load_applicability(classification)
    applicable_ids = set(applicability["applicable"])
    conditional = applicability.get("conditional", {})

    result = []
    for component in questions["components"]:
        for sub in component["sub_components"]:
            sub_id = sub["id"]
            rubric = _resolve_rubric(sub["scoring"], classification)

            if sub_id in applicable_ids:
                result.append({
                    "component_id": component["id"],
                    "component_name": component["name"],
                    "sub_component_id": sub_id,
                    "sub_component_name": sub["name"],
                    "question": sub.get("question", ""),
                    "rubric": rubric,
                    "is_conditional": False,
                })
            elif sub_id in conditional:
                result.append({
                    "component_id": component["id"],
                    "component_name": component["name"],
                    "sub_component_id": sub_id,
                    "sub_component_name": sub["name"],
                    "question": sub.get("question", ""),
                    "rubric": rubric,
                    "is_conditional": True,
                    "conditional_rule": conditional[sub_id],
                })
    return result


def get_components_with_subs(classification: str) -> list[dict]:
    """Return components with only their applicable sub-components included."""
    questions = load_questions()
    applicability = load_applicability(classification)
    applicable_ids = set(applicability["applicable"])
    conditional = applicability.get("conditional", {})
    all_relevant = applicable_ids | set(conditional.keys())

    result = []
    for component in questions["components"]:
        subs = []
        for sub in component["sub_components"]:
            if sub["id"] in all_relevant:
                sub_copy = dict(sub)
                sub_copy["is_conditional"] = sub["id"] in conditional
                resolved = _resolve_rubric(sub["scoring"], classification)
                sub_copy["rubric"] = resolved
                sub_copy["scoring"] = resolved
                subs.append(sub_copy)
        if subs:
            result.append({
                "id": component["id"],
                "name": component["name"],
                "sub_components": subs,
            })
    return result