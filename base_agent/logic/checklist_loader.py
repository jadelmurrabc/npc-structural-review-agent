"""Load classification-specific checklists and applicability rules from config/."""
import json
from base_agent.config import CONFIG_DIR


def load_applicability(classification: str) -> dict:
    """Return applicability rules for a given classification."""
    with open(CONFIG_DIR / "applicability.json", "r") as f:
        data = json.load(f)
    classification = classification.lower().strip()
    if classification not in data:
        raise ValueError(f"Unknown classification: {classification}. Must be entity, sectoral, or thematic.")
    return data[classification]


def load_checklist(classification: str) -> dict:
    """Return the full checklist (components + rubrics) for a classification."""
    classification = classification.lower().strip()
    filepath = CONFIG_DIR / f"checklist_{classification}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Checklist not found: {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


def get_applicable_sub_components(classification: str) -> list[dict]:
    """Return a flat list of applicable sub-components with their rubrics for a classification."""
    checklist = load_checklist(classification)
    applicability = load_applicability(classification)
    applicable_ids = set(applicability["applicable"])
    conditional = applicability.get("conditional", {})

    result = []
    for component in checklist["components"]:
        for sub in component["sub_components"]:
            sub_id = sub["id"]
            if sub_id in applicable_ids:
                result.append({
                    "component_id": component["id"],
                    "component_name": component["name"],
                    "sub_component_id": sub_id,
                    "sub_component_name": sub["name"],
                    "rubric": sub["rubric"],
                    "is_conditional": False,
                })
            elif sub_id in conditional:
                result.append({
                    "component_id": component["id"],
                    "component_name": component["name"],
                    "sub_component_id": sub_id,
                    "sub_component_name": sub["name"],
                    "rubric": sub["rubric"],
                    "is_conditional": True,
                    "conditional_rule": conditional[sub_id],
                })
    return result


def get_components_with_subs(classification: str) -> list[dict]:
    """Return components with only their applicable sub-components included."""
    checklist = load_checklist(classification)
    applicability = load_applicability(classification)
    applicable_ids = set(applicability["applicable"])
    conditional = applicability.get("conditional", {})
    all_relevant = applicable_ids | set(conditional.keys())

    result = []
    for component in checklist["components"]:
        subs = []
        for sub in component["sub_components"]:
            if sub["id"] in all_relevant:
                sub_copy = dict(sub)
                sub_copy["is_conditional"] = sub["id"] in conditional
                subs.append(sub_copy)
        if subs:
            result.append({
                "id": component["id"],
                "name": component["name"],
                "sub_components": subs,
            })
    return result