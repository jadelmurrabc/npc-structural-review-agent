"""Load evaluation criteria from the client-editable questions.json and applicability rules."""
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


def load_questions() -> dict:
    """Load the unified questions file (client-editable)."""
    filepath = CONFIG_DIR / "questions.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Questions file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checklist(classification: str) -> dict:
    """Return the full checklist from questions.json (backward-compatible interface)."""
    questions = load_questions()
    return {
        "classification": classification.lower().strip(),
        "components": questions["components"],
    }


def get_applicable_sub_components(classification: str) -> list[dict]:
    """Return a flat list of applicable sub-components with their rubrics and questions."""
    questions = load_questions()
    applicability = load_applicability(classification)
    applicable_ids = set(applicability["applicable"])
    conditional = applicability.get("conditional", {})

    result = []
    for component in questions["components"]:
        for sub in component["sub_components"]:
            sub_id = sub["id"]
            if sub_id in applicable_ids:
                result.append({
                    "component_id": component["id"],
                    "component_name": component["name"],
                    "sub_component_id": sub_id,
                    "sub_component_name": sub["name"],
                    "question": sub.get("question", ""),
                    "rubric": sub["scoring"],
                    "is_conditional": False,
                })
            elif sub_id in conditional:
                result.append({
                    "component_id": component["id"],
                    "component_name": component["name"],
                    "sub_component_id": sub_id,
                    "sub_component_name": sub["name"],
                    "question": sub.get("question", ""),
                    "rubric": sub["scoring"],
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
                if "scoring" in sub_copy and "rubric" not in sub_copy:
                    sub_copy["rubric"] = sub_copy["scoring"]
                subs.append(sub_copy)
        if subs:
            result.append({
                "id": component["id"],
                "name": component["name"],
                "sub_components": subs,
            })
    return result
