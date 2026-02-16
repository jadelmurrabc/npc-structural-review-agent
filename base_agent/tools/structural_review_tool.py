"""The main structural review tool. Called by the ADK agent."""
import json
import logging
import os
import re
import traceback
from google import genai
from google.genai import types
from base_agent.logic.checklist_loader import get_components_with_subs, load_applicability
from base_agent.logic.scoring import validate_score, aggregate_component, score_label
from base_agent.logic.report_generator import generate_report

logger = logging.getLogger(__name__)


def _get_client():
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    return genai.Client(vertexai=True, project=project, location=location)


def _call_gemini(prompt: str, system_instruction: str = "") -> str:
    client = _get_client()
    model = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-pro")
    config = types.GenerateContentConfig(
        temperature=0.1,
        system_instruction=system_instruction if system_instruction else None,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text


SYSTEM_INSTRUCTION = (
    "You are an expert strategy document reviewer for Qatars National Planning Council. "
    "You evaluate strategy documents against a structural checklist with precise scoring rubrics. "
    "You must be thorough, conservative (when in doubt score lower), and always cite page numbers. "
    "You MUST respond with valid JSON only. No markdown, no explanation outside the JSON."
)


def _build_component_prompt(component, classification, document_text):
    comp_name = component["name"]
    comp_id = component["id"]
    subs = component["sub_components"]

    sub_instructions = []
    for sub in subs:
        sub_id = sub["id"]
        sub_name = sub["name"]
        rubric = sub["rubric"]
        conditional_note = ""
        if sub.get("is_conditional"):
            conditional_note = " (CONDITIONAL: Not required. If you find relevant content score it. If absent set applicable to false.)"

        rubric_lines = []
        for score_val, desc in rubric.items():
            rubric_lines.append(f"      {score_val}: {desc}")
        rubric_text = "\n".join(rubric_lines)

        sub_instructions.append(
            f"  Sub-Component {sub_id} - {sub_name}{conditional_note}\n"
            f"    Scoring Rubric:\n"
            f"{rubric_text}"
        )

    subs_block = "\n\n".join(sub_instructions)

    json_example = json.dumps({
        "component_id": comp_id,
        "component_name": comp_name,
        "comment": "consolidated comment here",
        "sub_results": [{
            "sub_component_id": "X.X",
            "sub_component_name": "name",
            "applicable": True,
            "score": 0.5,
            "pages": [12, 15],
            "evidence": "what was found in the document",
            "evidence_summary": "one-line summary for the index table",
            "reasoning": "why this score and not higher",
            "gap_to_next": "what is needed to reach the next level",
            "recommendation": "specific actionable recommendation"
        }]
    }, indent=2)

    prompt = f"""Evaluate Component {comp_id}: {comp_name} of this {classification} strategy document.

SUB-COMPONENTS TO EVALUATE:

{subs_block}

INSTRUCTIONS:
For each sub-component above:
1. Search the document for relevant content. Quote brief evidence snippets.
2. Note the page numbers where evidence is found (look for [PAGE X] markers).
3. Compare evidence against the scoring rubric and select the best-fit score.
4. Score MUST be exactly one of: 0.0, 0.25, 0.5, 0.75, or 1.0
5. Explain your reasoning: why this score and not higher.
6. State what is needed to reach the next level.
7. Write a recommendation if score is below 1.0.

After all sub-components write a 1-2 sentence consolidated comment for the whole component.
Style: Demonstrates/Provides [level] coverage [by/through] [what was found]; however [what was missing].

Respond with this exact JSON structure:
{json_example}

DOCUMENT:
{document_text}"""

    return prompt


def _extract_json(text):
    """Extract the first valid JSON object from text that may have trailing content."""
    text = text.strip()
    # Remove markdown fences
    if text.startswith("`"):
        lines = text.split("\n")
        if lines[0].startswith("`"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "`":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # Find the JSON object by matching braces
    if not text.startswith("{"):
        # Try to find first {
        idx = text.find("{")
        if idx == -1:
            return None
        text = text[idx:]
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[:i+1]
    return None


def _parse_component_response(response_text, component):
    text = response_text.strip()
    json_str = _extract_json(text)
    if json_str is None:
        logger.error("Could not find JSON in LLM response")
        logger.error("Raw response: %s", text[:500])
        return _fallback_component(component)
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse extracted JSON: %s", e)
        logger.error("Extracted text: %s", json_str[:500])
        return _fallback_component(component)
    for sub in result.get("sub_results", []):
        if sub.get("applicable", True) and sub.get("score") is not None:
            sub["score"] = validate_score(float(sub["score"]))
    return result


def _fallback_component(component):
    sub_results = []
    for sub in component["sub_components"]:
        sub_results.append({
            "sub_component_id": sub["id"],
            "sub_component_name": sub["name"],
            "applicable": not sub.get("is_conditional", False),
            "score": 0.0,
            "pages": [],
            "evidence": "Error: Could not parse LLM response.",
            "evidence_summary": "Parse error",
            "reasoning": "LLM response could not be parsed. Manual review required.",
            "gap_to_next": "",
            "recommendation": "Manual review required due to processing error.",
        })
    return {
        "id": component["id"],
        "name": component["name"],
        "comment": "Error: Could not parse LLM evaluation for this component.",
        "sub_results": sub_results,
    }


def _add_na_subs(component_result, classification, all_sub_ids):
    applicability = load_applicability(classification)
    na_ids = set(applicability["not_applicable"])
    existing_ids = {s["sub_component_id"] for s in component_result.get("sub_results", [])}
    for sub_id in all_sub_ids:
        if sub_id in na_ids and sub_id not in existing_ids:
            component_result["sub_results"].append({
                "sub_component_id": sub_id,
                "sub_component_name": sub_id,
                "applicable": False,
                "score": None,
                "pages": [],
                "evidence": "",
                "evidence_summary": "",
                "reasoning": "",
                "gap_to_next": "",
                "recommendation": "",
                "na_reason": f"Not required for {classification.capitalize()} strategies",
            })
    component_result["sub_results"].sort(key=lambda x: x["sub_component_id"])
    return component_result


def structural_review(
    document_text: str,
    classification: str,
    strategy_title: str = "Untitled Strategy",
    entity_name: str = "Unknown Entity",
) -> dict:
    """Performs a full structural checklist review of a strategy document.

    Evaluates the document against 7 structural components and 20 sub-components,
    producing a scored markdown report with evidence, reasoning, and recommendations.

    Args:
        document_text: The full text of the strategy document to review.
        classification: The document classification. Must be one of: entity, sectoral, thematic.
        strategy_title: The title of the strategy document.
        entity_name: The name of the submitting government entity.

    Returns:
        A dict with the markdown report and overall score.
    """
    try:
        classification = classification.lower().strip()
        if classification not in ("entity", "sectoral", "thematic"):
            return {"error": True, "message": f"Invalid classification: {classification}. Must be entity, sectoral, or thematic."}

        if not document_text or len(document_text.strip()) < 100:
            return {"error": True, "message": "Document text is too short or empty. Please provide the full strategy document text."}

        logger.info("Starting structural review: %s (%s)", strategy_title, classification)

        components = get_components_with_subs(classification)

        from base_agent.logic.checklist_loader import load_checklist
        full_checklist = load_checklist(classification)
        comp_all_subs = {}
        for comp in full_checklist["components"]:
            comp_all_subs[comp["id"]] = [s["id"] for s in comp["sub_components"]]

        component_results = []
        for component in components:
            logger.info("Evaluating Component %d: %s", component["id"], component["name"])
            prompt = _build_component_prompt(component, classification, document_text)
            raw_response = _call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION)
            parsed = _parse_component_response(raw_response, component)
            parsed["id"] = component["id"]
            parsed["name"] = component["name"]
            all_sub_ids = comp_all_subs.get(component["id"], [])
            parsed = _add_na_subs(parsed, classification, all_sub_ids)
            component_results.append(parsed)

        report = generate_report(
            strategy_title=strategy_title,
            entity_name=entity_name,
            classification=classification,
            component_results=component_results,
        )

        all_scores = []
        for comp in component_results:
            for sub in comp.get("sub_results", []):
                if sub.get("applicable", True) and sub.get("score") is not None:
                    all_scores.append(sub["score"])

        overall = round(sum(all_scores) / len(all_scores) * 100, 2) if all_scores else 0.0
        logger.info("Review complete. Overall score: %.2f%%", overall)

        return {
            "error": False,
            "overall_score": overall,
            "report": report,
        }

    except Exception as e:
        logger.exception("Structural review failed")
        return {
            "error": True,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
