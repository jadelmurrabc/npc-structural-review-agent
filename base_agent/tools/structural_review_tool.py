"""The main structural review tool. Called by the ADK agent."""
import json
import logging
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from base_agent.logic.checklist_loader import get_applicable_sub_components, load_applicability, load_checklist
from base_agent.logic.scoring import validate_score
from base_agent.logic.report_generator import generate_report

logger = logging.getLogger(__name__)

MAX_WORKERS = 10


def _get_client():
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    return genai.Client(vertexai=True, project=project, location=location)


def _call_gemini(prompt, system_instruction=""):
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


def _build_sub_component_prompt(sub_info, classification, document_text):
    sub_id = sub_info["sub_component_id"]
    sub_name = sub_info["sub_component_name"]
    comp_name = sub_info["component_name"]
    rubric = sub_info["rubric"]
    question = sub_info.get("question", "")

    rubric_lines = []
    for score_val, desc in rubric.items():
        rubric_lines.append(f"  {score_val}: {desc}")
    rubric_text = chr(10).join(rubric_lines)

    conditional_note = ""
    if sub_info.get("is_conditional"):
        conditional_note = (
            chr(10) + chr(10) +
            "IMPORTANT: This sub-component is CONDITIONAL. It is not required for this "
            "classification. If you find relevant content in the document, evaluate and score it. "
            "If no relevant content exists at all, respond with applicable=false."
        )

    json_example = json.dumps({
        "sub_component_id": sub_id,
        "sub_component_name": sub_name,
        "applicable": True,
        "score": 0.5,
        "pages": [12, 15],
        "evidence": "Exact quotes or descriptions of what was found in the document",
        "evidence_summary": "One-line summary for the index table",
        "reasoning": "Why this score was selected and not a higher one, referencing the rubric levels",
        "gap_to_next": "What specifically is needed to reach the next scoring level",
        "recommendation": "Concrete, actionable recommendation for improvement"
    }, indent=2)

    prompt = (
        f"You are evaluating Sub-Component {sub_id}: {sub_name}" + chr(10) +
        f"Part of Component: {comp_name}" + chr(10) +
        f"Classification: {classification}" + chr(10) + chr(10) +
        "EVALUATION QUESTION:" + chr(10) +
        question + chr(10) + chr(10) +
        "SCORING RUBRIC (you MUST select exactly one of these scores):" + chr(10) +
        rubric_text +
        conditional_note + chr(10) + chr(10) +
        "DETAILED INSTRUCTIONS:" + chr(10) +
        "1. Read the ENTIRE document carefully looking for content related to this sub-component." + chr(10) +
        "2. Extract and quote the specific evidence you find. Note the exact page numbers using [PAGE X] markers." + chr(10) +
        "3. Compare the evidence against EACH rubric level from 0.0 to 1.0." + chr(10) +
        "4. Select the score that BEST matches the evidence. Be conservative - if between two levels, choose the lower one." + chr(10) +
        "5. Your score MUST be exactly one of: 0.0, 0.25, 0.5, 0.75, or 1.0. No other values." + chr(10) +
        "6. In your reasoning, explain:" + chr(10) +
        "   - What specific content from the document matches the selected score level" + chr(10) +
        "   - What the NEXT higher score level requires that is MISSING from the document" + chr(10) +
        "   - Why the evidence does NOT qualify for the next level up" + chr(10) +
        "7. In gap_to_next, state the single most important thing the entity needs to add or improve." + chr(10) +
        "8. Write a specific, actionable recommendation if the score is below 1.0." + chr(10) + chr(10) +
        "Respond with ONLY this JSON structure (no other text):" + chr(10) +
        json_example + chr(10) + chr(10) +
        "DOCUMENT TO EVALUATE:" + chr(10) +
        document_text
    )
    return prompt


def _extract_json(text):
    text = text.strip()
    if text.startswith("`"):
        lines = text.split(chr(10))
        if lines[0].startswith("`"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "`":
            lines = lines[:-1]
        text = chr(10).join(lines).strip()
    if not text.startswith("{"):
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
        if ch == chr(92):
            if in_string:
                escape = True
            continue
        if ch == chr(34) and not escape:
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


def _parse_sub_response(response_text, sub_info):
    text = response_text.strip()
    json_str = _extract_json(text)
    if json_str is None:
        logger.error("Could not find JSON in response for %s", sub_info["sub_component_id"])
        return _fallback_sub(sub_info)
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("JSON parse error for %s: %s", sub_info["sub_component_id"], e)
        return _fallback_sub(sub_info)
    if result.get("applicable", True) and result.get("score") is not None:
        result["score"] = validate_score(float(result["score"]))
    return result


def _fallback_sub(sub_info):
    return {
        "sub_component_id": sub_info["sub_component_id"],
        "sub_component_name": sub_info["sub_component_name"],
        "applicable": not sub_info.get("is_conditional", False),
        "score": 0.0,
        "pages": [],
        "evidence": "Error: Could not parse LLM response for this sub-component.",
        "evidence_summary": "Parse error",
        "reasoning": "LLM response could not be parsed. Manual review required.",
        "gap_to_next": "",
        "recommendation": "Manual review required due to processing error.",
    }


def _evaluate_sub_component(sub_info, classification, document_text):
    sub_id = sub_info["sub_component_id"]
    logger.info("  [Thread] Evaluating %s: %s", sub_id, sub_info["sub_component_name"])
    prompt = _build_sub_component_prompt(sub_info, classification, document_text)
    raw_response = _call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION)
    result = _parse_sub_response(raw_response, sub_info)
    result["sub_component_id"] = sub_id
    result["sub_component_name"] = sub_info["sub_component_name"]
    logger.info("  [Thread] Done %s: score=%s", sub_id, result.get("score"))
    return result


def _build_comment_prompt(comp_name, sub_results):
    summary_lines = []
    for sub in sub_results:
        if sub.get("applicable", True):
            score = sub.get("score", 0.0)
            pct = int((score or 0) * 100)
            summary_lines.append(f"  {sub['sub_component_id']} {sub['sub_component_name']}: {pct}% - {sub.get('evidence_summary', 'N/A')}")
        else:
            summary_lines.append(f"  {sub['sub_component_id']} {sub['sub_component_name']}: Not Applicable")
    summary_text = chr(10).join(summary_lines)

    return (
        f"Write a 1-2 sentence consolidated assessment comment for Component: {comp_name}" + chr(10) + chr(10) +
        "Sub-component results:" + chr(10) +
        summary_text + chr(10) + chr(10) +
        "Style: Demonstrates/Provides [level] coverage [by/through] [what was found]; however, [what was missing or insufficient]." + chr(10) + chr(10) +
        "Respond with ONLY the comment text, no JSON, no quotes, no prefix."
    )


def _add_na_subs(sub_results, classification, all_sub_ids):
    applicability = load_applicability(classification)
    na_ids = set(applicability["not_applicable"])
    existing_ids = {s["sub_component_id"] for s in sub_results}
    for sub_id in all_sub_ids:
        if sub_id in na_ids and sub_id not in existing_ids:
            sub_results.append({
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
    sub_results.sort(key=lambda x: x["sub_component_id"])
    return sub_results


def structural_review(
    document_text: str,
    classification: str,
    strategy_title: str = "Untitled Strategy",
    entity_name: str = "Unknown Entity",
) -> dict:
    """Performs a full structural checklist review of a strategy document.

    Evaluates the document against 7 structural components and 20 sub-components,
    producing a scored markdown report with evidence, reasoning, and recommendations.
    Each sub-component is evaluated independently with its own focused prompt, and
    evaluations run in parallel for speed.

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

        applicable_subs = get_applicable_sub_components(classification)
        logger.info("Evaluating %d sub-components in parallel (max %d threads)", len(applicable_subs), MAX_WORKERS)

        sub_results_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_sub = {
                executor.submit(_evaluate_sub_component, sub_info, classification, document_text): sub_info
                for sub_info in applicable_subs
            }
            for future in as_completed(future_to_sub):
                sub_info = future_to_sub[future]
                sub_id = sub_info["sub_component_id"]
                try:
                    result = future.result()
                    sub_results_map[sub_id] = result
                except Exception as e:
                    logger.error("Thread failed for %s: %s", sub_id, e)
                    sub_results_map[sub_id] = _fallback_sub(sub_info)

        full_checklist = load_checklist(classification)

        component_results = []
        for comp in full_checklist["components"]:
            comp_id = comp["id"]
            comp_name = comp["name"]
            all_sub_ids = [s["id"] for s in comp["sub_components"]]

            comp_sub_results = []
            for sub_id in all_sub_ids:
                if sub_id in sub_results_map:
                    comp_sub_results.append(sub_results_map[sub_id])

            comp_sub_results = _add_na_subs(comp_sub_results, classification, all_sub_ids)

            try:
                comment_prompt = _build_comment_prompt(comp_name, comp_sub_results)
                comment = _call_gemini(comment_prompt, system_instruction=SYSTEM_INSTRUCTION).strip()
                if comment.startswith(chr(34)) and comment.endswith(chr(34)):
                    comment = comment[1:-1]
            except Exception as e:
                logger.warning("Could not generate comment for component %d: %s", comp_id, e)
                comment = ""

            component_results.append({
                "id": comp_id,
                "name": comp_name,
                "comment": comment,
                "sub_results": comp_sub_results,
            })

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
        logger.info("Review complete. Overall score: %.2f%% (%d sub-components evaluated)", overall, len(all_scores))

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
