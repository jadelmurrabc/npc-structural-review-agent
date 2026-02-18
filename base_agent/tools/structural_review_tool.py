"""The main structural review tool. Called by the ADK agent."""
import json
import logging
import os
import random
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from google.adk.tools import ToolContext
from base_agent.logic.checklist_loader import get_applicable_sub_components, load_applicability, load_checklist
from base_agent.logic.scoring import validate_score, calculate_overall_score
from base_agent.logic.report_generator import generate_report

logger = logging.getLogger(__name__)

MAX_WORKERS = 4  # Keep low to avoid 429 rate limit storms on Agent Engine
MAX_RETRIES = 5
RETRY_BASE_DELAY = 8  # seconds — generous to survive rate limit windows


def _get_client():
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    return genai.Client(vertexai=True, project=project, location=location)


def _get_max_page(document_text: str) -> int:
    """Extract the highest [PAGE X] number found in the document text."""
    page_nums = re.findall(r'\[PAGE\s+(\d+)\]', document_text)
    if page_nums:
        return max(int(p) for p in page_nums)
    return 0


def _extract_entity_name(document_text: str) -> str | None:
    """Try to extract the entity/organization name from the first few pages."""
    header = document_text[:3000]
    patterns = [
        r'(?:prepared by|submitted by|issued by|published by)[:\s]+([A-Z][A-Za-z\s&,]+?)(?:\n|\.)',
        r'(Ministry of [A-Za-z\s&]+)',
        r'(National [A-Za-z\s&]+ (?:Committee|Council|Authority|Office|Commission|Agency|Board))',
        r'((?:General |Supreme )?Authority (?:of|for) [A-Za-z\s&]+)',
        r'(State of Qatar\s*[-\u2013\u2014]\s*[A-Za-z\s&]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, header, re.IGNORECASE)
        if match:
            name = match.group(1).strip().rstrip(',.')
            if 5 < len(name) < 100:
                return name
    return None


def _extract_strategy_title(document_text: str) -> str | None:
    """Try to extract the strategy title from the first page."""
    header = document_text[:2000]
    patterns = [
        r'(National [A-Za-z\s&]+ Strategy[\s\d\-\u2013\u2014]*)',
        r'([A-Za-z\s&]+ Strategic Plan[\s\d\-\u2013\u2014]*)',
        r'([A-Za-z\s&]+ Strategy\s+\d{4}\s*[-\u2013\u2014]\s*\d{4})',
        r'(Qatar [A-Za-z\s&]+ Strategy[\s\d\-\u2013\u2014]*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, header, re.IGNORECASE)
        if match:
            title = match.group(1).strip().rstrip(',.')
            if 10 < len(title) < 150:
                return title
    return None


def _call_gemini(prompt, system_instruction=""):
    """Call Gemini with robust retry logic for transient errors."""
    client = _get_client()
    model = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-pro")
    config = types.GenerateContentConfig(
        temperature=0.0,
        system_instruction=system_instruction if system_instruction else None,
    )
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            retryable = any(keyword in error_str for keyword in [
                "429", "resource_exhausted", "rate_limit",
                "500", "internal", "503", "unavailable",
                "capacity", "overloaded", "deadline", "timeout",
            ])
            if retryable:
                base_delay = RETRY_BASE_DELAY * (2 ** attempt)
                jitter = random.uniform(0, base_delay * 0.3)
                delay = base_delay + jitter
                logger.warning(
                    "Gemini call failed (attempt %d/%d). Retrying in %.1fs: %s",
                    attempt + 1, MAX_RETRIES, delay, str(e)[:200]
                )
                time.sleep(delay)
            else:
                logger.error("Non-retryable Gemini error: %s", str(e)[:300])
                raise
    logger.error("All %d retries exhausted for Gemini call.", MAX_RETRIES)
    raise last_exception


# ── CHANGE 1 of 2: Added grounding rule to system instruction ──
SYSTEM_INSTRUCTION = (
    "You are an expert strategy document reviewer for Qatar's National Planning Council. "
    "You evaluate strategy documents against a structural checklist with precise scoring rubrics. "
    "You must be thorough, accurate, and evidence-based. Always cite page numbers using [PAGE X] markers. "
    "Match the evidence to the rubric level it genuinely fits — do not default to any particular score. "
    "A document may score anywhere from 0.0 to 1.0 on any sub-criteria depending on the evidence. "
    "GROUNDING RULE: ONLY cite text, names, numbers, and acronyms that ACTUALLY appear verbatim in the "
    "provided document. NEVER fabricate or invent content. Use EXACT wording from the document for quotes. "
    "If you cannot find evidence, say it is absent — do not generate plausible-sounding content. "
    "You MUST respond with valid JSON only. No markdown, no explanation outside the JSON."
)


# ── Cross-criteria scope boundaries ──
SCOPE_BOUNDARIES = {
    "1.1": (
        "SCOPE: Look for a DEDICATED diagnostic/situational analysis section (e.g., SWOT, "
        "gap analysis, current-state assessment). This covers the entity/sector's existing "
        "conditions, problems, and baseline data. "
        "NOTE: This is DIFFERENT from Criteria 7 (Risks). A diagnostic describes WHERE WE ARE "
        "NOW; a risk register describes WHAT COULD GO WRONG during strategy implementation."
    ),
    "1.2": (
        "SCOPE: Look for EXPLICIT, data-driven comparisons against named peer countries or "
        "institutions. General references to 'international best practices' or 'global standards' "
        "without specific country names, data points, or lessons learned count as 0.25 at best."
    ),
    "1.3": (
        "SCOPE: Look for FORWARD-LOOKING trend analysis — global megatrends, technology shifts, "
        "demographic projections, regional developments. Historical data analysis (what happened "
        "in the past) belongs to sub-criteria 1.1 (Diagnostic), not here. "
        "KEY DISTINCTION between 0.25 and 0.5: At 0.25, trends are merely MENTIONED without analysis. "
        "At 0.5, the document IDENTIFIES specific trends AND connects them to interventions or projects, "
        "even if the analysis is surface-level. If specific trends are named and linked to specific "
        "projects/actions in the strategy, that is 0.5 minimum."
    ),
    "2.1": (
        "SCOPE: Evaluate the VISION STATEMENT specifically — its clarity, ambition, specificity, "
        "and alignment with Qatar's national direction (QNV 2030, NDS3). "
        "KEY DISTINCTION between 0.25 and 0.5: At 0.25, the vision is VAGUE and GENERIC (e.g., "
        "'be the best'). At 0.5, the vision IS stated and may be ambitious, but lacks SPECIFICITY "
        "or clear national alignment. If the document claims alignment with national goals elsewhere "
        "but the vision statement itself doesn't reflect it, that limits the score."
    ),
    "2.2": (
        "SCOPE: Evaluate the MISSION STATEMENT specifically. Check for three elements: "
        "1) Purpose (what it does), 2) Scope (how/where), 3) Target beneficiaries (for whom). "
        "If all three are present with minor gaps, score 0.75. If one element is completely "
        "missing, score 0.5 maximum."
    ),
    "2.3": (
        "SCOPE: Look for DEFINED VALUES or guiding principles. These may not be labeled 'Values' — "
        "look for 'Principles', 'Guiding Principles', or equivalent. Evaluate whether they are "
        "aligned with the entity/sector mandate and operationally relevant (i.e., linked to actual "
        "strategy components). If principles are listed AND the strategy's structure is explicitly "
        "built upon them, that indicates operational relevance."
    ),
    "3.1": (
        "SCOPE: Evaluate the STRATEGIC OBJECTIVES or PILLARS — not the KPIs (that's 3.2-3.4). "
        "Check if objectives are measurable and linked to the vision/mission. A clear hierarchy "
        "from vision \u2192 objectives \u2192 sub-objectives with quantified targets qualifies for high scores."
    ),
    "3.2": (
        "SCOPE: Evaluate KPI DEFINITIONS specifically — methodology, measurement cadence, data "
        "source, and calculation method. Simply LISTING indicator names without any of these details "
        "is 0.25. Having SOME methodology for SOME indicators is 0.5."
    ),
    "3.3": (
        "SCOPE: Look for QUANTIFIED BASELINES and TIME-BOUND TARGETS for KPIs. "
        "KEY DISTINCTION: High-level strategic targets (e.g., '50% reduction by 2030') count, "
        "but the score also depends on whether PROJECT-LEVEL indicators have baselines and targets. "
        "If high-level targets exist but project-level KPIs lack baselines/targets, that limits "
        "the score. At 0.5, PARTIAL coverage is present (some KPIs have baselines/targets, others don't)."
    ),
    "3.4": (
        "SCOPE: Evaluate whether KPIs/indicators are SMART (Specific, Measurable, Achievable, "
        "Relevant, Time-bound). Each KPI should have a clear calculation methodology. "
        "A list of indicator NAMES without baselines, targets, or methodology is NOT SMART. "
        "Output deliverables (e.g., 'Prepare a report') are NOT KPIs."
    ),
    "4.1": (
        "SCOPE: Evaluate INTERNAL governance — committees, internal roles, responsibilities, "
        "reporting flows, and decision-making within the strategy's own implementing bodies. "
        "This is about the governance structure WITHIN the organization or coordination body, "
        "not about external stakeholders (that's 4.2)."
    ),
    "4.2": (
        "SCOPE: Evaluate EXTERNAL stakeholder ecosystem — entities OUTSIDE the primary governance "
        "body. For a government strategy, external stakeholders include private sector, civil society, "
        "academia, international organizations, and other government entities NOT on the main committee. "
        "KEY DISTINCTION: If the document assigns 'Key Stakeholder' and 'Supporting Agency' roles for "
        "every action to specific named entities, that IS stakeholder mapping with roles. The score "
        "depends on how COMPREHENSIVE the mapping is and whether ENGAGEMENT MECHANISMS are defined. "
        "Assigning roles to every action across multiple agencies is at least 0.75-level mapping."
    ),
    "4.3": (
        "SCOPE: Evaluate COORDINATION MECHANISMS — meeting cadence, reporting format, escalation "
        "paths, and accountability structures. "
        "KEY DISTINCTION: If the document defines quarterly reports, annual assessments, a mid-term "
        "review, and clear accountability per action, that represents GOOD coordination mechanisms. "
        "The gap to 1.0 is typically the absence of FORMAL ESCALATION PATHS for issue resolution."
    ),
    "5.1": (
        "SCOPE: Look for TRANSFORMATIVE initiatives — these must go beyond business-as-usual "
        "operations. Initiatives must be explicitly linked to strategic objectives/pillars. "
        "Routine operational activities do not qualify as transformative initiatives. "
        "KEY DISTINCTION: If initiatives are linked to PILLARS or PROGRAMS but not explicitly "
        "to the specific STRATEGIC OBJECTIVES stated in the document, that limits the score. "
        "Look for whether the document creates a traceable link from each initiative back to "
        "the stated objectives."
    ),
    "5.2": (
        "SCOPE: Look for a clear HIERARCHY: initiatives \u2192 projects. Projects must be explicitly "
        "derived from the initiatives identified in 5.1, with traceable linkage back to objectives. "
        "A flat list of actions without hierarchical derivation does not fully satisfy this."
    ),
    "6.1": (
        "SCOPE: Look for project descriptions with DESIGNATED OWNERS (named entities/roles "
        "responsible). A list of actions with 'stakeholders' is partial — full compliance requires "
        "clear ownership, roles, and responsibilities per project. "
        "KEY DISTINCTION: If EVERY action has a single designated 'Key Stakeholder' as owner with "
        "supporting roles clearly defined, that is 1.0 level. If there are OVERLAPPING or AMBIGUOUS "
        "ownerships (e.g., same action assigned to two different leads), that limits to 0.75."
    ),
    "6.2": (
        "SCOPE: Look for DETAILED, project-level timelines with start/end dates, milestones, "
        "phases, and dependencies. An overall strategy timeframe (e.g., '2025-2030') without "
        "project-specific scheduling is only 0.25. Timelines MUST be at the individual project "
        "or action level, not just the strategy level."
    ),
    "6.3": (
        "SCOPE: Look for QUANTIFIED budget allocations — actual numbers. A statement that "
        "'stakeholders will acquire their own budgets' or 'funding will be sought' is NOT a "
        "budget allocation. Score 0.0 if no actual financial figures are provided anywhere."
    ),
    "7.1": (
        "SCOPE: Look for a DEDICATED risk management section — a risk register, risk matrix, "
        "or explicit 'Risks' section that identifies IMPLEMENTATION risks (e.g., funding "
        "shortfalls, capacity gaps, coordination failures, political changes, technology risks, "
        "stakeholder resistance, external shocks). "
        "CRITICAL DISTINCTION: A 'Challenges' or 'Problems' section that describes the sector's "
        "current issues (e.g., high accident rates, pedestrian fatalities) is DIAGNOSTIC CONTENT "
        "(Criteria 1), NOT risk identification. Sector problems that the strategy aims to SOLVE "
        "are NOT the same as risks that could PREVENT the strategy from being IMPLEMENTED. "
        "If the document has no dedicated risk section, risk register, or risk matrix, score 0.0 "
        "even if the diagnostic section mentions challenges."
    ),
    "7.2": (
        "SCOPE: Look for EXPLICIT mitigation strategies tied to identified IMPLEMENTATION risks "
        "(from 7.1). These must include specific actions, ownership, and ideally timelines and "
        "contingency plans. "
        "CRITICAL DISTINCTION: The strategy's PROJECTS and ACTION PLANS (Criteria 5-6) are the "
        "strategy's core IMPLEMENTATION activities — they are NOT mitigation measures for Criteria 7. "
        "Mitigation measures address risks TO implementation (e.g., 'If funding is delayed, we will "
        "prioritize Phase 1 projects'), not the problems the strategy solves. "
        "CRITICAL DEPENDENCY: If 7.1 scored 0.0 (no implementation risks identified), then this "
        "sub-criteria MUST also score 0.0 — you cannot have mitigation measures for risks that "
        "were never identified. Assigning RESPONSIBILITY for future risk management (e.g., 'the "
        "Secretariat will mitigate risks') is NOT a mitigation measure — it is a governance "
        "statement. Score 0.0 unless SPECIFIC risks are identified AND SPECIFIC mitigation "
        "actions are defined for them."
    ),
}


def _get_scope_guidance(sub_id: str) -> str:
    guidance = SCOPE_BOUNDARIES.get(sub_id, "")
    if guidance:
        return chr(10) + chr(10) + "EVIDENCE SCOPE BOUNDARY:" + chr(10) + guidance
    return ""


def _build_sub_component_prompt(sub_info, classification, document_text, max_page=0, text_source="pymupdf"):
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
            "IMPORTANT: This sub-criteria is CONDITIONAL. It is not required for this "
            "classification. If you find relevant content in the document, evaluate and score it. "
            "If no relevant content exists at all, respond with applicable=false."
        )

    scope_guidance = _get_scope_guidance(sub_id)

    json_example = """{
  "sub_component_id": \"""" + sub_id + """\",
  "sub_component_name": \"""" + sub_name + """\",
  "applicable": true,
  "rubric_walkthrough": {
    "0.0_assessment": "Does the document meet or fail the 0.0 (Absent) level? Explain briefly.",
    "0.25_assessment": "Does the document meet the 0.25 (Low) level? What evidence supports or contradicts this?",
    "0.5_assessment": "Does the document meet the 0.5 (Mid) level? What evidence supports or contradicts this?",
    "0.75_assessment": "Does the document meet the 0.75 (High) level? What evidence supports or contradicts this?",
    "1.0_assessment": "Does the document meet the 1.0 (Complete) level? What evidence supports or contradicts this?",
    "selected_score_justification": "Based on the walkthrough above, the highest level fully supported by evidence is X because..."
  },
  "score": 0.0,
  "pages": [12, 15],
  "evidence": "Exact quotes or descriptions of what was found, with [PAGE X] references for each piece of evidence.",
  "evidence_summary": "One-line summary for the evidence index table.",
  "reasoning": "Detailed explanation of why this specific score was selected. Must reference what the NEXT higher level requires and why the document does NOT meet it.",
  "gap_to_next": "What is missing to achieve the MAXIMUM score of 1.0 (100%). List ALL gaps, not just the next level up.",
  "recommendation": "Concrete, actionable recommendation describing everything needed to reach the MAXIMUM score of 1.0 (100%)."
}"""

    prompt = (
        f"You are evaluating Sub-Criteria {sub_id}: {sub_name}" + chr(10) +
        f"Part of Criteria: {comp_name}" + chr(10) +
        f"Classification: {classification}" + chr(10) + chr(10) +
        "EVALUATION QUESTION:" + chr(10) +
        question + chr(10) + chr(10) +
        "SCORING RUBRIC \u2014 You MUST select exactly one of these five scores:" + chr(10) +
        rubric_text +
        conditional_note +
        scope_guidance + chr(10) + chr(10) +

        "CRITICAL INSTRUCTIONS:" + chr(10) +
        "1. Read the ENTIRE document carefully for content related to this sub-criteria." + chr(10) +
        "2. Extract specific evidence with exact [PAGE X] references." + chr(10) +
        "3. RUBRIC WALKTHROUGH (MANDATORY): In rubric_walkthrough, evaluate the evidence against EACH " +
        "rubric level from 0.0 to 1.0 IN ORDER. For each level, state whether the document meets " +
        "that level and cite the specific evidence that supports or contradicts it." + chr(10) +
        "4. DETERMINE SCORE LAST: Your score must be the HIGHEST rubric level that the evidence " +
        "FULLY supports. Do not pick a score first and then rationalize it." + chr(10) +
        "5. SCORE RANGE REMINDER: Scores WILL vary across sub-criteria. A document may score " +
        "0.0 on some, 0.25 on some, 0.5 on some, 0.75 on some, and 1.0 on others. " +
        "Do NOT default to any single score. Match the evidence to the rubric." + chr(10) +
        "6. KEY DISTINCTIONS TO WATCH FOR:" + chr(10) +
        "   - 0.0 (Absent): The topic is simply not addressed AT ALL in the document." + chr(10) +
        "   - 0.25 (Low): The topic is mentioned but only in vague, high-level, or generic terms." + chr(10) +
        "   - 0.5 (Mid): Substantive content exists but is INCOMPLETE \u2014 key elements are missing." + chr(10) +
        "   - 0.75 (High): Mostly complete with only MINOR gaps \u2014 nearly everything is there." + chr(10) +
        "   - 1.0 (Complete): Fully comprehensive \u2014 all required elements are present and detailed." + chr(10) +
        "7. In reasoning, you MUST explain:" + chr(10) +
        "   - What specific content matches your selected score level" + chr(10) +
        "   - What the NEXT higher level requires that is MISSING" + chr(10) +
        "   - Why the evidence does NOT qualify for that next level" + chr(10) +
        "8. Your score MUST be exactly one of: 0.0, 0.25, 0.5, 0.75, or 1.0." + chr(10) +
        "9. For gap_to_next: Describe ALL gaps between the current score and the MAXIMUM score of 1.0 (100%). " +
        "If score is 1.0, write 'N/A \u2014 maximum score achieved'. " +
        "If score is 0.25, explain everything needed to reach 1.0, not just 0.5." + chr(10) +
        "10. For recommendation: Write a concrete, actionable recommendation describing ALL improvements " +
        "needed to achieve the MAXIMUM score of 1.0 (100%). Cover every gap, not just the next level. " +
        "If the score is 1.0, write 'No action required \u2014 this sub-criteria meets all requirements.' " +
        "Do NOT prefix recommendations with [MANDATORY] \u2014 that label is added separately." + chr(10) +
        "11. PAGE NUMBERS: The document has " + str(max_page) + " pages total. " +
        "All [PAGE X] references in your response MUST be between 1 and " + str(max_page) + ". " +
        "NEVER use [PAGE 0] \u2014 page numbers start at 1. " +
        "Do NOT cite page numbers higher than " + str(max_page) + ". " +
        "Do NOT confuse action numbers (e.g., Action 126) with page numbers. " +
        "Only use page numbers from the [PAGE X] markers in the document text. " +
        "If you cannot find a specific page for evidence, use the nearest relevant [PAGE X] marker rather than page 0. " +
        "For the 'pages' field in JSON, only include page numbers where you found actual evidence." + chr(10) +
        "12. ZERO HALLUCINATION: Every quote, name, number, and acronym in your response MUST appear " +
        "verbatim in the document text below. Do NOT invent content. If unsure, state it is not found." + chr(10) + chr(10) +
        (
            "*** CRITICAL WARNING — AGENT-PROVIDED TEXT ***" + chr(10) +
            "The text below was read and summarized by an AI agent, NOT extracted directly from the PDF. " +
            "It may contain FABRICATED content that does not exist in the real document. " +
            "You MUST follow these rules strictly:" + chr(10) +
            "- If the text CLAIMS something exists (e.g., 'The appendix contains...', 'over 200 actions...', " +
            "'the action plan includes...') but does NOT show the actual detailed content, " +
            "treat that content as NOT PRESENT. A claim that something exists is NOT evidence of its content." + chr(10) +
            "- ONLY score based on content you can see DIRECTLY in the text below. " +
            "Descriptions of what an appendix or action plan supposedly contains do NOT count as evidence." + chr(10) +
            "- If the text says 'Appendix A contains X' but Appendix A is not shown in the text, score as if X is ABSENT." + chr(10) +
            "- Score CONSERVATIVELY. When in doubt, score LOWER." + chr(10) +
            "- A document that only MENTIONS it has initiatives, KPIs, or budgets without LISTING them " +
            "should score 0.0 (Absent) on those criteria, not higher." + chr(10) +
            "***" + chr(10) + chr(10)
            if text_source == "agent" else ""
        ) +
        "Respond with ONLY this JSON structure:" + chr(10) +
        json_example + chr(10) + chr(10) +
        "DOCUMENT TO EVALUATE:" + chr(10) +
        document_text
    )
    return prompt


def _extract_json(text):
    """Extract the first complete JSON object from text."""
    text = text.strip()
    if text.startswith("`"):
        lines = text.split(chr(10))
        if lines[0].startswith("`"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("`"):
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


def _parse_sub_response(response_text, sub_info, max_page=0):
    """Parse the LLM response JSON and validate pages."""
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

    # Validate page numbers — remove any outside valid range [1, max_page]
    if max_page > 0 and "pages" in result and isinstance(result["pages"], list):
        original_pages = result["pages"]
        valid_pages = [p for p in original_pages if isinstance(p, (int, float)) and 1 <= int(p) <= max_page]
        if len(valid_pages) != len(original_pages):
            invalid = [p for p in original_pages if p not in valid_pages]
            logger.warning(
                "  [PageFix] %s: Removed invalid pages %s (max=%d). Kept %s",
                sub_info["sub_component_id"], invalid, max_page, valid_pages
            )
        result["pages"] = [int(p) for p in valid_pages]

        if result.get("evidence") and max_page > 0:
            def _clamp_page_ref(match):
                page_num = int(match.group(1))
                if page_num < 1:
                    return "[PAGE 1]"
                if page_num > max_page:
                    return f"[PAGE {max_page}]"
                return match.group(0)
            result["evidence"] = re.sub(r'\[PAGE\s+(\d+)\]', _clamp_page_ref, result["evidence"])

    elif "pages" in result and isinstance(result["pages"], list):
        result["pages"] = [max(1, int(p)) for p in result["pages"] if isinstance(p, (int, float)) and int(p) >= 0]

    # Handle bare integer pages
    if "pages" in result and not isinstance(result["pages"], list):
        val = result["pages"]
        if isinstance(val, (int, float)) and val > 0:
            result["pages"] = [max(1, int(val))]
        else:
            result["pages"] = []

    # Fix [PAGE 0] in all text fields
    for field in ("evidence", "reasoning", "gap_to_next", "recommendation"):
        if field in result and isinstance(result[field], str):
            result[field] = re.sub(r'\[PAGE\s+0\]', '[PAGE 1]', result[field])

    walkthrough = result.pop("rubric_walkthrough", None)
    if walkthrough:
        logger.info(
            "  [Walkthrough] %s: %s",
            sub_info["sub_component_id"],
            walkthrough.get("selected_score_justification", "N/A")[:200]
        )

    return result


def _fallback_sub(sub_info):
    """Return a fallback result when parsing fails."""
    return {
        "sub_component_id": sub_info["sub_component_id"],
        "sub_component_name": sub_info["sub_component_name"],
        "applicable": not sub_info.get("is_conditional", False),
        "score": 0.0,
        "pages": [],
        "evidence": "Error: Could not parse LLM response for this sub-criteria.",
        "evidence_summary": "Parse error",
        "reasoning": "LLM response could not be parsed. Manual review required.",
        "gap_to_next": "",
        "recommendation": "Manual review required due to processing error.",
    }


def _evaluate_sub_component(sub_info, classification, document_text, max_page=0, text_source="pymupdf"):
    """Evaluate a single sub-criteria by calling Gemini."""
    sub_id = sub_info["sub_component_id"]
    logger.info("  [Thread] Evaluating %s: %s", sub_id, sub_info["sub_component_name"])
    prompt = _build_sub_component_prompt(sub_info, classification, document_text, max_page=max_page, text_source=text_source)
    raw_response = _call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION)
    result = _parse_sub_response(raw_response, sub_info, max_page=max_page)
    result["sub_component_id"] = sub_id
    result["sub_component_name"] = sub_info["sub_component_name"]
    logger.info("  [Thread] Done %s: score=%s", sub_id, result.get("score"))
    return result


def _build_comment_prompt(comp_name, sub_results):
    """Build prompt for consolidated criteria comment."""
    summary_lines = []
    for sub in sub_results:
        if sub.get("applicable", True):
            score = sub.get("score", 0.0)
            pct = int((score or 0) * 100)
            label = {0: "Absent", 25: "Low", 50: "Mid", 75: "High", 100: "Complete"}.get(pct, "Unknown")
            summary_lines.append(
                f"  {sub['sub_component_id']} {sub['sub_component_name']}: {label} ({pct}%) - "
                f"{sub.get('evidence_summary', 'N/A')}"
            )
        else:
            summary_lines.append(f"  {sub['sub_component_id']} {sub['sub_component_name']}: Not Applicable")
    summary_text = chr(10).join(summary_lines)

    applicable_scores = [
        sub.get("score", 0.0) for sub in sub_results
        if sub.get("applicable", True) and sub.get("score") is not None
    ]
    if applicable_scores:
        raw_avg = sum(applicable_scores) / len(applicable_scores)
        if raw_avg >= 1.0:
            level_word = "complete"
        elif raw_avg >= 0.75:
            level_word = "high"
        elif raw_avg >= 0.50:
            level_word = "moderate"
        elif raw_avg >= 0.25:
            level_word = "low"
        else:
            level_word = "absent"
    else:
        level_word = "absent"

    return (
        f"Write a 1-2 sentence consolidated assessment comment for Criteria: {comp_name}" + chr(10) + chr(10) +
        "Sub-criteria results:" + chr(10) +
        summary_text + chr(10) + chr(10) +
        f"The criteria's overall assessment band is: {level_word.upper()} coverage." + chr(10) + chr(10) +
        f"Style: Write in NPC narrative style. Start with 'Provides {level_word} coverage' or "
        f"'Demonstrates {level_word} coverage'. Then describe what was found (strengths) "
        "followed by 'however,' and what was missing or insufficient (gaps). "
        "Be specific about what was found and what was missing. Reference the actual sub-criteria "
        "scores above \u2014 do NOT contradict them." + chr(10) + chr(10) +
        "Respond with ONLY the comment text, no JSON, no quotes, no prefix."
    )


def _add_na_subs(sub_results, classification, comp_sub_components):
    """Add Not Applicable placeholders for sub-criteria that don't apply."""
    applicability = load_applicability(classification)
    na_ids = set(applicability["not_applicable"])
    existing_ids = {s["sub_component_id"] for s in sub_results}
    name_lookup = {s["id"]: s["name"] for s in comp_sub_components}

    for sub_def in comp_sub_components:
        sub_id = sub_def["id"]
        if sub_id in na_ids and sub_id not in existing_ids:
            sub_results.append({
                "sub_component_id": sub_id,
                "sub_component_name": name_lookup.get(sub_id, sub_id),
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


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF, with [PAGE X] markers."""
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append(f"[PAGE {page_num + 1}]\n{text}")
    doc.close()
    return "\n\n".join(pages)


def _try_extract_from_artifacts(tool_context: ToolContext) -> str | None:
    """Try to load PDF from session artifacts and extract text."""
    try:
        artifact_names = tool_context.list_artifacts()
        if not artifact_names:
            logger.info("No artifacts found in session.")
            return None

        logger.info("Found %d artifacts: %s", len(artifact_names), artifact_names)

        for name in artifact_names:
            try:
                artifact_part = tool_context.load_artifact(name)
            except Exception as e:
                logger.warning("Failed to load artifact '%s': %s", name, e)
                continue
            if artifact_part is None:
                continue

            has_inline = hasattr(artifact_part, 'inline_data') and artifact_part.inline_data is not None
            has_text = hasattr(artifact_part, 'text') and artifact_part.text

            if has_inline:
                mime = getattr(artifact_part.inline_data, 'mime_type', '') or ""
                data = getattr(artifact_part.inline_data, 'data', None)
                if data and ("pdf" in mime.lower() or (len(data) > 100 and data[:5] == b'%PDF-')):
                    try:
                        text = _extract_text_from_pdf_bytes(data)
                        if text and len(text) > 500:
                            logger.info("Extracted %d chars from artifact '%s'", len(text), name)
                            return text
                    except Exception as e:
                        logger.error("PyMuPDF failed for artifact '%s': %s", name, e)

            if has_text and len(artifact_part.text) > 500:
                return artifact_part.text

        return None
    except Exception as e:
        logger.error("Failed to extract from artifacts: %s", e, exc_info=True)
        return None


def _try_extract_from_session_events(tool_context: ToolContext) -> str | None:
    """Scan session events for PDF file_uri or inline_data and extract text.

    This is the PRIMARY extraction method on Agent Engine.
    When a user uploads a file, Agent Engine stores it as file_data
    in the session events with a file_uri pointing to GCS.
    """
    try:
        # --- Deep inspection: log everything available on tool_context ---
        logger.info("=== DEEP TOOL_CONTEXT INSPECTION ===")
        logger.info("tool_context type: %s", type(tool_context).__name__)
        tc_attrs = [a for a in dir(tool_context) if not a.startswith('__')]
        logger.info("tool_context attributes: %s", tc_attrs)

        # Try to access function_call_id, agent_name, etc.
        for attr in ['function_call_id', 'agent_name', 'state', 'actions']:
            val = getattr(tool_context, attr, 'NOT_FOUND')
            if val != 'NOT_FOUND':
                logger.info("  tool_context.%s = %s", attr, repr(val)[:200])

        # --- Method A: Standard invocation_context path ---
        invocation_ctx = getattr(tool_context, '_invocation_context', None)
        if invocation_ctx is None:
            # Try without underscore
            invocation_ctx = getattr(tool_context, 'invocation_context', None)
        if invocation_ctx is None:
            logger.info("No invocation_context found. Trying alternative paths...")
            # Try to find it through other attributes
            for attr in tc_attrs:
                if 'invocation' in attr.lower() or 'context' in attr.lower() or 'session' in attr.lower():
                    val = getattr(tool_context, attr, None)
                    logger.info("  Found potentially useful attr: %s = %s (type: %s)",
                                attr, repr(val)[:100], type(val).__name__)
                    if val is not None and hasattr(val, 'session'):
                        invocation_ctx = val
                        logger.info("  Using %s as invocation_context!", attr)
                        break
        else:
            logger.info("Found invocation_context: type=%s", type(invocation_ctx).__name__)
            inv_attrs = [a for a in dir(invocation_ctx) if not a.startswith('__')]
            logger.info("  invocation_context attrs: %s", inv_attrs)

        if invocation_ctx is None:
            logger.info("Cannot find invocation_context via any path.")
            return None

        session = getattr(invocation_ctx, 'session', None)
        if session is None:
            logger.info("No session on invocation_context.")
            # Log what IS available
            for attr in dir(invocation_ctx):
                if not attr.startswith('__'):
                    val = getattr(invocation_ctx, attr, None)
                    logger.info("  invocation_ctx.%s = type=%s", attr, type(val).__name__)
            return None

        logger.info("Found session: type=%s", type(session).__name__)
        session_attrs = [a for a in dir(session) if not a.startswith('__')]
        logger.info("  session attrs: %s", session_attrs)

        # Log session ID if available
        session_id = getattr(session, 'id', None) or getattr(session, 'session_id', None)
        logger.info("  session id: %s", session_id)

        events = getattr(session, 'events', None) or []
        logger.info("Scanning %d session events for PDF file references...", len(events))

        # --- Scan all events for files ---
        file_uris_found = []
        for i, event in enumerate(events):
            content = getattr(event, 'content', None)
            if content is None:
                continue
            parts = getattr(content, 'parts', None) or []
            for j, part in enumerate(parts):
                # Check file_data
                file_data = getattr(part, 'file_data', None)
                if file_data:
                    file_uri = getattr(file_data, 'file_uri', None) or ""
                    mime_type = getattr(file_data, 'mime_type', '') or ""
                    logger.info("EVENT[%d] PART[%d]: file_data uri=%s mime=%s", i, j, file_uri, mime_type)
                    file_uris_found.append((file_uri, mime_type))

                    if "pdf" in mime_type.lower() or file_uri.lower().endswith(".pdf"):
                        pdf_bytes = _download_gcs_uri(file_uri)
                        if pdf_bytes:
                            try:
                                text = _extract_text_from_pdf_bytes(pdf_bytes)
                                if text and len(text) > 500:
                                    logger.info("SUCCESS via session events: Extracted %d chars from %s",
                                                len(text), file_uri[:200])
                                    return text
                            except Exception as e:
                                logger.error("PyMuPDF failed for file_uri %s: %s", file_uri, e)

                # Check inline_data
                inline_data = getattr(part, 'inline_data', None)
                if inline_data:
                    mime = getattr(inline_data, 'mime_type', '') or ""
                    data = getattr(inline_data, 'data', None)
                    logger.info("EVENT[%d] PART[%d]: inline_data mime=%s size=%d",
                                i, j, mime, len(data) if data else 0)
                    if data and ("pdf" in mime.lower() or (len(data) > 100 and data[:5] == b'%PDF-')):
                        try:
                            text = _extract_text_from_pdf_bytes(data)
                            if text and len(text) > 500:
                                logger.info("SUCCESS via inline_data: Extracted %d chars", len(text))
                                return text
                        except Exception as e:
                            logger.error("PyMuPDF failed for inline_data: %s", e)

                # Check for any other file-like attributes
                part_attrs = [a for a in dir(part) if not a.startswith('_') and a not in ('text',)]
                for attr in part_attrs:
                    val = getattr(part, attr, None)
                    if val is not None and attr not in ('file_data', 'inline_data', 'thought'):
                        logger.info("EVENT[%d] PART[%d]: part.%s = %s (type=%s)",
                                    i, j, attr, repr(val)[:100], type(val).__name__)

        if file_uris_found:
            logger.info("Found %d file_data references but none yielded PDF text: %s",
                         len(file_uris_found), file_uris_found)
        else:
            logger.info("No file_data or inline_data references found in any session events.")

        return None
    except Exception as e:
        logger.error("Failed to scan session events: %s", e, exc_info=True)
        return None


def _download_gcs_uri(uri: str) -> bytes | None:
    """Download a file from a GCS URI, HTTPS URL, or Vertex AI file URI.

    Handles multiple URI formats that Agent Engine may provide:
    - gs://bucket/path/to/file.pdf
    - https://storage.googleapis.com/bucket/path/to/file.pdf
    - https://storage.cloud.google.com/bucket/path/to/file.pdf
    - https://*.googleapis.com/... (authenticated)
    - Regular HTTP/HTTPS URLs
    """
    if not uri:
        return None

    uri = uri.strip()
    logger.info("Attempting to download from URI: %s", uri[:300])

    # Method 1: Direct GCS download (gs:// scheme)
    if uri.startswith("gs://"):
        try:
            from google.cloud import storage as gcs_storage
            parts = uri[5:].split("/", 1)
            if len(parts) != 2:
                logger.warning("Invalid GCS URI format: %s", uri)
                return None
            bucket_name, blob_path = parts
            logger.info("GCS download: bucket=%s, blob=%s", bucket_name, blob_path[:100])
            client = gcs_storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            data = blob.download_as_bytes()
            logger.info("SUCCESS: Downloaded %d bytes from GCS: %s", len(data), uri[:200])
            return data
        except Exception as e:
            logger.error("GCS download failed for %s: %s", uri[:200], e, exc_info=True)
            return None

    # Method 2: Google Storage HTTPS URLs → convert to GCS and download
    if "storage.googleapis.com" in uri or "storage.cloud.google.com" in uri:
        try:
            from urllib.parse import urlparse, unquote
            parsed = urlparse(uri)
            path = unquote(parsed.path).lstrip("/")

            if "storage.googleapis.com" in uri:
                # Format: https://storage.googleapis.com/bucket/path
                parts = path.split("/", 1)
            elif "storage.cloud.google.com" in uri:
                # Format: https://storage.cloud.google.com/bucket/path
                parts = path.split("/", 1)
            else:
                parts = []

            if len(parts) == 2:
                bucket_name, blob_path = parts
                gs_uri = f"gs://{bucket_name}/{blob_path}"
                logger.info("Converted HTTPS to GCS URI: %s", gs_uri[:200])
                return _download_gcs_uri(gs_uri)  # Recursive call with gs:// URI
        except Exception as e:
            logger.warning("Failed to convert storage URL: %s", e)
        # Fall through to generic HTTPS download

    # Method 3: Authenticated Google API HTTPS URLs
    if "googleapis.com" in uri:
        try:
            import google.auth
            import google.auth.transport.requests
            credentials, _ = google.auth.default()
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            token = credentials.token

            import urllib.request
            req = urllib.request.Request(uri)
            req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
                logger.info("SUCCESS: Downloaded %d bytes from authenticated URL: %s", len(data), uri[:200])
                return data
        except Exception as e:
            logger.warning("Authenticated download failed for %s: %s", uri[:200], e)
        # Fall through to unauthenticated HTTPS

    # Method 4: Generic HTTPS/HTTP download (unauthenticated)
    if uri.startswith("http://") or uri.startswith("https://"):
        try:
            import urllib.request
            req = urllib.request.Request(uri)
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
                logger.info("SUCCESS: Downloaded %d bytes from URL: %s", len(data), uri[:200])
                return data
        except Exception as e:
            logger.error("HTTP download failed for %s: %s", uri[:200], e)
            return None

    logger.warning("Unrecognized URI scheme: %s", uri[:100])
    return None


def structural_review(
    document_text: str,
    classification: str,
    strategy_title: str = "Untitled Strategy",
    entity_name: str = "Unknown Entity",
    file_gcs_uri: str = "",
    tool_context: ToolContext = None,
) -> dict:
    """Performs a full structural checklist review of a strategy document.

    Args:
        document_text: The full text content of the strategy document.
        classification: The classification of the strategy: entity, sectoral, or thematic.
        strategy_title: The title of the strategy document.
        entity_name: The name of the entity that owns the strategy.
        file_gcs_uri: A gs:// URI or URL pointing to the uploaded PDF file.
                      If provided, the tool will download and extract text directly
                      using PyMuPDF, which is more accurate than agent-provided text.
        tool_context: ADK tool context (injected automatically).

    Returns:
        A dict containing the full structural review report in markdown format.
    """
    try:
        classification = classification.lower().strip()
        if classification not in ("entity", "sectoral", "thematic"):
            return {"error": True, "message": f"Invalid classification: {classification}. Must be entity, sectoral, or thematic."}

        # --- PDF extraction: Try ALL methods ---
        # Priority: 1) file_gcs_uri  2) session events  3) artifacts  4) agent text
        text_source = "agent"
        agent_text_len = len(document_text.strip()) if document_text else 0
        logger.info("=" * 60)
        logger.info("PDF EXTRACTION — TRYING ALL METHODS")
        logger.info("  Agent document_text: %d chars", agent_text_len)
        logger.info("  file_gcs_uri: %s", repr(file_gcs_uri[:300]) if file_gcs_uri else "(empty)")
        logger.info("  tool_context: %s (type: %s)",
                     "PRESENT" if tool_context else "NONE",
                     type(tool_context).__name__ if tool_context else "NoneType")
        logger.info("=" * 60)

        # METHOD 1: Direct GCS download from file_gcs_uri
        if text_source == "agent" and file_gcs_uri and file_gcs_uri.strip():
            uri = file_gcs_uri.strip()
            logger.info("METHOD 1 — file_gcs_uri: %s", uri[:200])
            pdf_bytes = _download_gcs_uri(uri)
            if pdf_bytes:
                try:
                    extracted = _extract_text_from_pdf_bytes(pdf_bytes)
                    if extracted and len(extracted) > 500:
                        logger.info("SUCCESS METHOD 1: %d chars via file_gcs_uri", len(extracted))
                        document_text = extracted
                        text_source = "pymupdf_gcs"
                except Exception as e:
                    logger.error("METHOD 1 PyMuPDF failed: %s", e)

        # METHOD 2: Session events — PRIMARY method for Agent Engine
        if text_source == "agent" and tool_context is not None:
            logger.info("METHOD 2 — Session events (tool_context is %s)", type(tool_context).__name__)
            extracted_text = _try_extract_from_session_events(tool_context)
            if extracted_text and len(extracted_text) > 500:
                logger.info("SUCCESS METHOD 2: %d chars via session events", len(extracted_text))
                document_text = extracted_text
                text_source = "pymupdf_session"
        elif text_source == "agent":
            logger.warning("METHOD 2 SKIPPED: tool_context is None!")

        # METHOD 3: Artifacts — works on ADK Web
        if text_source == "agent" and tool_context is not None:
            logger.info("METHOD 3 — Artifacts")
            extracted_text = _try_extract_from_artifacts(tool_context)
            if extracted_text and len(extracted_text) > 500:
                logger.info("SUCCESS METHOD 3: %d chars via artifacts", len(extracted_text))
                document_text = extracted_text
                text_source = "pymupdf_artifacts"

        # METHOD 4: Check if agent text has real page markers (>5 pages = probably real)
        if text_source == "agent" and document_text:
            page_numbers = re.findall(r'\[PAGE\s+(\d+)\]', document_text)
            if page_numbers:
                max_page = max(int(p) for p in page_numbers)
                if max_page > 5:
                    logger.info("METHOD 4: Agent text has %d pages with markers — using it", max_page)
                    text_source = "agent_with_markers"
                else:
                    logger.warning("METHOD 4: Agent text has only %d pages — likely truncated", max_page)

        # Final logging
        if text_source == "agent":
            logger.warning("ALL METHODS FAILED — using raw agent text (%d chars). Anti-hallucination warning ACTIVE.",
                           agent_text_len)

        logger.info("=" * 60)
        logger.info("EXTRACTION RESULT: source=%s, length=%d", text_source, len(document_text) if document_text else 0)
        if document_text:
            logger.info("  First 200 chars: %s", document_text[:200])
        logger.info("=" * 60)

        if not document_text or len(document_text.strip()) < 100:
            return {"error": True, "message": "Document text is too short or empty. Please provide the full strategy document text."}

        # Auto-extract entity_name and strategy_title if defaults
        if entity_name in ("Unknown Entity", "", None):
            entity_name = _extract_entity_name(document_text) or "Unknown Entity"
            logger.info("Auto-detected entity_name: %s", entity_name)
        if strategy_title in ("Untitled Strategy", "", None):
            strategy_title = _extract_strategy_title(document_text) or "Untitled Strategy"
            logger.info("Auto-detected strategy_title: %s", strategy_title)

        logger.info("Starting structural review: %s (%s)", strategy_title, classification)
        logger.info("Document text length: %d characters", len(document_text))
        logger.info("Document text preview (first 200 chars): %s", document_text[:200])
        logger.info("Document text preview (last 200 chars): %s", document_text[-200:])

        def _update_status(msg: str):
            if tool_context is not None:
                try:
                    tool_context.state["ui:status_update"] = msg
                except Exception:
                    pass
            logger.info("Status: %s", msg)

        _update_status("Analyzing document structure...")

        max_page = _get_max_page(document_text)
        logger.info("Document max page number detected: %d", max_page)

        # ── CHANGE 2 of 2: Inject synthetic [PAGE X] markers if none found ──
        # Without markers, the LLM hallucinates page numbers and fabricates evidence.
        if max_page == 0:
            logger.warning("No [PAGE X] markers found! Injecting synthetic page markers.")
            lines = document_text.split('\n')
            new_lines = ["[PAGE 1]"]
            char_count = 0
            page_num = 1
            for line in lines:
                char_count += len(line) + 1
                new_lines.append(line)
                if char_count > 3000:
                    page_num += 1
                    new_lines.append(f"\n[PAGE {page_num}]")
                    char_count = 0
            document_text = '\n'.join(new_lines)
            max_page = _get_max_page(document_text)
            logger.info("Injected synthetic page markers. New max_page: %d", max_page)

        applicable_subs = get_applicable_sub_components(classification)
        logger.info("Evaluating %d sub-criteria in parallel (max %d threads)", len(applicable_subs), MAX_WORKERS)

        _update_status(f"Evaluating {len(applicable_subs)} sub-criteria against the document...")

        sub_results_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for i, sub_info in enumerate(applicable_subs):
                future = executor.submit(_evaluate_sub_component, sub_info, classification, document_text, max_page, text_source)
                futures[future] = sub_info
                if i > 0 and i % MAX_WORKERS == 0:
                    time.sleep(2)

            for future in as_completed(futures):
                sub_info = futures[future]
                sub_id = sub_info["sub_component_id"]
                try:
                    result = future.result()
                    sub_results_map[sub_id] = result
                    done_count = len(sub_results_map)
                    total_count = len(applicable_subs)
                    logger.info("Progress: %d/%d sub-criteria evaluated", done_count, total_count)
                    _update_status(f"Evaluated {done_count}/{total_count} sub-criteria...")
                except Exception as e:
                    logger.error("Thread failed for %s: %s", sub_id, e)
                    sub_results_map[sub_id] = _fallback_sub(sub_info)

        full_checklist = load_checklist(classification)

        _update_status("Generating criteria assessments and report...")

        component_results = []
        for comp in full_checklist["components"]:
            comp_id = comp["id"]
            comp_name = comp["name"]
            comp_subs = comp["sub_components"]
            all_sub_ids = [s["id"] for s in comp_subs]

            comp_sub_results = []
            for sub_id in all_sub_ids:
                if sub_id in sub_results_map:
                    comp_sub_results.append(sub_results_map[sub_id])

            comp_sub_results = _add_na_subs(comp_sub_results, classification, comp_subs)

            try:
                comment_prompt = _build_comment_prompt(comp_name, comp_sub_results)
                comment = _call_gemini(comment_prompt, system_instruction=SYSTEM_INSTRUCTION).strip()
                if comment.startswith(chr(34)) and comment.endswith(chr(34)):
                    comment = comment[1:-1]
            except Exception as e:
                logger.warning("Could not generate comment for criteria %d: %s", comp_id, e)
                applicable_scores = [
                    s.get("score", 0.0) for s in comp_sub_results
                    if s.get("applicable", True) and s.get("score") is not None
                ]
                if applicable_scores:
                    avg = sum(applicable_scores) / len(applicable_scores)
                    if avg >= 0.75: level = "high"
                    elif avg >= 0.5: level = "moderate"
                    elif avg >= 0.25: level = "low"
                    else: level = "absent"
                else:
                    level = "absent"
                scored_names = [
                    f"{s['sub_component_id']} {s['sub_component_name']}"
                    for s in comp_sub_results if s.get("applicable", True)
                ]
                comment = (
                    f"Provides {level} coverage across {len(scored_names)} sub-criteria: "
                    f"{', '.join(scored_names)}."
                )

            component_results.append({
                "id": comp_id,
                "name": comp_name,
                "comment": comment,
                "sub_results": comp_sub_results,
            })

        _update_status("Compiling final report...")

        report = generate_report(
            strategy_title=strategy_title,
            entity_name=entity_name,
            classification=classification,
            component_results=component_results,
        )

        all_scores = []
        na_count = 0
        for comp in component_results:
            for sub in comp.get("sub_results", []):
                if sub.get("applicable", True) and sub.get("score") is not None:
                    all_scores.append(sub["score"])
                elif not sub.get("applicable", True):
                    na_count += 1

        overall = calculate_overall_score(all_scores, na_count=na_count)
        logger.info(
            "Review complete. Overall score: %.2f%% (%d applicable + %d N/A = %d total)",
            overall, len(all_scores), na_count, len(all_scores) + na_count
        )

        return {
            "error": False,
            "overall_score": overall,
            "report": report,
        }

    except Exception as e:
        logger.exception("Structural review failed")
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg.upper():
            user_msg = (
                "The review could not be completed because the Gemini API rate limit was exceeded. "
                "Please wait 2-3 minutes and try again. "
                "If this persists, contact your administrator to check the Gemini API quota."
            )
        elif "timeout" in error_msg.lower() or "deadline" in error_msg.lower():
            user_msg = (
                "The review timed out. Large documents may take longer to process. "
                "Please try again. If this persists, the document may need to be split into sections."
            )
        else:
            user_msg = f"The structural review encountered an error: {error_msg}"
        return {
            "error": True,
            "message": user_msg,
            "technical_detail": traceback.format_exc()[-500:],
        }