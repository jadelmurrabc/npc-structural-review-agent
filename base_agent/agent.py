"""Top-level agent — exports root_agent for ADK.

KEY FIX: The wrapper function now includes `tool_context: ToolContext`.
ADK automatically detects this parameter type and injects the real ToolContext,
which gives the tool access to session events (where uploaded file URIs live).

Previous versions passed tool_context=None, which killed all extraction methods
that depend on reading session state.
"""
from __future__ import annotations
import json
import logging
import os
import traceback

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

from base_agent.prompts import return_instructions_root
from base_agent.tools.structural_review_tool import structural_review as _tool_impl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _make_json_safe(value):
    """Ensure the tool output is JSON-serializable."""
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return {"_non_serializable_tool_output": str(value)}


def structural_review(
    document_text: str = "",
    classification: str = "",
    strategy_title: str = "Untitled Strategy",
    entity_name: str = "Unknown Entity",
    file_gcs_uri: str = "",
    file_b64: str = "",
    tool_context: ToolContext = None,
) -> dict:
    """Performs a structural checklist review of a strategy document.

    Analyzes the document against 7 criteria and 20 sub-criteria, producing
    a detailed review report with scores, evidence, and recommendations.

    Args:
        document_text: The full text content of the strategy document.
                       Pass ALL text you can read from the uploaded file.
        classification: The strategy classification: entity, sectoral, or thematic.
        strategy_title: The title of the strategy document.
        entity_name: The name of the entity that owns the strategy.
        file_gcs_uri: The gs:// URI or https:// URL of the uploaded PDF file.
                      Pass the file's URI/path if available.
        file_b64: Base64-encoded contents of the uploaded PDF file.
                  If you have the raw file data, encode it as base64 and pass it here.
        tool_context: Automatically injected by ADK — do NOT pass this manually.

    Returns:
        A dict containing the structural review report in markdown format.
    """
    try:
        logger.info("=" * 60)
        logger.info("structural_review WRAPPER called")
        logger.info("  document_text: %d chars", len(document_text) if document_text else 0)
        logger.info("  classification: %s", classification)
        logger.info("  file_gcs_uri: %s", file_gcs_uri[:200] if file_gcs_uri else "(empty)")
        logger.info("  file_b64: %d chars", len(file_b64) if file_b64 else 0)
        logger.info("  tool_context: %s", type(tool_context).__name__ if tool_context else "None")
        logger.info("=" * 60)

        # If file_b64 is provided, decode and extract text with PyMuPDF
        if file_b64 and len(file_b64) > 100:
            logger.info("file_b64 provided (%d chars). Attempting decode + PyMuPDF extraction...", len(file_b64))
            try:
                import base64
                pdf_bytes = base64.b64decode(file_b64)
                logger.info("Decoded %d bytes from base64", len(pdf_bytes))
                if pdf_bytes[:5] == b'%PDF-' or len(pdf_bytes) > 1000:
                    from base_agent.tools.structural_review_tool import _extract_text_from_pdf_bytes
                    extracted = _extract_text_from_pdf_bytes(pdf_bytes)
                    if extracted and len(extracted) > 500:
                        logger.info("SUCCESS via file_b64: Extracted %d chars with PyMuPDF", len(extracted))
                        document_text = extracted
                        
                        file_gcs_uri = ""
            except Exception as e:
                logger.warning("file_b64 decode/extraction failed: %s", e)

        result = _tool_impl(
            document_text=document_text,
            classification=classification,
            strategy_title=strategy_title,
            entity_name=entity_name,
            file_gcs_uri=file_gcs_uri,
            tool_context=tool_context,  
        )

        if result is None:
            result = {
                "error": True,
                "error_type": "ToolReturnedNone",
                "error_message": "Tool returned None unexpectedly.",
            }

        return _make_json_safe(result)

    except Exception as e:
        logger.exception("structural_review wrapper failed")
        return _make_json_safe({
            "error": True,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        })


def get_root_agent() -> LlmAgent:
    logger.info("Initializing NPC Structural Review Agent...")
    model_name = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-pro")

    return LlmAgent(
        model=model_name,
        name="npc_structural_reviewer",
        instruction=return_instructions_root(),
        tools=[structural_review],
        generate_content_config={"temperature": 0.0},
    )


try:
    root_agent = get_root_agent()
    logger.info("root_agent initialized successfully.")
except Exception as e:
    logger.critical("Failed to initialize root_agent: %s", e, exc_info=True)
    raise