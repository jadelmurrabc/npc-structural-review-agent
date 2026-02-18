"""NPC Structural Review Agent — Root agent definition."""
import logging
import os
from google.adk.agents import LlmAgent
from google.genai import types

# --- Logging setup (from proven working agent) ---
import google.cloud.logging

IS_RUNNING_IN_GCP = os.getenv("K_SERVICE") is not None
if IS_RUNNING_IN_GCP:
    client = google.cloud.logging.Client()
    client.setup_logging()
    logging.basicConfig(level=logging.INFO)
    logging.info("Running in GCP. Configured Google Cloud Logging.")
else:
    logging.basicConfig(level=logging.INFO)
    logging.info("Running locally. Using console logging.")

logger = logging.getLogger(__name__)

from .prompts import return_instructions_root
from .tools.structural_review_tool import (
    extract_document,
    structural_review,
)


def get_root_agent() -> LlmAgent:
    tools = [extract_document, structural_review]
    agent = LlmAgent(
        model=os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-pro"),
        name="npc_structural_reviewer",
        instruction=return_instructions_root(),
        tools=tools,
        generate_content_config=types.GenerateContentConfig(temperature=0.01),
    )
    logger.info("root_agent initialized with 2 tools: extract_document, structural_review")
    return agent


root_agent = get_root_agent()