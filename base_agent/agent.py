from __future__ import annotations
import logging
import os
from google.adk.agents import LlmAgent
from base_agent.prompts import return_instructions_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_root_agent() -> LlmAgent:
    logger.info("Initializing NPC Structural Review Agent...")
    model_name = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-pro")

    return LlmAgent(
        model=model_name,
        name="npc_structural_reviewer",
        instruction=return_instructions_root(),
        tools=[],
        generate_content_config={"temperature": 0.0},
    )


root_agent = get_root_agent()
logger.info("root_agent initialized successfully.")
