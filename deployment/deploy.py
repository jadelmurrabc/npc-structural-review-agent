"""Deployment script for the NPC Structural Review Agent."""

from __future__ import annotations

import logging
import os
import time
from typing import Dict

from absl import app, flags
from dotenv import load_dotenv

import vertexai
from google.cloud import storage
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

from base_agent.agent import root_agent

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

FLAGS = flags.FLAGS
flags.DEFINE_string("project_id", None, "GCP project ID.")
flags.DEFINE_string("location", None, "GCP location.")
flags.DEFINE_string("bucket", None, "GCS staging bucket name (used by Agent Engine).")
flags.DEFINE_string("resource_id", None, "ReasoningEngine resource ID (for delete).")
flags.DEFINE_bool("create", False, "Create a new agent.")
flags.DEFINE_bool("delete", False, "Delete an existing agent.")

flags.mark_bool_flags_as_mutual_exclusive(["create", "delete"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_env(name: str, fallback: str = "") -> str:
    v = os.getenv(name)
    return (v or fallback).strip()


def setup_staging_bucket(project_id: str, location: str, bucket_name: str) -> str:
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.lookup_bucket(bucket_name)
    if not bucket:
        logger.info("Staging bucket %s not found. Creating...", bucket_name)
        bucket = storage_client.create_bucket(bucket_name, project=project_id, location=location)
        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        bucket.patch()
        logger.info("Created staging bucket %s", bucket_name)
    else:
        logger.info("Staging bucket %s already exists.", bucket_name)
    return f"gs://{bucket_name}"


def _build_env_vars(project_id: str, location: str) -> Dict[str, str]:
    env_vars: Dict[str, str] = {}
    env_vars["ROOT_AGENT_MODEL"] = _get_env("ROOT_AGENT_MODEL", "gemini-2.5-pro")
    env_vars["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
    return env_vars


def create(project_id: str, location: str, staging_bucket: str) -> None:
    import cloudpickle

    # Import ALL modules that need to be serialized
    import base_agent
    import base_agent.agent
    import base_agent.config
    import base_agent.prompts
    import base_agent.tools
    import base_agent.tools.structural_review_tool
    import base_agent.logic
    import base_agent.logic.checklist_loader
    import base_agent.logic.pdf_extractor
    import base_agent.logic.report_generator
    import base_agent.logic.scoring

    # Register ALL modules for pickle-by-value
    cloudpickle.register_pickle_by_value(base_agent)
    cloudpickle.register_pickle_by_value(base_agent.agent)
    cloudpickle.register_pickle_by_value(base_agent.config)
    cloudpickle.register_pickle_by_value(base_agent.prompts)
    cloudpickle.register_pickle_by_value(base_agent.tools)
    cloudpickle.register_pickle_by_value(base_agent.tools.structural_review_tool)
    cloudpickle.register_pickle_by_value(base_agent.logic)
    cloudpickle.register_pickle_by_value(base_agent.logic.checklist_loader)
    cloudpickle.register_pickle_by_value(base_agent.logic.pdf_extractor)
    cloudpickle.register_pickle_by_value(base_agent.logic.report_generator)
    cloudpickle.register_pickle_by_value(base_agent.logic.scoring)

    # Embed config JSON files so they survive serialization
    import json
    from base_agent.config import embed_config, CONFIG_DIR
    for config_file in ["questions.json", "applicability.json"]:
        config_path = os.path.join(CONFIG_DIR, config_file)
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                embed_config(config_file, json.load(f))
            logger.info("Embedded config: %s", config_file)
        else:
            logger.warning("Config file not found for embedding: %s", config_path)

    env_vars = _build_env_vars(project_id, location)

    # No plugins — tool extracts PDFs via ToolContext session events + GCS download
    adk_app = AdkApp(
        agent=root_agent,
        enable_tracing=False,
    )

    display_name = _get_env("DISPLAY_NAME", f"npc-structural-review-{int(time.time())}")

    requirements = [
        "google-cloud-aiplatform[adk,agent-engines]>=1.60.0",
        "google-adk",
        "cloudpickle",
        "google-genai",
        "google-cloud-storage",
        "pymupdf",
        "python-dotenv",
        "pydantic>=2.0.0",
        "google-auth",
    ]

    logger.info("Creating remote agent (display_name=%s)...", display_name)
    logger.info("Environment variables: %s", {k: v for k, v in env_vars.items()})
    logger.info("Requirements: %s", requirements)

    remote_agent = agent_engines.create(
        adk_app,
        display_name=display_name,
        requirements=requirements,
        env_vars=env_vars,
    )

    logger.info("Created remote agent: %s", remote_agent.resource_name)
    print(f"\n{'='*60}")
    print(f"Successfully created agent!")
    print(f"Resource name: {remote_agent.resource_name}")
    print(f"Display name:  {display_name}")
    print(f"{'='*60}")


def delete(resource_id: str) -> None:
    if not resource_id:
        raise app.UsageError("Missing --resource_id for delete.")
    remote_agent = agent_engines.get(resource_id)
    remote_agent.delete(force=True)
    print(f"\nSuccessfully deleted agent: {resource_id}")


def main(argv: list[str]) -> None:
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    project_id = FLAGS.project_id or _get_env("GOOGLE_CLOUD_PROJECT")
    location = FLAGS.location or _get_env("GOOGLE_CLOUD_LOCATION", "us-central1")
    bucket_name = FLAGS.bucket or _get_env("GOOGLE_CLOUD_STORAGE_BUCKET")

    if not project_id or not location:
        raise app.UsageError("Missing --project_id or --location.")
    if not bucket_name:
        raise app.UsageError(
            "Missing --bucket. Provide a GCS staging bucket name.\n"
            "  Example: --bucket=npc-review-staging\n"
            "  Or set GOOGLE_CLOUD_STORAGE_BUCKET in .env"
        )

    staging_uri = setup_staging_bucket(project_id, location, bucket_name)
    vertexai.init(project=project_id, location=location, staging_bucket=staging_uri)

    if FLAGS.create:
        create(project_id, location, staging_uri)
    elif FLAGS.delete:
        delete(FLAGS.resource_id)
    else:
        raise app.UsageError("Specify either --create or --delete.")


if __name__ == "__main__":
    app.run(main)