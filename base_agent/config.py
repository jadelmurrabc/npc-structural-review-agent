"""Configuration module for NPC Structural Review Agent.

IMPORTANT: This module is serialized via cloudpickle during deployment.
All paths are stored as STRINGS (via os.path), never as pathlib.Path objects.
This avoids the 'cannot instantiate WindowsPath on your system' error when
a Windows-pickled module is deserialized on a Linux container.
"""
import os
import json


def _get_env(name: str, fallback: str = "") -> str:
    v = os.getenv(name)
    return (v or fallback).strip()


# Store directory paths as STRINGS, not Path objects.
_BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_DIR: str = os.path.join(_BASE_DIR, "config")


def get_config_dir() -> str:
    """Return the config directory path as a string."""
    if os.path.isdir(_CONFIG_DIR):
        return _CONFIG_DIR
    # Fallback: environment variable override
    override = os.getenv("NPC_CONFIG_DIR")
    if override and os.path.isdir(override):
        return override
    # Fallback: current working directory
    cwd_config = os.path.join(os.getcwd(), "config")
    if os.path.isdir(cwd_config):
        return cwd_config
    # Return the original (will trigger fallback to embedded configs)
    return _CONFIG_DIR


# --- Embedded config support for Agent Engine ---
_EMBEDDED_CONFIGS: dict[str, dict] = {}


def embed_config(name: str, data: dict) -> None:
    """Store a config dict in memory (called during deployment serialization)."""
    _EMBEDDED_CONFIGS[name] = data


def load_config_json(filename: str) -> dict:
    """Load a JSON config file, preferring disk, falling back to embedded data."""
    filepath = os.path.join(get_config_dir(), filename)
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback to embedded config (Agent Engine deployment)
    if filename in _EMBEDDED_CONFIGS:
        return _EMBEDDED_CONFIGS[filename]
    raise FileNotFoundError(
        f"Config file not found: {filepath} "
        f"(also not embedded). Available embedded: {list(_EMBEDDED_CONFIGS.keys())}"
    )


# --- Backward compatibility ---
# Exposed as a string, not a Path object.
CONFIG_DIR = _CONFIG_DIR