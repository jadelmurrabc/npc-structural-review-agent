import os
from pathlib import Path


def _get_env(name: str, fallback: str = "") -> str:
    v = os.getenv(name)
    return (v or fallback).strip()


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
