from __future__ import annotations

import os
from pathlib import Path


def runtime_root() -> Path:
    configured = os.environ.get("VOICE_BRIDGE_RUNTIME_ROOT")
    if configured:
        return Path(configured)
    if os.name == "nt" and Path("I:/").exists():
        return Path("I:/voice_bridge")
    return Path(".")


def model_root() -> Path:
    configured = os.environ.get("VOICE_BRIDGE_MODEL_ROOT")
    if configured:
        return Path(configured)
    return runtime_root() / "models"


def model_path(*parts: str) -> Path:
    return model_root().joinpath(*parts)
