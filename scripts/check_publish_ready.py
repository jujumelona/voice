from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BLOCKED_DIRS = {
    "checkpoints",
    "checkpoints_v2",
    "models",
    "bin",
    "outputs",
}

BLOCKED_SUFFIXES = {
    ".bin",
    ".pt",
    ".pth",
    ".safetensors",
    ".onnx",
    ".ckpt",
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".zip",
}

MAX_SOURCE_FILE_BYTES = 2_000_000


def main() -> int:
    problems: list[str] = []
    for path in ROOT.rglob("*"):
        if _is_inside_hidden_or_cache(path):
            continue
        rel = path.relative_to(ROOT).as_posix()
        if path.is_dir():
            continue
        if _is_inside_blocked_dir(rel):
            continue
        if rel.startswith("vendor/") and _is_ignored_vendor_artifact(rel, path):
            continue
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            problems.append(f"blocked artifact suffix: {rel}")
        if path.stat().st_size > MAX_SOURCE_FILE_BYTES and not rel.startswith("vendor/"):
            problems.append(f"large source file: {rel} ({path.stat().st_size} bytes)")

    if problems:
        print("Publish readiness check failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Publish readiness check passed.")
    print("Reminder: keep models, binaries, outputs, and weight/media artifacts untracked.")
    return 0


def _is_inside_blocked_dir(rel: str) -> bool:
    return any(rel == blocked or rel.startswith(blocked + "/") for blocked in BLOCKED_DIRS)


def _is_inside_hidden_or_cache(path: Path) -> bool:
    rel_parts = path.relative_to(ROOT).parts
    return any(
        part in {
            ".git",
            ".agents",
            ".codex",
            ".venv",
            ".voice_bridge_runtime",
            "venv",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
        }
        or part.endswith(".egg-info")
        for part in rel_parts
    )


def _is_ignored_vendor_artifact(rel: str, path: Path) -> bool:
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
