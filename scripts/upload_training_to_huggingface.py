from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi

DEFAULT_SONG_COUNT = 20
DEFAULT_REPO_NAME = "voice-jp20"
DEFAULT_FINAL_DIRS = ("outputs", "final", "exports", "artifacts", "trained_model")
_SECRET_NAMES = {
    ".env",
    "token",
    "token.txt",
    "hf_token",
    "hf_token.txt",
    "credentials.json",
}
_EPOCH_RE = re.compile(r"(?:^|[-_])epoch[-_ ]?(\d+)(?:$|[-_.])", re.IGNORECASE)


@dataclass(frozen=True)
class EpochArtifact:
    epoch: int
    path: Path

    @property
    def remote_dir(self) -> str:
        return f"epochs/epoch-{self.epoch:04d}"


def _epoch_number(path: Path) -> int | None:
    candidates = [path.name, *[parent.name for parent in path.parents[:2]]]
    for candidate in candidates:
        match = _EPOCH_RE.search(candidate)
        if match:
            return int(match.group(1))
    return None


def discover_epoch_artifacts(checkpoint_root: str | Path) -> list[EpochArtifact]:
    root = Path(checkpoint_root).expanduser().resolve()
    if not root.exists():
        return []

    by_epoch: dict[int, Path] = {}
    for path in sorted(root.rglob("*")):
        epoch = _epoch_number(path)
        if epoch is None:
            continue
        if path.is_dir():
            by_epoch.setdefault(epoch, path)
        elif path.is_file() and epoch not in by_epoch:
            by_epoch[epoch] = path

    return [
        EpochArtifact(epoch=epoch, path=by_epoch[epoch])
        for epoch in sorted(by_epoch)
    ]


def _safe_upload_path(path: Path) -> bool:
    if path.name.casefold() in _SECRET_NAMES:
        return False
    if any(part.casefold() in {"dataset", "datasets", "training_data", "raw_audio"} for part in path.parts):
        return False
    return True


def _ignore_patterns() -> list[str]:
    # Model/checkpoint/result artifacts are uploaded. Raw source-song directories and
    # obvious secrets stay local.
    return [
        "**/.env",
        "**/*token*",
        "**/credentials.json",
        "**/dataset/**",
        "**/datasets/**",
        "**/training_data/**",
        "**/raw_audio/**",
        "**/__pycache__/**",
        "**/*.pyc",
    ]


def epoch_readme(epoch: int, *, song_count: int = DEFAULT_SONG_COUNT) -> str:
    return f"""# Epoch {epoch}

This checkpoint is from **epoch {epoch}**.

## Training data

- 학습 데이터: **일본곡 {song_count}곡 ({song_count} Japanese songs)**
- Dataset composition: **{song_count} Japanese songs**
- Language/music domain: Japanese songs
- This repository contains model/checkpoint artifacts only; the source songs are not uploaded here.

## Artifact layout

The files in this folder are the saved training artifacts for epoch {epoch}.
"""


def root_readme(
    epochs: Iterable[EpochArtifact],
    *,
    song_count: int = DEFAULT_SONG_COUNT,
) -> str:
    epoch_list = list(epochs)
    if epoch_list:
        rows = "\n".join(
            f"- Epoch {item.epoch}: `{item.remote_dir}/`"
            for item in epoch_list
        )
    else:
        rows = "- No epoch checkpoints were found at upload time."

    return f"""---
tags:
- audio
- voice
- japanese
- music
- training
---

# Voice JP20 training artifacts

학습 데이터: **일본곡 {song_count}곡 ({song_count} Japanese songs)**.

These artifacts were trained using **{song_count} Japanese songs**.

Only trained model/checkpoint/output artifacts are uploaded. The original songs are not included in this Hugging Face repository.

## Epoch checkpoints

{rows}

## Final artifacts

Completed outputs are uploaded under `final/`.
"""


def _write_temp_text(text: str, *, suffix: str = ".md") -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=suffix,
        delete=False,
    )
    try:
        handle.write(text)
        return Path(handle.name)
    finally:
        handle.close()


def _upload_text(
    api: HfApi,
    *,
    repo_id: str,
    path_in_repo: str,
    text: str,
    commit_message: str,
) -> None:
    tmp = _write_temp_text(text)
    try:
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(tmp),
            path_in_repo=path_in_repo,
            commit_message=commit_message,
        )
    finally:
        tmp.unlink(missing_ok=True)


def _upload_epoch(
    api: HfApi,
    *,
    repo_id: str,
    artifact: EpochArtifact,
    song_count: int = DEFAULT_SONG_COUNT,
) -> None:
    remote = artifact.remote_dir
    if artifact.path.is_dir():
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(artifact.path),
            path_in_repo=remote,
            ignore_patterns=_ignore_patterns(),
            commit_message=f"Upload epoch {artifact.epoch}",
        )
    else:
        if not _safe_upload_path(artifact.path):
            raise RuntimeError(f"Refusing to upload unsafe epoch artifact: {artifact.path}")
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(artifact.path),
            path_in_repo=f"{remote}/{artifact.path.name}",
            commit_message=f"Upload epoch {artifact.epoch}",
        )

    _upload_text(
        api,
        repo_id=repo_id,
        path_in_repo=f"{remote}/README.md",
        text=epoch_readme(artifact.epoch, song_count=song_count),
        commit_message=f"Document epoch {artifact.epoch}",
    )


def _upload_final_dirs(
    api: HfApi,
    *,
    repo_id: str,
    final_dirs: Iterable[str | Path],
) -> list[str]:
    uploaded: list[str] = []
    for raw in final_dirs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            continue
        if not _safe_upload_path(path):
            print(f"Skipping raw/secret training path: {path}")
            continue

        if path.is_dir():
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(path),
                path_in_repo="final",
                ignore_patterns=_ignore_patterns(),
                commit_message=f"Upload completed artifacts from {path.name}",
            )
        elif path.is_file():
            api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=str(path),
                path_in_repo=f"final/{path.name}",
                commit_message=f"Upload completed artifact {path.name}",
            )
        uploaded.append(path.name)
    return uploaded


def upload_training_artifacts(
    *,
    token: str,
    repo_id: str | None = None,
    checkpoint_root: str | Path = "checkpoints",
    final_dirs: Iterable[str | Path] = DEFAULT_FINAL_DIRS,
    private: bool = False,
    song_count: int = DEFAULT_SONG_COUNT,
    api: HfApi | None = None,
) -> dict[str, Any]:
    if not token or not token.strip():
        raise ValueError("A Hugging Face write token is required.")

    client = api or HfApi(token=token.strip())
    identity = client.whoami(token=token.strip())
    username = str(identity.get("name") or identity.get("fullname") or "").strip()
    if not username:
        raise RuntimeError("Could not resolve the Hugging Face account from the supplied token.")

    resolved_repo_id = (repo_id or f"{username}/{DEFAULT_REPO_NAME}").strip()
    if "/" not in resolved_repo_id:
        resolved_repo_id = f"{username}/{resolved_repo_id}"

    client.create_repo(
        repo_id=resolved_repo_id,
        repo_type="model",
        private=bool(private),
        exist_ok=True,
        token=token.strip(),
    )

    epochs = discover_epoch_artifacts(checkpoint_root)
    for artifact in epochs:
        _upload_epoch(
            client,
            repo_id=resolved_repo_id,
            artifact=artifact,
            song_count=song_count,
        )

    uploaded_final = _upload_final_dirs(
        client,
        repo_id=resolved_repo_id,
        final_dirs=final_dirs,
    )

    _upload_text(
        client,
        repo_id=resolved_repo_id,
        path_in_repo="README.md",
        text=root_readme(epochs, song_count=song_count),
        commit_message="Update training artifact model card",
    )

    manifest = {
        "schema_version": "voice/huggingface-training-upload-v1",
        "repo_id": resolved_repo_id,
        "training_dataset": {
            "description": f"{song_count} Japanese songs",
            "song_count": song_count,
            "language_domain": "Japanese",
            "raw_training_audio_uploaded": False,
        },
        "epochs": [
            {
                "epoch": artifact.epoch,
                "source_name": artifact.path.name,
                "remote_path": artifact.remote_dir,
            }
            for artifact in epochs
        ],
        "final_artifacts": uploaded_final,
    }
    _upload_text(
        client,
        repo_id=resolved_repo_id,
        path_in_repo="training-manifest.json",
        text=json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        commit_message="Upload training manifest",
    )
    return manifest


def _prompt_token() -> str:
    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        return env_token
    # In Colab/Jupyter this renders as a masked password input field.
    return getpass.getpass("Hugging Face write token: ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Upload every epoch checkpoint and completed training output to Hugging Face. "
            "The token is requested with a masked input field when HF_TOKEN is not set."
        )
    )
    parser.add_argument(
        "--repo-id",
        default="",
        help=(
            "Hugging Face model repo, e.g. username/voice-jp20. "
            "Default: <token-account>/voice-jp20"
        ),
    )
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument(
        "--final-dir",
        action="append",
        dest="final_dirs",
        default=None,
        help="Completed artifact directory/file. May be supplied more than once.",
    )
    parser.add_argument("--song-count", type=int, default=DEFAULT_SONG_COUNT)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if args.song_count <= 0:
        parser.error("--song-count must be positive")

    token = _prompt_token()
    result = upload_training_artifacts(
        token=token,
        repo_id=args.repo_id or None,
        checkpoint_root=args.checkpoint_root,
        final_dirs=args.final_dirs or DEFAULT_FINAL_DIRS,
        private=args.private,
        song_count=args.song_count,
    )
    print(
        f"Upload complete: https://huggingface.co/{result['repo_id']} "
        f"(epochs={len(result['epochs'])}, final_paths={len(result['final_artifacts'])})"
    )


if __name__ == "__main__":
    main()
