from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np


def encode_audio_payload(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Encode mono float audio for JSONL worker responses without temp wav I/O."""
    audio = np.nan_to_num(np.asarray(samples, dtype=np.float32))
    if audio.ndim > 1:
        audio = np.squeeze(audio)
    if audio.ndim != 1:
        raise RuntimeError(f"audio payload expects 1-D samples, got {audio.shape}")
    buf = io.BytesIO()
    np.save(buf, audio.astype(np.float32), allow_pickle=False)
    return {
        "sample_rate": int(sample_rate),
        "audio_npy_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "samples": int(audio.size),
    }


def decode_audio_payload(payload: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Decode JSONL inline audio returned by the persistent Qwen worker."""
    if "audio_npy_b64" not in payload:
        raise RuntimeError("worker response did not include inline audio payload")
    raw = base64.b64decode(str(payload["audio_npy_b64"]).encode("ascii"))
    audio = np.load(io.BytesIO(raw), allow_pickle=False)
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    if audio.ndim > 1:
        audio = np.squeeze(audio)
    if audio.ndim != 1 or audio.size == 0:
        raise RuntimeError(f"worker returned unsupported audio payload shape: {audio.shape}")
    return audio.astype(np.float32), int(payload.get("sample_rate") or 24000)
