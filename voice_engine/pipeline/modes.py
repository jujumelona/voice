from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineMode:
    name: str
    asr: str
    translation: str
    decoder: str
    purpose: str


MODES = {
    "fast": PipelineMode(
        name="fast",
        asr="whisper",
        translation="argos",
        decoder="qwen3-tts",
        purpose="real-time calls",
    ),
    "balanced": PipelineMode(
        name="balanced",
        asr="faster-whisper",
        translation="argos",
        decoder="auto",
        purpose="real-time and quality balance",
    ),
    "quality": PipelineMode(
        name="quality",
        asr="faster-whisper",
        translation="marian",
        decoder="auto",
        purpose="quality after utterance end",
    ),
}


def get_mode(name: str) -> PipelineMode:
    try:
        return MODES[name]
    except KeyError as exc:
        raise ValueError(f"unknown mode: {name}") from exc
