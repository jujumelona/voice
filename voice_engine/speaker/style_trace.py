from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np
from voice_engine.pipeline.types import (
    Transcript,
    StyleTokenTrace,
    ContentUnits,
    SpeakerProfile,
)
from voice_engine.prosody.extractor import extract_prosody
from voice_engine.prosody.events import extract_word_prosody_events
from voice_engine.prosody.style_plan import build_style_plan
from voice_engine.prosody.style_tokens import style_tokens_from_plan


def extract_utterance_style_trace(
    wav_path: Path | str,
    text: str,
    language: str | None,
    speaker: SpeakerProfile,
    content: ContentUnits,
) -> StyleTokenTrace:
    """Extracts the Pitch, Rhythm, and Emotion style trace from a recorded utterance.

    This function acts as a single gateway to consolidate acoustic extraction,
    word alignment parsing, style planning, and style token extraction.

    Args:
        wav_path: Path to the recorded source speech segment.
        text: Transcribed source text.
        language: Language of the transcription.
        speaker: Cumulative speaker profile containing voice identities.
        content: Target translated units content.

    Returns:
        A StyleTokenTrace object for the current translated utterance.
    """
    wav_path = Path(wav_path)

    # 1. Extract raw prosody frame trace
    prosody_result = extract_prosody(wav_path, speaker_median_f0=speaker.median_f0)
    prosody_trace = prosody_result.trace

    # 2. Group word-level prosodic events
    transcript = Transcript(text=text, language=language)
    source_events = extract_word_prosody_events(transcript, prosody_trace)

    # 3. Create target-aligned style plan
    source_unit_count = len([char for char in text if not char.isspace()])
    style_plan = build_style_plan(
        source_trace=prosody_trace,
        speaker=speaker,
        content=content,
        source_unit_count=source_unit_count,
        source_events=source_events,
    )

    # 4. Generate final style tokens for the speech decoder
    style_tokens = style_tokens_from_plan(
        plan=style_plan,
        source_trace=prosody_trace,
        speaker=speaker,
    )

    return style_tokens


def extract_utterance_style_trace_from_samples(
    samples: list[float] | np.ndarray,
    sample_rate: int,
    text: str,
    language: str | None,
    speaker: SpeakerProfile,
    content: ContentUnits,
) -> StyleTokenTrace:
    """Samples-based gateway for the live path.

    The current prosody extractor is path based, so this uses a deterministic
    runtime cache file instead of a per-turn temporary file.
    """
    audio = np.nan_to_num(np.asarray(samples, dtype=np.float32))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    if audio.size == 0:
        raise RuntimeError("cannot extract style trace from empty samples")

    cache_root = Path(".voice_bridge_runtime") / "style_samples"
    cache_root.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1()
    h.update(str(int(sample_rate)).encode("ascii"))
    h.update(audio[: min(len(audio), int(sample_rate) * 8)].tobytes())
    path = cache_root / ("style_" + h.hexdigest()[:16] + ".wav")
    if not path.exists():
        _write_float_wav(path, audio, int(sample_rate))
    return extract_utterance_style_trace(path, text, language, speaker, content)


def _write_float_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes((np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
