from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class WordTiming:
    text: str
    start_ms: float
    end_ms: float
    probability: float | None = None


@dataclass(frozen=True)
class SpeechChunk:
    path: Path
    sample_rate: int
    start_sec: float = 0.0
    end_sec: float | None = None


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None
    confidence: float | None = None
    words: list[WordTiming] = field(default_factory=list)


import numpy as np


@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    language: str | None
    is_final: bool
    chunk_index: int
    start_ms: float
    end_ms: float
    samples: np.ndarray | None = None


@dataclass(frozen=True)
class Translation:
    text: str
    source_language: str | None = None
    target_language: str | None = None


@dataclass(frozen=True)
class TranslationEvent:
    text: str
    source_language: str | None
    target_language: str | None
    is_final: bool
    chunk_index: int
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class StreamAudioChunk:
    samples: list[float]
    sample_rate: int
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class SpeakerProfile:
    speaker_embedding: list[float]
    spectral_envelope: list[float]
    formant_profile: list[float]
    average_f0: float | None
    median_f0: float | None
    pitch_range: float | None
    breathiness: float
    harmonic_noise_ratio: float
    speaking_rate_baseline: float
    pause_ratio_baseline: float
    energy_baseline: float
    style_habits: dict[str, float | str]
    timbre_code: list[float]
    speaker_embedding_backend: str = "wespeaker/eres2net-large"
    speaker_embedding_dim: int = 0
    embedding_l2_normalized: bool = True
    speaker_quality: float = 0.0
    speech_duration_sec: float = 0.0
    update_count: int = 1


@dataclass(frozen=True)
class ContentUnits:
    language: str
    text: str
    phonemes: list[str]
    semantic_tokens: list[int]
    units: list[str] = field(default_factory=list)
    unit_type: str = "word_or_phrase"


@dataclass(frozen=True)
class PatternFrame:
    time_ms: float
    energy: float
    entropy: float
    f0: float | None
    f0_hz: float | None
    attack: float
    pause: bool
    stress: bool


@dataclass(frozen=True)
class PatternAnchor:
    time_ms: float
    kind: str
    strength: float


@dataclass(frozen=True)
class PatternTrace:
    frame_ms: int
    hop_ms: int
    duration_ms: float
    median_f0: float | None
    f0_reference: float | None
    speaking_rate: float
    energy_p10: float
    energy_p90: float
    frames: list[PatternFrame]
    anchors: list[PatternAnchor]


@dataclass(frozen=True)
class ProsodyPattern:
    duration_sec: float
    speech_rate: float
    energy_mean: float
    energy_peak: float
    f0_mean: float | None
    f0_range: float | None
    pause_ratio: float
    attack: float
    energy_contour: list[float]
    f0_contour: list[float | None]
    emphasis: list[float]
    trace: PatternTrace


@dataclass(frozen=True)
class ProsodyEvent:
    source_text: str
    start_ms: float
    end_ms: float
    pause_before_ms: float
    pause_after_ms: float
    energy_mean: float
    energy_peak: float
    log_f0_start: float | None
    log_f0_end: float | None
    log_f0_slope: float
    breath: float
    attack_peak: float
    stress: float
    emotion_hint: str


@dataclass(frozen=True)
class TargetStyleEvent:
    target_text: str
    source_text: str
    start_ms: float
    end_ms: float
    pause_before_ms: float
    pause_after_ms: float
    energy_mean: float
    energy_peak: float
    log_f0_start: float | None
    log_f0_end: float | None
    log_f0_slope: float
    breath: float
    attack_peak: float
    stress: float
    emotion_hint: str


@dataclass(frozen=True)
class StylePlan:
    """Utterance-level control plan built before acoustic generation."""

    target_duration_sec: float
    suggested_tts_speed: float
    target_frame_count: int
    energy: list[float]
    log_f0_rel: list[float]
    pause: list[bool]
    stress: list[bool]
    breath: list[float]
    breathiness_target: float
    hnr_target: float
    diagnostics: dict[str, float | int | str]
    events: list[TargetStyleEvent] = field(default_factory=list)


@dataclass(frozen=True)
class StyleTokenTrace:
    """Frame controls that must be consumed before acoustic/mel generation."""

    frame_ms: int
    hop_ms: int
    duration_ms: float
    energy: list[float]
    log_f0_rel: list[float]
    pause: list[bool]
    stress: list[bool]
    breath: list[float]
    events: list[TargetStyleEvent]
    speaking_rate: float
    speaker_median_f0: float | None
    diagnostics: dict[str, float | int | str]


@dataclass(frozen=True)
class VoiceState:
    speaker: SpeakerProfile
    pattern_trace_q8: list[list[int]]
    content: ContentUnits
    frame_ms: int = 20
    hop_ms: int = 10
    source_unit_count: int = 0
    style_plan: StylePlan | None = None
    style_tokens: StyleTokenTrace | None = None


@dataclass(frozen=True)
class VoiceDecoderInput:
    content: ContentUnits
    speaker: SpeakerProfile | None
    prosody: PatternTrace | None
    voice_state: VoiceState | None = None
    style_plan: StylePlan | None = None
    style_tokens: StyleTokenTrace | None = None
    reference_audio_path: str | None = None
    reference_audio_samples: list[float] | np.ndarray | None = None
    reference_audio_sample_rate: int | None = None
