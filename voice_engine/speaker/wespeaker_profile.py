"""Backward-compatible import shim for older profiler/notebook code."""

from voice_engine.speaker.speaker_profile import (  # noqa: F401
    SpeakerSideMetrics,
    extract_speaker_profile_from_samples,
    extract_speaker_profile_from_wav,
    extract_speaker_side_metrics_from_wav,
    load_speaker_profile,
    save_speaker_profile,
    speaker_profile_from_dict,
    speaker_profile_to_dict,
    update_speaker_profile,
)
