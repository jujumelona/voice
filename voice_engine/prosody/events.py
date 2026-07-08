from __future__ import annotations

import re

from voice_engine.pipeline.types import PatternFrame, PatternTrace, ProsodyEvent, Transcript, WordTiming


def extract_word_prosody_events(transcript: Transcript, trace: PatternTrace) -> list[ProsodyEvent]:
    words = transcript.words or _estimated_word_timings(transcript.text, trace.duration_ms)
    if not words:
        return []
    events: list[ProsodyEvent] = []
    for index, word in enumerate(words):
        frames = _frames_in_span(trace.frames, word.start_ms, word.end_ms)
        previous_end = words[index - 1].end_ms if index > 0 else 0.0
        next_start = words[index + 1].start_ms if index + 1 < len(words) else trace.duration_ms
        pause_before = max(0.0, word.start_ms - previous_end)
        pause_after = max(0.0, next_start - word.end_ms)
        events.append(_event_from_frames(word, frames, pause_before, pause_after))
    return events


def _event_from_frames(
    word: WordTiming,
    frames: list[PatternFrame],
    pause_before_ms: float,
    pause_after_ms: float,
) -> ProsodyEvent:
    if not frames:
        return ProsodyEvent(
            source_text=word.text,
            start_ms=word.start_ms,
            end_ms=word.end_ms,
            pause_before_ms=pause_before_ms,
            pause_after_ms=pause_after_ms,
            energy_mean=0.0,
            energy_peak=0.0,
            log_f0_start=None,
            log_f0_end=None,
            log_f0_slope=0.0,
            breath=0.0,
            attack_peak=0.0,
            stress=0.0,
            emotion_hint="neutral",
        )
    voiced = [frame for frame in frames if not frame.pause]
    energy_values = [frame.energy for frame in voiced or frames]
    entropy_values = [frame.entropy for frame in voiced or frames]
    attack_values = [frame.attack for frame in voiced or frames]
    f0_values = [frame.f0 for frame in voiced if frame.f0 is not None]
    energy_mean = sum(energy_values) / len(energy_values)
    energy_peak = max(energy_values)
    attack_peak = max(attack_values)
    breath = max(0.0, sum(entropy_values) / len(entropy_values) - energy_mean)
    f0_start = f0_values[0] if f0_values else None
    f0_end = f0_values[-1] if f0_values else None
    f0_slope = (f0_end - f0_start) if f0_start is not None and f0_end is not None else 0.0
    stress = _stress_score(energy_peak, attack_peak, pause_before_ms, pause_after_ms, f0_slope)
    return ProsodyEvent(
        source_text=word.text,
        start_ms=word.start_ms,
        end_ms=word.end_ms,
        pause_before_ms=pause_before_ms,
        pause_after_ms=pause_after_ms,
        energy_mean=round(energy_mean, 4),
        energy_peak=round(energy_peak, 4),
        log_f0_start=round(f0_start, 4) if f0_start is not None else None,
        log_f0_end=round(f0_end, 4) if f0_end is not None else None,
        log_f0_slope=round(f0_slope, 4),
        breath=round(max(0.0, min(1.0, breath)), 4),
        attack_peak=round(attack_peak, 4),
        stress=round(stress, 4),
        emotion_hint=_emotion_hint(energy_mean, attack_peak, breath, f0_slope, pause_before_ms, pause_after_ms),
    )


def _frames_in_span(frames: list[PatternFrame], start_ms: float, end_ms: float) -> list[PatternFrame]:
    return [frame for frame in frames if start_ms <= frame.time_ms < max(start_ms + 1.0, end_ms)]


def _stress_score(
    energy_peak: float,
    attack_peak: float,
    pause_before_ms: float,
    pause_after_ms: float,
    f0_slope: float,
) -> float:
    pause_boost = min(0.25, max(pause_before_ms, pause_after_ms) / 1000.0)
    pitch_boost = min(0.20, abs(f0_slope) * 0.35)
    return max(0.0, min(1.0, energy_peak * 0.50 + attack_peak * 0.30 + pause_boost + pitch_boost))


def _emotion_hint(
    energy_mean: float,
    attack_peak: float,
    breath: float,
    f0_slope: float,
    pause_before_ms: float,
    pause_after_ms: float,
) -> str:
    if energy_mean >= 0.70 and attack_peak >= 0.24:
        return "urgent_or_excited"
    if breath >= 0.25 and energy_mean <= 0.45:
        return "breathy_or_tired"
    if pause_before_ms >= 300.0 or pause_after_ms >= 420.0:
        return "hesitant_or_emphatic_pause"
    if abs(f0_slope) >= 0.18:
        return "intonation_movement"
    return "neutral"


def _estimated_word_timings(text: str, duration_ms: float) -> list[WordTiming]:
    tokens = re.findall(r"[가-힣]+|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s]", text)
    if not tokens:
        return []
    duration_ms = max(duration_ms, len(tokens) * 220.0)
    step = duration_ms / len(tokens)
    return [
        WordTiming(text=token, start_ms=index * step, end_ms=(index + 1) * step)
        for index, token in enumerate(tokens)
    ]
