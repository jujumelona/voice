from __future__ import annotations

from dataclasses import asdict

from voice_engine.pipeline.types import ContentUnits, PatternTrace, SpeakerProfile, StylePlan, TargetStyleEvent
from voice_engine.prosody.alignment import align_events_to_content
from voice_engine.prosody.pattern_retime import target_frames_from_units
from voice_engine.prosody.q8_codec import Q8PatternTrace, decode_q8, encode_q8
from voice_engine.prosody.pattern_retime import retime_q8_anchor_locked
from voice_engine.pipeline.types import ProsodyEvent


def build_style_plan(
    *,
    source_trace: PatternTrace,
    speaker: SpeakerProfile,
    content: ContentUnits,
    source_unit_count: int,
    source_events: list[ProsodyEvent] | None = None,
    target_events: list[TargetStyleEvent] | None = None,
) -> StylePlan:
    """Build a control plan before TTS/acoustic generation.

    The plan intentionally keeps only measurable acoustic controls: duration,
    energy, relative pitch, pauses, stress, and breath-like noise regions. It is
    serializable and passed along with the current utterance reference audio.
    """

    aligned_events = target_events or (
        align_events_to_content(source_events, content) if source_events else []
    )
    if aligned_events:
        return _build_event_style_plan(
            source_trace=source_trace,
            speaker=speaker,
            content=content,
            source_unit_count=source_unit_count,
            events=aligned_events,
        )

    target_unit_count = len(content.units) or len([c for c in content.text if not c.isspace()])
    source_frames = len(source_trace.frames)
    target_frames = target_frames_from_units(
        source_frames=source_frames,
        source_unit_count=source_unit_count,
        target_unit_count=target_unit_count,
        min_ratio=0.70,
        max_ratio=1.65,
    )
    retimed = retime_q8_anchor_locked(
        encode_q8(source_trace),
        target_frames,
        max_energy_step=20,
        max_f0_step=14,
        smoothing=0.24,
    )
    decoded = decode_q8(retimed)
    breath = _breath_curve(decoded.entropy, decoded.energy, decoded.pause, speaker.breathiness)
    target_duration_sec = target_frames * max(1, source_trace.hop_ms) / 1000.0
    suggested_speed = _suggest_speed(
        source_duration_sec=source_trace.duration_ms / 1000.0,
        target_duration_sec=target_duration_sec,
        source_unit_count=source_unit_count,
        target_unit_count=target_unit_count,
    )

    diagnostics: dict[str, float | int | str] = {
        "source_frames": source_frames,
        "target_frames": target_frames,
        "source_unit_count": source_unit_count,
        "target_unit_count": target_unit_count,
        "source_duration_sec": round(source_trace.duration_ms / 1000.0, 4),
        "target_duration_sec": round(target_duration_sec, 4),
        "suggested_tts_speed": round(suggested_speed, 4),
        "pause_ratio": round(sum(1 for x in decoded.pause if x) / max(1, len(decoded.pause)), 4),
        "stress_frames": sum(1 for x in decoded.stress if x),
        "note": "style plan is prepared before TTS/acoustic generation",
    }
    return StylePlan(
        target_duration_sec=target_duration_sec,
        suggested_tts_speed=suggested_speed,
        target_frame_count=target_frames,
        energy=decoded.energy,
        log_f0_rel=decoded.log_f0_rel,
        pause=decoded.pause,
        stress=decoded.stress,
        breath=breath,
        breathiness_target=speaker.breathiness,
        hnr_target=speaker.harmonic_noise_ratio,
        diagnostics=diagnostics,
        events=[],
    )


def style_plan_to_dict(plan: StylePlan) -> dict[str, object]:
    return asdict(plan)


def _build_event_style_plan(
    *,
    source_trace: PatternTrace,
    speaker: SpeakerProfile,
    content: ContentUnits,
    source_unit_count: int,
    events: list[TargetStyleEvent],
) -> StylePlan:
    hop_ms = max(1, source_trace.hop_ms)
    target_duration_ms = max(event.end_ms + event.pause_after_ms for event in events)
    target_frames = max(1, round(target_duration_ms / hop_ms))
    energy = [0.0] * target_frames
    log_f0_rel = [0.0] * target_frames
    pause = [True] * target_frames
    stress = [False] * target_frames
    breath = [0.0] * target_frames

    for event in events:
        start_index = max(0, min(target_frames - 1, round(event.start_ms / hop_ms)))
        end_index = max(start_index + 1, min(target_frames, round(event.end_ms / hop_ms)))
        pause_before_frames = round(event.pause_before_ms / hop_ms)
        pause_after_frames = round(event.pause_after_ms / hop_ms)
        for frame_index in range(max(0, start_index - pause_before_frames), start_index):
            pause[frame_index] = True
            breath[frame_index] = max(breath[frame_index], event.breath)
        span = max(1, end_index - start_index)
        for offset, frame_index in enumerate(range(start_index, end_index)):
            pos = offset / max(1, span - 1)
            pause[frame_index] = False
            energy[frame_index] = max(energy[frame_index], _event_energy(event, pos))
            log_f0_rel[frame_index] = _event_f0(event, pos)
            stress[frame_index] = stress[frame_index] or (event.stress >= 0.62 and 0.20 <= pos <= 0.72)
            breath[frame_index] = max(breath[frame_index], event.breath * (0.45 if event.energy_peak > 0.65 else 1.0))
        for frame_index in range(end_index, min(target_frames, end_index + pause_after_frames)):
            pause[frame_index] = True
            breath[frame_index] = max(breath[frame_index], event.breath * 1.35)

    energy = _smooth(energy, keep=0.62)
    log_f0_rel = _smooth(log_f0_rel, keep=0.70)
    breath = _smooth(breath, keep=0.74)
    target_duration_sec = target_frames * hop_ms / 1000.0
    target_unit_count = len(content.units) or len(events)
    suggested_speed = _suggest_speed(
        source_duration_sec=source_trace.duration_ms / 1000.0,
        target_duration_sec=target_duration_sec,
        source_unit_count=source_unit_count,
        target_unit_count=target_unit_count,
    )
    diagnostics: dict[str, float | int | str] = {
        "algorithm": "word_event_alignment_v1",
        "source_frames": len(source_trace.frames),
        "target_frames": target_frames,
        "source_unit_count": source_unit_count,
        "target_unit_count": target_unit_count,
        "source_duration_sec": round(source_trace.duration_ms / 1000.0, 4),
        "target_duration_sec": round(target_duration_sec, 4),
        "event_count": len(events),
        "pause_ratio": round(sum(1 for x in pause if x) / max(1, len(pause)), 4),
        "stress_frames": sum(1 for x in stress if x),
        "note": "word events are aligned to target content before acoustic generation",
    }
    return StylePlan(
        target_duration_sec=target_duration_sec,
        suggested_tts_speed=suggested_speed,
        target_frame_count=target_frames,
        energy=[round(max(0.0, min(1.0, value)), 4) for value in energy],
        log_f0_rel=[round(max(-0.70, min(0.70, value)), 4) for value in log_f0_rel],
        pause=pause,
        stress=stress,
        breath=[round(max(0.0, min(1.0, value)), 4) for value in breath],
        breathiness_target=speaker.breathiness,
        hnr_target=speaker.harmonic_noise_ratio,
        diagnostics=diagnostics,
        events=events,
    )


def _event_energy(event: TargetStyleEvent, pos: float) -> float:
    attack_shape = max(0.0, 1.0 - abs(pos - 0.28) / 0.28)
    stress_shape = max(0.0, 1.0 - abs(pos - 0.50) / 0.34)
    value = event.energy_mean
    value += event.attack_peak * 0.30 * attack_shape
    value += event.stress * 0.24 * stress_shape
    return max(value, event.energy_peak * 0.82 if event.stress >= 0.62 and stress_shape > 0.6 else value)


def _event_f0(event: TargetStyleEvent, pos: float) -> float:
    if event.log_f0_start is None or event.log_f0_end is None:
        return max(-0.35, min(0.35, event.log_f0_slope * (pos - 0.5)))
    return event.log_f0_start * (1.0 - pos) + event.log_f0_end * pos


def _suggest_speed(
    *,
    source_duration_sec: float,
    target_duration_sec: float,
    source_unit_count: int,
    target_unit_count: int,
) -> float:
    if target_duration_sec <= 0.0:
        return 1.0
    length_ratio = max(0.25, min(4.0, target_unit_count / max(1, source_unit_count)))
    duration_ratio = max(0.25, min(4.0, target_duration_sec / max(0.1, source_duration_sec)))
    # Keep the speed control narrow so the generated audio stays close to the
    # current utterance rhythm.
    speed = 1.0 / ((duration_ratio + length_ratio) * 0.5)
    return max(0.78, min(1.22, speed))


def _breath_curve(
    entropy: list[float],
    energy: list[float],
    pause: list[bool],
    speaker_breathiness: float,
) -> list[float]:
    curve: list[float] = []
    base = max(0.0, min(1.0, speaker_breathiness))
    for index, ent in enumerate(entropy):
        e = energy[index] if index < len(energy) else 0.0
        is_pause = pause[index] if index < len(pause) else False
        value = base * max(0.0, ent - e)
        if is_pause:
            value *= 1.45
        curve.append(max(0.0, min(1.0, value)))
    return _smooth(curve, keep=0.72)


def _smooth(values: list[float], keep: float) -> list[float]:
    if not values:
        return values
    state = values[0]
    out = [state]
    keep = max(0.0, min(0.98, keep))
    for value in values[1:]:
        state = state * keep + value * (1.0 - keep)
        out.append(state)
    return out
