from __future__ import annotations

from voice_engine.pipeline.types import ContentUnits, ProsodyEvent, TargetStyleEvent


def align_events_to_content(
    source_events: list[ProsodyEvent],
    content: ContentUnits,
) -> list[TargetStyleEvent]:
    target_units = content.units or [content.text]
    if not source_events or not target_units:
        return []

    total_source_ms = max(source_events[-1].end_ms - source_events[0].start_ms, 1.0)
    target_duration_ms = _target_duration(source_events, target_units)
    aligned: list[TargetStyleEvent] = []

    for index, unit in enumerate(target_units):
        source_index = min(
            len(source_events) - 1,
            round(index * (len(source_events) - 1) / max(1, len(target_units) - 1)),
        )
        source = source_events[source_index]
        start = index * target_duration_ms / len(target_units)
        end = (index + 1) * target_duration_ms / len(target_units)
        duration_scale = (end - start) / max(1.0, source.end_ms - source.start_ms)
        aligned.append(
            TargetStyleEvent(
                target_text=unit,
                source_text=source.source_text,
                start_ms=round(start, 3),
                end_ms=round(end, 3),
                pause_before_ms=round(source.pause_before_ms * duration_scale, 3),
                pause_after_ms=round(source.pause_after_ms * duration_scale, 3),
                energy_mean=source.energy_mean,
                energy_peak=source.energy_peak,
                log_f0_start=source.log_f0_start,
                log_f0_end=source.log_f0_end,
                log_f0_slope=source.log_f0_slope,
                breath=source.breath,
                attack_peak=source.attack_peak,
                stress=source.stress,
                emotion_hint=source.emotion_hint,
            )
        )
    return _preserve_major_pauses(source_events, aligned, total_source_ms, target_duration_ms)


def _target_duration(source_events: list[ProsodyEvent], target_units: list[str]) -> float:
    source_duration = max(source_events[-1].end_ms - source_events[0].start_ms, 1.0)
    ratio = len(target_units) / max(1, len(source_events))
    ratio = max(0.72, min(1.55, ratio))
    return max(240.0 * len(target_units), source_duration * ratio)


def _preserve_major_pauses(
    source_events: list[ProsodyEvent],
    aligned: list[TargetStyleEvent],
    total_source_ms: float,
    target_duration_ms: float,
) -> list[TargetStyleEvent]:
    if not aligned:
        return aligned
    mutable = list(aligned)
    major_pauses = [
        (event.end_ms / total_source_ms, max(event.pause_after_ms, event.pause_before_ms))
        for event in source_events
        if max(event.pause_after_ms, event.pause_before_ms) >= 260.0
    ]
    for position, pause_ms in major_pauses:
        target_index = min(len(mutable) - 1, max(0, round(position * (len(mutable) - 1))))
        event = mutable[target_index]
        mutable[target_index] = TargetStyleEvent(
            target_text=event.target_text,
            source_text=event.source_text,
            start_ms=event.start_ms,
            end_ms=event.end_ms,
            pause_before_ms=event.pause_before_ms,
            pause_after_ms=round(max(event.pause_after_ms, min(pause_ms, target_duration_ms * 0.25)), 3),
            energy_mean=event.energy_mean,
            energy_peak=event.energy_peak,
            log_f0_start=event.log_f0_start,
            log_f0_end=event.log_f0_end,
            log_f0_slope=event.log_f0_slope,
            breath=event.breath,
            attack_peak=event.attack_peak,
            stress=event.stress,
            emotion_hint=event.emotion_hint,
        )
    return mutable
