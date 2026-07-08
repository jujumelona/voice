from __future__ import annotations

from dataclasses import dataclass

from voice_engine.prosody.q8_codec import Q8PatternTrace


@dataclass(frozen=True)
class RetimeStats:
    source_frames: int
    target_frames: int
    anchor_count: int
    pause_frames: int
    stress_frames: int
    max_energy_step: int
    max_f0_step: int


def retime_q8_anchor_locked(
    trace: Q8PatternTrace,
    target_frame_count: int,
    *,
    max_energy_step: int = 28,
    max_f0_step: int = 18,
    smoothing: float = 0.18,
) -> Q8PatternTrace:
    """Retimes utterance-local q8 prosody without a learned embedding.

    Continuous channels are interpolated between event anchors. Binary channels
    are copied from nearest source frames and lightly widened, so pause/stress
    structure survives language-length changes better than plain linear warp.
    """

    target_frame_count = max(1, target_frame_count)
    if not trace.frames:
        return Q8PatternTrace(trace.frame_ms, trace.hop_ms, [])
    anchors = _anchors(trace.frames)
    mapped = _map_anchors(anchors, len(trace.frames), target_frame_count)

    result: list[list[int]] = []
    for target_index in range(target_frame_count):
        source_position = _source_position_from_anchors(
            target_index,
            anchors,
            mapped,
            len(trace.frames),
            target_frame_count,
        )
        left = int(source_position)
        right = min(len(trace.frames) - 1, left + 1)
        weight = source_position - left
        nearest = left if weight < 0.5 else right
        result.append(
            [
                _lerp(trace.frames[left][0], trace.frames[right][0], weight),
                _lerp(trace.frames[left][1], trace.frames[right][1], weight),
                _lerp(trace.frames[left][2], trace.frames[right][2], weight),
                _lerp(trace.frames[left][3], trace.frames[right][3], weight),
                trace.frames[nearest][4],
                trace.frames[nearest][5],
            ]
        )

    result = _lock_binary_events(result, trace.frames, anchors, mapped)
    result = _smooth_continuous(result, smoothing)
    result = _limit_steps(result, channel=0, max_step=max_energy_step)
    result = _limit_steps(result, channel=1, max_step=max_f0_step)
    result = _recompute_attack(result)
    return Q8PatternTrace(trace.frame_ms, trace.hop_ms, result)


def retime_q8_with_stats(
    trace: Q8PatternTrace,
    target_frame_count: int,
) -> tuple[Q8PatternTrace, RetimeStats]:
    retimed = retime_q8_anchor_locked(trace, target_frame_count)
    stats = RetimeStats(
        source_frames=len(trace.frames),
        target_frames=len(retimed.frames),
        anchor_count=len(_anchors(trace.frames)),
        pause_frames=sum(1 for frame in retimed.frames if frame[4]),
        stress_frames=sum(1 for frame in retimed.frames if frame[5]),
        max_energy_step=_max_step(retimed.frames, 0),
        max_f0_step=_max_step(retimed.frames, 1),
    )
    return retimed, stats


def target_frames_from_units(
    source_frames: int,
    source_unit_count: int,
    target_unit_count: int,
    *,
    min_ratio: float = 0.65,
    max_ratio: float = 1.80,
) -> int:
    source_frames = max(1, source_frames)
    source_unit_count = max(1, source_unit_count)
    target_unit_count = max(1, target_unit_count)
    ratio = target_unit_count / source_unit_count
    ratio = max(min_ratio, min(max_ratio, ratio))
    return max(1, round(source_frames * ratio))


def _anchors(frames: list[list[int]]) -> list[int]:
    anchors = {0, len(frames) - 1}
    previous_pause = bool(frames[0][4])
    for index, frame in enumerate(frames):
        pause = bool(frame[4])
        if pause != previous_pause:
            anchors.add(index)
            if index > 0:
                anchors.add(index - 1)
        previous_pause = pause
        if frame[5] or frame[3] >= 56:
            anchors.add(index)
    return sorted(anchors)


def _map_anchors(
    anchors: list[int],
    source_count: int,
    target_count: int,
) -> list[int]:
    if source_count <= 1 or target_count <= 1:
        return [0 for _ in anchors]
    scale = (target_count - 1) / (source_count - 1)
    mapped = [round(anchor * scale) for anchor in anchors]
    for index in range(1, len(mapped)):
        if mapped[index] <= mapped[index - 1]:
            mapped[index] = mapped[index - 1] + 1
    overflow = mapped[-1] - (target_count - 1)
    if overflow > 0:
        mapped = [max(0, value - overflow) for value in mapped]
    mapped[0] = 0
    mapped[-1] = target_count - 1
    return mapped


def _source_position_from_anchors(
    target_index: int,
    source_anchors: list[int],
    target_anchors: list[int],
    source_count: int,
    target_count: int,
) -> float:
    if source_count <= 1 or target_count <= 1:
        return 0.0
    for index in range(len(target_anchors) - 1):
        target_left = target_anchors[index]
        target_right = target_anchors[index + 1]
        if target_left <= target_index <= target_right:
            source_left = source_anchors[index]
            source_right = source_anchors[index + 1]
            span = max(1, target_right - target_left)
            weight = (target_index - target_left) / span
            return source_left * (1.0 - weight) + source_right * weight
    return target_index * (source_count - 1) / max(1, target_count - 1)


def _lock_binary_events(
    target: list[list[int]],
    source: list[list[int]],
    anchors: list[int],
    mapped: list[int],
) -> list[list[int]]:
    result = [frame[:] for frame in target]
    for source_index, target_index in zip(anchors, mapped):
        if not (0 <= target_index < len(result)):
            continue
        source_frame = source[source_index]
        width = 1 if source_frame[4] else 0
        for offset in range(-width, width + 1):
            index = target_index + offset
            if 0 <= index < len(result):
                result[index][4] = source_frame[4]
                result[index][5] = max(result[index][5], source_frame[5])
                if source_frame[5]:
                    result[index][0] = max(result[index][0], source_frame[0])
                    result[index][3] = max(result[index][3], source_frame[3])
    return result


def _smooth_continuous(frames: list[list[int]], smoothing: float) -> list[list[int]]:
    if not frames or smoothing <= 0.0:
        return frames
    smoothing = max(0.0, min(0.95, smoothing))
    result = [frames[0][:]]
    state = [float(value) for value in frames[0][:4]]
    for frame in frames[1:]:
        next_frame = frame[:]
        if not frame[4]:
            for channel in range(4):
                state[channel] = state[channel] * smoothing + frame[channel] * (1.0 - smoothing)
                next_frame[channel] = _clamp_q8(round(state[channel]))
        else:
            state = [float(value) for value in frame[:4]]
        result.append(next_frame)
    return result


def _limit_steps(frames: list[list[int]], channel: int, max_step: int) -> list[list[int]]:
    if not frames:
        return frames
    result = [frames[0][:]]
    previous = frames[0][channel]
    for frame in frames[1:]:
        next_frame = frame[:]
        delta = next_frame[channel] - previous
        if delta > max_step:
            next_frame[channel] = previous + max_step
        elif delta < -max_step:
            next_frame[channel] = previous - max_step
        next_frame[channel] = _clamp_q8(next_frame[channel])
        previous = next_frame[channel]
        result.append(next_frame)
    return result


def _recompute_attack(frames: list[list[int]]) -> list[list[int]]:
    if not frames:
        return frames
    result = [frames[0][:]]
    previous_energy = frames[0][0]
    for frame in frames[1:]:
        next_frame = frame[:]
        next_frame[3] = _clamp_q8(max(0, next_frame[0] - previous_energy))
        if frame[5]:
            next_frame[3] = max(next_frame[3], frame[3])
        previous_energy = next_frame[0]
        result.append(next_frame)
    return result


def _max_step(frames: list[list[int]], channel: int) -> int:
    if len(frames) < 2:
        return 0
    return max(abs(right[channel] - left[channel]) for left, right in zip(frames, frames[1:]))


def _lerp(left: int, right: int, weight: float) -> int:
    return _clamp_q8(round(left * (1.0 - weight) + right * weight))


def _clamp_q8(value: int) -> int:
    return max(0, min(255, int(value)))
