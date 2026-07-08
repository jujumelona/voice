from __future__ import annotations

from dataclasses import dataclass

from voice_engine.pipeline.types import PatternTrace

LOG_F0_MIN = -0.55
LOG_F0_MAX = 0.55


@dataclass(frozen=True)
class Q8PatternTrace:
    frame_ms: int
    hop_ms: int
    frames: list[list[int]]


@dataclass(frozen=True)
class DecodedQ8Pattern:
    energy: list[float]
    log_f0_rel: list[float]
    entropy: list[float]
    attack: list[float]
    pause: list[bool]
    stress: list[bool]


def encode_q8(trace: PatternTrace) -> Q8PatternTrace:
    frames: list[list[int]] = []
    for frame in trace.frames:
        frames.append(
            [
                _float_to_q8(frame.energy),
                _log_f0_to_q8(frame.f0),
                _float_to_q8(frame.entropy),
                _float_to_q8(frame.attack),
                1 if frame.pause else 0,
                1 if frame.stress else 0,
            ]
        )
    return Q8PatternTrace(frame_ms=trace.frame_ms, hop_ms=trace.hop_ms, frames=frames)


def decode_q8(trace: Q8PatternTrace) -> DecodedQ8Pattern:
    energy: list[float] = []
    log_f0_rel: list[float] = []
    entropy: list[float] = []
    attack: list[float] = []
    pause: list[bool] = []
    stress: list[bool] = []

    for frame in trace.frames:
        energy.append(_q8_to_float(frame[0]))
        log_f0_rel.append(_q8_to_log_f0(frame[1]))
        entropy.append(_q8_to_float(frame[2]))
        attack.append(_q8_to_float(frame[3]))
        pause.append(bool(frame[4]))
        stress.append(bool(frame[5]))

    return DecodedQ8Pattern(
        energy=energy,
        log_f0_rel=log_f0_rel,
        entropy=entropy,
        attack=attack,
        pause=pause,
        stress=stress,
    )


def warp_q8_linear(trace: Q8PatternTrace, target_frame_count: int) -> Q8PatternTrace:
    target_frame_count = max(1, target_frame_count)
    if not trace.frames or len(trace.frames) == target_frame_count:
        return trace

    source_last = len(trace.frames) - 1
    target_last = max(1, target_frame_count - 1)
    frames: list[list[int]] = []
    for index in range(target_frame_count):
        position = index * source_last / target_last
        left = int(position)
        right = min(source_last, left + 1)
        weight = position - left
        frames.append(_interpolate_frame(trace.frames[left], trace.frames[right], weight))
    return Q8PatternTrace(frame_ms=trace.frame_ms, hop_ms=trace.hop_ms, frames=frames)


def _interpolate_frame(left: list[int], right: list[int], weight: float) -> list[int]:
    pause = left[4] if weight < 0.5 else right[4]
    stress = 1 if left[5] or right[5] else 0
    return [
        _lerp_q8(left[0], right[0], weight),
        _lerp_q8(left[1], right[1], weight),
        _lerp_q8(left[2], right[2], weight),
        _lerp_q8(left[3], right[3], weight),
        pause,
        stress,
    ]


def _float_to_q8(value: float | None) -> int:
    if value is None:
        return 0
    return max(0, min(255, round(value * 255.0)))


def _q8_to_float(value: int) -> float:
    return max(0, min(255, value)) / 255.0


def _log_f0_to_q8(value: float | None) -> int:
    if value is None:
        return 128
    clipped = max(LOG_F0_MIN, min(LOG_F0_MAX, value))
    return round((clipped - LOG_F0_MIN) / (LOG_F0_MAX - LOG_F0_MIN) * 255.0)


def _q8_to_log_f0(value: int) -> float:
    value = max(0, min(255, value))
    return LOG_F0_MIN + value / 255.0 * (LOG_F0_MAX - LOG_F0_MIN)


def _lerp_q8(left: int, right: int, weight: float) -> int:
    return max(0, min(255, round(left * (1.0 - weight) + right * weight)))

