from __future__ import annotations

import math
import wave
from pathlib import Path

from voice_engine.pipeline.types import PatternAnchor, PatternFrame, PatternTrace, ProsodyPattern


def _read_mono_pcm16(path: Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if width != 2:
        raise ValueError("Only PCM16 wav is supported by the lightweight extractor.")

    samples: list[float] = []
    step = width * channels
    for i in range(0, len(frames), step):
        channel_sum = 0
        for ch in range(channels):
            offset = i + ch * width
            raw = frames[offset : offset + width]
            channel_sum += int.from_bytes(raw, "little", signed=True)
        samples.append(channel_sum / channels / 32768.0)
    return sample_rate, samples


def extract_prosody(
    path: str | Path,
    frame_ms: int = 20,
    hop_ms: int = 10,
    speaker_median_f0: float | None = None,
) -> ProsodyPattern:
    wav_path = Path(path)
    sample_rate, samples = _read_mono_pcm16(wav_path)
    return extract_prosody_from_samples(
        samples,
        sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        speaker_median_f0=speaker_median_f0,
    )


def extract_prosody_from_samples(
    samples: list[float],
    sample_rate: int,
    frame_ms: int = 20,
    hop_ms: int = 10,
    speaker_median_f0: float | None = None,
) -> ProsodyPattern:
    if not samples:
        return ProsodyPattern(
            duration_sec=0.0,
            speech_rate=0.0,
            energy_mean=0.0,
            energy_peak=0.0,
            f0_mean=None,
            f0_range=None,
            pause_ratio=1.0,
            attack=0.0,
            energy_contour=[],
            f0_contour=[],
            emphasis=[],
            trace=PatternTrace(
                frame_ms=frame_ms,
                hop_ms=hop_ms,
                duration_ms=0.0,
                median_f0=None,
                f0_reference=speaker_median_f0,
                speaking_rate=0.0,
                energy_p10=0.0,
                energy_p90=0.0,
                frames=[],
                anchors=[],
            ),
        )

    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    hop_size = max(1, int(sample_rate * hop_ms / 1000))
    energies: list[float] = []
    f0_values: list[float | None] = []
    entropies: list[float] = []

    for start in range(0, max(1, len(samples) - frame_size + 1), hop_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        rms = math.sqrt(sum(x * x for x in frame) / len(frame))
        energies.append(rms)
        entropies.append(_estimate_entropy(frame))
        f0_values.append(_estimate_f0(frame, sample_rate) if rms > 0.005 else None)

    duration = len(samples) / sample_rate
    energy_mean = sum(energies) / len(energies) if energies else 0.0
    energy_peak = max(energies) if energies else 0.0
    energy_p10 = _percentile(energies, 10.0)
    energy_p90 = _percentile(energies, 90.0)

    normalized_energies = [
        _normalize_energy(value, energy_p10, energy_p90) for value in energies
    ]
    pause_flags = [
        energy < max(0.006, energy_p10 + (energy_p90 - energy_p10) * 0.12)
        for energy in energies
    ]
    f0_values = [None if pause else f0 for f0, pause in zip(f0_values, pause_flags)]
    valid_f0 = [x for x in f0_values if x is not None]
    f0_mean = sum(valid_f0) / len(valid_f0) if valid_f0 else None
    f0_range = max(valid_f0) - min(valid_f0) if valid_f0 else None
    median_f0 = _median(valid_f0) if valid_f0 else None
    f0_reference = speaker_median_f0 or median_f0
    voiced_frames = sum(1 for paused in pause_flags if not paused)
    pause_ratio = 1.0 - voiced_frames / len(energies) if energies else 1.0
    attack = _estimate_attack(normalized_energies)
    energy_contour = [round(x, 4) for x in normalized_energies]
    emphasis = [round(x, 4) for x in normalized_energies]
    speech_rate = voiced_frames / duration if duration else 0.0
    trace = _build_trace(
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        duration_ms=duration * 1000.0,
        normalized_energies=normalized_energies,
        entropies=entropies,
        f0_values=f0_values,
        median_f0=median_f0,
        f0_reference=f0_reference,
        energy_p10=energy_p10,
        energy_p90=energy_p90,
        pause_flags=pause_flags,
        speaking_rate=speech_rate,
    )

    return ProsodyPattern(
        duration_sec=duration,
        speech_rate=speech_rate,
        energy_mean=energy_mean,
        energy_peak=energy_peak,
        f0_mean=f0_mean,
        f0_range=f0_range,
        pause_ratio=pause_ratio,
        attack=attack,
        energy_contour=energy_contour,
        f0_contour=f0_values,
        emphasis=emphasis,
        trace=trace,
    )


def _estimate_f0(frame: list[float], sample_rate: int) -> float | None:
    lag_min = max(1, int(sample_rate / 420))
    lag_max = min(len(frame) - 1, int(sample_rate / 70))
    if lag_max <= lag_min:
        return None

    frame_mean = sum(frame) / len(frame)
    centered = [x - frame_mean for x in frame]
    energy = sum(x * x for x in centered)
    if energy <= 1e-9:
        return None

    best_lag = 0
    best_score = 0.0
    for lag in range(lag_min, lag_max + 1):
        score = 0.0
        for i in range(len(centered) - lag):
            score += centered[i] * centered[i + lag]
        score /= energy
        if score > best_score:
            best_score = score
            best_lag = lag

    if best_score < 0.28 or best_lag == 0:
        return None
    return sample_rate / best_lag


def _estimate_attack(energies: list[float]) -> float:
    if len(energies) < 2:
        return 0.0
    peak = max(energies)
    if peak <= 1e-9:
        return 0.0
    max_rise = max(
        max(0.0, current - previous)
        for previous, current in zip(energies, energies[1:])
    )
    return round(max_rise / peak, 4)


def _build_trace(
    frame_ms: int,
    hop_ms: int,
    duration_ms: float,
    normalized_energies: list[float],
    entropies: list[float],
    f0_values: list[float | None],
    median_f0: float | None,
    f0_reference: float | None,
    energy_p10: float,
    energy_p90: float,
    pause_flags: list[bool],
    speaking_rate: float,
) -> PatternTrace:
    frames: list[PatternFrame] = []
    previous_energy = 0.0
    for index, (energy, entropy, f0_hz, pause) in enumerate(
        zip(normalized_energies, entropies, f0_values, pause_flags)
    ):
        normalized_energy = energy
        attack = max(0.0, normalized_energy - previous_energy)
        f0 = math.log(f0_hz / f0_reference) if f0_hz and f0_reference else None
        frames.append(
            PatternFrame(
                time_ms=float(index * hop_ms),
                energy=round(normalized_energy, 4),
                entropy=round(entropy, 4),
                f0=round(f0, 4) if f0 is not None else None,
                f0_hz=f0_hz,
                attack=round(attack, 4),
                pause=pause,
                stress=False,
            )
        )
        previous_energy = normalized_energy

    anchors = _detect_anchors(frames)
    stress_times = {anchor.time_ms for anchor in anchors if anchor.kind == "stress_peak"}
    frames = [
        PatternFrame(
            time_ms=frame.time_ms,
            energy=frame.energy,
            entropy=frame.entropy,
            f0=frame.f0,
            f0_hz=frame.f0_hz,
            attack=frame.attack,
            pause=frame.pause,
            stress=frame.time_ms in stress_times,
        )
        for frame in frames
    ]
    return PatternTrace(
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        duration_ms=duration_ms,
        median_f0=median_f0,
        f0_reference=f0_reference,
        speaking_rate=speaking_rate,
        energy_p10=energy_p10,
        energy_p90=energy_p90,
        frames=frames,
        anchors=anchors,
    )


def _estimate_entropy(frame: list[float], bins: int = 16) -> float:
    if not frame:
        return 0.0
    counts = [0] * bins
    for sample in frame:
        index = int((sample + 1.0) * 0.5 * bins)
        index = max(0, min(bins - 1, index))
        counts[index] += 1
    total = len(frame)
    entropy = 0.0
    for count in counts:
        if count == 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy / math.log2(bins)


def _normalize_energy(value: float, low: float, high: float) -> float:
    span = high - low
    if span <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, (value - low) / span))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _median(values: list[float]) -> float:
    return _percentile(values, 50.0)


def _detect_anchors(frames: list[PatternFrame]) -> list[PatternAnchor]:
    anchors: list[PatternAnchor] = []
    previous_pause = True
    for index, frame in enumerate(frames):
        if frame.pause != previous_pause:
            anchors.append(
                PatternAnchor(
                    time_ms=frame.time_ms,
                    kind="pause_end" if previous_pause else "pause_start",
                    strength=1.0,
                )
            )
        previous_pause = frame.pause

        prev_energy = frames[index - 1].energy if index > 0 else frame.energy
        next_energy = frames[index + 1].energy if index + 1 < len(frames) else frame.energy
        local_peak = frame.energy >= prev_energy and frame.energy >= next_energy
        if not frame.pause and local_peak and (frame.energy >= 0.72 or frame.attack >= 0.22):
            anchors.append(
                PatternAnchor(
                    time_ms=frame.time_ms,
                    kind="stress_peak",
                    strength=round(max(frame.energy, frame.attack), 4),
                )
            )
    return anchors
