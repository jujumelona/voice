from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from voice_engine.prosody.extractor import extract_prosody
from voice_engine.speaker.speaker_profile import extract_speaker_profile_from_wav, extract_speaker_side_metrics_from_wav


@dataclass(frozen=True)
class StyleMetricReport:
    source_duration_sec: float
    output_duration_sec: float
    duration_ratio: float
    source_pause_ratio: float
    output_pause_ratio: float
    pause_delta: float
    source_f0_mean: float | None
    output_f0_mean: float | None
    f0_ratio: float | None
    source_energy_peak: float
    output_energy_peak: float
    energy_peak_ratio: float
    source_breathiness: float
    output_breathiness: float
    breathiness_delta: float
    source_hnr: float
    output_hnr: float
    hnr_delta: float
    energy_contour_corr: float
    f0_contour_corr: float | None
    pause_match: float
    speaker_similarity: float
    machine_risk_score: float
    passed: bool


def compare_style_audio(source_wav: str | Path, output_wav: str | Path) -> StyleMetricReport:
    source_prosody = extract_prosody(source_wav)
    output_prosody = extract_prosody(output_wav, speaker_median_f0=source_prosody.trace.median_f0)
    source_speaker = extract_speaker_side_metrics_from_wav(source_wav)
    output_speaker = extract_speaker_side_metrics_from_wav(output_wav)
    source_profile = extract_speaker_profile_from_wav(source_wav)
    output_profile = extract_speaker_profile_from_wav(output_wav)

    duration_ratio = _ratio(output_prosody.duration_sec, source_prosody.duration_sec)
    f0_ratio = (
        _ratio(output_prosody.f0_mean, source_prosody.f0_mean)
        if output_prosody.f0_mean and source_prosody.f0_mean
        else None
    )
    energy_peak_ratio = _ratio(output_prosody.energy_peak, source_prosody.energy_peak)
    pause_delta = abs(output_prosody.pause_ratio - source_prosody.pause_ratio)
    breathiness_delta = abs(output_speaker.breathiness - source_speaker.breathiness)
    hnr_delta = abs(output_speaker.harmonic_noise_ratio - source_speaker.harmonic_noise_ratio)
    energy_corr = _corr(
        [frame.energy for frame in source_prosody.trace.frames],
        [frame.energy for frame in output_prosody.trace.frames],
    )
    f0_corr = _corr_optional(
        [frame.f0 for frame in source_prosody.trace.frames],
        [frame.f0 for frame in output_prosody.trace.frames],
    )
    pause_match = _pause_match(
        [frame.pause for frame in source_prosody.trace.frames],
        [frame.pause for frame in output_prosody.trace.frames],
    )
    speaker_similarity = _cosine(source_profile.speaker_embedding, output_profile.speaker_embedding)
    machine_risk = _machine_risk(
        duration_ratio=duration_ratio,
        pause_delta=pause_delta,
        f0_ratio=f0_ratio,
        energy_peak_ratio=energy_peak_ratio,
        breathiness_delta=breathiness_delta,
        hnr_delta=hnr_delta,
        energy_corr=energy_corr,
        f0_corr=f0_corr,
        pause_match=pause_match,
        speaker_similarity=speaker_similarity,
    )
    return StyleMetricReport(
        source_duration_sec=round(source_prosody.duration_sec, 4),
        output_duration_sec=round(output_prosody.duration_sec, 4),
        duration_ratio=round(duration_ratio, 4),
        source_pause_ratio=round(source_prosody.pause_ratio, 4),
        output_pause_ratio=round(output_prosody.pause_ratio, 4),
        pause_delta=round(pause_delta, 4),
        source_f0_mean=_round_optional(source_prosody.f0_mean),
        output_f0_mean=_round_optional(output_prosody.f0_mean),
        f0_ratio=_round_optional(f0_ratio),
        source_energy_peak=round(source_prosody.energy_peak, 6),
        output_energy_peak=round(output_prosody.energy_peak, 6),
        energy_peak_ratio=round(energy_peak_ratio, 4),
        source_breathiness=round(source_speaker.breathiness, 4),
        output_breathiness=round(output_speaker.breathiness, 4),
        breathiness_delta=round(breathiness_delta, 4),
        source_hnr=round(source_speaker.harmonic_noise_ratio, 4),
        output_hnr=round(output_speaker.harmonic_noise_ratio, 4),
        hnr_delta=round(hnr_delta, 4),
        energy_contour_corr=round(energy_corr, 4),
        f0_contour_corr=_round_optional(f0_corr),
        pause_match=round(pause_match, 4),
        speaker_similarity=round(speaker_similarity, 4),
        machine_risk_score=round(machine_risk, 4),
        passed=machine_risk <= 0.22 and speaker_similarity >= 0.62,
    )


def write_style_metric_report(source_wav: str | Path, output_wav: str | Path, report_path: str | Path) -> StyleMetricReport:
    report = compare_style_audio(source_wav, output_wav)
    Path(report_path).write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _machine_risk(
    *,
    duration_ratio: float,
    pause_delta: float,
    f0_ratio: float | None,
    energy_peak_ratio: float,
    breathiness_delta: float,
    hnr_delta: float,
    energy_corr: float,
    f0_corr: float | None,
    pause_match: float,
    speaker_similarity: float,
) -> float:
    duration_penalty = min(1.0, abs(duration_ratio - 1.0) / 0.55)
    pause_penalty = min(1.0, pause_delta / 0.28)
    f0_penalty = 0.35 if f0_ratio is None else min(1.0, abs(f0_ratio - 1.0) / 0.45)
    energy_penalty = min(1.0, abs(energy_peak_ratio - 1.0) / 0.75)
    breath_penalty = min(1.0, breathiness_delta / 0.35)
    hnr_penalty = min(1.0, hnr_delta / 0.40)
    energy_shape_penalty = min(1.0, max(0.0, 0.72 - energy_corr) / 1.44)
    f0_shape_penalty = 0.45 if f0_corr is None else min(1.0, max(0.0, 0.55 - f0_corr) / 1.10)
    pause_shape_penalty = min(1.0, max(0.0, 0.82 - pause_match) / 0.82)
    speaker_penalty = min(1.0, max(0.0, 0.72 - speaker_similarity) / 0.72)
    return (
        0.08 * duration_penalty
        + 0.10 * pause_penalty
        + 0.10 * f0_penalty
        + 0.08 * energy_penalty
        + 0.07 * breath_penalty
        + 0.05 * hnr_penalty
        + 0.15 * energy_shape_penalty
        + 0.10 * f0_shape_penalty
        + 0.07 * pause_shape_penalty
        + 0.20 * speaker_penalty
    )


def _ratio(left: float | None, right: float | None) -> float:
    if left is None or right is None or abs(right) <= 1e-9:
        return 1.0
    return left / right


def _round_optional(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _corr(left: list[float], right: list[float]) -> float:
    n = max(2, min(len(left), len(right)))
    left_r = _resample_float(left, n)
    right_r = _resample_float(right, n)
    lm = sum(left_r) / n
    rm = sum(right_r) / n
    num = sum((a - lm) * (b - rm) for a, b in zip(left_r, right_r))
    ld = math_sqrt(sum((a - lm) ** 2 for a in left_r))
    rd = math_sqrt(sum((b - rm) ** 2 for b in right_r))
    if ld <= 1e-9 or rd <= 1e-9:
        return 0.0
    return max(-1.0, min(1.0, num / (ld * rd)))


def _corr_optional(left: list[float | None], right: list[float | None]) -> float | None:
    n = max(2, min(len(left), len(right)))
    left_r = _resample_optional(left, n)
    right_r = _resample_optional(right, n)
    pairs = [(a, b) for a, b in zip(left_r, right_r) if a is not None and b is not None]
    if len(pairs) < 4:
        return None
    return _corr([a for a, _ in pairs], [b for _, b in pairs])


def _pause_match(left: list[bool], right: list[bool]) -> float:
    n = max(1, min(len(left), len(right)))
    left_r = _resample_bool(left, n)
    right_r = _resample_bool(right, n)
    return sum(1 for a, b in zip(left_r, right_r) if a == b) / n


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    left_norm = math_sqrt(sum(value * value for value in left))
    right_norm = math_sqrt(sum(value * value for value in right))
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return -1.0
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)))


def _resample_float(values: list[float], count: int) -> list[float]:
    if not values:
        return [0.0] * count
    if len(values) == 1:
        return [values[0]] * count
    last = len(values) - 1
    target_last = max(1, count - 1)
    out = []
    for index in range(count):
        pos = index * last / target_last
        left = int(pos)
        right = min(last, left + 1)
        weight = pos - left
        out.append(values[left] * (1.0 - weight) + values[right] * weight)
    return out


def _resample_optional(values: list[float | None], count: int) -> list[float | None]:
    filled = [0.0 if value is None else value for value in values]
    mask = [0.0 if value is None else 1.0 for value in values]
    filled_r = _resample_float(filled, count)
    mask_r = _resample_float(mask, count)
    return [value if keep >= 0.5 else None for value, keep in zip(filled_r, mask_r)]


def _resample_bool(values: list[bool], count: int) -> list[bool]:
    if not values:
        return [False] * count
    if len(values) == 1:
        return [values[0]] * count
    last = len(values) - 1
    target_last = max(1, count - 1)
    out = []
    for index in range(count):
        source = round(index * last / target_last)
        out.append(values[max(0, min(last, source))])
    return out


def math_sqrt(value: float) -> float:
    return value ** 0.5
