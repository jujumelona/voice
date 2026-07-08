from __future__ import annotations

import hashlib
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voice_engine.pipeline.types import SpeakerProfile
from voice_engine.prosody.extractor import extract_prosody
from voice_engine.speaker.wespeaker_eres2net import WeSpeakerERes2NetLargeSpeakerEncoder, MODEL_BACKEND

_ENCODER_CACHE: dict[tuple[str | None, str | None], WeSpeakerERes2NetLargeSpeakerEncoder] = {}


@dataclass(frozen=True)
class SpeakerSideMetrics:
    breathiness: float
    harmonic_noise_ratio: float


def extract_speaker_profile_from_wav(
    path: str | Path,
    *,
    device: str | None = None,
    cache_dir: str | Path | None = None,
) -> SpeakerProfile:
    """Extract speaker identity with WeSpeaker ERes2Net-large only."""

    wav_path = Path(path)
    encoder = _get_encoder(device=device, cache_dir=cache_dir)
    embedding = encoder.encode_file(wav_path)
    prosody = extract_prosody(wav_path)
    _, samples = _read_mono_pcm16(wav_path)
    breathiness = _estimate_breathiness(samples)
    hnr = _estimate_harmonic_noise_ratio(samples)

    return SpeakerProfile(
        speaker_embedding=embedding.embedding,
        spectral_envelope=[],
        formant_profile=[],
        average_f0=prosody.f0_mean,
        median_f0=prosody.trace.median_f0,
        pitch_range=prosody.f0_range,
        breathiness=breathiness,
        harmonic_noise_ratio=hnr,
        speaking_rate_baseline=prosody.speech_rate,
        pause_ratio_baseline=prosody.pause_ratio,
        energy_baseline=prosody.energy_mean,
        style_habits=_style_habits_from_prosody(prosody),
        timbre_code=[],
        speaker_embedding_backend=embedding.backend,
        speaker_embedding_dim=embedding.dim,
        embedding_l2_normalized=embedding.l2_normalized,
        speaker_quality=embedding.quality,
        speech_duration_sec=embedding.speech_duration_sec,
        update_count=1,
    )



def extract_speaker_profile_from_samples(
    samples: list[float] | np.ndarray,
    sample_rate: int,
    *,
    wav_path: str | Path | None = None,
    device: str | None = None,
    cache_dir: str | Path | None = None,
) -> SpeakerProfile:
    """Extract a profile from in-memory audio.

    WeSpeaker itself is file based in this repo, so this function reuses the
    stable call reference path when available. If not available, it writes a
    deterministic cache wav under .voice_bridge_runtime instead of a temp file.
    """
    if wav_path is not None and Path(wav_path).exists():
        return extract_speaker_profile_from_wav(wav_path, device=device, cache_dir=cache_dir)

    audio = np.nan_to_num(np.asarray(samples, dtype=np.float32))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    if audio.size == 0:
        raise RuntimeError("cannot extract speaker profile from empty samples")

    cache_root = Path(".voice_bridge_runtime") / "speaker_samples"
    cache_root.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1()
    h.update(str(int(sample_rate)).encode("ascii"))
    h.update(audio[: min(len(audio), int(sample_rate) * 8)].tobytes())
    path = cache_root / ("speaker_" + h.hexdigest()[:16] + ".wav")
    if not path.exists():
        _write_float_wav(path, audio, int(sample_rate))
    return extract_speaker_profile_from_wav(path, device=device, cache_dir=cache_dir)


def _get_encoder(
    *,
    device: str | None = None,
    cache_dir: str | Path | None = None,
) -> WeSpeakerERes2NetLargeSpeakerEncoder:
    key = (device, str(cache_dir) if cache_dir is not None else None)
    encoder = _ENCODER_CACHE.get(key)
    if encoder is None:
        encoder = WeSpeakerERes2NetLargeSpeakerEncoder(device=device, cache_dir=cache_dir)
        _ENCODER_CACHE[key] = encoder
    return encoder

def extract_speaker_side_metrics_from_wav(path: str | Path) -> SpeakerSideMetrics:
    _, samples = _read_mono_pcm16(Path(path))
    return SpeakerSideMetrics(
        breathiness=_estimate_breathiness(samples),
        harmonic_noise_ratio=_estimate_harmonic_noise_ratio(samples),
    )


def update_speaker_profile(
    old: SpeakerProfile | None,
    new: SpeakerProfile,
    alpha: float = 0.02,
    *,
    min_update_similarity: float = 0.35,
) -> SpeakerProfile:
    """Slowly update cumulative speaker identity, never replacing it per chunk."""

    if old is None:
        return new
    if old.speaker_embedding_backend != MODEL_BACKEND or new.speaker_embedding_backend != MODEL_BACKEND:
        raise RuntimeError("speaker profile update requires WeSpeaker ERes2Net-large embeddings only")

    similarity = _cosine(old.speaker_embedding, new.speaker_embedding)
    if similarity < min_update_similarity:
        return old

    keep = 1.0 - alpha
    speaker_embedding = _l2_normalize(_ema_list(old.speaker_embedding, new.speaker_embedding, keep, alpha))
    return SpeakerProfile(
        speaker_embedding=speaker_embedding,
        spectral_envelope=[],
        formant_profile=[],
        average_f0=_ema_optional(old.average_f0, new.average_f0, keep, alpha),
        median_f0=_ema_optional(old.median_f0, new.median_f0, keep, alpha),
        pitch_range=_ema_optional(old.pitch_range, new.pitch_range, keep, alpha),
        breathiness=old.breathiness * keep + new.breathiness * alpha,
        harmonic_noise_ratio=old.harmonic_noise_ratio * keep + new.harmonic_noise_ratio * alpha,
        speaking_rate_baseline=old.speaking_rate_baseline * keep + new.speaking_rate_baseline * alpha,
        pause_ratio_baseline=old.pause_ratio_baseline * keep + new.pause_ratio_baseline * alpha,
        energy_baseline=old.energy_baseline * keep + new.energy_baseline * alpha,
        style_habits=_merge_style_habits(old.style_habits, new.style_habits, keep, alpha),
        timbre_code=[],
        speaker_embedding_backend=MODEL_BACKEND,
        speaker_embedding_dim=len(speaker_embedding),
        embedding_l2_normalized=True,
        speaker_quality=max(old.speaker_quality, new.speaker_quality),
        speech_duration_sec=old.speech_duration_sec + new.speech_duration_sec,
        update_count=old.update_count + 1,
    )


def load_speaker_profile(path: str | Path) -> SpeakerProfile | None:
    profile_path = Path(path)
    if not profile_path.exists():
        return None
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    return speaker_profile_from_dict(data)


def save_speaker_profile(path: str | Path, profile: SpeakerProfile) -> None:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(speaker_profile_to_dict(profile), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def speaker_profile_to_dict(profile: SpeakerProfile) -> dict[str, object]:
    return {
        "speaker_embedding_backend": profile.speaker_embedding_backend,
        "speaker_embedding_dim": profile.speaker_embedding_dim,
        "speaker_embedding": [round(x, 8) for x in profile.speaker_embedding],
        "embedding_l2_normalized": profile.embedding_l2_normalized,
        "speaker_quality": round(profile.speaker_quality, 6),
        "speech_duration_sec": round(profile.speech_duration_sec, 6),
        "average_f0": _round_optional(profile.average_f0),
        "median_f0": _round_optional(profile.median_f0),
        "pitch_range": _round_optional(profile.pitch_range),
        "breathiness": round(profile.breathiness, 6),
        "harmonic_noise_ratio": round(profile.harmonic_noise_ratio, 6),
        "speaking_rate_baseline": round(profile.speaking_rate_baseline, 6),
        "pause_ratio_baseline": round(profile.pause_ratio_baseline, 6),
        "energy_baseline": round(profile.energy_baseline, 6),
        "style_habits": profile.style_habits,
        "update_count": profile.update_count,
    }


def speaker_profile_from_dict(data: dict[str, object]) -> SpeakerProfile:
    embedding = [float(x) for x in data.get("speaker_embedding", [])]
    backend = str(data.get("speaker_embedding_backend", MODEL_BACKEND))
    if backend != MODEL_BACKEND:
        raise RuntimeError(f"unsupported speaker embedding backend: {backend}")
    embedding = _l2_normalize(embedding)
    return SpeakerProfile(
        speaker_embedding=embedding,
        spectral_envelope=[],
        formant_profile=[],
        average_f0=_float_or_none(data.get("average_f0")),
        median_f0=_float_or_none(data.get("median_f0")),
        pitch_range=_float_or_none(data.get("pitch_range")),
        breathiness=float(data.get("breathiness", 0.0)),
        harmonic_noise_ratio=float(data.get("harmonic_noise_ratio", 0.0)),
        speaking_rate_baseline=float(data.get("speaking_rate_baseline", 0.0)),
        pause_ratio_baseline=float(data.get("pause_ratio_baseline", 0.0)),
        energy_baseline=float(data.get("energy_baseline", 0.0)),
        style_habits={str(k): v for k, v in dict(data.get("style_habits", {})).items()},
        timbre_code=[],
        speaker_embedding_backend=MODEL_BACKEND,
        speaker_embedding_dim=int(data.get("speaker_embedding_dim", len(embedding))),
        embedding_l2_normalized=True,
        speaker_quality=float(data.get("speaker_quality", 0.0)),
        speech_duration_sec=float(data.get("speech_duration_sec", 0.0)),
        update_count=int(data.get("update_count", 1)),
    )


def _ema_optional(old: float | None, new: float | None, keep: float, alpha: float) -> float | None:
    if old is None:
        return new
    if new is None:
        return old
    return old * keep + new * alpha


def _ema_list(old: list[float], new: list[float], keep: float, alpha: float) -> list[float]:
    if len(old) != len(new):
        raise RuntimeError(f"speaker embedding dimension changed: {len(old)} != {len(new)}")
    return [left * keep + right * alpha for left, right in zip(old, new)]


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-8:
        raise RuntimeError("speaker embedding has zero norm")
    return [value / norm for value in values]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    return sum(a * b for a, b in zip(left, right))


def _style_habits_from_prosody(prosody) -> dict[str, float | str]:
    energy = prosody.energy_mean
    attack = prosody.attack
    pause = prosody.pause_ratio
    rate = prosody.speech_rate
    if energy >= 0.12 and attack >= 0.18 and rate >= 7.5:
        emotion = "animated"
    elif pause >= 0.30 and rate <= 5.5:
        emotion = "measured"
    elif energy <= 0.05:
        emotion = "soft"
    else:
        emotion = "neutral"
    return {
        "dominant_emotion_hint": emotion,
        "average_attack": round(attack, 6),
        "average_pause_ratio": round(pause, 6),
        "average_rate": round(rate, 6),
    }


def _merge_style_habits(
    old: dict[str, float | str],
    new: dict[str, float | str],
    keep: float,
    alpha: float,
) -> dict[str, float | str]:
    merged: dict[str, float | str] = {}
    for key in sorted(set(old) | set(new)):
        old_value = old.get(key)
        new_value = new.get(key)
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            merged[key] = round(float(old_value) * keep + float(new_value) * alpha, 6)
        else:
            merged[key] = new_value if new_value is not None else old_value
    return merged


def _estimate_breathiness(samples: list[float]) -> float:
    if len(samples) < 2:
        return 0.0
    diff_energy = sum((samples[i] - samples[i - 1]) ** 2 for i in range(1, len(samples)))
    signal_energy = sum(sample * sample for sample in samples) + 1e-9
    return max(0.0, min(1.0, math.sqrt(diff_energy / signal_energy) / 2.0))


def _estimate_harmonic_noise_ratio(samples: list[float]) -> float:
    if len(samples) < 3:
        return 0.0
    signal = sum(sample * sample for sample in samples)
    noise = sum((samples[i] - 2 * samples[i - 1] + samples[i - 2]) ** 2 for i in range(2, len(samples)))
    return max(0.0, min(1.0, signal / (signal + noise + 1e-9)))



def _write_float_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.nan_to_num(samples.astype(np.float32))
    samples = np.clip(samples, -1.0, 1.0)
    try:
        import soundfile as sf
        sf.write(str(path), samples, int(sample_rate))
        return
    except Exception:
        pass
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes((samples * 32767.0).astype("<i2").tobytes())

def _round_optional(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _read_mono_pcm16(path: Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if width != 2:
        raise ValueError("Only PCM16 wav is supported for speaker profile side metrics.")

    samples: list[float] = []
    step = width * channels
    for index in range(0, len(frames), step):
        total = 0
        for channel in range(channels):
            offset = index + channel * width
            total += int.from_bytes(frames[offset : offset + width], "little", signed=True)
        samples.append(total / channels / 32768.0)
    return sample_rate, samples
