from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import librosa
import numpy as np


VOICE_ADAPTER_DEFAULT = "B_spectral_delta_080"

VOICE_ADAPTER_CONFIG = {
    "sr": 16000,
    "n_fft": 1024,
    "hop_length": 256,
    "n_mels": 80,
    "fmin": 50.0,
    "strength": 0.80,
    "max_gain_db": 12.0,
}


@dataclass
class SpectralDeltaConfig:
    sr: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    fmin: float = 50.0
    strength: float = 0.80
    max_gain_db: float = 12.0
    eps: float = 1e-8
    peak: float = 0.98


@dataclass
class SpeakerSpectralProfile:
    cfg: SpectralDeltaConfig
    count: int = 0
    mean_logmel: Optional[np.ndarray] = None
    ref_wavs: list[np.ndarray] = field(default_factory=list)
    max_refs: int = 8

    def update(self, wav: np.ndarray) -> None:
        wav = np.nan_to_num(wav.astype(np.float32))
        if len(wav) < self.cfg.hop_length * 2:
            return

        current = logmel_mean(wav, self.cfg)
        if self.mean_logmel is None:
            self.mean_logmel = current
            self.count = 1
        else:
            self.count += 1
            alpha = 1.0 / min(self.count, 32)
            self.mean_logmel = (1.0 - alpha) * self.mean_logmel + alpha * current

        self.ref_wavs.append(wav)
        if len(self.ref_wavs) > self.max_refs:
            self.ref_wavs.pop(0)

    def ready(self) -> bool:
        return self.mean_logmel is not None and self.count > 0

    def get_reference_wav(self) -> Optional[np.ndarray]:
        if not self.ref_wavs:
            return None
        return self.ref_wavs[-1]


def normalize_peak(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    y = np.nan_to_num(y.astype(np.float32))
    max_abs = float(np.max(np.abs(y)) + 1e-9)
    if max_abs > peak:
        y = y / max_abs * peak
    return y.astype(np.float32)


def logmel_mean(y: np.ndarray, cfg: SpectralDeltaConfig) -> np.ndarray:
    y = np.nan_to_num(y.astype(np.float32))
    stft = librosa.stft(
        y,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window="hann",
    )
    power = np.abs(stft) ** 2
    mel = librosa.feature.melspectrogram(
        S=power,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.sr // 2,
    )
    return np.mean(np.log(np.maximum(mel, cfg.eps)), axis=1).astype(np.float32)


def spectral_delta_adapter(
    source: np.ndarray,
    target: np.ndarray,
    cfg: Optional[SpectralDeltaConfig] = None,
) -> np.ndarray:
    """B_spectral_delta_080.

    Moves only the average spectral color of source toward target. It preserves
    source phase, source length, source timing, and source pitch contour.
    """
    cfg = cfg or SpectralDeltaConfig()
    source = np.nan_to_num(source.astype(np.float32))
    target = np.nan_to_num(target.astype(np.float32))

    if len(source) < cfg.hop_length * 2:
        return normalize_peak(source, cfg.peak)
    if len(target) < cfg.hop_length * 2:
        return normalize_peak(source, cfg.peak)

    target_mean = logmel_mean(target, cfg)
    return spectral_delta_from_profile(source, target_mean, cfg)


def spectral_delta_from_profile(
    source: np.ndarray,
    target_logmel_mean: np.ndarray,
    cfg: SpectralDeltaConfig,
) -> np.ndarray:
    source = np.nan_to_num(source.astype(np.float32))
    source_len = len(source)
    if source_len < cfg.hop_length * 2:
        return normalize_peak(source, cfg.peak)

    x = librosa.stft(
        source,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window="hann",
    )
    src_mag = np.abs(x)
    src_phase = np.angle(x)

    src_mean = logmel_mean(source, cfg)
    delta = target_logmel_mean.astype(np.float32) - src_mean.astype(np.float32)

    max_log_gain = cfg.max_gain_db / 20.0 * np.log(10.0)
    delta = np.clip(cfg.strength * delta, -max_log_gain, max_log_gain)
    gain_mel = np.exp(delta).astype(np.float32)

    mel_freqs = librosa.mel_frequencies(
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.sr // 2,
    )
    fft_freqs = np.linspace(0, cfg.sr / 2, src_mag.shape[0], dtype=np.float32)
    gain_fft = np.interp(
        fft_freqs,
        mel_freqs,
        gain_mel,
        left=float(gain_mel[0]),
        right=float(gain_mel[-1]),
    ).astype(np.float32)

    new_mag = src_mag * gain_fft[:, None]
    y = librosa.istft(
        new_mag * np.exp(1j * src_phase),
        hop_length=cfg.hop_length,
        window="hann",
        length=source_len,
    )
    return normalize_peak(_match_length(y, source_len), cfg.peak)


def profile_strength(profile_count: int) -> float:
    if profile_count <= 0:
        return 0.0
    if profile_count == 1:
        return 0.35
    if profile_count == 2:
        return 0.55
    if profile_count == 3:
        return 0.70
    return 0.80


def apply_best_voice_adapter(
    source: np.ndarray,
    speaker_profile: SpeakerSpectralProfile | None,
) -> np.ndarray:
    if speaker_profile is None or not speaker_profile.ready() or speaker_profile.mean_logmel is None:
        return source.astype(np.float32)

    cfg = SpectralDeltaConfig(
        sr=speaker_profile.cfg.sr,
        n_fft=speaker_profile.cfg.n_fft,
        hop_length=speaker_profile.cfg.hop_length,
        n_mels=speaker_profile.cfg.n_mels,
        fmin=speaker_profile.cfg.fmin,
        strength=profile_strength(speaker_profile.count),
        max_gain_db=speaker_profile.cfg.max_gain_db,
        eps=speaker_profile.cfg.eps,
        peak=speaker_profile.cfg.peak,
    )
    return spectral_delta_from_profile(source, speaker_profile.mean_logmel, cfg)


def _match_length(y: np.ndarray, length: int) -> np.ndarray:
    if len(y) == length:
        return y.astype(np.float32)
    if len(y) > length:
        return y[:length].astype(np.float32)
    return np.pad(y, (0, length - len(y))).astype(np.float32)
