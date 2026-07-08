from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from voice_engine.adapters.spectral_delta_adapter import (
    SpeakerSpectralProfile,
    SpectralDeltaConfig,
    apply_best_voice_adapter,
)
from voice_engine.decoders.base import StreamingVoiceDecoder
from voice_engine.pipeline.types import StreamAudioChunk, VoiceDecoderInput


class SpectralDeltaDecoderAdapter(StreamingVoiceDecoder):
    """Post-TTS adapter for B_spectral_delta_080.

    This is block-streaming, not whole-utterance collection:
    - source audio is adapted every ``window_ms`` block;
    - reference audio is cached per sample-rate;
    - path references are decoded once with soundfile/wave fallback;
    - no per-turn full chunk collection is used.
    """

    def __init__(
        self,
        decoder: StreamingVoiceDecoder,
        cfg: SpectralDeltaConfig | None = None,
        chunk_ms: int = 500,
        window_ms: int = 900,
        adapter_sample_rate: int | None = 24000,
    ) -> None:
        self.decoder = decoder
        self.cfg = cfg or SpectralDeltaConfig()
        self.chunk_ms = int(chunk_ms)
        self.window_ms = int(window_ms)
        self.adapter_sample_rate = int(adapter_sample_rate) if adapter_sample_rate else None
        self._profiles_by_sr: dict[int, SpeakerSpectralProfile] = {}
        self._reference_path_cache: dict[tuple[str, int, int, int], np.ndarray] = {}
        self._updated_refs: set[tuple[str, int, int, int]] = set()
        self.speaker_profile = self._profile_for_sr(self.cfg.sr)

    def _profile_for_sr(self, sample_rate: int) -> SpeakerSpectralProfile:
        sample_rate = int(sample_rate)
        profile = self._profiles_by_sr.get(sample_rate)
        if profile is None:
            cfg = SpectralDeltaConfig(
                sr=sample_rate,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                n_mels=self.cfg.n_mels,
                fmin=self.cfg.fmin,
                strength=self.cfg.strength,
                max_gain_db=self.cfg.max_gain_db,
                eps=self.cfg.eps,
                peak=self.cfg.peak,
            )
            profile = SpeakerSpectralProfile(cfg)
            self._profiles_by_sr[sample_rate] = profile
        return profile

    def update_profile_from_wav(self, path: str | Path, sample_rate: int | None = None) -> None:
        target_sr = int(sample_rate or self.cfg.sr)
        path = Path(path)
        stat = path.stat()
        cache_key = (str(path.resolve()), target_sr, int(stat.st_mtime_ns), int(stat.st_size))
        if cache_key in self._updated_refs:
            return
        wav = self._load_reference_cached(path, target_sr, cache_key)
        profile = self._profile_for_sr(target_sr)
        profile.update(wav)
        self._updated_refs.add(cache_key)
        if target_sr == self.cfg.sr:
            self.speaker_profile = profile

    def update_profile_from_samples(self, samples: np.ndarray, sample_rate: int) -> None:
        sample_rate = int(sample_rate)
        samples = np.nan_to_num(samples.astype(np.float32))
        profile = self._profile_for_sr(sample_rate)
        profile.update(samples)
        if sample_rate == self.cfg.sr:
            self.speaker_profile = profile

    def decode(self, decoder_input: VoiceDecoderInput) -> Iterator[StreamAudioChunk]:
        source_rate: int | None = None
        buffered_parts: list[np.ndarray] = []
        buffered_samples = 0
        start_ms = 0.0

        for chunk in self.decoder.decode(decoder_input):
            source_rate = int(chunk.sample_rate)
            process_rate = self.adapter_sample_rate or source_rate
            self._ensure_reference_profile(decoder_input, process_rate)
            part = np.nan_to_num(np.asarray(chunk.samples, dtype=np.float32))
            if process_rate != source_rate:
                part = _resample(part, source_rate, process_rate)
            if part.size == 0:
                continue
            buffered_parts.append(part)
            buffered_samples += int(part.size)

            window_samples = max(1, int(process_rate * self.window_ms / 1000.0))
            while buffered_samples >= window_samples:
                block, buffered_parts, buffered_samples = _pop_samples(buffered_parts, buffered_samples, window_samples)
                for out_chunk in self._adapt_and_chunk(block, process_rate, start_ms):
                    start_ms = out_chunk.end_ms
                    yield out_chunk

        if source_rate is not None and buffered_samples > 0:
            tail = np.concatenate(buffered_parts).astype(np.float32)
            process_rate = self.adapter_sample_rate or source_rate
            for out_chunk in self._adapt_and_chunk(tail, process_rate, start_ms):
                start_ms = out_chunk.end_ms
                yield out_chunk

    def _ensure_reference_profile(self, decoder_input: VoiceDecoderInput, sample_rate: int) -> None:
        ref_samples = getattr(decoder_input, "reference_audio_samples", None)
        ref_sr = getattr(decoder_input, "reference_audio_sample_rate", None)
        if ref_samples is not None and ref_sr:
            samples = np.asarray(ref_samples, dtype=np.float32)
            if int(ref_sr) != int(sample_rate):
                samples = _resample(samples, int(ref_sr), int(sample_rate))
            self.update_profile_from_samples(samples, int(sample_rate))
            return

        if decoder_input.reference_audio_path:
            self.update_profile_from_wav(decoder_input.reference_audio_path, sample_rate=sample_rate)

    def _adapt_and_chunk(
        self,
        samples: np.ndarray,
        sample_rate: int,
        start_ms_offset: float,
    ) -> Iterator[StreamAudioChunk]:
        profile = self._profile_for_sr(sample_rate)
        if profile.ready():
            converted = apply_best_voice_adapter(samples, profile)
            converted = _match_length(converted, len(samples))
        else:
            converted = samples.astype(np.float32)

        chunk_size = max(1, int(sample_rate * self.chunk_ms / 1000.0))
        for start in range(0, len(converted), chunk_size):
            part = converted[start : start + chunk_size].astype(np.float32)
            yield StreamAudioChunk(
                samples=part.tolist(),
                sample_rate=sample_rate,
                start_ms=start_ms_offset + start / sample_rate * 1000.0,
                end_ms=start_ms_offset + (start + len(part)) / sample_rate * 1000.0,
            )

    def _load_reference_cached(self, path: Path, sample_rate: int, cache_key) -> np.ndarray:
        cached = self._reference_path_cache.get(cache_key)
        if cached is not None:
            return cached
        audio, sr = _read_audio_mono(path)
        if sr != sample_rate:
            audio = _resample(audio, sr, sample_rate)
        audio = np.nan_to_num(audio.astype(np.float32))
        self._reference_path_cache[cache_key] = audio
        return audio


def read_wav_mono(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    audio, sr = _read_audio_mono(Path(path))
    if sr != sample_rate:
        audio = _resample(audio, sr, sample_rate)
    return audio.astype(np.float32)


def write_wav_mono(path: str | Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    _write_audio_mono(Path(path), samples, sample_rate)


def _pop_samples(parts: list[np.ndarray], total: int, n: int):
    out: list[np.ndarray] = []
    need = int(n)
    while need > 0 and parts:
        first = parts.pop(0)
        if len(first) <= need:
            out.append(first)
            need -= len(first)
        else:
            out.append(first[:need])
            parts.insert(0, first[need:])
            need = 0
    block = np.concatenate(out).astype(np.float32) if out else np.zeros(0, dtype=np.float32)
    return block, parts, total - len(block)


def _match_length(samples: np.ndarray, length: int) -> np.ndarray:
    if len(samples) == length:
        return samples.astype(np.float32)
    if len(samples) > length:
        return samples[:length].astype(np.float32)
    return np.pad(samples, (0, length - len(samples))).astype(np.float32)


def _read_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        audio = np.asarray(data, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.nan_to_num(audio).astype(np.float32), int(sr)
    except Exception:
        with wave.open(str(path), "rb") as wav:
            sr = wav.getframerate()
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
        if sampwidth != 2:
            raise RuntimeError("only PCM16 fallback wav is supported")
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio.astype(np.float32), int(sr)


def _write_audio_mono(path: Path, samples: np.ndarray, sample_rate: int) -> None:
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


def _resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if int(src_sr) == int(dst_sr):
        return samples.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(int(src_sr), int(dst_sr))
        return resample_poly(samples, int(dst_sr) // g, int(src_sr) // g).astype(np.float32)
    except Exception:
        import librosa
        return librosa.resample(samples.astype(np.float32), orig_sr=int(src_sr), target_sr=int(dst_sr)).astype(np.float32)
