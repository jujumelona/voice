from __future__ import annotations

import hashlib
import inspect
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from voice_engine.paths import model_path
from voice_engine.pipeline.types import StreamAudioChunk, VoiceDecoderInput
from voice_engine.decoders.base import StreamingVoiceDecoder


class VoxCPM2Decoder(StreamingVoiceDecoder):
    """VoxCPM2 voice decoder optimized for call latency.

    Speed changes, without lowering inference_timesteps:
    - model is loaded once and reused;
    - CUDA is forced when device="auto" on a GPU runtime;
    - true streaming API is used when the installed VoxCPM runtime exposes it;
    - when true streaming is unavailable, text is split into short phrases so the
      first generated phrase can be yielded before the full translated sentence;
    - reference audio is clipped to a stable 3s cache file once, not rewritten per turn;
    - cfg defaults lower for call mode, while inference_timesteps stays unchanged.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "auto",
        *,
        chunk_ms: int = 500,
        cfg_value: float = 1.0,
        inference_timesteps: int = 10,
        load_denoiser: bool = False,
        max_reference_sec: float = 3.0,
        split_text: bool = True,
        max_segment_chars: int = 24,
        fast_backend: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else None
        self.default_model_dir = model_path("voxcpm2")
        self.model_id = "openbmb/VoxCPM2"
        self.device = _resolve_device(device)
        self.chunk_ms = int(chunk_ms)
        self.cfg_value = float(cfg_value)
        self.inference_timesteps = int(inference_timesteps)
        self.load_denoiser = bool(load_denoiser)
        self.max_reference_sec = float(max_reference_sec)
        self.split_text = bool(split_text)
        self.max_segment_chars = int(max_segment_chars)
        self.fast_backend = fast_backend
        self._model = None
        self._ref_cache: dict[tuple[str, int, int, float], str] = {}

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        try:
            from voxcpm import VoxCPM
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "VoxCPM2 runtime is not installed. Install the runtime Python "
                "packages, then run: python -m pip install voxcpm"
            ) from exc

        source = self._model_source()
        kwargs = {
            "device": self.device,
            "load_denoiser": self.load_denoiser,
        }
        if self.fast_backend:
            kwargs["backend"] = self.fast_backend

        try:
            self._model = _call_with_supported_kwargs(VoxCPM.from_pretrained, source, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load VoxCPM2 from {source!r}. "
                "Keep model files outside the repo or allow Hugging Face cache download."
            ) from exc

    def _model_source(self) -> str:
        if self.model_dir is not None:
            if not self.model_dir.exists():
                raise FileNotFoundError(f"VoxCPM2 model directory not found: {self.model_dir}")
            return str(self.model_dir)
        if self.default_model_dir.exists():
            return str(self.default_model_dir)
        return self.model_id

    def decode(self, decoder_input: VoiceDecoderInput) -> Iterator[StreamAudioChunk]:
        self._ensure_model()

        text = decoder_input.content.text.strip()
        ref_audio = decoder_input.reference_audio_path

        if not text:
            return
        if not ref_audio:
            raise ValueError("VoxCPM2 decoder requires reference_audio_path for call voice cloning.")
        ref_path = Path(ref_audio)
        if not ref_path.exists():
            raise FileNotFoundError(f"VoxCPM2 reference audio not found: {ref_path}")

        cached_ref = self._cached_reference_path(ref_path)
        segments = _split_text_for_tts(text, self.max_segment_chars) if self.split_text else [text]
        output_start_ms = 0.0

        for segment in segments:
            yielded = False
            for chunk in self._decode_segment_stream(segment, cached_ref, start_ms_offset=output_start_ms):
                yielded = True
                output_start_ms = chunk.end_ms
                yield chunk

            # If the runtime reported no streaming chunks, use full segment generation.
            if not yielded:
                audio_samples = self._generate_segment(segment, cached_ref)
                sample_rate = _model_sample_rate(self._model)
                for chunk in _iter_audio_chunks(audio_samples, sample_rate, self.chunk_ms, output_start_ms):
                    output_start_ms = chunk.end_ms
                    yield chunk

    def _decode_segment_stream(
        self,
        text: str,
        ref_audio: str,
        *,
        start_ms_offset: float,
    ) -> Iterator[StreamAudioChunk]:
        stream_fn = _find_streaming_generate(self._model)
        if stream_fn is None:
            return

        sample_rate = _model_sample_rate(self._model)
        start_ms = float(start_ms_offset)
        kwargs = self._generate_kwargs(text, ref_audio)

        for item in _call_with_supported_kwargs(stream_fn, **kwargs):
            audio_samples = _audio_to_numpy(item)
            if audio_samples.size == 0:
                continue
            end_ms = start_ms + audio_samples.size / sample_rate * 1000.0
            yield StreamAudioChunk(
                samples=audio_samples.astype(np.float32).tolist(),
                sample_rate=sample_rate,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            start_ms = end_ms

    def _generate_segment(self, text: str, ref_audio: str) -> np.ndarray:
        generate_fn = getattr(self._model, "generate")
        kwargs = self._generate_kwargs(text, ref_audio)
        with _torch_inference_context():
            value = _call_with_supported_kwargs(generate_fn, **kwargs)
        return _audio_to_numpy(value)

    def _generate_kwargs(self, text: str, ref_audio: str) -> dict[str, object]:
        return {
            "text": text,
            "reference_wav_path": str(ref_audio),
            "cfg_value": self.cfg_value,
            "inference_timesteps": self.inference_timesteps,
        }

    def _cached_reference_path(self, path: Path) -> str:
        stat = path.stat()
        cache_key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size), self.max_reference_sec)
        cached = self._ref_cache.get(cache_key)
        if cached and Path(cached).exists():
            return cached

        cache_dir = Path(".voice_bridge_runtime") / "voxcpm2_refs"
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(repr(cache_key).encode("utf-8")).hexdigest()[:16]
        out_path = cache_dir / f"ref_{digest}.wav"

        if not out_path.exists():
            try:
                samples, sr = _read_wav_float(path)
                max_samples = max(1, int(sr * self.max_reference_sec))
                if samples.size > max_samples:
                    # Last clean seconds tend to reflect the current call state better.
                    samples = samples[-max_samples:]
                _write_wav_float(out_path, samples, sr)
            except Exception:
                # If decoding/ref clipping fails, use the original reference path.
                self._ref_cache[cache_key] = str(path)
                return str(path)

        self._ref_cache[cache_key] = str(out_path)
        return str(out_path)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _find_streaming_generate(model):
    for name in ("generate_stream", "stream_generate", "streaming_generate"):
        fn = getattr(model, name, None)
        if callable(fn):
            return fn
    return None


def _call_with_supported_kwargs(fn, *args, **kwargs):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        try:
            return fn(*args, **kwargs)
        except TypeError:
            # Conservative fallback for runtimes with older signatures.
            minimal = {k: v for k, v in kwargs.items() if k in {"text", "reference_wav_path"}}
            return fn(*args, **minimal)

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **filtered)


class _torch_inference_context:
    def __enter__(self):
        try:
            import torch
            self._ctx = torch.inference_mode()
            return self._ctx.__enter__()
        except Exception:
            self._ctx = None
            return None

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc, tb)
        return False


def _audio_to_numpy(value) -> np.ndarray:
    if isinstance(value, dict):
        for key in ("audio", "wav", "samples", "speech"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, tuple):
        value = value[0]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    audio = np.asarray(value, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.squeeze(audio)
    if audio.ndim != 1:
        raise RuntimeError(f"VoxCPM2 returned unsupported audio shape: {audio.shape}")
    if audio.size == 0:
        return audio.astype(np.float32)
    return np.nan_to_num(audio).astype(np.float32)


def _model_sample_rate(model) -> int:
    tts_model = getattr(model, "tts_model", None)
    for owner in (tts_model, model):
        if owner is None:
            continue
        sample_rate = getattr(owner, "sample_rate", None)
        if sample_rate:
            return int(sample_rate)
    return 48000


def _iter_audio_chunks(
    audio_samples: np.ndarray,
    sample_rate: int,
    chunk_ms: int,
    start_ms_offset: float = 0.0,
) -> Iterator[StreamAudioChunk]:
    chunk_size = max(1, int(sample_rate * chunk_ms / 1000.0))
    for i in range(0, len(audio_samples), chunk_size):
        part = audio_samples[i : i + chunk_size].astype(np.float32)
        start_ms = start_ms_offset + i / sample_rate * 1000.0
        end_ms = start_ms_offset + (i + len(part)) / sample_rate * 1000.0
        yield StreamAudioChunk(
            samples=part.tolist(),
            sample_rate=sample_rate,
            start_ms=start_ms,
            end_ms=end_ms,
        )


def _split_text_for_tts(text: str, max_chars: int) -> list[str]:
    """Split translated text for lower first-response latency.

    This does not change inference_timesteps. It only shortens each VoxCPM2
    generate call so a non-streaming runtime can emit the first phrase earlier.
    Korean particles/endings and punctuation are treated as safe-ish cut points.
    """
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]

    hard = set(".?!。？！;；\n")
    soft = set(",，:：")
    korean_endings = ("요", "다", "죠", "까", "네", "음", "함", "해", "야")

    pieces: list[str] = []
    current: list[str] = []
    last_soft_cut = -1

    def flush_at(idx: int | None = None) -> None:
        nonlocal current, last_soft_cut
        if idx is None or idx >= len(current):
            piece = "".join(current).strip()
            current = []
        else:
            piece = "".join(current[:idx]).strip()
            current = current[idx:]
        if piece:
            pieces.append(piece)
        last_soft_cut = -1

    for char in text:
        current.append(char)
        if char in soft or char.isspace():
            last_soft_cut = len(current)
        if char in hard:
            flush_at()
            continue
        if len(current) >= max_chars:
            tail = "".join(current).strip()
            if last_soft_cut >= max(4, max_chars // 2):
                flush_at(last_soft_cut)
            elif tail.endswith(korean_endings) and len(current) >= max(8, max_chars // 2):
                flush_at()
            else:
                flush_at()

    if current:
        flush_at()

    # Avoid extremely tiny final fragments by merging only sub-6-char tails.
    merged: list[str] = []
    for piece in pieces:
        if merged and len(piece) < 6 and len(merged[-1]) + 1 + len(piece) <= max_chars:
            merged[-1] = merged[-1] + " " + piece
        else:
            merged.append(piece)
    return merged or [text]


def _read_wav_float(path: Path) -> tuple[np.ndarray, int]:
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


def _write_wav_float(path: Path, samples: np.ndarray, sample_rate: int) -> None:
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
