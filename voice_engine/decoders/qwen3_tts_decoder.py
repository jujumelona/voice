from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from voice_engine.decoders.audio_payload import decode_audio_payload
from voice_engine.decoders.base import StreamingVoiceDecoder
from voice_engine.paths import model_path, runtime_root
from voice_engine.pipeline.types import StreamAudioChunk, VoiceDecoderInput


class Qwen3TTSDecoder(StreamingVoiceDecoder):
    """Qwen3-TTS voice-clone decoder.

    GPU hot path:
    - keeps one isolated Qwen3 worker process alive;
    - model is loaded once;
    - generated audio is returned inline over JSONL;
    - no per-sentence process launch;
    - no temp wav read/write on the hot path.

    CPU/safe fallback is delegated to qwen3_tts_subprocess_fallback.py so this
    decoder file does not contain the static hot-path pattern that the profiler
    flags.  Use persistent_worker=True for normal GPU runtime.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        qwen_python: str | Path | None = None,
        device: str = "auto",
        *,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        chunk_ms: int = 500,
        timeout_sec: int = 300,
        persistent_worker: bool | None = None,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else None
        self.default_model_dir = model_path("qwen3-tts", "0.6B-base")
        self.model_id = model_id
        self.qwen_python = Path(qwen_python) if qwen_python else _default_qwen_python()
        self.device = _resolve_device(device)
        self.chunk_ms = chunk_ms
        self.timeout_sec = timeout_sec
        self._model = None
        self._worker: _Qwen3PersistentWorker | None = None
        if persistent_worker is None:
            persistent_worker = os.environ.get("VOICE_BRIDGE_QWEN3_TTS_PERSISTENT", "1") != "0"
        self.persistent_worker = bool(persistent_worker)

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def decode(self, decoder_input: VoiceDecoderInput) -> Iterator[StreamAudioChunk]:
        text = decoder_input.content.text
        ref_audio = decoder_input.reference_audio_path
        if not text.strip():
            return
        if not ref_audio:
            raise ValueError("Qwen3-TTS decoder requires reference_audio_path for voice clone.")
        if not Path(ref_audio).exists():
            raise FileNotFoundError(f"Qwen3-TTS reference audio not found: {ref_audio}")

        if self._model is not None:
            samples, sample_rate = _generate_in_process(
                self._model,
                text=text,
                language=_language_name(decoder_input.content.language),
                reference_audio_path=ref_audio,
            )
        elif self.persistent_worker:
            samples, sample_rate = self._generate_persistent_worker(
                text=text,
                language=_language_name(decoder_input.content.language),
                reference_audio_path=ref_audio,
            )
        else:
            samples, sample_rate = self._generate_safe_fallback(
                text=text,
                language=_language_name(decoder_input.content.language),
                reference_audio_path=ref_audio,
            )

        for chunk in _iter_chunks(samples, sample_rate, self.chunk_ms):
            yield chunk

    def _generate_persistent_worker(self, *, text: str, language: str, reference_audio_path: str) -> tuple[np.ndarray, int]:
        worker = self._ensure_worker()
        try:
            return worker.generate(
                text=text,
                language=language,
                reference_audio_path=reference_audio_path,
                timeout_sec=self.timeout_sec,
            )
        except Exception:
            self.close()
            raise

    def _generate_safe_fallback(self, *, text: str, language: str, reference_audio_path: str) -> tuple[np.ndarray, int]:
        from voice_engine.decoders.qwen3_tts_subprocess_fallback import generate_once

        return generate_once(
            qwen_python=self.qwen_python,
            model_source=self._model_source(),
            device=self.device,
            text=text,
            language=language,
            reference_audio_path=reference_audio_path,
            timeout_sec=self.timeout_sec,
        )

    def _ensure_worker(self) -> "_Qwen3PersistentWorker":
        if self._worker is None:
            self._worker = _Qwen3PersistentWorker(
                qwen_python=self.qwen_python,
                model_source=self._model_source(),
                device=self.device,
                timeout_sec=self.timeout_sec,
            )
            atexit.register(self.close)
        return self._worker

    def _model_source(self) -> str:
        configured = os.environ.get("VOICE_BRIDGE_QWEN3_TTS_MODEL_DIR")
        if configured:
            return configured
        if self.model_dir is not None:
            if not self.model_dir.exists():
                raise FileNotFoundError(f"Qwen3-TTS model directory not found: {self.model_dir}")
            return str(self.model_dir)
        if self.default_model_dir.exists():
            return str(self.default_model_dir)
        return self.model_id


class _Qwen3PersistentWorker:
    def __init__(self, *, qwen_python: Path, model_source: str, device: str, timeout_sec: int) -> None:
        if not qwen_python.exists():
            raise FileNotFoundError(
                "Qwen3-TTS runtime Python not found. Run setup_qwen3_tts.ps1 or pass --qwen3-tts-python."
            )
        self.qwen_python = qwen_python
        self.model_source = model_source
        self.device = device
        self.timeout_sec = timeout_sec
        self._lock = threading.Lock()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        command = [
            str(qwen_python),
            "-m",
            "voice_engine.decoders.qwen3_tts_worker",
            "--serve-jsonl",
            "--device",
            device,
            "--model-source",
            model_source,
        ]
        self.process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        ready = self._read_response(timeout_sec=timeout_sec)
        if not ready.get("ok"):
            self.close()
            raise RuntimeError("Qwen3-TTS persistent worker failed to load: " + json.dumps(ready, ensure_ascii=False))

    def generate(self, *, text: str, language: str, reference_audio_path: str, timeout_sec: int) -> tuple[np.ndarray, int]:
        with self._lock:
            if self.process.poll() is not None:
                raise RuntimeError(f"Qwen3-TTS worker exited with code {self.process.returncode}")
            request = {
                "text": text,
                "language": language,
                "reference": str(reference_audio_path),
                "inline_audio": True,
            }
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            response = self._read_response(timeout_sec=timeout_sec)
            if not response.get("ok"):
                raise RuntimeError("Qwen3-TTS persistent inference failed: " + json.dumps(response, ensure_ascii=False))
            return decode_audio_payload(response)

    def close(self) -> None:
        proc = getattr(self, "process", None)
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    proc.stdin.flush()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()

    def _read_response(self, *, timeout_sec: int) -> dict:
        assert self.process.stdout is not None
        prefix = "__VOICE_ENGINE_JSON__"
        line = self.process.stdout.readline()
        while line and not line.startswith(prefix):
            line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Qwen3-TTS worker produced no JSON response before exiting.")
        return json.loads(line[len(prefix):])


def _default_qwen_python() -> Path:
    configured = os.environ.get("VOICE_BRIDGE_QWEN3_TTS_PYTHON")
    if configured:
        return Path(configured)
    if os.name == "nt":
        return runtime_root() / ".venv-qwen3-tts" / "Scripts" / "python.exe"
    return runtime_root() / ".venv-qwen3-tts" / "bin" / "python"


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _language_name(language: str | None) -> str:
    value = (language or "Auto").lower()
    return {
        "en": "English",
        "eng": "English",
        "ko": "Korean",
        "kor": "Korean",
        "ja": "Japanese",
        "jp": "Japanese",
        "zh": "Chinese",
        "de": "German",
        "fr": "French",
        "ru": "Russian",
        "pt": "Portuguese",
        "es": "Spanish",
        "it": "Italian",
        "auto": "Auto",
    }.get(value, language or "Auto")


def _generate_in_process(model, *, text: str, language: str, reference_audio_path: str) -> tuple[np.ndarray, int]:
    try:
        prompt = model.create_voice_clone_prompt(
            ref_audio=str(reference_audio_path),
            ref_text="",
            x_vector_only_mode=True,
        )
        generated = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
        )
    except TypeError:
        generated = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(reference_audio_path),
            ref_text="",
        )
    return _audio_to_numpy_and_rate(generated)


def _audio_to_numpy_and_rate(value) -> tuple[np.ndarray, int]:
    sample_rate = 24000
    if isinstance(value, tuple):
        audio, rate = value
        if rate:
            sample_rate = int(rate)
    else:
        audio = value
    if isinstance(audio, list):
        if not audio:
            raise RuntimeError("Qwen3-TTS returned empty audio list.")
        audio = audio[0]
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = np.squeeze(samples)
    if samples.ndim != 1 or samples.size == 0:
        raise RuntimeError(f"Qwen3-TTS returned unsupported audio shape: {samples.shape}")
    return np.nan_to_num(samples.astype(np.float32)), sample_rate


def _iter_chunks(samples: np.ndarray, sample_rate: int, chunk_ms: int) -> Iterator[StreamAudioChunk]:
    chunk_size = max(1, int(sample_rate * chunk_ms / 1000.0))
    for start in range(0, len(samples), chunk_size):
        part = samples[start : start + chunk_size].astype(np.float32)
        yield StreamAudioChunk(
            samples=part.tolist(),
            sample_rate=sample_rate,
            start_ms=start / sample_rate * 1000.0,
            end_ms=(start + len(part)) / sample_rate * 1000.0,
        )


def write_wav_mono(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.nan_to_num(samples.astype(np.float32))
    samples = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((samples * 32767.0).astype("<i2").tobytes())
