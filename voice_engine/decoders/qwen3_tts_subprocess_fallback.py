from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from voice_engine.decoders.qwen3_tts_decoder import write_wav_mono


def generate_once(
    *,
    qwen_python: Path,
    model_source: str,
    device: str,
    text: str,
    language: str,
    reference_audio_path: str,
    timeout_sec: int,
) -> tuple[np.ndarray, int]:
    """CPU/safe single-shot fallback only. Not used by GPU persistent hot path."""
    if not qwen_python.exists():
        raise FileNotFoundError(
            "Qwen3-TTS runtime Python not found. Run setup_qwen3_tts.ps1 or pass --qwen3-tts-python."
        )

    import wave

    with tempfile.TemporaryDirectory(prefix="ve_qwen3_fallback_") as tmp:
        out_wav = Path(tmp) / "qwen3_tts.wav"
        command = [
            str(qwen_python),
            "-m",
            "voice_engine.decoders.qwen3_tts_worker",
            "--text",
            text,
            "--language",
            language,
            "--reference",
            str(reference_audio_path),
            "--out",
            str(out_wav),
            "--device",
            device,
            "--model-source",
            model_source,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        result = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            tail = "\n".join(((result.stdout or "") + "\n" + (result.stderr or "")).splitlines()[-20:])
            raise RuntimeError(f"Qwen3-TTS fallback inference failed with code {result.returncode}:\n{tail}")

        with wave.open(str(out_wav), "rb") as wav:
            sample_rate = int(wav.getframerate())
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
        if sample_width != 2:
            raise RuntimeError("Qwen3-TTS fallback produced non-PCM16 wav")
        values = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            values = values.reshape(-1, channels).mean(axis=1)
        return np.nan_to_num(values.astype(np.float32)), sample_rate
