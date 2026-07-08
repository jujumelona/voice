from __future__ import annotations

import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import atexit
import json
import threading
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from voice_engine.paths import model_path

MODEL_BACKEND = "wespeaker/eres2net-large"
MIN_SPEECH_SEC = 1.2


@dataclass(frozen=True)
class SpeakerEmbeddingResult:
    embedding: list[float]
    backend: str
    dim: int
    l2_normalized: bool
    quality: float
    speech_duration_sec: float


class WeSpeakerERes2NetLargeSpeakerEncoder:
    """WeSpeaker ERes2Net speaker embedding adapter."""

    def __init__(
        self,
        *,
        device: str | None = None,
        cache_dir: str | Path | None = None,
        pretrain_dir: str | Path | None = None,
    ) -> None:
        self.device = device or os.environ.get("VOICE_BRIDGE_WESPEAKER_DEVICE", "auto")
        self.cache_dir = Path(cache_dir or os.environ.get("VOICE_BRIDGE_WESPEAKER_HOME", str(model_path("wespeaker"))))
        self.pretrain_dir = Path(
            pretrain_dir
            or os.environ.get(
                "VOICE_BRIDGE_WESPEAKER_ERES2NET_DIR",
                str(model_path("speaker", "wespeaker", "eres2net-large")),
            )
        )

    def encode_file(self, wav_path: str | Path) -> SpeakerEmbeddingResult:
        wav = Path(wav_path)
        if not wav.exists():
            raise FileNotFoundError(f"speaker wav not found: {wav}")
        embedding = _extract_embedding_cached(
            str(wav.resolve()),
            _resolve_device(self.device),
            str(self.cache_dir.resolve()),
            str(self.pretrain_dir.resolve()) if _is_pretrain_dir(self.pretrain_dir) else "",
        )
        speech_duration = _wav_duration_sec(wav)
        if speech_duration < MIN_SPEECH_SEC:
            raise RuntimeError(
                f"speaker segment is too short for WeSpeaker ERes2Net-large: "
                f"{speech_duration:.3f}s < {MIN_SPEECH_SEC:.1f}s"
            )
        embedding = _l2_normalize(embedding)
        return SpeakerEmbeddingResult(
            embedding=embedding,
            backend=MODEL_BACKEND,
            dim=len(embedding),
            l2_normalized=True,
            quality=_quality_score(speech_duration, len(embedding)),
            speech_duration_sec=round(speech_duration, 6),
        )


@lru_cache(maxsize=16)
def _extract_embedding_cached(wav_path: str, device: str, cache_dir: str, pretrain_dir: str) -> tuple[float, ...]:
    worker_python = _wespeaker_python()
    if worker_python:
        return _worker_embedding(worker_python, wav_path, device, cache_dir, pretrain_dir)

    command = _wespeaker_command()
    with tempfile.TemporaryDirectory(prefix="voice_engine_wespeaker_") as tmp:
        output_file = Path(tmp) / "embedding.txt"
        env = os.environ.copy()
        env["WESPEAKER_HOME"] = cache_dir
        src_path = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        args = [
            *command,
            "--task",
            "embedding",
            "--audio_file",
            wav_path,
            "--output_file",
            str(output_file),
            "--device",
            device,
            "--eres2net",
        ]
        if pretrain_dir:
            args.extend(["--pretrain", pretrain_dir])
        completed = subprocess.run(args, check=False, capture_output=True, text=True, env=env)
        if completed.returncode != 0:
            raise RuntimeError(
                "WeSpeaker ERes2Net-large encoder failed\n"
                f"command: {' '.join(args)}\n"
                f"stdout: {completed.stdout.strip()}\n"
                f"stderr: {completed.stderr.strip()}"
            )
        if not output_file.exists():
            raise FileNotFoundError(f"WeSpeaker did not write embedding file: {output_file}")
        values = _parse_embedding(output_file.read_text(encoding="utf-8", errors="ignore"))
    if len(values) < 64:
        raise RuntimeError(f"WeSpeaker embedding output is too small: {len(values)} values")
    return tuple(values)


def _wespeaker_command() -> list[str]:
    configured_command = os.environ.get("VOICE_BRIDGE_WESPEAKER_COMMAND")
    if configured_command:
        return shlex.split(configured_command)
    worker_python = _wespeaker_python()
    if worker_python:
        return [worker_python, "-m", "voice_engine.speaker.wespeaker_eres2net_runner"]
    command = shutil.which("wespeaker")
    if command:
        return [command]
    raise RuntimeError(
        "WeSpeaker runtime was not found. Install with scripts/setup_speaker_encoder.ps1 "
        "or set VOICE_BRIDGE_WESPEAKER_PYTHON to the speaker encoder Python."
    )


def _wespeaker_python() -> str | None:
    configured_python = os.environ.get("VOICE_BRIDGE_WESPEAKER_PYTHON")
    if configured_python:
        return configured_python
    local_python = Path("I:/voice_engine/.venv-speaker/Scripts/python.exe")
    if local_python.exists():
        return str(local_python)
    return None


_WORKERS: dict[tuple[str, str, str, str], "_EmbeddingWorker"] = {}
_WORKERS_LOCK = threading.Lock()


def _worker_embedding(
    python: str,
    wav_path: str,
    device: str,
    cache_dir: str,
    pretrain_dir: str,
) -> tuple[float, ...]:
    key = (python, device, cache_dir, pretrain_dir)
    with _WORKERS_LOCK:
        worker = _WORKERS.get(key)
        if worker is None or worker.closed:
            worker = _EmbeddingWorker(python, device, cache_dir, pretrain_dir)
            _WORKERS[key] = worker
    return worker.extract(wav_path)


class _EmbeddingWorker:
    def __init__(self, python: str, device: str, cache_dir: str, pretrain_dir: str) -> None:
        self.closed = False
        self._lock = threading.Lock()
        env = os.environ.copy()
        env["WESPEAKER_HOME"] = cache_dir
        src_path = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        args = [
            python,
            "-m",
            "voice_engine.speaker.wespeaker_eres2net_worker",
            "--device",
            device,
        ]
        if pretrain_dir:
            args.extend(["--pretrain", pretrain_dir])
        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )
        ready = self._read_json()
        if not ready.get("ok"):
            self.close()
            raise RuntimeError(f"WeSpeaker worker failed to start: {ready.get('error')}\n{ready.get('traceback', '')}")
        atexit.register(self.close)

    def extract(self, wav_path: str) -> tuple[float, ...]:
        with self._lock:
            request_id = uuid.uuid4().hex
            self._write_json({"request_id": request_id, "audio_file": wav_path})
            response = self._read_json()
            if not response.get("ok"):
                raise RuntimeError(
                    f"WeSpeaker worker failed: {response.get('error')}\n{response.get('traceback', '')}"
                )
            values = response.get("embedding")
            if not isinstance(values, list):
                raise RuntimeError("WeSpeaker worker response did not include an embedding list")
            return tuple(float(value) for value in values)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self._process.stdin:
                self._write_json({"quit": True})
        except Exception:
            pass
        try:
            self._process.terminate()
        except Exception:
            pass

    def _write_json(self, payload: dict[str, object]) -> None:
        if self._process.stdin is None:
            raise RuntimeError("WeSpeaker worker stdin is closed")
        self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._process.stdin.flush()

    def _read_json(self) -> dict[str, object]:
        if self._process.stdout is None:
            raise RuntimeError("WeSpeaker worker stdout is closed")
        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(f"WeSpeaker worker exited before response. stderr:\n{stderr}")
        return json.loads(line)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "", "-1"} else "cpu"
    return device


def _is_pretrain_dir(path: Path) -> bool:
    return (path / "avg_model.pt").exists() and (path / "config.yaml").exists()


def _parse_embedding(text: str) -> list[float]:
    return [float(item) for item in re.findall(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]


def _l2_normalize(values: list[float] | tuple[float, ...]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if norm <= 1e-8:
        raise RuntimeError("WeSpeaker returned a zero speaker embedding")
    return [float(value) / norm for value in values]


def _quality_score(speech_duration_sec: float, dim: int) -> float:
    duration_score = min(1.0, speech_duration_sec / 6.0)
    dim_score = min(1.0, dim / 192.0)
    return round(max(0.0, min(1.0, 0.72 * duration_score + 0.28 * dim_score)), 6)


def _wav_duration_sec(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / max(1, wav.getframerate())
