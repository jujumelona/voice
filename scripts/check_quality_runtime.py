from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voice_engine.adapters.spectral_delta_adapter import (  # noqa: E402
    SpectralDeltaConfig,
    spectral_delta_adapter,
)
from voice_engine.mt.argos import ArgosTranslation  # noqa: E402
from voice_engine.paths import runtime_root  # noqa: E402
from voice_engine.pipeline.types import Transcript  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real runtime quality checks")
    parser.add_argument("--runtime-root", type=Path, default=runtime_root())
    parser.add_argument("--out", type=Path)
    parser.add_argument("--decoder", choices=["auto", "voxcpm2", "qwen3-tts"], default="auto")
    parser.add_argument("--voxcpm2-model-dir", type=Path)
    parser.add_argument("--voxcpm2-reference", type=Path)
    parser.add_argument("--voxcpm2-timeout-sec", type=int, default=240)
    parser.add_argument("--qwen3-tts-model-dir", type=Path)
    parser.add_argument("--qwen3-tts-python", type=Path)
    parser.add_argument("--qwen3-tts-timeout-sec", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-decoder", action="store_true")
    parser.add_argument("--skip-voxcpm2", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    out = args.out or args.runtime_root / "test" / "quality_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "ok": False,
        "runtime_root": str(args.runtime_root),
        "system": _system_info(),
        "checks": {},
    }

    checks: dict[str, object] = {}
    checks["asr_whisper_cpp"] = _check_whisper_cpp(args.runtime_root)
    checks["translation_argos"] = _check_argos_translation()
    checks["spectral_delta"] = _check_spectral_delta()
    decoder = _selected_decoder(args.decoder, report["system"], args.device)
    checks["speech_decoder_generation"] = (
        {"ok": True, "skipped": True, "decoder": decoder}
        if args.skip_decoder or args.skip_voxcpm2
        else _check_speech_decoder(args, out.parent, decoder)
    )

    report["checks"] = checks
    report["ok"] = all(bool(check.get("ok")) for check in checks.values() if isinstance(check, dict))

    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"quality_report={out}")

    if not report["ok"]:
        raise SystemExit(1)


def _system_info() -> dict[str, object]:
    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - diagnostic only
        info["torch_error"] = repr(exc)
    return info


def _check_whisper_cpp(root: Path) -> dict[str, object]:
    binary = root / "bin" / "whisper-cli.exe"
    model = root / "models" / "whisper" / "ggml-base.bin"
    sample = root / "src" / "whisper.cpp-v1.9.1" / "samples" / "jfk.wav"
    missing = [str(p) for p in (binary, model, sample) if not p.exists()]
    if missing:
        return {"ok": False, "missing": missing}

    start = time.perf_counter()
    result = subprocess.run(
        [str(binary), "-m", str(model), "-f", str(sample), "-nt", "-np"],
        text=True,
        capture_output=True,
        timeout=90,
    )
    elapsed = time.perf_counter() - start
    text = (result.stdout + "\n" + result.stderr).strip()
    normalized = text.lower()
    expected = ["ask not", "country", "fellow americans"]
    return {
        "ok": result.returncode == 0 and all(term in normalized for term in expected),
        "returncode": result.returncode,
        "elapsed_sec": round(elapsed, 3),
        "text": text,
    }


def _check_argos_translation() -> dict[str, object]:
    samples = [
        ("en", "ko", "Hello, can you hear me clearly? I will call you back in five minutes."),
        ("ko", "en", "안녕하세요. 제 말이 잘 들리나요? 5분 뒤에 다시 전화할게요."),
    ]
    samples = [
        ("en", "ko", "Hello, can you hear me clearly? I will call you back in five minutes."),
        (
            "ko",
            "en",
            "\uc548\ub155\ud558\uc138\uc694. \uc81c \ub9d0\uc774 \uc798 \ub4e4\ub9ac\ub098\uc694? 5\ubd84 \ub4a4\uc5d0 \ub2e4\uc2dc \uc804\ud654\ud560\uac8c\uc694.",
        ),
    ]
    outputs: list[dict[str, object]] = []
    ok = True
    for source, target, text in samples:
        try:
            translated = ArgosTranslation(source_language=source).translate(
                Transcript(text=text, language=source),
                target,
            )
            value = translated.text
            script_ok = _contains_hangul(value) if target == "ko" else _contains_latin(value)
            clean_ok = bool(value.strip()) and "\ufffd" not in value
            # This is a syntax/legibility gate, not a human naturalness score.
            sample_ok = script_ok and clean_ok
            ok = ok and sample_ok
            outputs.append(
                {
                    "ok": sample_ok,
                    "source_language": source,
                    "target_language": target,
                    "source": text,
                    "translation": value,
                    "script_ok": script_ok,
                    "clean_ok": clean_ok,
                }
            )
        except Exception as exc:
            ok = False
            outputs.append(
                {
                    "ok": False,
                    "source_language": source,
                    "target_language": target,
                    "source": text,
                    "error": repr(exc),
                }
            )
    return {
        "ok": ok,
        "note": "Argos pass only means offline translation is legible. It does not prove natural call-style Korean.",
        "samples": outputs,
    }


def _check_spectral_delta() -> dict[str, object]:
    sr = 16000
    duration = 3.0
    t = _linspace(duration, sr)
    source = (0.22 * _sin(2.0 * 3.141592653589793 * 220.0 * t)).astype("float32")
    target = (
        0.12 * _sin(2.0 * 3.141592653589793 * 120.0 * t)
        + 0.05 * _sin(2.0 * 3.141592653589793 * 240.0 * t)
    ).astype("float32")

    # Exclude librosa/numba cold-start compilation from steady-state DSP timing.
    spectral_delta_adapter(source, target, SpectralDeltaConfig(sr=sr))

    start = time.perf_counter()
    out = spectral_delta_adapter(source, target, SpectralDeltaConfig(sr=sr))
    elapsed = time.perf_counter() - start
    peak = float(abs(out).max()) if len(out) else 0.0
    finite = bool(_is_finite(out))
    duration_sec = len(out) / sr
    rtf = elapsed / max(duration_sec, 1e-6)
    return {
        "ok": finite and len(out) == len(source) and 0.0 < peak <= 0.98 and rtf < 0.25,
        "elapsed_sec": round(elapsed, 4),
        "audio_duration_sec": round(duration_sec, 4),
        "rtf": round(rtf, 4),
        "peak": round(peak, 6),
        "length_match": len(out) == len(source),
        "finite": finite,
    }


def _selected_decoder(decoder: str, system: dict[str, object], device: str) -> str:
    if decoder != "auto":
        return decoder
    if device.startswith("cuda"):
        return "voxcpm2"
    if device == "cpu":
        return "qwen3-tts"
    return "voxcpm2" if system.get("cuda_available") else "qwen3-tts"


def _check_speech_decoder(args, out_dir: Path, decoder: str) -> dict[str, object]:
    if decoder == "voxcpm2":
        return _check_voxcpm2(args, out_dir)
    if decoder == "qwen3-tts":
        return _check_qwen3_tts(args, out_dir)
    raise ValueError(f"unknown decoder: {decoder}")


def _check_voxcpm2(args, out_dir: Path) -> dict[str, object]:
    model_dir = args.voxcpm2_model_dir or args.runtime_root / "models" / "voxcpm2"
    reference = args.voxcpm2_reference or args.runtime_root / "test" / "ref_synthetic.wav"
    out_wav = out_dir / "voxcpm2_quality_check.wav"
    log_path = out_dir / "voxcpm2_quality_check.log"
    missing = [str(p) for p in (model_dir, reference) if not p.exists()]
    if missing:
        return {"ok": False, "decoder": "voxcpm2", "missing": missing}

    script = ROOT / "scripts" / "smoke_voxcpm2_generate.py"
    command = [
        sys.executable,
        str(script),
        "--model-dir",
        str(model_dir),
        "--reference",
        str(reference),
        "--out",
        str(out_wav),
        "--text",
        "Hello, can you hear me clearly? I will call you back in five minutes.",
        "--device",
        args.device,
        "--inference-timesteps",
        "4",
    ]
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=args.voxcpm2_timeout_sec,
        )
        elapsed = time.perf_counter() - start
        log = (result.stdout or "") + ("\n" if result.stdout and result.stderr else "") + (result.stderr or "")
        log_path.write_text(log, encoding="utf-8", errors="replace")
        return {
            "ok": result.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 44,
            "decoder": "voxcpm2",
            "returncode": result.returncode,
            "elapsed_sec": round(elapsed, 3),
            "out": str(out_wav),
            "out_exists": out_wav.exists(),
            "out_size": out_wav.stat().st_size if out_wav.exists() else 0,
            "log": str(log_path),
            "log_tail": _tail_text(log, 30),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        log = ((exc.stdout or "") + "\n" + (exc.stderr or "")).strip()
        if isinstance(log, bytes):
            log = log.decode("utf-8", errors="replace")
        log_path.write_text(str(log), encoding="utf-8", errors="replace")
        return {
            "ok": False,
            "decoder": "voxcpm2",
            "timeout": True,
            "elapsed_sec": round(elapsed, 3),
            "log": str(log_path),
            "log_tail": _tail_text(str(log), 30),
        }


def _check_qwen3_tts(args, out_dir: Path) -> dict[str, object]:
    model_dir = args.qwen3_tts_model_dir or args.runtime_root / "models" / "qwen3-tts" / "0.6B-base"
    qwen_python = args.qwen3_tts_python or args.runtime_root / ".venv-qwen3-tts" / "Scripts" / "python.exe"
    reference = args.voxcpm2_reference or args.runtime_root / "test" / "ref_synthetic.wav"
    out_wav = out_dir / "qwen3_tts_quality_check.wav"
    log_path = out_dir / "qwen3_tts_quality_check.log"
    missing = [str(p) for p in (qwen_python, reference) if not p.exists()]
    if not model_dir.exists():
        missing.append(str(model_dir))
    if missing:
        return {"ok": False, "decoder": "qwen3-tts", "missing": missing}

    script = ROOT / "scripts" / "smoke_qwen3_tts_generate.py"
    command = [
        sys.executable,
        str(script),
        "--model-dir",
        str(model_dir),
        "--qwen-python",
        str(qwen_python),
        "--reference",
        str(reference),
        "--out",
        str(out_wav),
        "--text",
        "Hello, can you hear me clearly? I will call you back in five minutes.",
        "--language",
        "en",
        "--device",
        args.device,
        "--timeout-sec",
        str(args.qwen3_tts_timeout_sec),
    ]
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=args.qwen3_tts_timeout_sec + 30,
        )
        elapsed = time.perf_counter() - start
        log = (result.stdout or "") + ("\n" if result.stdout and result.stderr else "") + (result.stderr or "")
        log_path.write_text(log, encoding="utf-8", errors="replace")
        return {
            "ok": result.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 44,
            "decoder": "qwen3-tts",
            "returncode": result.returncode,
            "elapsed_sec": round(elapsed, 3),
            "out": str(out_wav),
            "out_exists": out_wav.exists(),
            "out_size": out_wav.stat().st_size if out_wav.exists() else 0,
            "log": str(log_path),
            "log_tail": _tail_text(log, 30),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        log = ((exc.stdout or "") + "\n" + (exc.stderr or "")).strip()
        if isinstance(log, bytes):
            log = log.decode("utf-8", errors="replace")
        log_path.write_text(str(log), encoding="utf-8", errors="replace")
        return {
            "ok": False,
            "decoder": "qwen3-tts",
            "timeout": True,
            "elapsed_sec": round(elapsed, 3),
            "log": str(log_path),
            "log_tail": _tail_text(str(log), 30),
        }


def _contains_hangul(value: str) -> bool:
    return any("\uac00" <= c <= "\ud7a3" for c in value)


def _contains_latin(value: str) -> bool:
    return any(("a" <= c.lower() <= "z") for c in value)


def _tail_text(value: str, lines: int) -> list[str]:
    return value.splitlines()[-lines:]


def _linspace(duration: float, sr: int):
    import numpy as np

    return np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)


def _sin(value):
    import numpy as np

    return np.sin(value)


def _is_finite(value) -> bool:
    import numpy as np

    return bool(np.isfinite(value).all())


if __name__ == "__main__":
    main()
