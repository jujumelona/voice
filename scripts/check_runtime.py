from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXPECTED_PYTHON = "3.11"

CHECKS = [
    ("voice_engine", "voice_engine", "local", True),
    ("numpy", "numpy", "2.4.6", True),
    ("librosa", "librosa", "0.11.0", True),
    ("soundfile", "soundfile", "0.14.0", True),
    ("sounddevice", "sounddevice", "0.5.5", True),
    ("faster-whisper", "faster_whisper", "1.2.1", True),
    ("ctranslate2", "ctranslate2", "4.8.1", True),
    ("onnxruntime", "onnxruntime", "1.27.0", True),
    ("tokenizers", "tokenizers", "0.22.2", True),
    ("huggingface-hub", "huggingface_hub", "1.22.0", True),
    ("argostranslate", "argostranslate", "1.11.0", True),
    ("stanza", "stanza", "1.10.1", True),
    ("spacy", "spacy", "3.8.14", True),
    ("sentencepiece", "sentencepiece", "0.2.1", True),
    ("sacremoses", "sacremoses", "0.1.1", True),
    ("torch", "torch", "2.12.1", True),
    ("tqdm", "tqdm", "4.68.4", True),
    ("voxcpm", "voxcpm", "2.0.3", True),
    ("transformers", "transformers", "5.13.0", True),
    ("torchaudio", "torchaudio", None, False),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Voice Engine runtime versions and IO pipeline")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON only")
    parser.add_argument("--strict", action="store_true", help="exit non-zero when required runtime packages are missing")
    args = parser.parse_args()

    report = build_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report)

    if args.strict and not report["ok"]:
        raise SystemExit(1)


def build_report() -> dict[str, Any]:
    packages = [_package_status(dist, module, expected, required) for dist, module, expected, required in CHECKS]
    pipeline_import = _import_status("voice_engine.pipeline.realtime_call_translate")
    adapter_import = _import_status("voice_engine.adapters.spectral_delta_adapter")
    package_ok = all(not item["required"] or (item["ok"] and item["version_ok"]) for item in packages)
    python_ok = sys.version_info[:2] == (3, 11)
    ok = python_ok and package_ok and pipeline_import["ok"] and adapter_import["ok"]
    return {
        "ok": ok,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "expected": EXPECTED_PYTHON,
            "ok": python_ok,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "paths": {
            "repo": str(ROOT),
            "runtime_root": os.environ.get("VOICE_BRIDGE_RUNTIME_ROOT", _default_runtime_root()),
            "model_root": os.environ.get("VOICE_BRIDGE_MODEL_ROOT", str(Path(_default_runtime_root()) / "models")),
            "wespeaker_home": os.environ.get("VOICE_BRIDGE_WESPEAKER_HOME")
            or os.environ.get("WESPEAKER_HOME")
            or str(Path(_default_runtime_root()) / "models" / "wespeaker"),
        },
        "packages": packages,
        "imports": {
            "pipeline": pipeline_import,
            "spectral_delta_adapter": adapter_import,
        },
        "external_tools": {
            "git": _external_tool("git", ["git", "--version"], required=False),
            "ffmpeg": _external_tool("ffmpeg", ["ffmpeg", "-version"], required=False, known_rel="bin/ffmpeg/ffmpeg.exe"),
            "cmake": _external_tool("cmake", ["cmake", "--version"], required=False),
            "whisper_cli": _external_tool("whisper-cli", ["whisper-cli", "--help"], required=False, known_rel="bin/whisper-cli.exe"),
            "marian_decoder": _external_tool("marian-decoder", ["marian-decoder", "--help"], required=False, known_rel="bin/marian-decoder.exe"),
        },
        "pipeline_io": pipeline_io_contract(),
        "commands": {
            "install_all": r".\scripts\setup_all.ps1 -RuntimeRoot I:\voice_bridge -Torch cpu -WhisperModel base",
            "check_runtime": f"{sys.executable} scripts\\check_runtime.py --strict",
            "validate_pipeline": (
                f"{sys.executable} -m voice_engine.pipeline.realtime_call_translate "
                r"--validate-only --direction both --mode fast --voice-adapter spectral_delta"
            ),
            "wav_adapter_test": (
                f"{sys.executable} scripts\\apply_spectral_delta_adapter.py "
                r"--source generated.wav --target target_speaker.wav --out converted.wav"
            ),
        },
    }


def pipeline_io_contract() -> dict[str, Any]:
    return {
        "outbound": [
            "physical microphone",
            "ASR stream",
            "translation",
            "ContentUnits",
            "current utterance StyleTokenTrace",
            "cumulative SpeakerProfile",
            "auto speech decoder output wav/source stream",
            "B_spectral_delta_080 voice color adapter",
            "virtual microphone for call apps",
        ],
        "inbound": [
            "call app speaker or WASAPI loopback",
            "ASR stream",
            "translation",
            "ContentUnits",
            "current utterance StyleTokenTrace",
            "cumulative SpeakerProfile",
            "auto speech decoder output wav/source stream",
            "B_spectral_delta_080 voice color adapter",
            "local headphones",
        ],
        "adapter_rules": {
            "source": "TTS-generated audio",
            "target": "current speaker reference chunk or cumulative spectral profile",
            "preserves": ["source phase", "source length", "source timing", "source pronunciation", "source pitch contour"],
            "does_not_do_by_default": ["target phase", "target timing", "formant warp", "pitch shift", "global energy matching"],
        },
    }


def print_human(report: dict[str, Any]) -> None:
    print("Voice Engine runtime check")
    print("==========================")
    print(f"ok: {report['ok']}")
    python_status = "OK" if report["python"]["ok"] else "WRONG"
    print(
        f"python: {python_status} {report['python']['version']} "
        f"(expected {report['python']['expected']}.x) {report['python']['executable']}"
    )
    print(f"runtime_root: {report['paths']['runtime_root']}")
    print(f"model_root: {report['paths']['model_root']}")
    print()
    print("packages:")
    for item in report["packages"]:
        status = "OK" if item["ok"] else "MISSING"
        if item["ok"] and not item["version_ok"]:
            status = "VERSION"
        required = "required" if item["required"] else "optional"
        version = item.get("version") or "-"
        expected = item.get("expected") or "-"
        print(f"  {status:7} {item['module']:18} {version:12} expected={expected:10} {required}")
        if item.get("error"):
            print(f"          {item['error']}")
    print()
    print("external tools:")
    for name, item in report["external_tools"].items():
        status = "OK" if item["found"] else "MISSING"
        print(f"  {status:7} {name:15} {item.get('path') or '-'}")
        if item.get("version"):
            print(f"          {item['version']}")
    print()
    print("pipeline:")
    for direction, stages in report["pipeline_io"].items():
        if isinstance(stages, list):
            print(f"  {direction}:")
            for stage in stages:
                print(f"    -> {stage}")
    print()
    print("commands:")
    for label, command in report["commands"].items():
        print(f"  {label}: {command}")


def _package_status(dist_name: str, module_name: str, expected: str | None, required: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "distribution": dist_name,
        "module": module_name,
        "expected": expected,
        "required": required,
        "ok": False,
        "version_ok": expected is None,
        "version": None,
    }
    try:
        importlib.import_module(module_name)
        result["ok"] = True
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    try:
        result["version"] = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        if module_name == "voice_engine":
            result["version"] = "local"
    if expected is not None and result["version"] is not None:
        result["version_ok"] = result["version"] == expected
    return result


def _import_status(module_name: str) -> dict[str, Any]:
    try:
        importlib.import_module(module_name)
        return {"ok": True, "module": module_name}
    except BaseException as exc:
        return {"ok": False, "module": module_name, "error": f"{type(exc).__name__}: {exc}"}


def _default_runtime_root() -> str:
    if os.name == "nt" and Path("I:/").exists():
        return "I:/voice_bridge"
    return str(ROOT / ".runtime")


def _external_tool(name: str, command: list[str], required: bool, known_rel: str | None = None) -> dict[str, Any]:
    path = shutil.which(name)
    if path is None and known_rel:
        runtime_root = Path(os.environ.get("VOICE_BRIDGE_RUNTIME_ROOT", _default_runtime_root()))
        known_path = runtime_root / Path(known_rel)
        if known_path.exists():
            path = str(known_path)
            command = [str(known_path), *command[1:]]
    result: dict[str, Any] = {"required": required, "found": path is not None, "path": path}
    if path is None:
        return result
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            encoding="utf-8",
            errors="replace",
        )
        output = (completed.stdout or completed.stderr or "").splitlines()
        if output:
            result["version"] = output[0].strip()
    except BaseException as exc:
        result["version_error"] = f"{type(exc).__name__}: {exc}"
    return result


if __name__ == "__main__":
    main()
