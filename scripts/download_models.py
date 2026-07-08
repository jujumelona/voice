import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "model_manifest.json"


def runtime_root() -> Path:
    configured = os.environ.get("VOICE_BRIDGE_RUNTIME_ROOT")
    if configured:
        return Path(configured)
    if os.name == "nt" and Path("I:/").exists():
        return Path("I:/voice_bridge")
    return Path(".")


def model_root() -> Path:
    configured = os.environ.get("VOICE_BRIDGE_MODEL_ROOT")
    if configured:
        return Path(configured)
    return runtime_root() / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", action="store_true", help="print runtime contract and local paths")
    args = parser.parse_args()

    if args.runtime:
        print_runtime_contract()
        return

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print("Model download policy")
    print("=====================")
    print("This script intentionally does not fetch weights.")
    print("Keep weights/checkpoints under VOICE_BRIDGE_MODEL_ROOT or user cache, not in git.")
    print(f"Current model root: {model_root()}\n")

    for group, entries in data.items():
        print(f"[{group}]")
        for entry in entries:
            print(f"- {entry['name']}")
            print(f"  license: {entry['license']}")
            print(f"  source: {entry['source']}")
            print(f"  expected_path: {entry['expected_path']}")
            print(f"  weights_note: {entry['weights_note']}")
        print()


def print_runtime_contract() -> None:
    print("Voice Engine runtime contract")
    print("============================")
    print("Default fast decoder: qwen3-tts + B_spectral_delta_080")
    print("Auto decoder:")
    print("  CPU  -> qwen3-tts")
    print("  CUDA -> voxcpm2")
    print("Set one of:")
    print(f"  VOICE_BRIDGE_RUNTIME_ROOT={runtime_root()}")
    print(f"  VOICE_BRIDGE_MODEL_ROOT={model_root()}")
    print(f"  WESPEAKER_HOME={model_root() / 'wespeaker'}")
    print(f"  VOICE_BRIDGE_WESPEAKER_PYTHON={runtime_root() / '.venv-speaker' / 'Scripts' / 'python.exe'}")
    print("  VOICE_BRIDGE_WESPEAKER_DEVICE=auto")
    print()
    print("Default call translation command:")
    print("  --mode fast --voice-adapter spectral_delta")
    print()
    print("Speaker encoder:")
    print("  WeSpeaker ERes2Net-large only")
    print("  Runtime: python -m voice_engine.speaker.wespeaker_eres2net_runner --task embedding --eres2net")
    print("  Pinned stack: Python 3.11 + torch/torchaudio 2.5.1")
    print(f"  optional pretrain dir: {model_root() / 'speaker' / 'wespeaker' / 'eres2net-large'}")
    print()
    print("Pipeline:")
    print("  audio -> ASR -> MT -> ContentUnits + StyleTokenTrace + SpeakerProfile")
    print("  -> TTS decoder -> B_spectral_delta_080 -> output device")
    print()
    print("Recommended local path for engine weights:")
    print(f"  {model_root()}")


if __name__ == "__main__":
    main()
