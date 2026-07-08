from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VoxCPM2 model weights from Hugging Face.")
    parser.add_argument("--repo-id", default="openbmb/VoxCPM2", help="Hugging Face repository ID")
    parser.add_argument("--output-dir", type=Path, help="Local directory to store weights")
    args = parser.parse_args()

    # Determine output path, falling back to VOICE_ENGINE_MODEL_ROOT/voxcpm2
    output_dir = args.output_dir
    if not output_dir:
        model_root = Path(os.environ.get("VOICE_ENGINE_MODEL_ROOT", "models"))
        output_dir = model_root / "voxcpm2"

    print(f"Targeting Hugging Face Repo: {args.repo_id}")
    print(f"Downloading weights to: {output_dir.resolve()}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("\n[Error] huggingface_hub is not installed.")
        print("Please install it to download weights: pip install huggingface_hub\n")
        return

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_path = snapshot_download(
            repo_id=args.repo_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.md", "*.txt", "LICENSE*"],
        )
        print(f"\n[Success] VoxCPM2 model weights downloaded successfully at: {saved_path}")
    except Exception as e:
        print(f"\n[Failure] Download failed due to error: {e}")
        print("Please check your internet connection or model repository accessibility.")


if __name__ == "__main__":
    main()
