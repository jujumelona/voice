from __future__ import annotations

import argparse
import os
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml

from wespeaker.cli.hub import Hub
from wespeaker.models import eres2net
from wespeaker.utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["embedding"], default="embedding")
    parser.add_argument("--audio_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eres2net", action="store_true")
    parser.add_argument("--pretrain", default="")
    parser.add_argument("--resample_rate", type=int, default=16000)
    args = parser.parse_args()

    model_dir = Path(args.pretrain) if args.pretrain else Path(Hub.get_model("eres2net"))
    model = _load_eres2net_model(model_dir, args.device)
    embedding = _extract_embedding(model, Path(args.audio_file), args.device, args.resample_rate)
    np.savetxt(args.output_file, embedding.detach().cpu().numpy())


def _load_eres2net_model(model_dir: Path, device: str) -> torch.nn.Module:
    config_path = model_dir / "config.yaml"
    checkpoint_path = model_dir / "avg_model.pt"
    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"WeSpeaker ERes2Net files missing under {model_dir}. "
            "Need config.yaml and avg_model.pt."
        )
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    model_name = config["model"]
    if not model_name.startswith("ERes2Net"):
        raise ValueError(f"Expected an ERes2Net model config, got {model_name}")
    model = getattr(eres2net, model_name)(**config["model_args"])
    load_checkpoint(model, str(checkpoint_path))
    model.eval()
    return model.to(torch.device(device))


def _extract_embedding(
    model: torch.nn.Module,
    audio_path: Path,
    device: str,
    sample_rate: int,
) -> torch.Tensor:
    wavform, original_rate = _read_pcm16_wav(audio_path)
    if original_rate != sample_rate:
        wavform = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=sample_rate)(wavform)
    feats = kaldi.fbank(
        wavform.to(torch.float),
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        sample_frequency=sample_rate,
        window_type="povey",
    )
    feats = feats - torch.mean(feats, dim=0)
    feats = feats.unsqueeze(0).to(torch.device(device))
    with torch.no_grad():
        outputs = model(feats)
        return outputs[-1][0] if isinstance(outputs, tuple) else outputs[0]


def _read_pcm16_wav(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"WeSpeaker ERes2Net runner expects PCM16 wav: {path}")
    values: list[float] = []
    step = sample_width * channels
    for index in range(0, len(frames), step):
        total = 0
        for channel in range(channels):
            offset = index + channel * sample_width
            total += int.from_bytes(frames[offset : offset + sample_width], "little", signed=True)
        values.append(total / channels / 32768.0)
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0), sample_rate


if __name__ == "__main__":
    main()
