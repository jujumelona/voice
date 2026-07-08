from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voice_engine.content.units import content_units_from_text
from voice_engine.decoders.qwen3_tts_decoder import Qwen3TTSDecoder
from voice_engine.pipeline.types import VoiceDecoderInput


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Qwen3-TTS generation through Voice Engine decoder")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--qwen-python", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--text", default="Hello, this is a voice engine test.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--timeout-sec", type=int, default=300)
    args = parser.parse_args()

    decoder = Qwen3TTSDecoder(
        model_dir=args.model_dir,
        qwen_python=args.qwen_python,
        device=args.device,
        timeout_sec=args.timeout_sec,
    )
    request = VoiceDecoderInput(
        content=content_units_from_text(args.text, args.language),
        speaker=None,
        prosody=None,
        reference_audio_path=str(args.reference),
    )

    start = time.perf_counter()
    chunks = list(decoder.decode(request))
    elapsed = time.perf_counter() - start

    if not chunks:
        raise RuntimeError("Qwen3-TTS returned no chunks")

    sample_rate = chunks[0].sample_rate
    samples = [sample for chunk in chunks for sample in chunk.samples]
    _write_wav(args.out, samples, sample_rate)
    duration = len(samples) / sample_rate
    rtf = elapsed / max(duration, 1e-6)
    print(f"sample_rate={sample_rate}")
    print(f"samples={len(samples)}")
    print(f"audio_duration_sec={duration:.3f}")
    print(f"elapsed_sec={elapsed:.3f}")
    print(f"rtf={rtf:.3f}")
    print(f"out={args.out}")


def _write_wav(path: Path, samples: list[float], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for sample in samples:
            value = max(-1.0, min(1.0, float(sample)))
            wav.writeframes(int(value * 32767.0).to_bytes(2, "little", signed=True))


if __name__ == "__main__":
    main()
