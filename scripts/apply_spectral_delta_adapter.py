from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply B_spectral_delta_080 to a generated wav")
    parser.add_argument("--source", required=True, help="TTS/generated wav")
    parser.add_argument("--target", required=True, help="target speaker wav")
    parser.add_argument("--out", required=True, help="converted output wav")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--strength", type=float, default=0.80)
    parser.add_argument("--max_gain_db", type=float, default=12.0)
    args = parser.parse_args()

    from voice_engine.adapters.decoder_wrapper import read_wav_mono, write_wav_mono
    from voice_engine.adapters.spectral_delta_adapter import (
        SpectralDeltaConfig,
        spectral_delta_adapter,
    )

    source = read_wav_mono(args.source, sample_rate=args.sr)
    target = read_wav_mono(args.target, sample_rate=args.sr)
    cfg = SpectralDeltaConfig(
        sr=args.sr,
        strength=args.strength,
        max_gain_db=args.max_gain_db,
    )
    converted = spectral_delta_adapter(source, target, cfg)
    write_wav_mono(Path(args.out), converted, sample_rate=args.sr)


if __name__ == "__main__":
    main()
