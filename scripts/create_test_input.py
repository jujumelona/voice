from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/test/input.wav"))
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = int(args.seconds * args.sample_rate)
    with wave.open(str(args.out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(args.sample_rate)
        for index in range(total):
            t = index / args.sample_rate
            envelope = min(1.0, index / (args.sample_rate * 0.05), (total - index) / (args.sample_rate * 0.08))
            f0 = 170.0 + 45.0 * math.sin(2.0 * math.pi * 1.7 * t)
            energy = 0.22 + 0.18 * math.sin(2.0 * math.pi * 3.3 * t) ** 2
            if 0.80 < t < 1.00:
                energy *= 0.08
            sample = envelope * energy * math.sin(2.0 * math.pi * f0 * t)
            wav.writeframes(int(max(-1.0, min(1.0, sample)) * 32767.0).to_bytes(2, "little", signed=True))
    print(args.out)


if __name__ == "__main__":
    main()
