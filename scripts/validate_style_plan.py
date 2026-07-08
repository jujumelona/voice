from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voice_engine.content.units import content_units_from_text
from voice_engine.paths import model_path
from voice_engine.pipeline.types import Transcript, Translation
from voice_engine.prosody.events import extract_word_prosody_events
from voice_engine.prosody.extractor import extract_prosody
from voice_engine.prosody.style_plan import build_style_plan, style_plan_to_dict
from voice_engine.prosody.style_tokens import style_tokens_from_plan, style_tokens_to_dict
from voice_engine.speaker.speaker_profile import extract_speaker_profile_from_wav


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/style_plan_validation"))
    parser.add_argument("--source-text", default="hello now")
    parser.add_argument("--translation-text", default="annyeonghaseyo")
    parser.add_argument("--target-language", default="ko")
    parser.add_argument("--speaker-encoder-device", default="auto")
    parser.add_argument("--speaker-encoder-cache-dir", type=Path, default=model_path("wespeaker"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    source = args.input or args.out / "synthetic_reference.wav"
    if args.input is None:
        _write_synthetic_reference(source)

    speaker = extract_speaker_profile_from_wav(
        source,
        device=args.speaker_encoder_device,
        cache_dir=args.speaker_encoder_cache_dir,
    )
    trace = extract_prosody(source, speaker_median_f0=speaker.median_f0).trace
    transcript = Transcript(text=args.source_text, language="en")
    source_events = extract_word_prosody_events(transcript, trace)
    translation = Translation(text=args.translation_text, source_language="en", target_language=args.target_language)
    content = content_units_from_text(translation.text, translation.target_language or "ko")
    source_unit_count = len(source_events) or len([token for token in args.source_text.split() if token])
    plan = build_style_plan(
        source_trace=trace,
        speaker=speaker,
        content=content,
        source_unit_count=source_unit_count,
        source_events=source_events,
    )
    tokens = style_tokens_from_plan(plan=plan, source_trace=trace, speaker=speaker)

    summary = {
        "algorithm": plan.diagnostics.get("algorithm"),
        "source_event_count": len(source_events),
        "target_event_count": len(plan.events),
        "content_units": content.units,
        "style_plan": style_plan_to_dict(plan),
        "style_tokens": style_tokens_to_dict(tokens),
        "note": "This validates event extraction and pre-acoustic token payload only. It does not synthesize comparison audio.",
    }
    (args.out / "style_plan.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: summary[k] for k in ("algorithm", "source_event_count", "target_event_count", "content_units", "note")},
            ensure_ascii=False,
            indent=2,
        )
    )


def _write_synthetic_reference(path: Path, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    segments = [
        (0.74, 190.0, 0.24),
        (0.16, 0.0, 0.0),
        (0.82, 245.0, 0.36),
    ]
    samples: list[float] = []
    phase = 0.0
    for duration, f0, amp in segments:
        count = round(duration * sample_rate)
        for index in range(count):
            if f0 <= 0.0:
                samples.append(0.0)
                continue
            envelope = min(1.0, index / max(1, count * 0.12), (count - index) / max(1, count * 0.10))
            phase += 2.0 * math.pi * f0 / sample_rate
            sample = amp * envelope * (math.sin(phase) + 0.28 * math.sin(2.0 * phase))
            samples.append(max(-0.98, min(0.98, sample)))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for sample in samples:
            wav.writeframes(int(sample * 32767.0).to_bytes(2, "little", signed=True))


if __name__ == "__main__":
    main()
