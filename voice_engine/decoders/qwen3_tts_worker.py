from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from voice_engine.decoders.audio_payload import encode_audio_payload
from voice_engine.decoders.qwen3_tts_decoder import _audio_to_numpy_and_rate, write_wav_mono

JSON_PREFIX = "__VOICE_ENGINE_JSON__"


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-TTS isolated inference worker")
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--text")
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--serve-jsonl", action="store_true", help="Keep model loaded and handle JSONL requests on stdin.")
    args = parser.parse_args()

    if args.serve_jsonl:
        _serve_jsonl(args.model_source, args.device)
        return

    if args.text is None or args.reference is None or args.out is None:
        raise SystemExit("single-shot mode requires --text, --reference, and --out")
    if not args.reference.exists():
        raise FileNotFoundError(f"reference audio not found: {args.reference}")

    model = _load_model(args.model_source, args.device)
    samples, sample_rate = _generate(
        model,
        text=args.text,
        language=args.language,
        reference=args.reference,
    )
    write_wav_mono(args.out, samples, sample_rate)


def _serve_jsonl(model_source: str, device: str) -> None:
    """Persistent worker: load Qwen3 once, then synthesize one request per JSON line.

    Response lines are prefixed so accidental library prints on stdout do not break
    the parent process parser.
    """
    try:
        model = _load_model(model_source, device)
        _emit({"ok": True, "event": "ready"})
    except Exception as exc:
        _emit({"ok": False, "event": "load_error", "error": repr(exc), "traceback": traceback.format_exc()})
        return

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
            if request.get("cmd") == "shutdown":
                _emit({"ok": True, "event": "shutdown"})
                return
            text = str(request["text"])
            language = str(request.get("language") or "Auto")
            reference = Path(request["reference"])
            if not reference.exists():
                raise FileNotFoundError(f"reference audio not found: {reference}")
            samples, sample_rate = _generate(model, text=text, language=language, reference=reference)
            if request.get("inline_audio", True):
                payload = encode_audio_payload(samples, sample_rate)
                payload.update({"ok": True, "event": "generated"})
                _emit(payload)
            else:
                out = Path(request["out"])
                write_wav_mono(out, samples, sample_rate)
                _emit({"ok": True, "event": "generated", "out": str(out), "sample_rate": int(sample_rate), "samples": int(len(samples))})
        except Exception as exc:
            _emit({"ok": False, "event": "generate_error", "error": repr(exc), "traceback": traceback.format_exc()})


def _emit(payload: dict) -> None:
    sys.stdout.write(JSON_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _load_model(model_source: str, device: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    kwargs = {"device_map": device}
    if device.startswith("cuda"):
        kwargs["dtype"] = torch.bfloat16
    return Qwen3TTSModel.from_pretrained(model_source, **kwargs)


def _generate(model, *, text: str, language: str, reference: Path) -> tuple[np.ndarray, int]:
    try:
        prompt = model.create_voice_clone_prompt(
            ref_audio=str(reference),
            ref_text="",
            x_vector_only_mode=True,
        )
        generated = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
        )
    except TypeError:
        generated = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(reference),
            ref_text="",
        )
    return _audio_to_numpy_and_rate(generated)


if __name__ == "__main__":
    main()
