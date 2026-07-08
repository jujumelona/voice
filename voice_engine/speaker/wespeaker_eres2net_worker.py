from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path

from wespeaker.cli.hub import Hub

from voice_engine.speaker.wespeaker_eres2net_runner import _extract_embedding, _load_eres2net_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrain", default="")
    parser.add_argument("--resample_rate", type=int, default=16000)
    args = parser.parse_args()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            model_dir = Path(args.pretrain) if args.pretrain else Path(Hub.get_model("eres2net"))
            model = _load_eres2net_model(model_dir, args.device)
        _write({"ok": True, "ready": True, "model_dir": str(model_dir)})
    except BaseException as exc:
        _write({"ok": False, "ready": False, "error": str(exc), "traceback": traceback.format_exc()})
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("quit"):
                _write({"ok": True, "quit": True})
                return
            embedding = _extract_embedding(
                model,
                Path(str(request["audio_file"])),
                args.device,
                args.resample_rate,
            )
            _write(
                {
                    "ok": True,
                    "request_id": request.get("request_id"),
                    "embedding": [float(value) for value in embedding.detach().cpu().numpy().tolist()],
                }
            )
        except BaseException as exc:
            _write(
                {
                    "ok": False,
                    "request_id": request.get("request_id") if "request" in locals() else None,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )


def _write(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
