from __future__ import annotations

import subprocess
from pathlib import Path

from voice_engine.mt.base import TranslationAdapter
from voice_engine.paths import model_path
from voice_engine.pipeline.types import Transcript, Translation


class MarianTranslation(TranslationAdapter):
    def __init__(
        self,
        binary: str | Path = "marian-decoder",
        model: str | Path | None = None,
        vocab: str | Path | None = None,
    ) -> None:
        self.binary = str(binary)
        self.model = str(model or model_path("marian", "model.npz"))
        self.vocab = str(vocab or model_path("marian", "vocab.yml"))

    def translate(self, transcript: Transcript, target_language: str) -> Translation:
        command = [
            self.binary,
            "-m",
            self.model,
            "-v",
            self.vocab,
            self.vocab,
        ]
        result = subprocess.run(
            command,
            input=transcript.text,
            text=True,
            capture_output=True,
            check=True,
        )
        return Translation(
            text=result.stdout.strip(),
            source_language=transcript.language,
            target_language=target_language,
        )
