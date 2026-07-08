from __future__ import annotations

from abc import ABC, abstractmethod

from voice_engine.pipeline.types import Transcript, Translation


class TranslationAdapter(ABC):
    @abstractmethod
    def translate(self, transcript: Transcript, target_language: str) -> Translation:
        raise NotImplementedError

