from __future__ import annotations

from voice_engine.mt.base import TranslationAdapter
from voice_engine.pipeline.types import Transcript, Translation, TranscriptEvent, TranslationEvent


class ArgosTranslation(TranslationAdapter):
    """Argos Translate adapter.

    Argos itself and downloaded language packages stay outside this repository.
    Install with the optional `translate` extra, then install language packages
    into the user's local Argos cache.
    """

    def __init__(self, source_language: str | None = None) -> None:
        self.source_language = source_language
        try:
            import argostranslate.translate as argos_translate
        except ImportError as exc:
            raise RuntimeError(
                "Argos Translate is not installed. Install optional dependency: "
                "python -m pip install .[translate]"
            ) from exc
        self._translate = argos_translate

    def translate(self, transcript: Transcript, target_language: str) -> Translation:
        source_language = transcript.language or self.source_language
        if not source_language:
            raise ValueError("Argos Translate requires a source language code")
        text = self._translate.translate(transcript.text, source_language, target_language)
        return Translation(
            text=text,
            source_language=source_language,
            target_language=target_language,
        )

    def translate_event(self, event: TranscriptEvent, target_language: str) -> TranslationEvent:
        translation = self.translate(
            Transcript(text=event.text, language=event.language),
            target_language=target_language,
        )
        return TranslationEvent(
            text=translation.text,
            source_language=translation.source_language,
            target_language=translation.target_language,
            is_final=event.is_final,
            chunk_index=event.chunk_index,
            start_ms=event.start_ms,
            end_ms=event.end_ms,
        )

