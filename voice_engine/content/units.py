from __future__ import annotations

import re

from voice_engine.pipeline.types import ContentUnits, Translation


def content_units_from_translation(translation: Translation) -> ContentUnits:
    return content_units_from_text(
        translation.text,
        language=translation.target_language or "und",
    )


def content_units_from_text(text: str, language: str) -> ContentUnits:
    units = _word_or_phrase_units(text)
    return ContentUnits(
        language=language,
        text=text,
        phonemes=[],
        semantic_tokens=[],
        units=units,
        unit_type="word_or_phrase",
    )


def _word_or_phrase_units(text: str) -> list[str]:
    units = re.findall(r"[가-힣]+|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s]", text)
    return [unit for unit in units if unit.strip()]
