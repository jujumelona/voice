from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from voice_engine.decoders.qwen3_tts_decoder import Qwen3TTSDecoder
from voice_engine.pipeline.types import ContentUnits, VoiceDecoderInput


class _FakeQwen3TTS:
    def create_voice_clone_prompt(self, **kwargs):
        if not Path(kwargs["ref_audio"]).exists():
            raise AssertionError(kwargs)
        return {"prompt": "ok"}

    def generate_voice_clone(self, **kwargs):
        if kwargs["text"] != "hello":
            raise AssertionError(kwargs)
        if kwargs["language"] != "English":
            raise AssertionError(kwargs)
        return [np.ones(1600, dtype=np.float32) * 0.05], 16000


class Qwen3TTSDecoderTest(unittest.TestCase):
    def test_decode_uses_loaded_model_and_chunks_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.wav"
            _write_empty_wav(ref)

            decoder = Qwen3TTSDecoder(chunk_ms=50)
            decoder._model = _FakeQwen3TTS()
            chunks = list(decoder.decode(_decoder_input("hello", ref)))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].sample_rate, 16000)
        self.assertEqual(len(chunks[0].samples), 800)

    def test_missing_reference_audio_is_an_error(self) -> None:
        decoder = Qwen3TTSDecoder()
        decoder._model = _FakeQwen3TTS()
        with self.assertRaises(ValueError):
            list(decoder.decode(_decoder_input("hello", None)))


def _decoder_input(text: str, ref: Path | None) -> VoiceDecoderInput:
    return VoiceDecoderInput(
        content=ContentUnits(
            language="en",
            text=text,
            phonemes=[],
            semantic_tokens=[],
            units=[text],
        ),
        speaker=None,
        prosody=None,
        reference_audio_path=str(ref) if ref else None,
    )


def _write_empty_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 160)


if __name__ == "__main__":
    unittest.main()
