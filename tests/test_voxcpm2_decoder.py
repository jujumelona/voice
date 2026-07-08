from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from voice_engine.decoders.voxcpm2_decoder import VoxCPM2Decoder
from voice_engine.pipeline.types import ContentUnits, VoiceDecoderInput


class _FakeTTSModel:
    sample_rate = 16000


class _FakeVoxCPM2:
    tts_model = _FakeTTSModel()

    def generate(self, **kwargs):
        if kwargs["text"] != "hello":
            raise AssertionError(kwargs)
        if not Path(kwargs["reference_wav_path"]).exists():
            raise AssertionError(kwargs)
        return np.ones(1600, dtype=np.float32) * 0.05


class VoxCPM2DecoderTest(unittest.TestCase):
    def test_decode_uses_loaded_model_and_chunks_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.wav"
            _write_empty_wav(ref)

            decoder = VoxCPM2Decoder(chunk_ms=50)
            decoder._model = _FakeVoxCPM2()
            chunks = list(decoder.decode(_decoder_input("hello", ref)))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].sample_rate, 16000)
        self.assertEqual(len(chunks[0].samples), 800)

    def test_missing_reference_audio_is_an_error(self) -> None:
        decoder = VoxCPM2Decoder()
        decoder._model = _FakeVoxCPM2()
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
