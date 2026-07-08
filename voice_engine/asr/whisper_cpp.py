from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from collections.abc import Iterable, Iterator
from pathlib import Path

from voice_engine.asr.base import ASRAdapter, StreamingASRAdapter
from voice_engine.paths import model_path
from voice_engine.pipeline.types import SpeechChunk, Transcript, TranscriptEvent, WordTiming


import numpy as np
from voice_engine.audio.vad import AudioVADSegmenter


class WhisperCppASR(ASRAdapter):
    def __init__(
        self,
        binary: str | Path = "whisper-cli",
        model: str | Path | None = None,
        language: str | None = None,
    ) -> None:
        self.binary = str(binary)
        self.model = str(model or model_path("whisper", "ggml-base.bin"))
        self.language = language

    def transcribe(self, chunk: SpeechChunk) -> Transcript:
        out_prefix = chunk.path.with_suffix("")
        command = [
            self.binary,
            "-m",
            self.model,
            "-f",
            str(chunk.path),
            "-oj",
            "-of",
            str(out_prefix),
        ]
        if self.language:
            command.extend(["-l", self.language])

        subprocess.run(command, check=True)
        json_path = out_prefix.with_suffix(".json")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        text = _extract_text(data)
        language = data.get("result", {}).get("language") or self.language
        words = _extract_word_timings(data, text)
        return Transcript(text=text.strip(), language=language, words=words)

    def transcribe_samples(self, samples: np.ndarray) -> Transcript:
        with tempfile.TemporaryDirectory(prefix="voice_engine_wcpp_tmp_") as tmp:
            tmp_path = Path(tmp) / "input.wav"
            _write_wav_pcm16(tmp_path, samples, 16000)
            return self.transcribe(SpeechChunk(path=tmp_path, sample_rate=16000))


class WhisperCppStreamingASR(StreamingASRAdapter):
    """Chunked streaming adapter for whisper.cpp.

    This keeps whisper.cpp out of the repository. For live mic mode, use
    whisper.cpp's local binaries and feed SpeechChunk windows from the audio
    layer. This class intentionally uses `whisper-cli` per chunk as a portable
    integration baseline; a lower-latency implementation can swap in
    `whisper-stream` or a persistent server later.
    """

    def __init__(self, batch_asr: WhisperCppASR) -> None:
        self.batch_asr = batch_asr

    def transcribe_stream(self, chunks: Iterable[SpeechChunk]) -> Iterator[TranscriptEvent]:
        for index, chunk in enumerate(chunks):
            transcript = self.batch_asr.transcribe(chunk)
            yield TranscriptEvent(
                text=transcript.text,
                language=transcript.language,
                is_final=True,
                chunk_index=index,
                start_ms=chunk.start_sec * 1000.0,
                end_ms=(chunk.end_sec if chunk.end_sec is not None else chunk.start_sec) * 1000.0,
            )

    def transcribe_file_stream(
        self,
        path: str | Path,
        chunk_ms: int = 2000,
    ) -> Iterator[TranscriptEvent]:
        source = Path(path)
        with tempfile.TemporaryDirectory(prefix="voice_engine_whisper_") as tmp_dir:
            chunks = list(_split_wav_pcm16(source, Path(tmp_dir), chunk_ms=chunk_ms))
            yield from self.transcribe_stream(chunks)

    def transcribe_samples_stream(
        self,
        samples_generator: Iterator[np.ndarray],
    ) -> Iterator[TranscriptEvent]:
        segmenter = AudioVADSegmenter(sample_rate=16000)
        chunk_index = 0

        for samples in samples_generator:
            events = segmenter.push(samples)
            for speech_samples, is_final in events:
                if len(speech_samples) == 0:
                    continue
                transcript = self.batch_asr.transcribe_samples(speech_samples)
                yield TranscriptEvent(
                    text=transcript.text,
                    language=transcript.language,
                    is_final=is_final,
                    chunk_index=chunk_index,
                    start_ms=0.0,
                    end_ms=len(speech_samples) / 16000.0 * 1000.0,
                    samples=speech_samples,
                )

                if is_final:
                    chunk_index += 1


def _write_wav_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for sample in samples:
            val = max(-1.0, min(1.0, float(sample)))
            wav.writeframes(int(val * 32767.0).to_bytes(2, "little", signed=True))


def _extract_text(data: dict) -> str:
    transcription = data.get("transcription", "")
    if isinstance(transcription, str):
        return transcription
    if isinstance(transcription, list):
        return " ".join(
            str(segment.get("text", "")).strip()
            for segment in transcription
            if isinstance(segment, dict)
        )
    return ""


def _extract_word_timings(data: dict, text: str) -> list[WordTiming]:
    words: list[WordTiming] = []
    transcription = data.get("transcription", [])
    if not isinstance(transcription, list):
        return words

    for segment in transcription:
        if not isinstance(segment, dict):
            continue
        segment_words = segment.get("words")
        if isinstance(segment_words, list):
            for item in segment_words:
                if not isinstance(item, dict):
                    continue
                token = str(item.get("word") or item.get("text") or "").strip()
                start = _timestamp_to_ms(item.get("start") or item.get("from"))
                end = _timestamp_to_ms(item.get("end") or item.get("to"))
                if token and start is not None and end is not None:
                    words.append(WordTiming(text=token, start_ms=start, end_ms=end))
        if words:
            continue
        segment_text = str(segment.get("text", "")).strip()
        timestamps = segment.get("timestamps") if isinstance(segment.get("timestamps"), dict) else {}
        start = _timestamp_to_ms(timestamps.get("from") or segment.get("from") or segment.get("start"))
        end = _timestamp_to_ms(timestamps.get("to") or segment.get("to") or segment.get("end"))
        if segment_text and start is not None and end is not None:
            words.extend(_spread_words(segment_text, start, end))
    if words:
        return words
    return _spread_words(text, 0.0, 0.0)


def _timestamp_to_ms(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number * 1000.0 if number < 10_000 else number
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2].replace(",", "."))
            return (hours * 3600.0 + minutes * 60.0 + seconds) * 1000.0
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1].replace(",", "."))
            return (minutes * 60.0 + seconds) * 1000.0
        number = float(text.replace(",", "."))
    except ValueError:
        return None
    return number * 1000.0 if number < 10_000 else number


def _spread_words(text: str, start_ms: float, end_ms: float) -> list[WordTiming]:
    tokens = [token for token in text.split() if token]
    if not tokens:
        return []
    if end_ms <= start_ms:
        end_ms = start_ms + 280.0 * len(tokens)
    step = (end_ms - start_ms) / len(tokens)
    return [
        WordTiming(text=token, start_ms=start_ms + index * step, end_ms=start_ms + (index + 1) * step)
        for index, token in enumerate(tokens)
    ]


def _split_wav_pcm16(path: Path, tmp_dir: Path, chunk_ms: int) -> Iterator[SpeechChunk]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        total_frames = wav.getnframes()
        if sample_width != 2:
            raise ValueError("whisper.cpp streaming demo expects PCM16 wav chunks")

        frames_per_chunk = max(1, int(sample_rate * chunk_ms / 1000))
        chunk_index = 0
        start_frame = 0
        while start_frame < total_frames:
            wav.setpos(start_frame)
            data = wav.readframes(min(frames_per_chunk, total_frames - start_frame))
            if not data:
                break
            chunk_path = tmp_dir / f"chunk_{chunk_index:05d}.wav"
            with wave.open(str(chunk_path), "wb") as out:
                out.setnchannels(channels)
                out.setsampwidth(sample_width)
                out.setframerate(sample_rate)
                out.writeframes(data)
            end_frame = start_frame + len(data) // (channels * sample_width)
            yield SpeechChunk(
                path=chunk_path,
                sample_rate=sample_rate,
                start_sec=start_frame / sample_rate,
                end_sec=end_frame / sample_rate,
            )
            start_frame = end_frame
            chunk_index += 1
