from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from voice_engine.asr.base import ASRAdapter, StreamingASRAdapter
from voice_engine.asr.whisper_cpp import _split_wav_pcm16
from voice_engine.pipeline.types import SpeechChunk, Transcript, TranscriptEvent, WordTiming


import numpy as np

class FasterWhisperASR(ASRAdapter):
    def __init__(
        self,
        model_size_or_path: str = "small",
        language: str | None = None,
        device: str = "auto",
        compute_type: str = "default",
        download_root: str | None = None,
    ) -> None:
        self.model_size_or_path = model_size_or_path
        self.language = language
        self.download_root = download_root
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install optional dependency: "
                "python -m pip install .[balanced]"
            ) from exc
        self.model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )

    def transcribe(self, chunk: SpeechChunk) -> Transcript:
        segments, info = self.model.transcribe(
            str(chunk.path),
            language=self.language,
            word_timestamps=True,
        )
        return self._process_segments(segments, info)

    def transcribe_samples(self, samples: np.ndarray) -> Transcript:
        segments, info = self.model.transcribe(
            samples,
            language=self.language,
            word_timestamps=True,
        )
        return self._process_segments(segments, info)

    def _process_segments(self, segments, info) -> Transcript:
        segment_list = list(segments)
        text = " ".join(segment.text.strip() for segment in segment_list).strip()
        words: list[WordTiming] = []
        for segment in segment_list:
            for word in getattr(segment, "words", None) or []:
                token = str(getattr(word, "word", "")).strip()
                if not token:
                    continue
                words.append(
                    WordTiming(
                        text=token,
                        start_ms=float(getattr(word, "start", 0.0)) * 1000.0,
                        end_ms=float(getattr(word, "end", 0.0)) * 1000.0,
                        probability=getattr(word, "probability", None),
                    )
                )
        language = getattr(info, "language", None) or self.language
        probability = getattr(info, "language_probability", None)
        return Transcript(text=text, language=language, confidence=probability, words=words)


class FasterWhisperStreamingASR(StreamingASRAdapter):
    def __init__(self, batch_asr: FasterWhisperASR) -> None:
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
        import tempfile

        with tempfile.TemporaryDirectory(prefix="voice_engine_fw_") as tmp_dir:
            chunks = list(_split_wav_pcm16(Path(path), Path(tmp_dir), chunk_ms=chunk_ms))
            yield from self.transcribe_stream(chunks)

    def transcribe_samples_stream(
        self,
        samples_generator: Iterator[np.ndarray],
    ) -> Iterator[TranscriptEvent]:
        from voice_engine.audio.vad import AudioVADSegmenter

        segmenter = AudioVADSegmenter(sample_rate=16000)  # standard sample rate
        chunk_index = 0

        for samples in samples_generator:
            events = segmenter.push(samples)
            for speech_samples, is_final in events:
                if len(speech_samples) == 0:
                    continue
                if not is_final:
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
