from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from voice_engine.pipeline.types import SpeechChunk, Transcript, TranscriptEvent


import numpy as np

class ASRAdapter(ABC):
    @abstractmethod
    def transcribe(self, chunk: SpeechChunk) -> Transcript:
        raise NotImplementedError


class StreamingASRAdapter(ABC):
    @abstractmethod
    def transcribe_stream(self, chunks: Iterable[SpeechChunk]) -> Iterator[TranscriptEvent]:
        raise NotImplementedError

    @abstractmethod
    def transcribe_samples_stream(
        self,
        samples_generator: Iterator[np.ndarray],
    ) -> Iterator[TranscriptEvent]:
        raise NotImplementedError
