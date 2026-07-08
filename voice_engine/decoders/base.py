from __future__ import annotations

from abc import ABC, abstractmethod

from voice_engine.pipeline.types import StreamAudioChunk, VoiceDecoderInput


from collections.abc import Iterator

class StreamingVoiceDecoder(ABC):
    @abstractmethod
    def decode(self, decoder_input: VoiceDecoderInput) -> Iterator[StreamAudioChunk]:
        raise NotImplementedError

