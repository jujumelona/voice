from __future__ import annotations

import numpy as np


class EnergyVAD:
    """Light-weight energy-based Voice Activity Detection using RMS thresholding."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        threshold_db: float = -38.0,
        speech_lead_frames: int = 4,      # ~120ms of speech to trigger SPEECH state
        silence_tail_frames: int = 25,    # ~750ms of silence to trigger SILENCE state
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.threshold_amplitude = 10 ** (threshold_db / 20.0)
        self.speech_lead_frames = speech_lead_frames
        self.silence_tail_frames = silence_tail_frames

        self.is_speech = False
        self.active_frames = 0
        self.silent_frames = 0

    def is_speech_active(self, frame: np.ndarray) -> bool:
        """Processes a frame of audio samples and updates the VAD state.

        Args:
            frame: 1D float32 numpy array representing audio samples.

        Returns:
            True if the VAD detects that speech is currently active, False otherwise.
        """
        if len(frame) == 0:
            return self.is_speech

        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        is_active = rms > self.threshold_amplitude

        if is_active:
            self.silent_frames = 0
            if not self.is_speech:
                self.active_frames += 1
                if self.active_frames >= self.speech_lead_frames:
                    self.is_speech = True
                    self.active_frames = 0
        else:
            self.active_frames = 0
            if self.is_speech:
                self.silent_frames += 1
                if self.silent_frames >= self.silence_tail_frames:
                    self.is_speech = False
                    self.silent_frames = 0

        return self.is_speech


class AudioVADSegmenter:
    """Buffers continuous raw float32 audio and groups them into speech segments based on VAD state transitions."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        threshold_db: float = -38.0,
        speech_lead_frames: int = 4,
        silence_tail_frames: int = 25,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.vad = EnergyVAD(
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            threshold_db=threshold_db,
            speech_lead_frames=speech_lead_frames,
            silence_tail_frames=silence_tail_frames,
        )
        self._buffer = np.array([], dtype=np.float32)
        self._speech_buffer: list[np.ndarray] = []
        self._was_speech = False

    def push(self, samples: np.ndarray) -> list[tuple[np.ndarray, bool]]:
        """Pushes new raw samples, processes them by VAD frames, and returns speech segment events.

        Returns:
            A list of tuples: (speech_samples_chunk, is_final)
            - speech_samples_chunk: A piece of active speech.
            - is_final: True if this chunk concludes a complete speech utterance (sentence end).
        """
        self._buffer = np.concatenate([self._buffer, samples])
        events: list[tuple[np.ndarray, bool]] = []

        while len(self._buffer) >= self.frame_size:
            frame = self._buffer[:self.frame_size]
            self._buffer = self._buffer[self.frame_size:]

            is_speech = self.vad.is_speech_active(frame)

            if is_speech:
                self._speech_buffer.append(frame)
                self._was_speech = True
            else:
                if self._was_speech:
                    # Speech has just ended
                    if self._speech_buffer:
                        full_segment = np.concatenate(self._speech_buffer)
                        self._speech_buffer.clear()
                        events.append((full_segment, True))
                    self._was_speech = False
                # If silent and not speech, we discard the silent frame (or could emit as partial empty, but we skip)

        # Output partial segments if still speaking to allow ASR partial streaming
        if self._was_speech and self._speech_buffer:
            # Return current accumulated speech as non-final (partial) event
            # We don't clear the buffer yet, so we just return a copy of what's there so far
            # To avoid sending the duplicate prefix, we can send only the newly added frames
            # but standard streaming ASR often takes the whole prefix.
            # To make it simple: we emit the full accumulated buffer with is_final=False.
            events.append((np.concatenate(self._speech_buffer), False))

        return events
