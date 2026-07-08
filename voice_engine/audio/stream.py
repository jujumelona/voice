from __future__ import annotations

import queue
import sys
import tempfile
import threading
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from voice_engine.audio.devices import find_device
from voice_engine.pipeline.types import SpeechChunk, StreamAudioChunk


@dataclass(frozen=True)
class LiveAudioConfig:
    input_device: str | int | None
    output_device: str | int | None
    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 2000


class LiveAudioBridge:
    def __init__(self, config: LiveAudioConfig) -> None:
        self.config = config
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Install audio support: python -m pip install .[audio]") from exc
        self._sd = sd
        self._input_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._output_buffer = np.array([], dtype=np.float32)
        self._play_cursor = 0
        self._lock = threading.Lock()
        self._input_stream: sd.InputStream | None = None
        self._output_stream: sd.OutputStream | None = None

    def start(self) -> None:
        """Starts the background non-blocking input and output streams."""
        input_device = find_device(self.config.input_device, need_input=True)
        output_device = find_device(self.config.output_device, need_output=True)

        self._input_stream = self._sd.InputStream(
            device=input_device,
            channels=self.config.channels,
            samplerate=self.config.sample_rate,
            callback=self._input_callback,
            dtype="float32",
        )
        self._output_stream = self._sd.OutputStream(
            device=output_device,
            channels=self.config.channels,
            samplerate=self.config.sample_rate,
            callback=self._output_callback,
            dtype="float32",
        )
        self._input_stream.start()
        self._output_stream.start()

    def stop(self) -> None:
        """Stops the audio streams and releases device resources."""
        if self._input_stream:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
        if self._output_stream:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None

    def _input_callback(self, indata: np.ndarray, frames: int, time: object, status: object) -> None:
        if status:
            print(f"Audio input error status: {status}", file=sys.stderr)
        self._input_queue.put(indata.copy())

    def _output_callback(self, outdata: np.ndarray, frames: int, time: object, status: object) -> None:
        if status:
            print(f"Audio output error status: {status}", file=sys.stderr)
        with self._lock:
            available = len(self._output_buffer) - self._play_cursor
            if available >= frames:
                outdata[:, 0] = self._output_buffer[self._play_cursor : self._play_cursor + frames]
                self._play_cursor += frames
            else:
                if available > 0:
                    outdata[:available, 0] = self._output_buffer[self._play_cursor :]
                    outdata[available:, 0] = 0.0
                    self._play_cursor += available
                else:
                    outdata.fill(0.0)

            # Periodically truncate the played-out parts of the buffer
            if self._play_cursor > 160000:  # Truncate after ~10 seconds of speech
                self._output_buffer = self._output_buffer[self._play_cursor :]
                self._play_cursor = 0

    def stream_samples(self) -> Iterator[np.ndarray]:
        """Yields raw audio sample chunks from the background input stream as they arrive."""
        while self._input_stream and self._input_stream.active:
            try:
                data = self._input_queue.get(timeout=0.1)
                # Convert multi-channel to mono if necessary
                if data.ndim > 1:
                    data = data.mean(axis=1)
                yield data
            except queue.Empty:
                continue

    def play(self, chunk: StreamAudioChunk) -> None:
        """Appends new speech samples to the output playback buffer.

        Decoder output sample rates may differ from the live audio device rate
        when the post-TTS adapter is run at 24kHz for speed. Resample here once
        at the output boundary instead of forcing every decoder to match the
        device rate.
        """
        samples = np.array(chunk.samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if int(chunk.sample_rate) != int(self.config.sample_rate):
            samples = _resample_audio(samples, int(chunk.sample_rate), int(self.config.sample_rate))

        with self._lock:
            remaining = len(self._output_buffer) - self._play_cursor
            if remaining <= 0:
                self._output_buffer = samples
                self._play_cursor = 0
            else:
                # Apply crossfade at the boundary (80ms fade width @ 16kHz is ~1280 samples)
                fade_width = min(int(self.config.sample_rate * 0.08), remaining, len(samples))
                if fade_width > 0:
                    fade_out = np.linspace(1.0, 0.0, fade_width)
                    fade_in = np.linspace(0.0, 1.0, fade_width)

                    overlap_start = self._play_cursor + remaining - fade_width
                    overlap_orig = self._output_buffer[overlap_start : overlap_start + fade_width]
                    overlap_new = samples[:fade_width]

                    mixed = (overlap_orig * fade_out) + (overlap_new * fade_in)
                    self._output_buffer[overlap_start : overlap_start + fade_width] = mixed

                    # Append the rest of the new chunk
                    self._output_buffer = np.concatenate([
                        self._output_buffer[: overlap_start + fade_width],
                        samples[fade_width:],
                    ])
                else:
                    self._output_buffer = np.concatenate([self._output_buffer, samples])

    # Synchronous chunk API for scripts that do not need the callback stream.
    def input_chunks(self) -> Iterator[SpeechChunk]:
        """Synchronously record chunk_ms blocks of audio."""
        input_device = find_device(self.config.input_device, need_input=True)
        frames_per_chunk = max(1, int(self.config.sample_rate * self.config.chunk_ms / 1000))
        chunk_index = 0
        start_frame = 0
        runtime_dir = Path(".voice_bridge_runtime") / "live_chunks"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        while True:
            recording = self._sd.rec(
                frames_per_chunk,
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype="float32",
                device=input_device,
            )
            self._sd.wait()
            chunk_path = runtime_dir / f"mic_{chunk_index:06d}.wav"
            _write_float_wav(chunk_path, recording, self.config.sample_rate)
            end_frame = start_frame + frames_per_chunk
            yield SpeechChunk(
                path=chunk_path,
                sample_rate=self.config.sample_rate,
                start_sec=start_frame / self.config.sample_rate,
                end_sec=end_frame / self.config.sample_rate,
            )
            start_frame = end_frame
            chunk_index += 1


def _resample_audio(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if int(src_sr) == int(dst_sr):
        return samples.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(int(src_sr), int(dst_sr))
        return resample_poly(samples, int(dst_sr) // g, int(src_sr) // g).astype(np.float32)
    except Exception:
        try:
            import librosa
            return librosa.resample(samples.astype(np.float32), orig_sr=int(src_sr), target_sr=int(dst_sr)).astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"cannot resample output audio {src_sr}->{dst_sr}") from exc


def _write_float_wav(path: Path, frames: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for frame in frames:
            if hasattr(frame, "__len__"):
                sample = float(frame[0])
            else:
                sample = float(frame)
            sample = max(-1.0, min(1.0, sample))
            wav.writeframes(int(sample * 32767.0).to_bytes(2, "little", signed=True))
