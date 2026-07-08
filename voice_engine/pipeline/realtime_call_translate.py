from __future__ import annotations

import argparse
import json
import platform
import queue
import shutil
import threading
import uuid
import wave
from dataclasses import asdict
from pathlib import Path
from collections.abc import Iterator
import numpy as np

from voice_engine.asr.faster_whisper import FasterWhisperASR, FasterWhisperStreamingASR
from voice_engine.asr.whisper_cpp import WhisperCppASR, WhisperCppStreamingASR
from voice_engine.audio.devices import list_audio_devices
from voice_engine.audio.routing import load_call_routing_config, validate_call_routing
from voice_engine.audio.stream import LiveAudioBridge, LiveAudioConfig
from voice_engine.content.units import content_units_from_translation
from voice_engine.mt.argos import ArgosTranslation
from voice_engine.mt.marian import MarianTranslation
from voice_engine.paths import model_path
from voice_engine.pipeline.modes import MODES, get_mode
from voice_engine.pipeline.types import Transcript, TranscriptEvent, TranslationEvent, VoiceDecoderInput
from voice_engine.speaker.speaker_profile import (
    extract_speaker_profile_from_samples,
    extract_speaker_profile_from_wav,
    update_speaker_profile,
)
from voice_engine.speaker.style_trace import (
    extract_utterance_style_trace,
    extract_utterance_style_trace_from_samples,
)
from voice_engine.decoders.base import StreamingVoiceDecoder


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time speech translation bridge")
    parser.add_argument("--config", default=Path("configs/call_mode.example.json"), type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list-audio-devices", action="store_true")
    parser.add_argument("--demo-wav", type=Path)
    parser.add_argument("--direction", choices=["outbound", "inbound", "both"], default="outbound")
    parser.add_argument("--mode", choices=sorted(MODES), default="fast")

    # ASR / MT
    parser.add_argument("--asr", choices=["whisper", "faster-whisper"])
    parser.add_argument("--mt", choices=["argos", "marian"])
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--target-language")
    parser.add_argument("--whisper-binary", default=_default_backend_binary("whisper-cli.exe", "whisper-cli"))
    parser.add_argument("--whisper-model", default=str(model_path("whisper", "ggml-base.bin")))
    parser.add_argument("--faster-whisper-model", default="small")
    parser.add_argument("--faster-whisper-device", default="auto")
    parser.add_argument("--faster-whisper-compute-type", default="default")
    parser.add_argument("--marian-binary", default=_default_backend_binary("marian-decoder.exe", "marian-decoder"))
    parser.add_argument("--marian-model", default=str(model_path("marian", "model.npz")))
    parser.add_argument("--marian-vocab", default=str(model_path("marian", "vocab.yml")))

    # Decoder
    parser.add_argument("--decoder", choices=["auto", "none", "voxcpm2", "qwen3-tts"])
    parser.add_argument("--voxcpm2-model-dir", type=Path)
    parser.add_argument("--qwen3-tts-model-dir", type=Path)
    parser.add_argument("--qwen3-tts-python", type=Path)
    parser.add_argument("--qwen3-tts-timeout-sec", type=int, default=300)
    parser.add_argument("--voxcpm2-cfg-value", type=float, default=1.0)
    parser.add_argument("--voxcpm2-inference-timesteps", type=int, default=10)
    parser.add_argument("--voxcpm2-max-reference-sec", type=float, default=3.0)
    parser.add_argument("--voxcpm2-max-segment-chars", type=int, default=24)
    parser.add_argument("--voxcpm2-fast-backend")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--voice-adapter", choices=["none", "spectral_delta"], default="spectral_delta")
    parser.add_argument("--voice-adapter-strength", type=float, default=0.80)
    parser.add_argument("--voice-adapter-max-gain-db", type=float, default=12.0)
    parser.add_argument("--voice-adapter-window-ms", type=int, default=900)
    parser.add_argument("--voice-adapter-sample-rate", type=int, default=24000)

    # Live/test separation: live mode is clean by default; artifacts are written only when requested.
    parser.add_argument("--test-mode", action="store_true", help="Save debug artifacts/JSONL for offline quality tests. Default live mode saves nothing.")
    parser.add_argument("--live-artifacts-dir", type=Path, default=Path(".voice_bridge_runtime/test_artifacts"))

    # Audio devices
    parser.add_argument("--chunk-ms", type=int, default=2000)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--input-device")
    parser.add_argument("--output-device")
    parser.add_argument("--outbound-input-device")
    parser.add_argument("--outbound-output-device")
    parser.add_argument("--inbound-input-device")
    parser.add_argument("--inbound-output-device")
    parser.add_argument("--outbound-source-language")
    parser.add_argument("--outbound-target-language")
    parser.add_argument("--inbound-source-language")
    parser.add_argument("--inbound-target-language")

    # Speaker profile
    parser.add_argument("--speaker-encoder-device", default="auto")
    parser.add_argument("--speaker-encoder-cache-dir", type=Path, default=model_path("wespeaker"))
    parser.add_argument("--speaker-profile-update-interval-sec", type=float, default=4.0)
    parser.add_argument("--speaker-profile-sync-first", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    if args.list_audio_devices:
        print(json.dumps(list_audio_devices(), ensure_ascii=False, indent=2))
        return

    selected = _selected_runtime(args.mode, args.asr, args.mt, args.decoder)

    config = load_call_routing_config(args.config)
    errors = validate_call_routing(config)
    if errors:
        raise SystemExit("Invalid call routing:\n" + "\n".join(f"- {e}" for e in errors))

    summary = {
        "status": "valid",
        "mode": selected["mode"],
        "asr": selected["asr"],
        "translation": selected["mt"],
        "decoder": _describe_decoder(selected["decoder"], args.device),
        "voice_adapter": args.voice_adapter,
        "test_mode": args.test_mode,
        "outbound": {
            "capture": config.outbound.source_microphone,
            "process": _process_summary(selected["decoder"], args.device, args.voice_adapter),
            "send_to_call": config.outbound.call_input_virtual_mic,
        },
        "inbound": {
            "capture": config.inbound.call_output_loopback,
            "send_to_user": config.inbound.local_monitor_output,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.validate_only:
        return

    target_language = args.target_language or config.outbound.target_language

    if args.demo_wav:
        for event in _run_demo(args, selected, target_language):
            print(json.dumps(asdict(event), ensure_ascii=False))
        return

    jobs = _live_jobs(args, config, selected)
    if len(jobs) == 1:
        run_live_call(**jobs[0])
    else:
        _run_live_jobs(jobs)


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

def _run_demo(args, selected, target_language):
    translator = _translator(selected["mt"], args)
    asr = _streaming_asr(selected["asr"], args)
    for event in asr.transcribe_file_stream(args.demo_wav, chunk_ms=args.chunk_ms):
        if not event.text.strip():
            continue
        yield _translate_event(translator, event, target_language)


# ---------------------------------------------------------------------------
# Live call loop (real-time streaming)
# ---------------------------------------------------------------------------

def run_live_call(
    selected: dict[str, str],
    source_language: str,
    target_language: str,
    input_device: str | int | None,
    output_device: str | int | None,
    sample_rate: int,
    chunk_ms: int,
    speaker_encoder_device: str,
    speaker_encoder_cache_dir: Path,
    speaker_profile_update_interval_sec: float,
    speaker_profile_sync_first: bool = True,
    test_mode: bool = False,
    live_artifacts_dir: Path | None = None,
    voxcpm2_model_dir: Path | None = None,
    qwen3_tts_model_dir: Path | None = None,
    qwen3_tts_python: Path | None = None,
    qwen3_tts_timeout_sec: int = 300,
    voxcpm2_cfg_value: float = 1.0,
    voxcpm2_inference_timesteps: int = 10,
    voxcpm2_max_reference_sec: float = 3.0,
    voxcpm2_max_segment_chars: int = 24,
    voxcpm2_fast_backend: str | None = None,
    device: str = "auto",
    voice_adapter: str = "spectral_delta",
    voice_adapter_strength: float = 0.80,
    voice_adapter_max_gain_db: float = 12.0,
    voice_adapter_window_ms: int = 900,
    voice_adapter_sample_rate: int | None = 24000,
    # ASR params forwarded
    whisper_binary: str = "whisper-cli",
    whisper_model: str = "",
    faster_whisper_model: str = "small",
    faster_whisper_device: str = "auto",
    faster_whisper_compute_type: str = "default",
    # MT params forwarded
    marian_binary: str = "marian-decoder",
    marian_model: str = "",
    marian_vocab: str = "",
) -> None:
    translator = _translator_direct(
        selected["mt"],
        source_language=source_language,
        marian_binary=marian_binary,
        marian_model=marian_model,
        marian_vocab=marian_vocab,
    )
    asr = _streaming_asr_direct(
        selected["asr"],
        source_language=source_language,
        whisper_binary=whisper_binary,
        whisper_model=whisper_model,
        faster_whisper_model=faster_whisper_model,
        faster_whisper_device=faster_whisper_device,
        faster_whisper_compute_type=faster_whisper_compute_type,
    )
    audio = LiveAudioBridge(
        LiveAudioConfig(
            input_device=input_device,
            output_device=output_device,
            sample_rate=sample_rate,
            chunk_ms=chunk_ms,
        )
    )
    decoder = _make_decoder(
        selected["decoder"],
        voxcpm2_model_dir=voxcpm2_model_dir,
        qwen3_tts_model_dir=qwen3_tts_model_dir,
        qwen3_tts_python=qwen3_tts_python,
        qwen3_tts_timeout_sec=qwen3_tts_timeout_sec,
        voxcpm2_cfg_value=voxcpm2_cfg_value,
        voxcpm2_inference_timesteps=voxcpm2_inference_timesteps,
        voxcpm2_max_reference_sec=voxcpm2_max_reference_sec,
        voxcpm2_max_segment_chars=voxcpm2_max_segment_chars,
        voxcpm2_fast_backend=voxcpm2_fast_backend,
        device=device,
        voice_adapter=voice_adapter,
        voice_adapter_strength=voice_adapter_strength,
        voice_adapter_max_gain_db=voice_adapter_max_gain_db,
        voice_adapter_window_ms=voice_adapter_window_ms,
        voice_adapter_sample_rate=voice_adapter_sample_rate,
    )

    turn_queue: queue.Queue[TranscriptEvent | object] | None = None
    worker_stop: object | None = None
    synth_thread: threading.Thread | None = None
    async_profile: _AsyncSpeakerProfileUpdater | None = None

    audio.start()
    try:
        speaker_profile = None
        last_update_sec: float | None = None
        reference_cache = _CallReferenceCache(sample_rate=sample_rate, max_sec=voxcpm2_max_reference_sec)
        test_artifacts = _LiveTestArtifacts(live_artifacts_dir) if test_mode else None
        async_profile = _AsyncSpeakerProfileUpdater(
            sample_rate=sample_rate,
            device=speaker_encoder_device,
            cache_dir=speaker_encoder_cache_dir,
            enabled=not speaker_profile_sync_first,
        )

        turn_queue = queue.Queue(maxsize=2)
        worker_stop = object()
        worker_errors: list[BaseException] = []

        def synthesize_worker() -> None:
            while True:
                item = turn_queue.get()
                try:
                    if item is worker_stop:
                        return
                    _process_final_event(item)
                except BaseException as exc:
                    worker_errors.append(exc)
                finally:
                    turn_queue.task_done()

        def _process_final_event(event: TranscriptEvent) -> None:
            nonlocal speaker_profile, last_update_sec
            if event.samples is None or len(event.samples) == 0:
                return

            event_samples = _mono_float_array(event.samples)
            event_sample_list = event_samples.astype(np.float32).tolist()
            t = event.end_ms / 1000.0
            profile_due = (
                speaker_profile is None
                or last_update_sec is None
                or t - last_update_sec >= speaker_profile_update_interval_sec
            )

            if profile_due or reference_cache.current_path is None:
                reference_path = reference_cache.update(event_samples)
            else:
                reference_path = reference_cache.current_path

            if profile_due:
                if speaker_profile is None and speaker_profile_sync_first:
                    # First profile can be synchronous so voice cloning has a stable identity.
                    new_profile = extract_speaker_profile_from_samples(
                        event_sample_list,
                        sample_rate,
                        wav_path=reference_path,
                        device=speaker_encoder_device,
                        cache_dir=speaker_encoder_cache_dir,
                    )
                    speaker_profile = new_profile
                    last_update_sec = t
                else:
                    async_profile.submit(event_samples, reference_path)
                    ready_profile = async_profile.take_ready()
                    if ready_profile is not None:
                        speaker_profile = (
                            ready_profile
                            if speaker_profile is None
                            else update_speaker_profile(speaker_profile, ready_profile, alpha=0.02)
                        )
                        last_update_sec = t

            ready_profile = async_profile.take_ready()
            if ready_profile is not None:
                speaker_profile = (
                    ready_profile
                    if speaker_profile is None
                    else update_speaker_profile(speaker_profile, ready_profile, alpha=0.02)
                )

            if speaker_profile is None:
                return

            print(f"\n[source] {event.text}", flush=True)
            translation = translator.translate(
                Transcript(text=event.text, language=event.language),
                target_language,
            )
            print(f"[translated] {translation.text}", flush=True)

            content = content_units_from_translation(translation)

            style_tokens = extract_utterance_style_trace_from_samples(
                samples=event_sample_list,
                sample_rate=sample_rate,
                text=event.text,
                language=event.language,
                speaker=speaker_profile,
                content=content,
            )

            if decoder is None:
                return

            request = VoiceDecoderInput(
                content=content,
                speaker=speaker_profile,
                prosody=None,
                style_tokens=style_tokens,
                reference_audio_path=str(reference_path),
                reference_audio_samples=event_sample_list,
                reference_audio_sample_rate=sample_rate,
            )
            if test_artifacts is not None:
                test_artifacts.log_turn(event.text, translation.text, str(reference_path))

            for chunk in decoder.decode(request):
                if test_artifacts is not None:
                    test_artifacts.write_chunk(chunk)
                audio.play(chunk)

        synth_thread = threading.Thread(target=synthesize_worker, daemon=True)
        synth_thread.start()

        for event in asr.transcribe_samples_stream(audio.stream_samples()):
            if worker_errors:
                raise worker_errors[0]
            if not event.text.strip():
                continue

            if not event.is_final:
                print(f"[partial] {event.text}", flush=True)
                continue

            turn_queue.put(event)
    finally:
        if turn_queue is not None and worker_stop is not None:
            try:
                turn_queue.put(worker_stop, timeout=0.5)
            except Exception:
                pass
        if synth_thread is not None and synth_thread.is_alive():
            synth_thread.join(timeout=0.5)
        try:
            if async_profile is not None:
                async_profile.close()
        except Exception:
            pass
        audio.stop()


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_decoder(
    mode: str,
    *,
    voxcpm2_model_dir: Path | None = None,
    qwen3_tts_model_dir: Path | None = None,
    qwen3_tts_python: Path | None = None,
    qwen3_tts_timeout_sec: int = 300,
    voxcpm2_cfg_value: float = 1.0,
    voxcpm2_inference_timesteps: int = 10,
    voxcpm2_max_reference_sec: float = 3.0,
    voxcpm2_max_segment_chars: int = 24,
    voxcpm2_fast_backend: str | None = None,
    device: str = "auto",
    voice_adapter: str = "spectral_delta",
    voice_adapter_strength: float = 0.80,
    voice_adapter_max_gain_db: float = 12.0,
    voice_adapter_window_ms: int = 900,
    voice_adapter_sample_rate: int | None = 24000,
) -> StreamingVoiceDecoder | None:
    if mode == "auto":
        mode = _auto_decoder(device)
    if mode == "none":
        return None
    if mode == "voxcpm2":
        from voice_engine.decoders.voxcpm2_decoder import VoxCPM2Decoder
        return _wrap_voice_adapter(
            VoxCPM2Decoder(
                model_dir=voxcpm2_model_dir,
                device=device,
                cfg_value=voxcpm2_cfg_value,
                inference_timesteps=voxcpm2_inference_timesteps,
                max_reference_sec=voxcpm2_max_reference_sec,
                max_segment_chars=voxcpm2_max_segment_chars,
                fast_backend=voxcpm2_fast_backend,
            ),
            voice_adapter=voice_adapter,
            strength=voice_adapter_strength,
            max_gain_db=voice_adapter_max_gain_db,
            window_ms=voice_adapter_window_ms,
            adapter_sample_rate=voice_adapter_sample_rate,
        )
    if mode == "qwen3-tts":
        from voice_engine.decoders.qwen3_tts_decoder import Qwen3TTSDecoder
        return _wrap_voice_adapter(
            Qwen3TTSDecoder(
                model_dir=qwen3_tts_model_dir,
                qwen_python=qwen3_tts_python,
                device=device,
                timeout_sec=qwen3_tts_timeout_sec,
            ),
            voice_adapter=voice_adapter,
            strength=voice_adapter_strength,
            max_gain_db=voice_adapter_max_gain_db,
            window_ms=voice_adapter_window_ms,
        )
    raise ValueError(f"unknown decoder: {mode}")


def _wrap_voice_adapter(
    decoder: StreamingVoiceDecoder,
    *,
    voice_adapter: str,
    strength: float,
    max_gain_db: float,
    window_ms: int = 900,
    adapter_sample_rate: int | None = 24000,
) -> StreamingVoiceDecoder:
    if voice_adapter == "none":
        return decoder
    if voice_adapter == "spectral_delta":
        try:
            from voice_engine.adapters.decoder_wrapper import SpectralDeltaDecoderAdapter
            from voice_engine.adapters.spectral_delta_adapter import SpectralDeltaConfig
        except ModuleNotFoundError as exc:
            if exc.name == "librosa":
                raise ModuleNotFoundError(
                    "spectral_delta voice adapter requires librosa. "
                    "Install requirements.txt or run: python -m pip install 'librosa>=0.11,<0.12'"
                ) from exc
            raise
        return SpectralDeltaDecoderAdapter(
            decoder,
            cfg=SpectralDeltaConfig(strength=strength, max_gain_db=max_gain_db),
            window_ms=window_ms,
            adapter_sample_rate=adapter_sample_rate,
        )
    raise ValueError(f"unknown voice adapter: {voice_adapter}")


def _translator(mode: str, args):
    return _translator_direct(
        mode,
        source_language=args.source_language,
        marian_binary=args.marian_binary,
        marian_model=args.marian_model,
        marian_vocab=args.marian_vocab,
    )


def _translator_direct(mode, *, source_language, marian_binary, marian_model, marian_vocab):
    if mode == "argos":
        return ArgosTranslation(source_language=source_language)
    if mode == "marian":
        _validate_marian(marian_binary, marian_model, marian_vocab)
        return MarianTranslation(binary=marian_binary, model=marian_model, vocab=marian_vocab)
    raise ValueError(f"unknown MT: {mode}")


def _streaming_asr(mode: str, args):
    return _streaming_asr_direct(
        mode,
        source_language=args.source_language,
        whisper_binary=args.whisper_binary,
        whisper_model=args.whisper_model,
        faster_whisper_model=args.faster_whisper_model,
        faster_whisper_device=args.faster_whisper_device,
        faster_whisper_compute_type=args.faster_whisper_compute_type,
    )


def _streaming_asr_direct(
    mode, *, source_language, whisper_binary, whisper_model,
    faster_whisper_model, faster_whisper_device, faster_whisper_compute_type,
):
    if mode == "whisper":
        _validate_whisper(whisper_binary, whisper_model)
        return WhisperCppStreamingASR(
            WhisperCppASR(binary=whisper_binary, model=whisper_model, language=source_language)
        )
    if mode == "faster-whisper":
        return FasterWhisperStreamingASR(
            FasterWhisperASR(
                model_size_or_path=faster_whisper_model,
                language=source_language,
                device=faster_whisper_device,
                compute_type=faster_whisper_compute_type,
            )
        )
    raise ValueError(f"unknown ASR: {mode}")


def _selected_runtime(mode, asr, mt, decoder):
    if mode:
        preset = get_mode(mode)
        return {
            "mode": mode,
            "asr": asr or preset.asr,
            "mt": mt or preset.translation,
            "decoder": decoder or preset.decoder,
        }
    return {"mode": "fast", "asr": asr or "whisper", "mt": mt or "argos", "decoder": decoder or "auto"}


def _process_summary(decoder: str, device: str, voice_adapter: str) -> str:
    process = f"ASR -> MT -> SpeakerProfile + StyleTrace -> {_describe_decoder(decoder, device)}"
    if voice_adapter == "spectral_delta":
        process += " -> B_spectral_delta_080"
    return process


def _auto_decoder(device: str) -> str:
    if device.startswith("cuda"):
        return "voxcpm2"
    if device == "cpu":
        return "qwen3-tts"
    try:
        import torch
    except ModuleNotFoundError:
        return "qwen3-tts"
    return "voxcpm2" if torch.cuda.is_available() else "qwen3-tts"


def _describe_decoder(decoder: str, device: str) -> str:
    if decoder == "auto":
        return f"auto({_auto_decoder(device)})"
    return decoder


# ---------------------------------------------------------------------------
# Job builder (outbound / inbound / both)
# ---------------------------------------------------------------------------

def _live_jobs(args, config, selected):
    base = {
        "selected": selected,
        "whisper_binary": args.whisper_binary,
        "whisper_model": args.whisper_model,
        "faster_whisper_model": args.faster_whisper_model,
        "faster_whisper_device": args.faster_whisper_device,
        "faster_whisper_compute_type": args.faster_whisper_compute_type,
        "marian_binary": args.marian_binary,
        "marian_model": args.marian_model,
        "marian_vocab": args.marian_vocab,
        "sample_rate": args.sample_rate,
        "chunk_ms": args.chunk_ms,
        "speaker_encoder_device": args.speaker_encoder_device,
        "speaker_encoder_cache_dir": args.speaker_encoder_cache_dir,
        "speaker_profile_update_interval_sec": args.speaker_profile_update_interval_sec,
        "speaker_profile_sync_first": args.speaker_profile_sync_first,
        "test_mode": args.test_mode,
        "live_artifacts_dir": args.live_artifacts_dir,
        "voxcpm2_model_dir": args.voxcpm2_model_dir,
        "qwen3_tts_model_dir": args.qwen3_tts_model_dir,
        "qwen3_tts_python": args.qwen3_tts_python,
        "qwen3_tts_timeout_sec": args.qwen3_tts_timeout_sec,
        "voxcpm2_cfg_value": args.voxcpm2_cfg_value,
        "voxcpm2_inference_timesteps": args.voxcpm2_inference_timesteps,
        "voxcpm2_max_reference_sec": args.voxcpm2_max_reference_sec,
        "voxcpm2_max_segment_chars": args.voxcpm2_max_segment_chars,
        "voxcpm2_fast_backend": args.voxcpm2_fast_backend,
        "device": args.device,
        "voice_adapter": args.voice_adapter,
        "test_mode": args.test_mode,
        "voice_adapter_strength": args.voice_adapter_strength,
        "voice_adapter_max_gain_db": args.voice_adapter_max_gain_db,
        "voice_adapter_window_ms": args.voice_adapter_window_ms,
        "voice_adapter_sample_rate": args.voice_adapter_sample_rate,
    }
    jobs: list[dict] = []
    if args.direction in {"outbound", "both"}:
        jobs.append({
            **base,
            "source_language": args.outbound_source_language or args.source_language,
            "target_language": args.outbound_target_language or args.target_language or config.outbound.target_language,
            "input_device": args.outbound_input_device or args.input_device or config.outbound.source_microphone,
            "output_device": args.outbound_output_device or args.output_device or config.outbound.call_input_virtual_mic,
        })
    if args.direction in {"inbound", "both"}:
        jobs.append({
            **base,
            "source_language": args.inbound_source_language or args.source_language,
            "target_language": args.inbound_target_language or config.inbound.target_language,
            "input_device": args.inbound_input_device or args.input_device or config.inbound.call_output_loopback,
            "output_device": args.inbound_output_device or args.output_device or config.inbound.local_monitor_output,
        })
    return jobs


def _run_live_jobs(jobs):
    errors: list[BaseException] = []
    def worker(job):
        try:
            run_live_call(**job)
        except BaseException as exc:
            errors.append(exc)
    threads = [threading.Thread(target=worker, args=(j,)) for j in jobs]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=0.5)
        if errors:
            raise errors[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _LiveTestArtifacts:
    """Optional artifact writer for test mode only.

    Live mode never creates metric CSV/HTML/wav dumps. This class is only
    constructed when --test-mode is supplied.
    """

    def __init__(self, root: Path | None) -> None:
        self.root = root or Path(".voice_bridge_runtime/test_artifacts")
        self.root.mkdir(parents=True, exist_ok=True)
        self.turn_index = 0
        self.chunk_index = 0
        self.jsonl = self.root / "turns.jsonl"

    def log_turn(self, source_text: str, translated_text: str, reference_path: str) -> None:
        payload = {
            "turn_index": self.turn_index,
            "source_text": source_text,
            "translated_text": translated_text,
            "reference_path": reference_path,
        }
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.turn_index += 1

    def write_chunk(self, chunk) -> None:
        # Small debug dump only when explicitly requested. Avoids any file I/O in live mode.
        path = self.root / f"chunk_{self.chunk_index:06d}.wav"
        self.chunk_index += 1
        samples = _mono_float_array(chunk.samples)
        _write_float_wav(path, samples, int(chunk.sample_rate))


class _AsyncSpeakerProfileUpdater:
    """Low-frequency non-blocking WeSpeaker update helper."""

    def __init__(self, *, sample_rate: int, device: str, cache_dir: Path, enabled: bool) -> None:
        self.sample_rate = int(sample_rate)
        self.device = device
        self.cache_dir = cache_dir
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ready = None
        self._error: BaseException | None = None

    def submit(self, samples: np.ndarray, wav_path: Path) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            audio = np.asarray(samples, dtype=np.float32).copy()
            path = Path(wav_path)
            self._thread = threading.Thread(target=self._worker, args=(audio, path), daemon=True)
            self._thread.start()

    def _worker(self, samples: np.ndarray, wav_path: Path) -> None:
        try:
            profile = extract_speaker_profile_from_samples(
                samples,
                self.sample_rate,
                wav_path=wav_path,
                device=self.device,
                cache_dir=self.cache_dir,
            )
            with self._lock:
                self._ready = profile
        except BaseException as exc:
            with self._lock:
                self._error = exc

    def take_ready(self):
        with self._lock:
            profile = self._ready
            self._ready = None
            return profile

    def close(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.1)


class _CallReferenceCache:
    """Keeps the current clone reference wav outside the hot temp-dir path.

    Qwen3-TTS still needs a wav path for voice cloning, so the live loop writes
    one stable reference file only when the speaker profile is due for update.
    The adapter receives the same reference as in-memory samples and does not
    reload this file.
    """

    def __init__(self, sample_rate: int, base_dir: Path | None = None, max_sec: float = 6.0) -> None:
        self.sample_rate = int(sample_rate)
        self.max_sec = float(max_sec)
        root = base_dir or Path(".voice_bridge_runtime") / "call_refs"
        root.mkdir(parents=True, exist_ok=True)
        self.current_path = root / ("ref_" + uuid.uuid4().hex + ".wav")

    def update(self, samples: np.ndarray) -> Path:
        samples = _mono_float_array(samples)
        max_samples = max(1, int(self.sample_rate * self.max_sec))
        if len(samples) > max_samples:
            samples = samples[-max_samples:]
        _write_float_wav(self.current_path, samples, self.sample_rate)
        return self.current_path


def _mono_float_array(samples: np.ndarray | list[float]) -> np.ndarray:
    audio = np.nan_to_num(np.asarray(samples, dtype=np.float32))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)

def _write_float_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for s in samples:
            v = max(-1.0, min(1.0, float(s)))
            w.writeframes(int(v * 32767).to_bytes(2, "little", signed=True))


def _validate_whisper(binary, model):
    bp = _resolve_binary(binary)
    if bp is None:
        raise FileNotFoundError(f"whisper.cpp binary not found: {binary}")
    if platform.system() != "Windows" and bp.suffix.lower() == ".exe":
        raise RuntimeError(f"Windows .exe cannot run on {platform.system()}: {bp}")
    if not Path(model).exists():
        raise FileNotFoundError(f"whisper.cpp model not found: {model}")


def _validate_marian(binary, model, vocab):
    if _resolve_binary(binary) is None:
        raise FileNotFoundError(f"Marian binary not found: {binary}")
    for label, val in (("model", model), ("vocab", vocab)):
        if not Path(val).exists():
            raise FileNotFoundError(f"Marian {label} not found: {val}")


def _resolve_binary(binary):
    p = Path(binary)
    if p.exists():
        return p
    found = shutil.which(binary)
    return Path(found) if found else None


def _default_backend_binary(exe_name: str, command_name: str) -> str:
    runtime_bin = model_path().parent / "bin" / exe_name
    if runtime_bin.exists():
        return str(runtime_bin)
    return command_name


def _translate_event(translator, event, target_language):
    translation = translator.translate(
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


if __name__ == "__main__":
    main()
