# Local Runtime

This repo keeps source code only. Runtime, model, cache, and generated files
stay outside the checkout. On Windows the default runtime root is:

```text
I:\voice_bridge
```

## One Command Install

CPU/default:

```powershell
.\scripts\setup_all.ps1 -RuntimeRoot I:\voice_bridge -Torch cpu -WhisperModel base
```

CUDA speaker/optional runtime:

```powershell
.\scripts\setup_all.ps1 -RuntimeRoot I:\voice_bridge -Torch cuda -WhisperModel base
```

The setup script installs:

```text
base Python runtime: I:\Python311 if present, otherwise I:\voice_bridge\python311
base packages: requirements.txt
FFmpeg: I:\voice_bridge\bin\ffmpeg
whisper.cpp CLI: I:\voice_bridge\bin\whisper-cli.exe
Whisper model: I:\voice_bridge\models\whisper
Argos language packages: installed into the Argos package store
speaker encoder venv: I:\voice_bridge\.venv-speaker
Qwen3-TTS CPU decoder venv: I:\voice_bridge\.venv-qwen3-tts
model/cache root: I:\voice_bridge\models
```

Marian C++ is optional because its Windows build needs extra native
dependencies. To attempt it explicitly:

```powershell
.\scripts\setup_all.ps1 -RuntimeRoot I:\voice_bridge -Torch cpu -WhisperModel base -BuildMarian
```

Backend install status:

```text
I:\voice_bridge\backend_status.json
I:\voice_bridge\backend_status_argos.json
I:\voice_bridge\backend_status_qwen3_tts.json
```

See `docs/RUNTIME_VERSIONS.md` for pinned package versions.

## Version Check

Use the runtime Python, not the system `python`:

```powershell
I:\Python311\python.exe scripts\check_runtime.py --strict
```

The system `python` on this machine currently reports Python 3.14.4, which is
not the supported project runtime. The project runtime is Python 3.11.9.

If setup created the runtime under `I:\voice_bridge\python311`, use:

```powershell
I:\voice_bridge\python311\python.exe scripts\check_runtime.py --strict
```

## Fast CPU Commands

Validate the fast CPU pipeline:

```powershell
.\scripts\run_fast_cpu.ps1 -ValidateOnly
```

List audio devices:

```powershell
.\scripts\run_fast_cpu.ps1 -ListAudioDevices
```

Run both directions:

```powershell
.\scripts\run_fast_cpu.ps1 -SourceLanguage en -TargetLanguage ko
```

## Input/Output Pipeline

Outbound:

```text
physical microphone
-> ASR stream
-> translation
-> ContentUnits
-> current utterance StyleTokenTrace
-> cumulative SpeakerProfile
-> auto speech decoder output wav/source stream
-> B_spectral_delta_080 voice color adapter
-> virtual microphone for Discord/call apps
```

Inbound:

```text
call app output / WASAPI loopback
-> ASR stream
-> translation
-> ContentUnits
-> current utterance StyleTokenTrace
-> cumulative SpeakerProfile
-> auto speech decoder output wav/source stream
-> B_spectral_delta_080 voice color adapter
-> local headphones
```

Windows audio routing:

```text
Discord input  = virtual microphone / cable output
Discord output = normal speakers or loopback-capturable device
Voice Engine outbound input  = physical microphone
Voice Engine outbound output = virtual microphone
Voice Engine inbound input   = Discord output loopback
Voice Engine inbound output  = headphones
```

## Validate Pipeline

```powershell
I:\Python311\python.exe -m voice_engine.pipeline.realtime_call_translate `
  --validate-only `
  --direction both `
  --mode fast `
  --voice-adapter spectral_delta
```

List audio devices:

```powershell
I:\Python311\python.exe -m voice_engine.pipeline.realtime_call_translate `
  --list-audio-devices
```

Run live call bridge directly:

```powershell
I:\Python311\python.exe -m voice_engine.pipeline.realtime_call_translate `
  --direction both `
  --mode fast `
  --voice-adapter spectral_delta `
  --source-language en `
  --target-language ko
```

## Spectral Delta Adapter Test

```powershell
I:\Python311\python.exe scripts\apply_spectral_delta_adapter.py `
  --source generated.wav `
  --target target_speaker.wav `
  --out converted.wav
```

Adapter defaults:

```text
sr = 16000
n_fft = 1024
hop_length = 256
n_mels = 80
fmin = 50.0
strength = 0.80
max_gain_db = 12.0
phase = source phase
length = source length
```

The adapter does not use target phase, target timing, formant warp, pitch shift,
or global energy matching by default.

## Model Locations

Recommended root:

```text
I:\voice_bridge\models
```

Subdirectories:

```text
I:\voice_bridge\models\whisper
I:\voice_bridge\models\faster-whisper
I:\voice_bridge\models\argos
I:\voice_bridge\models\marian
I:\voice_bridge\models\wespeaker
I:\voice_bridge\models\speaker\wespeaker\eres2net-large
I:\voice_bridge\models\voxcpm2
I:\voice_bridge\models\qwen3-tts\0.6B-base
I:\voice_bridge\models\qwen3-tts\tokenizer-12Hz
I:\voice_bridge\models\huggingface
```

Qwen3-TTS and VoxCPM2 can also be loaded from the Hugging Face cache by their
runtime packages. Keep all model weights outside this git checkout.

## Disk Use

Practical minimum:

```text
base Python/packages:        5-10 GB
speaker encoder venv/cache:  2-6 GB
ASR/MT models:               1-8 GB
VoxCPM2 model/cache:         5-20+ GB
Qwen3-TTS model/cache:       2-8+ GB
recommended free space:      30+ GB
```
