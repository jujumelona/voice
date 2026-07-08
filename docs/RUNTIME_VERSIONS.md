# Runtime Versions

Verified on Windows with `I:\Python311\python.exe`:

```text
Python 3.11.9
pip 24.0 before setup, upgraded by setup_runtime.ps1 to pip==26.1.2
```

The normal `python` on this machine currently reports Python 3.14.4. Do not use
that for this project runtime. Use the setup script runtime Python:

```text
I:\Python311\python.exe
```

## Base Runtime

Installed by:

```powershell
.\scripts\setup_all.ps1 -RuntimeRoot I:\voice_bridge -Torch cpu -WhisperModel base
```

Pinned direct/runtime packages:

```text
pip==26.1.2
setuptools==81.0.0
wheel==0.47.0
numpy==2.4.6
librosa==0.11.0
soundfile==0.14.0
sounddevice==0.5.5
faster-whisper==1.2.1
ctranslate2==4.8.1
onnxruntime==1.27.0
tokenizers==0.22.2
huggingface-hub==1.22.0
argostranslate==1.11.0
stanza==1.10.1
spacy==3.8.14
sentencepiece==0.2.1
sacremoses==0.1.1
torch==2.12.1
tqdm==4.68.4
voxcpm==2.0.3
transformers==5.13.0
```

## Speech Decoder Runtime

The fast call pipeline uses Qwen3-TTS by default:

```text
fast: Qwen3-TTS 0.6B Base via qwen-tts==0.1.1
auto on CPU: Qwen3-TTS 0.6B Base
auto on CUDA: VoxCPM2 via voxcpm==2.0.3
```

Qwen3-TTS:

```text
package: qwen-tts==0.1.1
model: Qwen/Qwen3-TTS-12Hz-0.6B-Base
tokenizer: Qwen/Qwen3-TTS-Tokenizer-12Hz
license: Apache-2.0
runtime: I:\voice_bridge\.venv-qwen3-tts
model path: I:\voice_bridge\models\qwen3-tts\0.6B-base
```

Qwen3-TTS is isolated in a separate venv because it requires
`transformers==4.57.3`, while the main runtime has its own transformer stack.

VoxCPM2:

```text
package: voxcpm==2.0.3
model: openbmb/VoxCPM2
license: Apache-2.0
model path: I:\voice_bridge\models\voxcpm2
```

VoxCPM2 requires Python 3.10+, PyTorch 2.5.0+, and CUDA 12.0+ for practical
GPU inference. CPU execution is not the target runtime for live call quality.
Keep model files under `I:\voice_bridge\models\voxcpm2` or the Hugging Face
cache, not in the repo.

## Speaker Encoder Runtime

Installed by `setup_all.ps1` unless `-SkipSpeaker` is passed.

Separate venv:

```text
I:\voice_bridge\.venv-speaker
```

Pinned packages:

```text
torch==2.5.1
torchaudio==2.5.1
numpy==1.26.4
scipy==1.17.1
scikit-learn==1.9.0
tqdm==4.68.4
```

Reason for separate venv: WeSpeaker ERes2Net is tested against the older
torch/torchaudio stack. The main runtime can use newer torch pulled by
Argos/Stanza without breaking the speaker encoder.

## External Programs

The setup script installs backend binaries under `I:\voice_bridge\bin`, so they
do not need to be globally installed on PATH.

```text
cmake: required for local source builds
MinGW gcc/g++/mingw32-make: required for local source builds
git: not required
ffmpeg: I:\voice_bridge\bin\ffmpeg\ffmpeg.exe
whisper-cli: I:\voice_bridge\bin\whisper-cli.exe
marian-decoder: optional, only when setup is run with -BuildMarian
```

Backend source/model defaults:

```text
FFmpeg: gyan.dev essentials build
whisper.cpp: ggml-org/whisper.cpp v1.9.1 source zip
Whisper ASR model: ggml-base.bin by default
Argos Translate language pairs: en:ko, ko:en
Marian: optional C++ backend, attempted only with -BuildMarian
```

The call pipeline auto-detects the runtime backend binaries. Manual override is
still available:

```text
--whisper-binary
--marian-binary
```

## Verify

Run:

```powershell
I:\Python311\python.exe scripts\check_runtime.py --strict
```

This checks Python version, package versions, importability, external tool
presence, and the current input/output pipeline contract.
