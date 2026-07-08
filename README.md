# Voice Engine

Real-time call translation bridge with current-utterance style transfer and
speaker-color correction.

## What It Does

Voice Engine takes speech from a microphone or call app, translates the spoken
content, generates speech in the target language, then adjusts the generated
voice toward the current speaker's color before sending it to a virtual mic or
headphones.

The project is for call translation experiments. It keeps model weights and
runtime downloads outside the GitHub repo.

## Pipeline

```text
microphone / call audio
-> ASR
-> translation
-> Qwen3-TTS fast CPU speech decoder
-> B_spectral_delta_080 speaker-color adapter
-> virtual mic / headphones
```

Fast mode uses Qwen3-TTS 0.6B on CPU. Model files stay outside this repo under
`I:\voice_bridge`.

## Modes

```text
fast
  ASR: whisper.cpp
  Translation: Argos Translate
  Speech: Qwen3-TTS 0.6B
  Use: fastest CPU call mode

balanced
  ASR: faster-whisper
  Translation: Argos Translate
  Speech: auto
  Use: better ASR while keeping call-mode latency

quality
  ASR: faster-whisper
  Translation: Marian
  Speech: auto
  Use: higher-quality utterance-end translation
```

`auto` speech decoder means VoxCPM2 on CUDA and Qwen3-TTS on CPU. The simple
CPU command uses `fast`.

## Supported Languages

A language is listed here only when the whole pipeline can handle it:

```text
ASR language
+ translation package/model
+ speech decoder output language
```

The default install only downloads these Argos translation pairs:

```text
en:ko
ko:en
```

So the default ready-to-run pair is English <-> Korean. Install more translation
pairs before using other languages.

Full-pipeline language set by mode:

```text
fast
  ASR: whisper.cpp
  Translation: Argos Translate
  Speech: Qwen3-TTS 0.6B
  Languages if Argos pair is installed:
    zh, en, ja, ko, de, fr, ru, pt, es, it

balanced
  ASR: faster-whisper
  Translation: Argos Translate
  CPU speech: Qwen3-TTS 0.6B
  CPU languages if Argos pair is installed:
    zh, en, ja, ko, de, fr, ru, pt, es, it
  CUDA speech: VoxCPM2
  CUDA languages if Argos pair is installed:
    ar, zh, da, nl, en, fi, fr, de, el, he, hi, id, it,
    ja, ko, ms, pl, pt, ru, es, sw, sv, tl, th, tr, vi

quality
  ASR: faster-whisper
  Translation: Marian
  Speech: auto
  Languages:
    installed Marian model pair
    intersected with Qwen3-TTS on CPU or VoxCPM2 on CUDA
```

Language names used above:

```text
ar  Arabic
zh  Chinese
da  Danish
nl  Dutch
en  English
fi  Finnish
fr  French
de  German
el  Greek
he  Hebrew
hi  Hindi
id  Indonesian
it  Italian
ja  Japanese
ko  Korean
ms  Malay
pl  Polish
pt  Portuguese
ru  Russian
es  Spanish
sw  Swahili
sv  Swedish
tl  Tagalog
th  Thai
tr  Turkish
vi  Vietnamese
```

Install more Argos pairs with one command:

```powershell
.\scripts\setup_all.ps1 `
  -RuntimeRoot I:\voice_bridge `
  -Torch cpu `
  -WhisperModel base `
  -ArgosPair "en:ja,ja:en,en:de,de:en,en:fr,fr:en,en:es,es:en"
```

Use any installed pair by passing language codes:

```powershell
.\scripts\run_fast_cpu.ps1 -SourceLanguage ja -TargetLanguage en
```

## Component Roles

```text
ASR
  Converts incoming speech audio into text.

Translation
  Converts recognized text from source language to target language.

Qwen3-TTS
  Generates target-language speech from translated text.

SpeakerProfile / StyleTrace
  Tracks speaker identity and current utterance style controls.

B_spectral_delta_080
  Moves generated speech toward the speaker's spectral color while preserving
  the generated speech timing and pronunciation.

Audio routing
  Sends outbound translated speech to a virtual microphone and inbound
  translated speech to local headphones.
```

## Runtime Scheduling

```text
direction=both
  outbound and inbound run in separate threads

live audio
  microphone/call audio capture uses non-blocking audio callbacks
  output playback uses a buffered callback stream

per direction
  ASR keeps reading input audio
  final utterances are queued to a synthesis worker
  translation + speech decoder + voice adapter run on that worker
  speaker profile updates run at a low rate and can run asynchronously

speech output
  Qwen3-TTS keeps a persistent worker process loaded
  VoxCPM2 keeps the model loaded after first use
  B_spectral_delta_080 adapts audio in streaming blocks
```

## Install

```powershell
.\scripts\setup_all.ps1 -RuntimeRoot I:\voice_bridge -Torch cpu -WhisperModel base
```

## Check

```powershell
.\scripts\run_fast_cpu.ps1 -ValidateOnly
```

List audio devices:

```powershell
.\scripts\run_fast_cpu.ps1 -ListAudioDevices
```

## Run

Use any installed source/target pair:

```powershell
.\scripts\run_fast_cpu.ps1 -SourceLanguage en -TargetLanguage ko
```

Use specific devices if needed:

```powershell
.\scripts\run_fast_cpu.ps1 `
  -SourceLanguage en `
  -TargetLanguage ko `
  -InputDevice "Your physical microphone" `
  -OutputDevice "VB-CABLE / virtual microphone"
```

## Runtime Files

Do not commit models, checkpoints, binaries, or generated audio. Runtime files
belong under:

```text
I:\voice_bridge
I:\voice_bridge\models
```

More runtime details are in `docs/LOCAL_RUNTIME.md`.
