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

Speech output depends on the selected mode and device:

```text
fast
  Speech decoder: Qwen3-TTS 0.6B
  Speech languages: zh, en, ja, ko, de, fr, ru, pt, es, it

balanced
  Speech decoder: auto
  CPU speech languages: zh, en, ja, ko, de, fr, ru, pt, es, it
  CUDA speech languages: ar, my, zh, da, nl, en, fi, fr, de, el,
                         he, hi, id, it, ja, km, ko, lo, ms, no,
                         pl, pt, ru, es, sw, sv, tl, th, tr, vi

quality
  Speech decoder: auto
  CPU speech languages: zh, en, ja, ko, de, fr, ru, pt, es, it
  CUDA speech languages: ar, my, zh, da, nl, en, fi, fr, de, el,
                         he, hi, id, it, ja, km, ko, lo, ms, no,
                         pl, pt, ru, es, sw, sv, tl, th, tr, vi
```

Qwen3-TTS language names:

```text
zh  Chinese
en  English
ja  Japanese
ko  Korean
de  German
fr  French
ru  Russian
pt  Portuguese
es  Spanish
it  Italian
```

VoxCPM2 CUDA language names:

```text
ar  Arabic
my  Burmese
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
km  Khmer
ko  Korean
lo  Lao
ms  Malay
no  Norwegian
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

The full call pipeline also needs an installed translation package for the
source and target pair. The default install downloads:

```text
en:ko
ko:en
```

`fast` and `balanced` use Argos Translate by default, so install the Argos pair
you want. `quality` uses Marian, so its translation languages depend on the
Marian model files you place under the runtime model directory.

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
