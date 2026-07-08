# Voice Engine

Real-time call translation bridge with current-utterance style transfer and
speaker-color correction.

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

English to Korean outbound, Korean to English inbound:

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
