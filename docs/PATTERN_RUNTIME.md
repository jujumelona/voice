# Pattern Runtime

The runtime path is:

```text
source audio
-> ASR with word timestamps
-> PatternTrace frames
-> word-level ProsodyEvent extraction
-> target ContentUnits
-> monotonic source-event to target-unit alignment
-> StyleTokenTrace
-> VoxCPM2 generation request
-> B_spectral_delta_080 adapter
```

## Runtime State

```text
SpeakerProfile
  slow cumulative identity state
  produced by WeSpeaker ERes2Net-large
  updated across the call

ProsodyEvent
  current utterance word event
  pause_before_ms / pause_after_ms
  energy_mean / energy_peak
  log_f0_start / log_f0_end / log_f0_slope
  breath
  attack_peak
  stress
  emotion_hint

TargetStyleEvent
  source ProsodyEvent mapped onto target word/phrase unit

StyleTokenTrace
  frame controls rendered from TargetStyleEvent
  current utterance only
  discarded after decode
```

## Files

```text
voice_engine/prosody/events.py
  Extracts word-level ProsodyEvent objects from ASR word timing + PatternTrace.

voice_engine/prosody/alignment.py
  Maps source ProsodyEvent objects to translated ContentUnits.

voice_engine/prosody/style_plan.py
  Renders aligned word events into frame-level controls.

voice_engine/prosody/style_tokens.py
  Serializes the final StyleTokenTrace payload.

voice_engine/decoders/voxcpm2_decoder.py
  Calls VoxCPM2 with translated text and current utterance reference audio.

voice_engine/speaker/speaker_profile.py
  Extracts speaker identity with WeSpeaker ERes2Net-large.
```

## Timing

ASR word timestamps should be used when available. If a transcript has no word
timestamps, the code creates monotonic estimated word timings so the rest of the
pipeline can still build a StyleTokenTrace.
