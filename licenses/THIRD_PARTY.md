# Third-party license notes

This repository should keep source code and integration glue only. Model weights stay outside git.

| Component | Role | Code license | Source |
| --- | --- | --- | --- |
| whisper.cpp | ASR runtime | MIT | https://github.com/ggml-org/whisper.cpp |
| OpenAI Whisper | ASR models/code | MIT | https://github.com/openai/whisper |
| Marian NMT | Translation runtime | MIT | https://github.com/marian-nmt/marian |
| Argos Translate | Offline translation | MIT or CC0 | https://github.com/argosopentech/argos-translate |
| WeSpeaker | Speaker embedding | Apache-2.0 | https://github.com/wenet-e2e/wespeaker |
| VoxCPM2 | Multilingual voice generation | Apache-2.0 | https://github.com/OpenBMB/VoxCPM |
| B_spectral_delta_080 | In-repo speaker color adapter | Project license | voice_engine/adapters/spectral_delta_adapter.py |

Important: translation, ASR, speaker, and voice generation checkpoints can have separate model-card terms from the engine code.
