# Publishing Notes

This repository is intended to be published as source code only.

Commit:

- source code under `vendor/`
- integration adapters
- manifests and docs

Do not commit:

- model weights
- generated wav/mp3/flac/ogg files
- prebuilt binaries
- private voice samples
- call recordings
- model files such as `.bin`, `.pt`, `.pth`, `.safetensors`, `.onnx`, `.ckpt`

Ignored local paths:

```text
models/
bin/
outputs/
```

`vendor/` is intentionally committed so the repository can be used after clone. Weight and media artifacts inside it are still excluded by extension rules.

The public repository should include only integration code, manifests, license notes, and reproducible setup scripts.

Before publishing:

```powershell
python scripts/check_publish_ready.py
python -m compileall src scripts
```
