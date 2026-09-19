from __future__ import annotations

import json
from pathlib import Path

from scripts.upload_training_to_huggingface import (
    discover_epoch_artifacts,
    epoch_readme,
    root_readme,
    upload_training_artifacts,
)


class _FakeApi:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.folder_uploads: list[dict] = []
        self.file_uploads: list[dict] = []

    def whoami(self, *, token: str):
        assert token == "hf_test_token"
        return {"name": "tester"}

    def create_repo(self, **kwargs):
        self.created.append(dict(kwargs))
        return {"url": "https://huggingface.co/tester/voice-jp20"}

    def upload_folder(self, **kwargs):
        self.folder_uploads.append(dict(kwargs))
        return {"ok": True}

    def upload_file(self, **kwargs):
        entry = dict(kwargs)
        source = Path(entry["path_or_fileobj"])
        if source.exists() and source.is_file() and source.suffix in {".md", ".json"}:
            entry["uploaded_text"] = source.read_text(encoding="utf-8")
        self.file_uploads.append(entry)
        return {"ok": True}


def test_discover_epoch_artifacts_sorted(tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "epoch-10").mkdir(parents=True)
    (checkpoints / "epoch-2").mkdir()
    (checkpoints / "epoch-2" / "model.safetensors").write_bytes(b"2")
    (checkpoints / "epoch-10" / "model.safetensors").write_bytes(b"10")

    found = discover_epoch_artifacts(checkpoints)

    assert [item.epoch for item in found] == [2, 10]
    assert found[0].path.name == "epoch-2"
    assert found[1].remote_dir == "epochs/epoch-0010"


def test_epoch_readme_states_japanese_song_count() -> None:
    text = epoch_readme(7)

    assert "epoch 7" in text.lower()
    assert "20 Japanese songs" in text
    assert "source songs are not uploaded" in text


def test_root_readme_lists_every_epoch(tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "epoch-1").mkdir(parents=True)
    (checkpoints / "epoch-3").mkdir()

    text = root_readme(discover_epoch_artifacts(checkpoints))

    assert "20 Japanese songs" in text
    assert "epochs/epoch-0001/" in text
    assert "epochs/epoch-0003/" in text
    assert "final/" in text


def test_upload_training_artifacts_uploads_epochs_final_and_model_cards(tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    epoch1 = checkpoints / "epoch-1"
    epoch2 = checkpoints / "epoch_2"
    epoch1.mkdir(parents=True)
    epoch2.mkdir(parents=True)
    (epoch1 / "model.safetensors").write_bytes(b"epoch1")
    (epoch2 / "model.safetensors").write_bytes(b"epoch2")

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "final.safetensors").write_bytes(b"final")
    (outputs / "config.json").write_text("{}", encoding="utf-8")

    api = _FakeApi()
    result = upload_training_artifacts(
        token="hf_test_token",
        checkpoint_root=checkpoints,
        final_dirs=(outputs,),
        api=api,
    )

    assert result["repo_id"] == "tester/voice-jp20"
    assert [item["epoch"] for item in result["epochs"]] == [1, 2]
    assert result["training_dataset"]["song_count"] == 20
    assert result["training_dataset"]["raw_training_audio_uploaded"] is False

    assert api.created
    assert api.created[0]["repo_id"] == "tester/voice-jp20"
    assert api.created[0]["repo_type"] == "model"
    assert api.created[0]["exist_ok"] is True

    remote_folders = {item["path_in_repo"] for item in api.folder_uploads}
    assert "epochs/epoch-0001" in remote_folders
    assert "epochs/epoch-0002" in remote_folders
    assert "final" in remote_folders

    readmes = {
        item["path_in_repo"]: item.get("uploaded_text", "")
        for item in api.file_uploads
        if item["path_in_repo"].endswith("README.md")
    }
    assert "20 Japanese songs" in readmes["epochs/epoch-0001/README.md"]
    assert "20 Japanese songs" in readmes["epochs/epoch-0002/README.md"]
    assert "20 Japanese songs" in readmes["README.md"]

    manifest_entry = next(
        item for item in api.file_uploads
        if item["path_in_repo"] == "training-manifest.json"
    )
    manifest = json.loads(manifest_entry["uploaded_text"])
    assert manifest["training_dataset"]["description"] == "20 Japanese songs"


def test_uploader_does_not_upload_dataset_or_raw_audio_dirs(tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    epoch = checkpoints / "epoch-1"
    epoch.mkdir(parents=True)
    (epoch / "model.safetensors").write_bytes(b"model")

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "song.mp3").write_bytes(b"audio")

    api = _FakeApi()
    upload_training_artifacts(
        token="hf_test_token",
        checkpoint_root=checkpoints,
        final_dirs=(dataset,),
        api=api,
    )

    assert all(item["path_in_repo"] != "final" for item in api.folder_uploads)
