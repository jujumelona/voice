from __future__ import annotations


def list_audio_devices() -> list[dict]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is not installed. Install audio extra: python -m pip install .[audio]"
        ) from exc

    devices = sd.query_devices()
    result: list[dict] = []
    for index, device in enumerate(devices):
        result.append(
            {
                "index": index,
                "name": device["name"],
                "max_input_channels": int(device["max_input_channels"]),
                "max_output_channels": int(device["max_output_channels"]),
                "default_samplerate": float(device["default_samplerate"]),
            }
        )
    return result


def find_device(name_or_index: str | int | None, *, need_input: bool = False, need_output: bool = False):
    if name_or_index is None:
        return None
    if isinstance(name_or_index, int):
        return name_or_index
    if str(name_or_index).isdigit():
        return int(name_or_index)

    needle = str(name_or_index).lower()
    matches = []
    for device in list_audio_devices():
        if needle not in device["name"].lower():
            continue
        if need_input and device["max_input_channels"] <= 0:
            continue
        if need_output and device["max_output_channels"] <= 0:
            continue
        matches.append(device)
    if not matches:
        raise ValueError(f"audio device not found: {name_or_index}")
    return matches[0]["index"]

