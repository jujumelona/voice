from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutboundRoute:
    source_microphone: str
    call_input_virtual_mic: str
    send_raw_microphone_to_call: bool
    target_language: str


@dataclass(frozen=True)
class InboundRoute:
    call_output_loopback: str
    local_monitor_output: str
    send_translated_inbound_back_to_call: bool
    target_language: str


@dataclass(frozen=True)
class CallSafety:
    require_headphones: bool
    mute_physical_mic_in_call_app: bool
    block_if_raw_mic_enabled: bool


@dataclass(frozen=True)
class CallRoutingConfig:
    mode: str
    outbound: OutboundRoute
    inbound: InboundRoute
    safety: CallSafety


def load_call_routing_config(path: str | Path) -> CallRoutingConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CallRoutingConfig(
        mode=data["mode"],
        outbound=OutboundRoute(**data["outbound"]),
        inbound=InboundRoute(**data["inbound"]),
        safety=CallSafety(**data["safety"]),
    )


def validate_call_routing(config: CallRoutingConfig) -> list[str]:
    errors: list[str] = []
    if config.mode != "translated_only_call":
        errors.append("mode must be translated_only_call")
    if config.safety.block_if_raw_mic_enabled and config.outbound.send_raw_microphone_to_call:
        errors.append("raw microphone must not be sent to the call app")
    if config.inbound.send_translated_inbound_back_to_call:
        errors.append("inbound translation must not be sent back into the call")
    if not config.outbound.call_input_virtual_mic:
        errors.append("call input virtual microphone is required")
    if not config.inbound.local_monitor_output:
        errors.append("local monitor output is required")
    if config.safety.mute_physical_mic_in_call_app and (
        config.outbound.source_microphone == config.outbound.call_input_virtual_mic
    ):
        errors.append("call app input must be a virtual mic, not the physical microphone")
    return errors

