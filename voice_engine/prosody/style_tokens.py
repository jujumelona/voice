from __future__ import annotations

from dataclasses import asdict

from voice_engine.pipeline.types import PatternTrace, SpeakerProfile, StylePlan, StyleTokenTrace


def style_tokens_from_plan(
    *,
    plan: StylePlan,
    source_trace: PatternTrace,
    speaker: SpeakerProfile,
) -> StyleTokenTrace:
    diagnostics = dict(plan.diagnostics)
    diagnostics["injection_stage"] = "pre_acoustic"
    diagnostics["rule"] = "do not treat this as a postprocess audio effect"
    return StyleTokenTrace(
        frame_ms=source_trace.frame_ms,
        hop_ms=source_trace.hop_ms,
        duration_ms=plan.target_duration_sec * 1000.0,
        energy=plan.energy,
        log_f0_rel=plan.log_f0_rel,
        pause=plan.pause,
        stress=plan.stress,
        breath=plan.breath,
        events=plan.events,
        speaking_rate=source_trace.speaking_rate,
        speaker_median_f0=speaker.median_f0,
        diagnostics=diagnostics,
    )


def style_tokens_to_dict(tokens: StyleTokenTrace) -> dict[str, object]:
    return asdict(tokens)
