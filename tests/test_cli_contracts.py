from __future__ import annotations

import subprocess
import sys
import unittest

from voice_engine.pipeline.realtime_call_translate import _auto_decoder, _selected_runtime


class CliContractsTest(unittest.TestCase):
    def test_spectral_delta_script_help_runs_from_repo_root(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/apply_spectral_delta_adapter.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--source", completed.stdout)

    def test_validate_only_pipeline(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "voice_engine.pipeline.realtime_call_translate",
                "--validate-only",
                "--direction",
                "both",
                "--decoder",
                "auto",
                "--voice-adapter",
                "none",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"decoder": "auto(', completed.stdout)

    def test_fast_mode_uses_qwen3_tts(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "voice_engine.pipeline.realtime_call_translate",
                "--validate-only",
                "--direction",
                "both",
                "--mode",
                "fast",
                "--voice-adapter",
                "none",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"decoder": "qwen3-tts"', completed.stdout)

    def test_mode_runtime_contracts(self) -> None:
        self.assertEqual(
            _selected_runtime("fast", None, None, None),
            {"mode": "fast", "asr": "whisper", "mt": "argos", "decoder": "qwen3-tts"},
        )
        self.assertEqual(
            _selected_runtime("balanced", None, None, None),
            {"mode": "balanced", "asr": "faster-whisper", "mt": "argos", "decoder": "auto"},
        )
        self.assertEqual(
            _selected_runtime("quality", None, None, None),
            {"mode": "quality", "asr": "faster-whisper", "mt": "marian", "decoder": "auto"},
        )

    def test_auto_decoder_contracts(self) -> None:
        self.assertEqual(_auto_decoder("cpu"), "qwen3-tts")
        self.assertEqual(_auto_decoder("cuda"), "voxcpm2")


if __name__ == "__main__":
    unittest.main()
