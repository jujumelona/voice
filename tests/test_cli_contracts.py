from __future__ import annotations

import subprocess
import sys
import unittest


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


if __name__ == "__main__":
    unittest.main()
