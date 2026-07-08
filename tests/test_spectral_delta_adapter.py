from __future__ import annotations

import unittest
import importlib.util

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("librosa"), "librosa is not installed")
class SpectralDeltaAdapterTest(unittest.TestCase):
    def test_output_is_finite_and_same_length(self) -> None:
        from voice_engine.adapters.spectral_delta_adapter import SpectralDeltaConfig, spectral_delta_adapter

        sr = 16000
        t = np.linspace(0.0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
        source = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        target = (
            0.15 * np.sin(2.0 * np.pi * 140.0 * t)
            + 0.05 * np.sin(2.0 * np.pi * 280.0 * t)
        ).astype(np.float32)

        out = spectral_delta_adapter(source, target, SpectralDeltaConfig(sr=sr))

        self.assertEqual(len(out), len(source))
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(float(np.max(np.abs(out))), 0.0)


if __name__ == "__main__":
    unittest.main()
