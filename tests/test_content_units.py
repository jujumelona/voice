from __future__ import annotations

import unittest

from voice_engine.content.units import content_units_from_text


class ContentUnitsTest(unittest.TestCase):
    def test_korean_and_english_units(self) -> None:
        units = content_units_from_text("그만해! Stop it.", "ko").units
        self.assertEqual(units, ["그만해", "!", "Stop", "it", "."])


if __name__ == "__main__":
    unittest.main()
