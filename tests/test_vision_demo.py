import unittest

from vision_demo import crowding_level


class CrowdingLevelTests(unittest.TestCase):
    def test_low(self):
        self.assertEqual(crowding_level(3), "LOW")

    def test_medium_boundaries(self):
        self.assertEqual(crowding_level(4), "MEDIUM")
        self.assertEqual(crowding_level(7), "MEDIUM")

    def test_high_boundary(self):
        self.assertEqual(crowding_level(8), "HIGH")


if __name__ == "__main__":
    unittest.main()
