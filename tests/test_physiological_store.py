import tempfile
import unittest
from pathlib import Path

from physiological_store import read_context, write_context


class PhysiologicalStoreTests(unittest.TestCase):
    def test_shared_context_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "latest.json"
            saved = write_context(
                {
                    "device_id": "iphone_test",
                    "heart_rate": 125,
                    "spo2": 94,
                    "body_temperature": 38.1,
                    "stress_index": 85,
                },
                state_path,
            )
            self.assertEqual(read_context(state_path), saved)

    def test_invalid_value_is_rejected(self):
        with self.assertRaises(ValueError):
            write_context(
                {
                    "heart_rate": 75,
                    "spo2": 120,
                    "body_temperature": 36.5,
                    "stress_index": 20,
                }
            )


if __name__ == "__main__":
    unittest.main()
