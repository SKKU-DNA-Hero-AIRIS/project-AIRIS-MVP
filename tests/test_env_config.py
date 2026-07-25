import tempfile
import unittest
from pathlib import Path

from env_config import read_env_value, save_env_value


class EnvConfigTests(unittest.TestCase):
    def test_value_round_trip_and_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            save_env_value("GROQ_API_KEY", "first", path)
            self.assertEqual(read_env_value("GROQ_API_KEY", path), "first")
            save_env_value("GROQ_API_KEY", "second", path)
            self.assertEqual(read_env_value("GROQ_API_KEY", path), "second")
            self.assertEqual(
                path.read_text(encoding="utf-8").count("GROQ_API_KEY="),
                1,
            )

    def test_blank_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            with self.assertRaises(ValueError):
                save_env_value("GROQ_API_KEY", "", path)


if __name__ == "__main__":
    unittest.main()
