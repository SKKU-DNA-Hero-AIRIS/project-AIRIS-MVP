import unittest

from integrated_context import analyze_live_context


class IntegratedContextTests(unittest.TestCase):
    def test_high_multimodal_risk_produces_high_control(self):
        _, result = analyze_live_context(
            {"person_count": 8, "mask_rate": 0.5, "crowding_level": "HIGH"},
            {
                "heart_rate": 125,
                "spo2": 94,
                "body_temperature": 38.1,
                "stress_index": 85,
            },
            {
                "temperature": 32,
                "humidity": 80,
                "pm25": 85,
                "air_quality": "BAD",
            },
            {
                "facility_type": "병원",
                "risk_level": "HIGH",
                "vulnerable_population": True,
            },
        )
        self.assertEqual(result.overall_risk, "HIGH")
        self.assertEqual(result.control_level, "HIGH")

    def test_normal_context_produces_low_control(self):
        _, result = analyze_live_context(
            {"person_count": 1, "mask_rate": 1.0, "crowding_level": "LOW"},
            {
                "heart_rate": 75,
                "spo2": 98,
                "body_temperature": 36.5,
                "stress_index": 20,
            },
            {
                "temperature": 24,
                "humidity": 45,
                "pm25": 10,
                "air_quality": "GOOD",
            },
            {
                "facility_type": "사무실",
                "risk_level": "LOW",
                "vulnerable_population": False,
            },
        )
        self.assertEqual(result.control_level, "LOW")


if __name__ == "__main__":
    unittest.main()
