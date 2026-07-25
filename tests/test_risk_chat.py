import unittest

from integrated_context import build_live_multimodal_context
from risk_chat import build_result_snapshot, build_system_prompt, local_answer
from airis import analyze_context


class RiskChatTests(unittest.TestCase):
    def setUp(self):
        self.context = build_live_multimodal_context(
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
        self.snapshot = build_result_snapshot(
            self.context,
            analyze_context(self.context),
        )

    def test_prompt_contains_current_result(self):
        prompt = build_system_prompt(self.snapshot)
        self.assertIn('"overall_risk": "HIGH"', prompt)
        self.assertIn('"body_temperature": 38.1', prompt)

    def test_local_temperature_answer_uses_measurement(self):
        answer = local_answer("체온이 왜 높아?", self.snapshot)
        self.assertIn("38.1℃", answer)
        self.assertIn("발열", answer)

    def test_local_summary_contains_control(self):
        answer = local_answer("결과를 요약해줘", self.snapshot)
        self.assertIn("종합 위험도는 HIGH", answer)
        self.assertIn("AIR GATE HIGH", answer)


if __name__ == "__main__":
    unittest.main()
