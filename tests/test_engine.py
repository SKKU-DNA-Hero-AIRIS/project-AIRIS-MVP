import unittest

from airis import analyze_context, build_context


class ContextAwareDecisionTests(unittest.TestCase):
    def test_same_sensor_data_changes_with_facility_context(self):
        office = build_context("신체 활동", "쾌적", "여유", "일반 사무실")
        hospital = build_context("신체 활동", "쾌적", "여유", "병원")

        self.assertEqual(office.health_context, hospital.health_context)
        self.assertEqual(analyze_context(office).control_level, "MEDIUM")
        self.assertEqual(analyze_context(hospital).control_level, "HIGH")

    def test_normal_context_is_low(self):
        context = build_context("정상", "쾌적", "여유", "일반 사무실")
        self.assertEqual(analyze_context(context).control_level, "LOW")

    def test_multiple_non_health_risks_escalate_control(self):
        context = build_context("정상", "대기질 나쁨", "혼잡", "일반 사무실")
        result = analyze_context(context)
        self.assertEqual(result.environment_risk, "HIGH")
        self.assertEqual(result.crowding_risk, "HIGH")
        self.assertEqual(result.control_level, "HIGH")

    def test_high_risk_health_is_high_in_any_facility(self):
        context = build_context("고위험", "쾌적", "여유", "일반 사무실")
        self.assertEqual(analyze_context(context).control_level, "HIGH")


if __name__ == "__main__":
    unittest.main()
