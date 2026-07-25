import unittest

from airis.gate_scenarios import BY_KEY, legacy_rule_level
from airis.policy import POLICIES, decide
from airis.risk_model import RiskModel


class RiskPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RiskModel.load()

    def _run(self, key, facility_type=None):
        s = BY_KEY[key]
        facility = dict(s.record["facility_context"])
        if facility_type:
            facility["facility_type"] = facility_type
        risk = self.model.estimate(s.person["health_context"], s.record["environment_context"])
        return s, risk, decide(risk, self.model.thresholds, facility,
                               s.record["environment_context"], s.record["vision_context"])

    def test_ambient_correction_removes_summer_false_positive(self):
        """주변온도 38.1℃에서 측정체온 38.1℃ — 룰은 발동, 보정 후 개인 위험은 LOW."""
        s, _, d = self._run("summer_false_positive")
        self.assertEqual(s.truth["is_abnormal"], 0)
        self.assertEqual(legacy_rule_level(s.person["health_context"]), "MEDIUM")
        self.assertEqual(d.personal_risk, "LOW")

    def test_environment_risk_is_independent_of_personal_risk(self):
        """개인은 통과시키되 환경 때문에 환기만 올린다 — 두 축이 섞이면 안 된다."""
        _, _, d = self._run("summer_false_positive")
        self.assertEqual(d.personal_risk, "LOW")
        self.assertEqual(d.environment_risk, "MEDIUM")
        self.assertEqual(d.control_level, "MEDIUM")

    def test_beta_blocker_fever_missed_by_rule_is_caught(self):
        """체온 결측 + 베타차단제로 억제된 심박 — 룰은 놓치고 잔차 모델은 잡는다."""
        s, _, d = self._run("beta_blocker_fever")
        self.assertEqual(s.truth["is_abnormal"], 1)
        self.assertEqual(legacy_rule_level(s.person["health_context"]), "LOW")
        self.assertNotEqual(d.personal_risk, "LOW")

    def test_elderly_blunted_fever_missed_by_rule_is_caught(self):
        s, _, d = self._run("elderly_blunted_fever")
        self.assertEqual(legacy_rule_level(s.person["health_context"]), "LOW")
        self.assertNotEqual(d.personal_risk, "LOW")

    def test_zero_channels_defers_instead_of_guessing(self):
        """측정 채널 0개면 판별력이 무작위와 같다 — 판정하지 말고 재측정을 요구한다."""
        _, risk, d = self._run("measurement_failure")
        self.assertEqual(risk.n_valid_channels, 0)
        self.assertEqual(risk.confidence, 0.0)
        self.assertTrue(d.needs_remeasure)
        self.assertIn("재측정", " ".join(d.reasons))

    def test_confidence_reflects_measurement_not_risk(self):
        """신뢰도는 '몇 채널을 쟀는가'이지 '얼마나 위험한가'가 아니다."""
        _, low, _ = self._run("elderly_blunted_fever")     # 1채널
        _, high, _ = self._run("clear_high_risk")          # 2채널
        self.assertAlmostEqual(low.confidence, 1 / 3)
        self.assertAlmostEqual(high.confidence, 2 / 3)

    def test_same_person_different_facility_different_control(self):
        """연구질문 Q2 — 같은 생체값이 시설 비용비에 따라 다른 제어를 낳는다."""
        _, risk_h, hospital = self._run("beta_blocker_fever", "hospital")
        _, risk_c, cafe = self._run("beta_blocker_fever", "cafe")
        self.assertEqual(risk_h.probability, risk_c.probability)   # 위험도는 동일
        self.assertEqual(hospital.personal_risk, "HIGH")
        self.assertEqual(cafe.personal_risk, "LOW")                # 정책만 다르다

    def test_facility_thresholds_are_ordered_by_cost_ratio(self):
        """비용비가 높은 시설일수록 낮은 임계값에서 발동해야 한다."""
        thr = self.model.thresholds
        ordered = sorted(POLICIES.values(), key=lambda p: -p.miss_cost_ratio)
        mediums = [thr[p.medium_at] for p in ordered]
        self.assertEqual(mediums, sorted(mediums))

    def test_clear_high_risk_agrees_with_rule(self):
        """명확한 사례에서는 룰도 맞는다 — 모델의 가치는 경계 사례에 있다."""
        s, _, d = self._run("clear_high_risk")
        self.assertEqual(legacy_rule_level(s.person["health_context"]), "HIGH")
        self.assertEqual(d.personal_risk, "HIGH")


if __name__ == "__main__":
    unittest.main()
