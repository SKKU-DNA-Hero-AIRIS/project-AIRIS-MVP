import unittest

from airis.control import analyze_hybrid, baseline_control_level, plan_actuation
from airis.dataset import build_dataset, label_context, sample_context
from airis.features import FEATURE_NAMES, extract_features
from airis.ml_policy import load_policy, predict_batch
from airis.models import MultimodalContext
from airis.rules import ceiling_level, evaluate_rules, floor_level
from airis.scenarios import build_context, load_context_samples


POLICY = load_policy()
SAMPLES = {payload["scenario_id"]: context for payload, context in load_context_samples()}


class FeatureTests(unittest.TestCase):
    def test_feature_layout_matches_extraction(self):
        features = extract_features(build_context("정상", "쾌적", "여유", "일반 사무실"))
        self.assertEqual(tuple(features), FEATURE_NAMES)

    def test_features_stay_normalized(self):
        for context in SAMPLES.values():
            for name, value in extract_features(context).items():
                self.assertGreaterEqual(value, 0.0, name)
                self.assertLessEqual(value, 1.0, name)

    def test_activity_signal_only_fires_in_low_sensitivity_facility(self):
        gym = extract_features(SAMPLES["S2"])
        hospital = extract_features(SAMPLES["S3"])
        self.assertGreater(gym["activity_plausible"], 0.5)
        self.assertEqual(hospital["activity_plausible"], 0.0)


class RuleLayerTests(unittest.TestCase):
    def test_hypoxia_sets_high_floor(self):
        hits = evaluate_rules(SAMPLES["S7"])
        self.assertIn("F1", [hit.rule_id for hit in hits])
        self.assertEqual(floor_level(hits), "HIGH")

    def test_fever_in_sensitive_facility_sets_high_floor(self):
        hits = evaluate_rules(SAMPLES["S4"])
        self.assertIn("F2", [hit.rule_id for hit in hits])
        self.assertEqual(floor_level(hits), "HIGH")

    def test_gym_workout_is_capped_at_medium(self):
        hits = evaluate_rules(SAMPLES["S10"])
        self.assertIn("C1", [hit.rule_id for hit in hits])
        self.assertEqual(ceiling_level(hits), "MEDIUM")

    def test_no_rules_fire_without_signals(self):
        hits = evaluate_rules(SAMPLES["S1"])
        self.assertEqual(floor_level(hits), "LOW")
        self.assertEqual([hit.rule_id for hit in hits], ["C2"])


class HybridDecisionTests(unittest.TestCase):
    def test_identical_health_context_diverges_by_facility(self):
        gym, hospital = SAMPLES["S2"], SAMPLES["S3"]
        self.assertEqual(gym.health_context, hospital.health_context)
        self.assertEqual(analyze_hybrid(gym, POLICY).control_level, "MEDIUM")
        self.assertEqual(analyze_hybrid(hospital, POLICY).control_level, "HIGH")

    def test_flat_baseline_cannot_separate_that_pair(self):
        self.assertEqual(baseline_control_level(SAMPLES["S2"]), "HIGH")
        self.assertEqual(baseline_control_level(SAMPLES["S3"]), "HIGH")

    def test_safety_floor_overrides_a_low_ml_prediction(self):
        decision = analyze_hybrid(SAMPLES["S7"], POLICY)
        self.assertEqual(decision.ml_level, "LOW")
        self.assertEqual(decision.control_level, "HIGH")
        self.assertEqual(decision.decision_source, "RULE_FLOOR")

    def test_damping_ceiling_lowers_an_ml_high(self):
        decision = analyze_hybrid(SAMPLES["S10"], POLICY)
        self.assertEqual(decision.ml_level, "HIGH")
        self.assertEqual(decision.control_level, "MEDIUM")
        self.assertEqual(decision.decision_source, "RULE_CEILING")

    def test_floor_beats_ceiling(self):
        """A ceiling must never talk a safety floor down."""
        context = MultimodalContext.from_dict(
            {
                **SAMPLES["S9"].to_dict(),
                "health_context": {
                    "heart_rate": 98,
                    "spo2": 88,
                    "body_temperature": 36.6,
                    "stress_index": 28,
                },
            }
        )
        hits = evaluate_rules(context)
        self.assertEqual(floor_level(hits), "HIGH")
        self.assertEqual(analyze_hybrid(context, POLICY).control_level, "HIGH")

    def test_every_sample_produces_a_reason_and_a_plan(self):
        for scenario_id, context in SAMPLES.items():
            decision = analyze_hybrid(context, POLICY)
            self.assertTrue(decision.reasons, scenario_id)
            self.assertIn(decision.control_level, ("LOW", "MEDIUM", "HIGH"))
            self.assertEqual(
                decision.actuation, plan_actuation(decision.control_level, context)
            )

    def test_sensitive_facility_tightens_actuation(self):
        office = plan_actuation("MEDIUM", SAMPLES["S8"])
        daycare = plan_actuation("MEDIUM", SAMPLES["S9"])
        self.assertGreater(daycare.fan_level, office.fan_level)
        self.assertTrue(daycare.air_shower)
        self.assertFalse(office.air_shower)


class PolicyTests(unittest.TestCase):
    def test_policy_reproduces_the_rubric_on_fresh_samples(self):
        features, _, contexts = build_dataset(400, seed=99, noisy_labels=False)
        labels = [label_context(context) for context in contexts]
        predicted = predict_batch(POLICY, features)
        accuracy = sum(int(a == b) for a, b in zip(labels, predicted)) / len(labels)
        self.assertGreater(accuracy, 0.88)

    def test_probabilities_are_calibrated_to_one(self):
        prediction = POLICY.predict(SAMPLES["S5"])
        self.assertAlmostEqual(sum(prediction.probabilities.values()), 1.0, places=3)

    def test_sampled_contexts_stay_in_physical_range(self):
        import numpy as np

        rng = np.random.default_rng(7)
        for _ in range(200):
            context = sample_context(rng)
            self.assertGreaterEqual(context.health_context.spo2, 80)
            self.assertLessEqual(context.health_context.spo2, 100)
            self.assertGreaterEqual(context.vision_context.person_count, 0)


if __name__ == "__main__":
    unittest.main()
