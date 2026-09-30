import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import live_context_store
from airis.body_profile import REFERENCE_PROFILE, load_user_profiles
from airis.gate_physics import DEFAULT_GATE, DEFAULT_PARAMS, map_gate_to_body, simulate
from airis.models import LEVELS
from airis.personalization import (
    TARGET_TOLERANCE,
    design_point_removal,
    plan_personalized,
)
from airis.surrogate import build_features, load_surrogate, sample_setpoints
from integrated_context import analyze_personalized_context


PERSONAS = {profile.profile_id: profile for profile in load_user_profiles()}
SURROGATE = load_surrogate()

HIGH_RISK_INPUTS = (
    {"person_count": 8, "mask_rate": 0.5, "crowding_level": "HIGH"},
    {"heart_rate": 125, "spo2": 94, "body_temperature": 38.1, "stress_index": 85},
    {"temperature": 32, "humidity": 80, "pm25": 85, "pm10": 120, "air_quality": "BAD"},
    {"facility_type": "병원", "risk_level": "HIGH", "vulnerable_population": True},
)


class PhysicsOptimizerTests(unittest.TestCase):
    def test_reference_adult_gets_the_same_cleaning_for_less_energy(self):
        plan = plan_personalized(REFERENCE_PROFILE, "HIGH", use_surrogate=False)
        self.assertEqual(plan.target_status, "MET")
        self.assertEqual(plan.baseline_violations, ())
        self.assertLess(plan.personalized.energy_wh, plan.baseline.energy_wh)
        self.assertGreater(plan.energy_saving_pct, 10)

    def test_target_is_what_the_fixed_plan_delivers_at_the_design_point(self):
        plan = plan_personalized(REFERENCE_PROFILE, "MEDIUM", use_surrogate=False)
        self.assertAlmostEqual(
            plan.target_removal, plan.baseline.removal_efficiency, places=4
        )
        self.assertEqual(plan.target_removal, round(design_point_removal(65, 5), 4))

    def test_fixed_high_plan_is_unsafe_for_a_child_and_the_fitted_one_is_not(self):
        plan = plan_personalized(PERSONAS["P2"], "HIGH", use_surrogate=False)
        self.assertEqual(len(plan.baseline_violations), 2)
        self.assertLessEqual(
            plan.personalized.face_velocity_mps, plan.limits.face_velocity_mps
        )
        self.assertLessEqual(
            plan.personalized.lateral_load_ratio, plan.limits.lateral_load_ratio + 1e-4
        )
        # Safety wins over the target, and the plan says so instead of hiding it.
        self.assertEqual(plan.target_status, "SHORTFALL")
        self.assertTrue(plan.needs_attendant)
        self.assertIn("관리자 확인 필요", plan.reasons[-1])

    def test_nozzle_rows_above_the_user_are_shut(self):
        plan = plan_personalized(PERSONAS["P2"], "MEDIUM", use_surrogate=False)
        for row in plan.personalized.rows:
            if row.height_m > PERSONAS["P2"].body.height_m + 0.02:
                self.assertEqual(row.nozzle_velocity_mps, 0.0)
        self.assertTrue(all(row.nozzle_velocity_mps > 0 for row in plan.baseline.rows))

    def test_under_cleaned_body_is_corrected_upward(self):
        plan = plan_personalized(PERSONAS["P4"], "MEDIUM", use_surrogate=False)
        self.assertLess(plan.baseline.removal_efficiency, plan.target_removal - 0.05)
        self.assertEqual(plan.target_status, "MET")
        self.assertLess(plan.energy_saving_pct, 0)

    def test_no_zone_is_abandoned_to_save_energy(self):
        plan = plan_personalized(REFERENCE_PROFILE, "MEDIUM", use_surrogate=False)
        for zone, removal in plan.personalized.zone_removal.items():
            self.assertGreater(removal, 0.5 * plan.target_removal, zone)

    def test_crowding_trades_hold_time_for_fan_speed(self):
        quiet = plan_personalized(
            REFERENCE_PROFILE, "MEDIUM", crowding_level="LOW", use_surrogate=False
        )
        crowded = plan_personalized(
            REFERENCE_PROFILE, "MEDIUM", crowding_level="HIGH", use_surrogate=False
        )
        self.assertLess(crowded.setpoints.hold_seconds, quiet.setpoints.hold_seconds)
        self.assertEqual(crowded.target_status, "MET")

    def test_same_seed_gives_the_same_plan(self):
        first = plan_personalized(PERSONAS["P3"], "MEDIUM", use_surrogate=False, seed=3)
        second = plan_personalized(PERSONAS["P3"], "MEDIUM", use_surrogate=False, seed=3)
        self.assertEqual(first.setpoints, second.setpoints)


class SafetyEnvelopeTests(unittest.TestCase):
    def test_no_plan_ever_exceeds_the_user_limits(self):
        for use_surrogate in (False, True):
            for profile in PERSONAS.values():
                for level in LEVELS:
                    plan = plan_personalized(profile, level, use_surrogate=use_surrogate)
                    label = f"{profile.profile_id}/{level}/surrogate={use_surrogate}"
                    self.assertLessEqual(
                        plan.personalized.face_velocity_mps,
                        plan.limits.face_velocity_mps,
                        label,
                    )
                    self.assertLessEqual(
                        plan.personalized.lateral_load_ratio,
                        plan.limits.lateral_load_ratio + 1e-4,
                        label,
                    )
                    self.assertLessEqual(
                        plan.setpoints.hold_seconds, DEFAULT_GATE.max_hold_seconds, label
                    )


class SurrogateTests(unittest.TestCase):
    def test_committed_surrogate_matches_the_default_gate(self):
        self.assertIsNotNone(SURROGATE)
        self.assertTrue(SURROGATE.matches(DEFAULT_GATE, DEFAULT_PARAMS))
        self.assertGreater(SURROGATE.metrics["holdout_r2"]["removal_efficiency"], 0.99)

    def test_surrogate_tracks_the_simulator_on_unseen_personas(self):
        rng = np.random.default_rng(11)
        for profile in PERSONAS.values():
            body_map = map_gate_to_body(profile)
            setpoints = sample_setpoints(rng, 200)
            truth = simulate(profile, setpoints, body_map=body_map).removal_efficiency
            predicted = SURROGATE.predict(
                build_features(profile, setpoints, body_map)
            )[:, 0]
            self.assertLess(np.abs(predicted - truth).mean(), 0.01, profile.profile_id)

    def test_surrogate_assisted_plan_agrees_with_physics_only(self):
        for profile in PERSONAS.values():
            for level in LEVELS:
                physics = plan_personalized(profile, level, use_surrogate=False)
                fused = plan_personalized(profile, level, surrogate=SURROGATE)
                label = f"{profile.profile_id}/{level}"
                self.assertEqual(fused.method, "SURROGATE+PHYSICS", label)
                self.assertEqual(fused.target_status, physics.target_status, label)
                self.assertAlmostEqual(
                    fused.personalized.removal_efficiency,
                    physics.personalized.removal_efficiency,
                    delta=TARGET_TOLERANCE,
                    msg=label,
                )
                self.assertLess(fused.simulator_calls, physics.simulator_calls / 5, label)

    def test_stale_surrogate_falls_back_to_physics(self):
        wider_gate = type(DEFAULT_GATE)(passage_width_m=1.0)
        plan = plan_personalized(
            REFERENCE_PROFILE, "MEDIUM", surrogate=SURROGATE, gate=wider_gate
        )
        self.assertEqual(plan.method, "PHYSICS")


class IntegrationTests(unittest.TestCase):
    def test_context_decision_drives_the_personalized_plan(self):
        context, decision, plan = analyze_personalized_context(
            *HIGH_RISK_INPUTS, PERSONAS["P5"].to_dict()
        )
        self.assertEqual(decision.control_level, "HIGH")
        self.assertEqual(plan.control_level, "HIGH")
        self.assertEqual(plan.profile, PERSONAS["P5"])
        # The baseline is the tightened plan the decision layer actually issued.
        self.assertEqual(
            plan.baseline.passage_time_s,
            round(1.2 / 0.8 + decision.actuation.gate_hold_seconds, 2),
        )
        self.assertLessEqual(plan.personalized.face_velocity_mps, 6.0)

    def test_missing_profile_falls_back_to_the_reference_adult(self):
        _, _, plan = analyze_personalized_context(*HIGH_RISK_INPUTS, None)
        self.assertEqual(plan.profile, REFERENCE_PROFILE)
        _, _, plan = analyze_personalized_context(*HIGH_RISK_INPUTS, {})
        self.assertEqual(plan.profile, REFERENCE_PROFILE)

    def test_plan_serializes_to_json_ready_dict(self):
        payload = plan_personalized(PERSONAS["P1"], "LOW").to_dict()
        self.assertEqual(payload["profile"]["profile_id"], "P1")
        self.assertEqual(len(payload["personalized"]["rows"]), DEFAULT_GATE.row_count)
        self.assertIn(payload["target_status"], ("MET", "WITHIN_TOLERANCE", "SHORTFALL"))

    def test_user_profile_store_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "user_profile.json"
            with mock.patch.object(live_context_store, "USER_PROFILE_PATH", path):
                self.assertEqual(live_context_store.read_user_profile(), {})
                live_context_store.write_user_profile(PERSONAS["P6"].to_dict())
                self.assertEqual(
                    live_context_store.read_user_profile(), PERSONAS["P6"].to_dict()
                )


if __name__ == "__main__":
    unittest.main()
