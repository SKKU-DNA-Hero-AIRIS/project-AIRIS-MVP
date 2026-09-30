import unittest

import numpy as np

from airis.body_profile import (
    REFERENCE_PROFILE,
    UserProfile,
    UserTraits,
    body_from_measurements,
)
from airis.gate_physics import (
    DEFAULT_GATE,
    comfort_limits,
    describe,
    fan_power,
    fine_fraction_from_pm,
    map_gate_to_body,
    simulate,
    uniform_rows,
)


CHILD = UserProfile(
    body_from_measurements(1.15, 0.29, mass_kg=21), UserTraits(1.0, age_group="CHILD")
)
BROAD = UserProfile(body_from_measurements(1.85, 0.52, mass_kg=98), UserTraits(1.3))


class GateBodyMapTests(unittest.TestCase):
    def test_rows_above_a_child_face_nobody(self):
        child_map = map_gate_to_body(CHILD)
        adult_map = map_gate_to_body(REFERENCE_PROFILE)
        self.assertGreater(child_map.row_zone.count(None), adult_map.row_zone.count(None))
        for height, zone in zip(DEFAULT_GATE.row_heights_m, child_map.row_zone):
            self.assertEqual(zone is None, height > 1.15 + 0.02, height)

    def test_every_zone_of_the_reference_adult_has_a_nozzle_row(self):
        self.assertTrue(np.all(map_gate_to_body(REFERENCE_PROFILE).rows_per_zone >= 1))

    def test_broad_torso_sits_closer_and_is_hit_harder(self):
        broad = map_gate_to_body(BROAD)
        slim_child = map_gate_to_body(CHILD)
        self.assertGreater(broad.zone_decay[1], slim_child.zone_decay[1])
        self.assertLessEqual(broad.zone_decay[1], 1.0)


class SimulationTests(unittest.TestCase):
    def test_removal_rises_with_velocity_and_with_hold(self):
        velocity_sweep = np.array([[v, v, v, 5.0] for v in (4, 8, 12, 16, 20)])
        removal = simulate(REFERENCE_PROFILE, velocity_sweep).removal_efficiency
        self.assertTrue(np.all(np.diff(removal) > 0))

        hold_sweep = np.array([[15, 15, 15, hold] for hold in (0, 3, 6, 12)])
        removal = simulate(REFERENCE_PROFILE, hold_sweep).removal_efficiency
        self.assertTrue(np.all(np.diff(removal) > 0))

    def test_removal_is_a_fraction(self):
        rng = np.random.default_rng(7)
        setpoints = rng.uniform(0, [20, 20, 20, 12], size=(500, 4))
        batch = simulate(BROAD, setpoints)
        self.assertTrue(np.all(batch.removal_efficiency >= 0))
        self.assertTrue(np.all(batch.removal_efficiency <= 1))
        self.assertTrue(np.all(batch.zone_removal <= 1))

    def test_shut_nozzles_clean_nothing_and_draw_nothing(self):
        batch = simulate(REFERENCE_PROFILE, np.array([[0.2, 0.0, 0.4, 8.0]]))
        self.assertEqual(batch.removal_efficiency[0], 0.0)
        self.assertEqual(batch.power_w[0], 0.0)

    def test_fan_power_follows_the_cube_law(self):
        slow = fan_power(uniform_rows(10.0))[0]
        fast = fan_power(uniform_rows(20.0))[0]
        self.assertAlmostEqual(fast / slow, 8.0, places=6)

    def test_same_jets_push_a_light_user_harder(self):
        rows, hold = uniform_rows(20.0)[0], 9
        child = describe(CHILD, rows, hold)
        adult = describe(REFERENCE_PROFILE, rows, hold)
        self.assertGreater(child.lateral_load_ratio, 2 * adult.lateral_load_ratio)

    def test_bulky_clothing_is_harder_to_clean(self):
        bulky = UserProfile(REFERENCE_PROFILE.body, UserTraits(1.3, clothing="BULKY"))
        setpoints = np.array([[10.0, 16.0, 16.0, 6.0]])
        self.assertLess(
            simulate(bulky, setpoints).removal_efficiency[0],
            simulate(REFERENCE_PROFILE, setpoints).removal_efficiency[0],
        )

    def test_finer_dust_is_harder_to_clean(self):
        setpoints = np.array([[10.0, 16.0, 16.0, 6.0]])
        coarse = simulate(REFERENCE_PROFILE, setpoints, fine_fraction=0.2)
        fine = simulate(REFERENCE_PROFILE, setpoints, fine_fraction=0.8)
        self.assertLess(fine.removal_efficiency[0], coarse.removal_efficiency[0])

    def test_describe_matches_the_batch_simulation(self):
        body_map = map_gate_to_body(REFERENCE_PROFILE)
        setpoints = np.array([[8.0, 18.0, 15.0, 7.0]])
        batch = simulate(REFERENCE_PROFILE, setpoints, body_map=body_map)
        rows = [
            0.0 if zone is None else setpoints[0, zone] for zone in body_map.row_zone
        ]
        report = describe(REFERENCE_PROFILE, np.array(rows), 7.0)
        self.assertAlmostEqual(
            report.removal_efficiency, batch.removal_efficiency[0], places=4
        )
        self.assertEqual(len(report.rows), DEFAULT_GATE.row_count)


class ComfortLimitTests(unittest.TestCase):
    def test_vulnerable_users_get_stricter_limits(self):
        adult = comfort_limits(UserTraits())
        for traits in (
            UserTraits(age_group="CHILD"),
            UserTraits(age_group="SENIOR"),
            UserTraits(comfort_sensitivity="SENSITIVE"),
            UserTraits(mobility="ASSISTED"),
        ):
            limits = comfort_limits(traits)
            self.assertLess(limits.face_velocity_mps, adult.face_velocity_mps)
            self.assertLess(limits.lateral_load_ratio, adult.lateral_load_ratio)

    def test_slow_walker_sits_between_normal_and_vulnerable(self):
        slow = comfort_limits(UserTraits(mobility="SLOW"))
        self.assertEqual(slow.face_velocity_mps, 8.0)

    def test_fine_fraction_uses_pm_ratio_when_pm10_is_known(self):
        self.assertEqual(fine_fraction_from_pm(30, 50), 0.6)
        self.assertEqual(fine_fraction_from_pm(30), 0.4)
        self.assertEqual(fine_fraction_from_pm(90, 100), 0.8)


if __name__ == "__main__":
    unittest.main()
