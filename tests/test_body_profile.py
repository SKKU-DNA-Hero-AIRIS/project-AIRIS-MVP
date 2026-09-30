import unittest

import numpy as np

from airis.body_profile import (
    ANKLE_HEIGHT_RATIO,
    HIP_HEIGHT_RATIO,
    NOSE_HEIGHT_RATIO,
    SHOULDER_HEIGHT_RATIO,
    BodyProfile,
    BodyProfileTracker,
    UserProfile,
    UserTraits,
    body_from_measurements,
    estimate_body_profile,
    load_user_profiles,
    traits_from_gait,
)


METERS_PER_PIXEL = 0.005


def standing_pose(
    height_m: float,
    shoulder_width_m: float,
    *,
    side_view: bool = False,
    floor_y: float = 450.0,
    center_x: float = 320.0,
) -> np.ndarray:
    """A COCO pose for a person of known size at the calibrated distance."""

    def y(ratio: float) -> float:
        return floor_y - height_m * ratio / METERS_PER_PIXEL

    half_shoulder = shoulder_width_m / 2.0 / METERS_PER_PIXEL
    if side_view:
        half_shoulder *= 0.1
    half_hip = half_shoulder * 0.6

    points = np.zeros((17, 3), dtype=float)
    points[:, 2] = 0.9
    points[0, :2] = (center_x, y(NOSE_HEIGHT_RATIO))
    points[5, :2] = (center_x - half_shoulder, y(SHOULDER_HEIGHT_RATIO))
    points[6, :2] = (center_x + half_shoulder, y(SHOULDER_HEIGHT_RATIO))
    points[11, :2] = (center_x - half_hip, y(HIP_HEIGHT_RATIO))
    points[12, :2] = (center_x + half_hip, y(HIP_HEIGHT_RATIO))
    points[15, :2] = (center_x - half_hip, y(ANKLE_HEIGHT_RATIO))
    points[16, :2] = (center_x + half_hip, y(ANKLE_HEIGHT_RATIO))
    return points


class BodySegmentTests(unittest.TestCase):
    def test_segments_tile_the_body_from_floor_to_crown(self):
        body = body_from_measurements(1.70, 0.42, mass_kg=65)
        head, upper, lower = body.segments()
        self.assertEqual((head.zone, upper.zone, lower.zone), ("HEAD", "UPPER", "LOWER"))
        self.assertEqual(lower.z_low_m, 0.0)
        self.assertEqual(lower.z_high_m, upper.z_low_m)
        self.assertEqual(upper.z_high_m, head.z_low_m)
        self.assertEqual(head.z_high_m, 1.70)

    def test_segment_areas_sum_to_body_surface_area(self):
        body = body_from_measurements(1.70, 0.42, mass_kg=65)
        total = sum(segment.surface_area_m2 for segment in body.segments())
        self.assertAlmostEqual(total, body.surface_area_m2, places=2)
        # Du Bois for 65 kg / 1.70 m is about 1.75 m^2.
        self.assertAlmostEqual(body.surface_area_m2, 1.75, delta=0.05)

    def test_child_carries_more_area_in_the_head(self):
        child = body_from_measurements(1.15, 0.29)
        adult = body_from_measurements(1.72, 0.43)
        child_share = child.segments()[0].surface_area_m2 / child.surface_area_m2
        adult_share = adult.segments()[0].surface_area_m2 / adult.surface_area_m2
        self.assertGreater(child_share, adult_share)

    def test_seated_lower_zone_is_as_wide_as_the_wheelchair(self):
        seated = body_from_measurements(1.30, 0.42, mass_kg=70, posture="SEATED")
        standing = body_from_measurements(1.70, 0.42, mass_kg=70)
        self.assertGreater(
            seated.segments()[2].half_width_m, standing.segments()[2].half_width_m
        )
        self.assertEqual(seated.segments()[0].z_high_m, 1.30)

    def test_build_follows_shoulder_to_height_ratio(self):
        self.assertEqual(body_from_measurements(1.80, 0.40).build, "SLIM")
        self.assertEqual(body_from_measurements(1.70, 0.44).build, "AVERAGE")
        self.assertEqual(body_from_measurements(1.70, 0.50).build, "BROAD")

    def test_invalid_body_is_rejected(self):
        with self.assertRaises(ValueError):
            BodyProfile(height_m=3.0, shoulder_width_m=0.4, hip_width_m=0.3, mass_kg=60)
        with self.assertRaises(ValueError):
            UserTraits(clothing="ARMOR")


class PoseEstimationTests(unittest.TestCase):
    def test_frontal_pose_recovers_height_and_shoulder_width(self):
        body = estimate_body_profile(standing_pose(1.70, 0.42), METERS_PER_PIXEL)
        self.assertEqual(body.source, "POSE_FRONT")
        self.assertAlmostEqual(body.height_m, 1.70, delta=0.02)
        self.assertAlmostEqual(body.shoulder_width_m, 0.42, delta=0.01)

    def test_side_view_falls_back_to_anthropometric_shoulder_ratio(self):
        body = estimate_body_profile(
            standing_pose(1.70, 0.42, side_view=True), METERS_PER_PIXEL
        )
        self.assertEqual(body.source, "POSE_SIDE")
        self.assertAlmostEqual(body.height_m, 1.70, delta=0.02)
        self.assertAlmostEqual(body.shoulder_width_m, 1.70 * 0.259, delta=0.01)
        self.assertLess(body.confidence, 0.9)

    def test_child_pose_is_estimated_as_child_sized(self):
        body = estimate_body_profile(standing_pose(1.15, 0.29), METERS_PER_PIXEL)
        self.assertTrue(body.is_child_sized)
        self.assertLess(body.mass_kg, 30)

    def test_hidden_ankles_return_no_estimate(self):
        points = standing_pose(1.70, 0.42)
        points[[15, 16], 2] = 0.1
        self.assertIsNone(estimate_body_profile(points, METERS_PER_PIXEL))

    def test_tracker_waits_for_enough_frames_and_rejects_one_outlier(self):
        tracker = BodyProfileTracker(minimum=3)
        self.assertIsNone(tracker.update(standing_pose(1.70, 0.42), METERS_PER_PIXEL))
        self.assertIsNone(tracker.update(standing_pose(1.71, 0.42), METERS_PER_PIXEL))
        tracker.update(standing_pose(2.10, 0.42), METERS_PER_PIXEL)
        body = tracker.update(standing_pose(1.69, 0.42), METERS_PER_PIXEL)
        self.assertAlmostEqual(body.height_m, 1.70, delta=0.03)


class UserTraitTests(unittest.TestCase):
    def test_slow_gait_selects_slow_mobility(self):
        body = body_from_measurements(1.60, 0.38)
        traits = traits_from_gait({"walking_speed_mps": 0.6}, body)
        self.assertEqual(traits.mobility, "SLOW")
        self.assertTrue(traits.is_vulnerable)

    def test_child_sized_body_selects_child_limits(self):
        body = body_from_measurements(1.15, 0.29)
        traits = traits_from_gait({"walking_speed_mps": 1.1}, body)
        self.assertEqual(traits.age_group, "CHILD")

    def test_unmeasured_gait_keeps_opt_in_traits(self):
        body = body_from_measurements(1.62, 0.39)
        base = UserTraits(walking_speed_mps=0.9, comfort_sensitivity="SENSITIVE")
        traits = traits_from_gait({"walking_speed_mps": None}, body, base)
        self.assertEqual(traits.walking_speed_mps, 0.9)
        self.assertEqual(traits.comfort_sensitivity, "SENSITIVE")


class PersonaTests(unittest.TestCase):
    def test_personas_load_in_order_and_round_trip(self):
        profiles = load_user_profiles()
        self.assertEqual(
            [profile.profile_id for profile in profiles],
            ["P1", "P2", "P3", "P4", "P5", "P6"],
        )
        for profile in profiles:
            self.assertEqual(UserProfile.from_dict(profile.to_dict()), profile)


if __name__ == "__main__":
    unittest.main()
