import unittest

import numpy as np

from posture_analysis import TemporalGaitAnalyzer, analyze_pose


def base_pose() -> np.ndarray:
    points = np.zeros((17, 3), dtype=float)
    points[:, 2] = 0.9
    points[0, :2] = (100, 45)
    points[3, :2] = (88, 55)
    points[4, :2] = (112, 55)
    points[5, :2] = (75, 100)
    points[6, :2] = (125, 100)
    points[11, :2] = (85, 185)
    points[12, :2] = (115, 185)
    points[13, :2] = (85, 250)
    points[14, :2] = (115, 250)
    points[15, :2] = (85, 315)
    points[16, :2] = (115, 315)
    return points


class PostureAnalysisTests(unittest.TestCase):
    def test_balanced_frontal_pose(self):
        result = analyze_pose(base_pose())
        self.assertEqual(result.posture_status, "양호")
        self.assertEqual(result.view, "FRONT")
        self.assertIn("측면 촬영 필요", " ".join(result.observations))

    def test_shoulder_tilt_is_flagged(self):
        points = base_pose()
        points[6, 1] = 120
        result = analyze_pose(points)
        self.assertEqual(result.posture_status, "관찰 필요")
        self.assertIn("어깨 높이 비대칭 경향", result.observations)

    def test_missing_torso_is_insufficient(self):
        points = base_pose()
        points[11, 2] = 0.1
        result = analyze_pose(points)
        self.assertEqual(result.posture_status, "측정 불충분")


class TemporalGaitAnalyzerTests(unittest.TestCase):
    def test_side_walk_emits_speed_and_temporal_metrics(self):
        analyzer = TemporalGaitAnalyzer(calibration_width_m=2.0)
        result = None
        for index in range(41):
            points = base_pose()
            points[5, 0], points[6, 0] = 98, 102
            points[11, 0], points[12, 0] = 99, 101
            phase = np.sin(index * np.pi / 5)
            offset = index * 4
            points[:, 0] += offset
            points[15, 0] += phase * 25
            points[16, 0] -= phase * 25
            points[9, :2] = (95 + offset - phase * 18, 170)
            points[10, :2] = (105 + offset + phase * 10, 170)
            result = analyzer.update(points, index * 0.1, 640)

        self.assertIsNotNone(result)
        self.assertEqual(result.view, "SIDE")
        self.assertEqual(result.measurement_status, "측정 완료")
        self.assertIsNotNone(result.walking_speed_mps)
        self.assertGreaterEqual(result.observed_strides, 2)
        self.assertIsNotNone(result.gait_asymmetry)
        self.assertIsNotNone(result.arm_swing_asymmetry)

    def test_front_view_does_not_guess_walking_speed(self):
        analyzer = TemporalGaitAnalyzer()
        result = None
        for index in range(25):
            points = base_pose()
            points[15, 0] += np.sin(index) * 10
            points[16, 0] -= np.sin(index) * 10
            result = analyzer.update(points, index * 0.1, 640)

        self.assertIsNotNone(result)
        self.assertEqual(result.view, "FRONT")
        self.assertIsNone(result.walking_speed_mps)
        self.assertIsNotNone(result.base_of_support_ratio)

    def test_short_sequence_stays_in_progress(self):
        analyzer = TemporalGaitAnalyzer()
        result = None
        for index in range(4):
            result = analyzer.update(base_pose(), index * 0.1, 640)
        self.assertEqual(result.measurement_status, "측정 중")
        self.assertIn("2초 이상 보행 데이터 필요", result.observations)


if __name__ == "__main__":
    unittest.main()
