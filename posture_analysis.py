"""Realtime, non-diagnostic posture cues from a single RGB camera."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import av
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from airis.body_profile import BodyProfileTracker


ROOT = Path(__file__).resolve().parent
POSE_WEIGHTS = ROOT / "models/yolo11n-pose.pt"

# COCO pose skeleton. Keypoints: nose, eyes, ears, shoulders, elbows, wrists,
# hips, knees, ankles.
SKELETON = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)


@dataclass(frozen=True)
class PostureResult:
    person_detected: bool
    posture_status: str
    view: str
    head_forward_angle: Optional[float]
    shoulder_tilt_angle: Optional[float]
    trunk_lean_angle: Optional[float]
    movement_level: str
    observations: tuple[str, ...]


@dataclass(frozen=True)
class GaitResult:
    measurement_status: str
    view: str
    duration_sec: float
    observed_strides: int
    walking_speed_mps: Optional[float]
    gait_asymmetry: Optional[float]
    arm_swing_asymmetry: Optional[float]
    trunk_flexion_deg: Optional[float]
    base_of_support_ratio: Optional[float]
    measurement_confidence: float
    observations: tuple[str, ...]


EMPTY_RESULT = PostureResult(
    person_detected=False,
    posture_status="측정 대기",
    view="UNKNOWN",
    head_forward_angle=None,
    shoulder_tilt_angle=None,
    trunk_lean_angle=None,
    movement_level="UNKNOWN",
    observations=("전신이 보이도록 카메라 앞에 서 주세요.",),
)

EMPTY_GAIT_RESULT = GaitResult(
    measurement_status="측정 대기",
    view="UNKNOWN",
    duration_sec=0.0,
    observed_strides=0,
    walking_speed_mps=None,
    gait_asymmetry=None,
    arm_swing_asymmetry=None,
    trunk_flexion_deg=None,
    base_of_support_ratio=None,
    measurement_confidence=0.0,
    observations=("전신이 보이도록 측면에서 통과해 주세요.",),
)


def _visible(points: np.ndarray, *indices: int, threshold: float = 0.35) -> bool:
    return all(points[index, 2] >= threshold for index in indices)


def _center(points: np.ndarray, left: int, right: int) -> np.ndarray:
    return (points[left, :2] + points[right, :2]) / 2.0


def _deviation_from_vertical(top: np.ndarray, bottom: np.ndarray) -> float:
    dx = abs(float(top[0] - bottom[0]))
    dy = max(abs(float(top[1] - bottom[1])), 1e-6)
    return math.degrees(math.atan2(dx, dy))


def analyze_pose(points: np.ndarray, movement_level: str = "STILL") -> PostureResult:
    """Convert one COCO pose into camera-sensitive posture observations."""
    if points.shape != (17, 3):
        raise ValueError("COCO pose keypoints must have shape (17, 3).")
    if not _visible(points, 5, 6, 11, 12):
        return PostureResult(
            person_detected=True,
            posture_status="측정 불충분",
            view="UNKNOWN",
            head_forward_angle=None,
            shoulder_tilt_angle=None,
            trunk_lean_angle=None,
            movement_level=movement_level,
            observations=("어깨와 골반이 모두 보이도록 거리를 조정해 주세요.",),
        )

    shoulder_center = _center(points, 5, 6)
    hip_center = _center(points, 11, 12)
    shoulder_width = float(np.linalg.norm(points[5, :2] - points[6, :2]))
    torso_length = max(float(np.linalg.norm(shoulder_center - hip_center)), 1.0)
    view = "SIDE" if shoulder_width / torso_length < 0.45 else "FRONT"

    shoulder_dx = max(abs(float(points[6, 0] - points[5, 0])), 1e-6)
    shoulder_dy = abs(float(points[6, 1] - points[5, 1]))
    shoulder_tilt = math.degrees(math.atan2(shoulder_dy, shoulder_dx))
    trunk_lean = _deviation_from_vertical(shoulder_center, hip_center)

    head_forward: Optional[float] = None
    if view == "SIDE":
        candidates = []
        for ear, shoulder in ((3, 5), (4, 6)):
            if _visible(points, ear, shoulder):
                candidates.append(
                    (
                        points[ear, 2] + points[shoulder, 2],
                        _deviation_from_vertical(
                            points[ear, :2],
                            points[shoulder, :2],
                        ),
                    )
                )
        if candidates:
            head_forward = max(candidates, key=lambda item: item[0])[1]

    observations: list[str] = []
    if shoulder_tilt >= 8:
        observations.append("어깨 높이 비대칭 경향")
    if trunk_lean >= 12:
        observations.append("상체 기울어짐 경향")
    if head_forward is not None and head_forward >= 18:
        observations.append("전방 머리 자세 경향")
    if movement_level == "MOVING":
        observations.append("보행 움직임 감지")
    if view != "SIDE":
        observations.append("목 정렬 평가는 측면 촬영 필요")

    posture_issue = any(
        label in observations
        for label in (
            "어깨 높이 비대칭 경향",
            "상체 기울어짐 경향",
            "전방 머리 자세 경향",
        )
    )
    status = "관찰 필요" if posture_issue else "양호"
    if not observations:
        observations.append("뚜렷한 자세 경향 없음")
    return PostureResult(
        person_detected=True,
        posture_status=status,
        view=view,
        head_forward_angle=round(head_forward, 1)
        if head_forward is not None
        else None,
        shoulder_tilt_angle=round(shoulder_tilt, 1),
        trunk_lean_angle=round(trunk_lean, 1),
        movement_level=movement_level,
        observations=tuple(observations),
    )


class TemporalGaitAnalyzer:
    """Estimate short-passage gait biomarkers from a tracked pose sequence.

    ``calibration_width_m`` is the real-world width represented by the full
    image. Walking speed is only emitted for a side view with enough horizontal
    travel; it is intentionally left ``None`` for frontal/depth motion.
    """

    def __init__(
        self,
        calibration_width_m: float = 2.0,
        history_seconds: float = 8.0,
    ) -> None:
        if calibration_width_m <= 0:
            raise ValueError("calibration_width_m must be positive.")
        self.calibration_width_m = calibration_width_m
        self.history_seconds = history_seconds
        self.samples: deque[tuple[float, np.ndarray, int]] = deque()

    def reset(self) -> None:
        self.samples.clear()

    def update(
        self,
        points: np.ndarray,
        timestamp: float,
        frame_width: int,
    ) -> GaitResult:
        if points.shape != (17, 3):
            raise ValueError("COCO pose keypoints must have shape (17, 3).")
        self.samples.append((float(timestamp), points.copy(), int(frame_width)))
        cutoff = timestamp - self.history_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        return self.result()

    def result(self) -> GaitResult:
        if not self.samples:
            return EMPTY_GAIT_RESULT

        valid = [
            sample
            for sample in self.samples
            if _visible(sample[1], 5, 6, 11, 12, 15, 16)
        ]
        if len(valid) < 3:
            return GaitResult(
                measurement_status="전신 인식 필요",
                view="UNKNOWN",
                duration_sec=0.0,
                observed_strides=0,
                walking_speed_mps=None,
                gait_asymmetry=None,
                arm_swing_asymmetry=None,
                trunk_flexion_deg=None,
                base_of_support_ratio=None,
                measurement_confidence=0.1,
                observations=("어깨·골반·양쪽 발목이 모두 보여야 합니다.",),
            )

        duration = max(valid[-1][0] - valid[0][0], 0.0)
        views = [analyze_pose(points).view for _, points, _ in valid]
        side_ratio = views.count("SIDE") / len(views)
        view = "SIDE" if side_ratio >= 0.6 else "FRONT"

        ankle_signal: list[float] = []
        hip_x: list[float] = []
        left_leg: list[float] = []
        right_leg: list[float] = []
        left_arm: list[float] = []
        right_arm: list[float] = []
        trunk_angles: list[float] = []
        stance_ratios: list[float] = []
        visible_scores: list[float] = []

        for _, points, _ in valid:
            shoulder_center = _center(points, 5, 6)
            hip_center = _center(points, 11, 12)
            torso = max(float(np.linalg.norm(shoulder_center - hip_center)), 1.0)
            hip_x.append(float(hip_center[0]))
            ankle_signal.append(float((points[15, 0] - points[16, 0]) / torso))
            left_leg.append(float((points[15, 0] - hip_center[0]) / torso))
            right_leg.append(float((points[16, 0] - hip_center[0]) / torso))
            trunk_angles.append(_deviation_from_vertical(shoulder_center, hip_center))

            shoulder_width = max(
                float(np.linalg.norm(points[5, :2] - points[6, :2])),
                1.0,
            )
            stance_ratios.append(
                float(abs(points[15, 0] - points[16, 0]) / shoulder_width)
            )
            if _visible(points, 5, 6, 9, 10):
                left_arm.append(float((points[9, 0] - points[5, 0]) / torso))
                right_arm.append(float((points[10, 0] - points[6, 0]) / torso))
            visible_scores.append(
                float(np.mean(points[[5, 6, 9, 10, 11, 12, 15, 16], 2] >= 0.35))
            )

        observed_strides = self._count_strides(ankle_signal)
        enough_time = duration >= 2.0
        enough_strides = observed_strides >= 2
        walking_speed = self._walking_speed(valid, hip_x, view, duration)
        gait_asymmetry = (
            self._symmetry_index(left_leg, right_leg)
            if view == "SIDE" and enough_time
            else None
        )
        arm_asymmetry = (
            self._symmetry_index(left_arm, right_arm)
            if view == "SIDE" and enough_time and len(left_arm) >= 3
            else None
        )
        trunk_flexion = (
            round(float(np.median(trunk_angles)), 1) if view == "SIDE" else None
        )
        base_of_support = (
            round(float(np.median(stance_ratios)), 2)
            if view == "FRONT"
            else None
        )

        confidence = min(duration / 4.0, 1.0) * float(np.mean(visible_scores))
        if view == "SIDE":
            confidence *= 0.7 + 0.3 * side_ratio
        confidence = round(min(max(confidence, 0.0), 1.0), 2)

        observations: list[str] = []
        if view == "FRONT":
            observations.append("속도·보행 비대칭 측정은 측면 촬영 필요")
        elif walking_speed is None:
            observations.append("속도 계산을 위해 화면을 가로질러 걸어 주세요")
        if not enough_time:
            observations.append("2초 이상 보행 데이터 필요")
        if not enough_strides:
            observations.append("비대칭 확정을 위해 2개 스트라이드 필요")
        if gait_asymmetry is not None and gait_asymmetry >= 0.2:
            observations.append("좌우 보행 비대칭 경향")
        if arm_asymmetry is not None and arm_asymmetry >= 0.25:
            observations.append("팔 흔들림 비대칭 경향")
        if trunk_flexion is not None and trunk_flexion >= 12:
            observations.append("체간 전방굴곡 경향")
        if base_of_support is not None and base_of_support >= 1.2:
            observations.append("넓은 지지기저면 경향")
        if not observations:
            observations.append("뚜렷한 보행 비대칭 경향 없음")

        status = "측정 완료" if enough_time and enough_strides else "측정 중"
        return GaitResult(
            measurement_status=status,
            view=view,
            duration_sec=round(duration, 1),
            observed_strides=observed_strides,
            walking_speed_mps=walking_speed,
            gait_asymmetry=gait_asymmetry,
            arm_swing_asymmetry=arm_asymmetry,
            trunk_flexion_deg=trunk_flexion,
            base_of_support_ratio=base_of_support,
            measurement_confidence=confidence,
            observations=tuple(observations),
        )

    def _walking_speed(
        self,
        samples: list[tuple[float, np.ndarray, int]],
        hip_x: list[float],
        view: str,
        duration: float,
    ) -> Optional[float]:
        if view != "SIDE" or duration < 1.0:
            return None
        frame_width = max(samples[-1][2], 1)
        travel_ratio = abs(hip_x[-1] - hip_x[0]) / frame_width
        if travel_ratio < 0.08:
            return None
        speed = travel_ratio * self.calibration_width_m / duration
        return round(float(speed), 2)

    @staticmethod
    def _symmetry_index(left: list[float], right: list[float]) -> Optional[float]:
        if len(left) < 3 or len(right) < 3:
            return None
        left_range = float(np.percentile(left, 90) - np.percentile(left, 10))
        right_range = float(np.percentile(right, 90) - np.percentile(right, 10))
        denominator = left_range + right_range
        if denominator < 0.08:
            return None
        return round(abs(left_range - right_range) / denominator, 2)

    @staticmethod
    def _count_strides(signal: list[float]) -> int:
        if len(signal) < 5:
            return 0
        smoothed = np.convolve(np.asarray(signal), np.ones(3) / 3.0, mode="valid")
        amplitude = float(np.percentile(smoothed, 90) - np.percentile(smoothed, 10))
        if amplitude < 0.12:
            return 0
        center = float(np.median(smoothed))
        signs = np.sign(smoothed - center)
        crossings = int(np.sum(signs[1:] * signs[:-1] < 0))
        return max(crossings // 2, 0)


class LivePostureState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payload: dict[str, Any] = {
            "timestamp": None,
            "posture_context": asdict(EMPTY_RESULT),
            "gait_context": asdict(EMPTY_GAIT_RESULT),
            "body_context": None,
        }

    def update(
        self,
        result: PostureResult,
        gait_result: GaitResult = EMPTY_GAIT_RESULT,
        body_context: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._payload = {
                "timestamp": round(time.time(), 3),
                "posture_context": asdict(result),
                "gait_context": asdict(gait_result),
                # Keep the last good 체형 estimate while the person is out of frame.
                "body_context": body_context or self._payload["body_context"],
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            body_context = self._payload["body_context"]
            return {
                "timestamp": self._payload["timestamp"],
                "posture_context": dict(self._payload["posture_context"]),
                "gait_context": dict(self._payload["gait_context"]),
                "body_context": dict(body_context) if body_context else None,
            }


class WebcamPostureProcessor:
    """Run lightweight pose inference and draw the selected person's skeleton."""

    def __init__(
        self,
        state: LivePostureState,
        model: YOLO,
        analyze_every: int = 4,
        calibration_width_m: float = 2.0,
    ) -> None:
        self.state = state
        self.model = model
        self.analyze_every = analyze_every
        self.frame_index = 0
        self.points: Optional[np.ndarray] = None
        self.previous_hip_y: Optional[float] = None
        self.motion_score = 0.0
        self.gait_analyzer = TemporalGaitAnalyzer(calibration_width_m)
        self.body_tracker = BodyProfileTracker()
        self._inference_lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        with self._inference_lock:
            if self.frame_index % self.analyze_every == 0:
                prediction = self.model.predict(
                    image,
                    conf=0.35,
                    imgsz=416,
                    verbose=False,
                )[0]
                self.points = self._largest_pose(prediction)
                if self.points is None:
                    self.state.update(EMPTY_RESULT, self.gait_analyzer.result())
                else:
                    movement = self._movement_level(self.points)
                    gait_result = self.gait_analyzer.update(
                        self.points,
                        time.monotonic(),
                        image.shape[1],
                    )
                    body = self.body_tracker.update(
                        self.points,
                        self.gait_analyzer.calibration_width_m / image.shape[1],
                    )
                    self.state.update(
                        analyze_pose(self.points, movement),
                        gait_result,
                        body.to_dict() if body is not None else None,
                    )

            annotated = image.copy()
            if self.points is not None:
                self._draw_pose(annotated, self.points)
            self.frame_index += 1
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    @staticmethod
    def _largest_pose(prediction: Any) -> Optional[np.ndarray]:
        if prediction.keypoints is None or len(prediction.keypoints.xy) == 0:
            return None
        boxes = prediction.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        index = int(np.argmax(areas))
        xy = prediction.keypoints.xy[index].cpu().numpy()
        confidence = prediction.keypoints.conf[index].cpu().numpy()
        return np.column_stack((xy, confidence))

    def _movement_level(self, points: np.ndarray) -> str:
        if not _visible(points, 11, 12):
            return "UNKNOWN"
        hip_y = float(_center(points, 11, 12)[1])
        if self.previous_hip_y is not None:
            delta = abs(hip_y - self.previous_hip_y)
            self.motion_score = 0.75 * self.motion_score + 0.25 * delta
        self.previous_hip_y = hip_y
        return "MOVING" if self.motion_score >= 2.5 else "STILL"

    @staticmethod
    def _draw_pose(image: np.ndarray, points: np.ndarray) -> None:
        for start, end in SKELETON:
            if _visible(points, start, end):
                start_point = tuple(points[start, :2].astype(int))
                end_point = tuple(points[end, :2].astype(int))
                cv2.line(image, start_point, end_point, (84, 214, 199), 2)
        for x, y, confidence in points:
            if confidence >= 0.35:
                cv2.circle(image, (int(x), int(y)), 4, (255, 190, 70), -1)

def load_pose_model() -> YOLO:
    POSE_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    weights = attempt_download_asset(POSE_WEIGHTS)
    return YOLO(str(weights))
