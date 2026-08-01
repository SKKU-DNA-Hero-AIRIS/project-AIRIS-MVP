"""Realtime webcam inference helpers for the AIRIS Streamlit demo."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Any

import av
import cv2

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency on restricted environments
    YOLO = None

from live_context_store import write_vision_context
from vision_demo import (
    VisionContextResult,
    boxes_for_class,
    crowding_level,
    detect_masks_in_people,
    download_models,
    draw_detection,
)


class LiveVisionState:
    """Thread-safe bridge between the WebRTC callback and Streamlit UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payload: dict[str, Any] = {
            "timestamp": None,
            "vision_context": asdict(VisionContextResult(0, 0, 0.0, "LOW")),
        }

    def update(self, result: VisionContextResult) -> None:
        with self._lock:
            self._payload = {
                "timestamp": round(time.time(), 3),
                "vision_context": asdict(result),
            }
        write_vision_context(asdict(result))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "timestamp": self._payload["timestamp"],
                "vision_context": dict(self._payload["vision_context"]),
            }


class WebcamVisionProcessor:
    """Run person and mask detection on incoming browser-camera frames."""

    def __init__(
        self,
        state: LiveVisionState,
        person_model: YOLO,
        mask_model: YOLO,
        person_confidence: float = 0.28,
        mask_confidence: float = 0.08,
        analyze_every: int = 6,
        mask_every: int = 18,
    ) -> None:
        self.state = state
        self.person_model = person_model
        self.mask_model = mask_model
        self.person_confidence = person_confidence
        self.mask_confidence = mask_confidence
        self.analyze_every = analyze_every
        self.mask_every = mask_every
        self.frame_index = 0
        self.people: list[tuple[int, int, int, int, float]] = []
        self.masks: list[tuple[int, int, int, int, float]] = []
        self._inference_lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        with self._inference_lock:
            if self.frame_index % self.analyze_every == 0:
                if self.person_model is None or self.mask_model is None:
                    result = VisionContextResult(0, 0, 0.0, "LOW")
                    self.people = []
                    self.masks = []
                    self.state.update(result)
                else:
                    person_result = self.person_model.predict(
                        image,
                        classes=[0],
                        conf=self.person_confidence,
                        imgsz=416,
                        verbose=False,
                    )[0]
                    self.people = boxes_for_class(person_result, {0})
                    if self.frame_index % self.mask_every == 0:
                        self.masks = detect_masks_in_people(
                            image,
                            self.people,
                            self.mask_model,
                            self.mask_confidence,
                            image_size=256,
                        )
                    self.masks = self.masks[: len(self.people)]

                    people_count = len(self.people)
                    masked_count = min(len(self.masks), people_count)
                    result = VisionContextResult(
                        person_count=people_count,
                        masked_person_count=masked_count,
                        mask_rate=round(masked_count / people_count, 3)
                        if people_count
                        else 0.0,
                        crowding_level=crowding_level(people_count),
                    )
                    self.state.update(result)

            annotated = image.copy()
            for detection in self.people:
                draw_detection(annotated, detection, "person", (75, 220, 120))
            for detection in self.masks:
                draw_detection(annotated, detection, "mask", (255, 190, 60))

            context = self.state.snapshot()["vision_context"]
            cv2.rectangle(annotated, (16, 14), (410, 96), (16, 22, 28), -1)
            cv2.putText(
                annotated,
                (
                    f"people {context['person_count']} | "
                    f"masks {context['masked_person_count']} | "
                    f"rate {context['mask_rate']:.2f}"
                ),
                (30, 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"crowding {context['crowding_level']}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (75, 220, 220),
                2,
                cv2.LINE_AA,
            )
            self.frame_index += 1

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def load_webcam_models() -> tuple[YOLO, YOLO]:
    if YOLO is None:
        return None, None
    try:
        person_weights, mask_weights = download_models()
        return YOLO(str(person_weights)), YOLO(str(mask_weights))
    except Exception:
        return None, None
