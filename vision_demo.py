"""Extract AIRIS vision_context from a video and create an annotated demo.

Usage:
    python vision_demo.py input.mp4 --output-dir outputs/vision_demo
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


@dataclass(frozen=True)
class VisionContextResult:
    person_count: int
    masked_person_count: int
    mask_rate: float
    crowding_level: str


def crowding_level(person_count: int) -> str:
    if person_count >= 8:
        return "HIGH"
    if person_count >= 4:
        return "MEDIUM"
    return "LOW"


def download_models() -> tuple[Path, Path]:
    """Download pinned, reusable model artifacts into the project."""
    person_model = MODEL_DIR / "person" / "yolo11n.pt"
    mask_model = MODEL_DIR / "mask" / "best.pt"
    if not person_model.exists():
        person_model = Path(
            hf_hub_download(
                repo_id="Ultralytics/YOLO11",
                filename="yolo11n.pt",
                local_dir=MODEL_DIR / "person",
            )
        )
    if not mask_model.exists():
        mask_model = Path(
            hf_hub_download(
                repo_id="JJJHHHH/cloth_surgical_mask_detection_yolov8",
                filename="best.pt",
                local_dir=MODEL_DIR / "mask",
            )
        )
    return person_model, mask_model


def boxes_for_class(result: Any, class_ids: set[int]) -> list[tuple[int, int, int, int, float]]:
    detections: list[tuple[int, int, int, int, float]] = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        class_id = int(box.cls.item())
        if class_id not in class_ids:
            continue
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        detections.append((x1, y1, x2, y2, float(box.conf.item())))
    return detections


def draw_detection(
    frame: np.ndarray,
    box: tuple[int, int, int, int, float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2, confidence = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"{label} {confidence:.2f}",
        (x1, max(22, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def detect_masks_in_people(
    frame: np.ndarray,
    people: list[tuple[int, int, int, int, float]],
    model: YOLO,
    confidence: float,
    image_size: int = 384,
) -> list[tuple[int, int, int, int, float]]:
    """Detect masks on enlarged upper-body crops and map boxes to the frame."""
    height, width = frame.shape[:2]
    crops: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    for x1, y1, x2, y2, _ in people:
        person_width = x2 - x1
        person_height = y2 - y1
        crop_x1 = max(0, x1 - int(person_width * 0.08))
        crop_x2 = min(width, x2 + int(person_width * 0.08))
        crop_y1 = max(0, y1)
        crop_y2 = min(height, y1 + int(person_height * 0.58))
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size:
            crops.append(crop)
            origins.append((crop_x1, crop_y1))

    if not crops:
        return []

    results = model.predict(
        crops,
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )
    masks: list[tuple[int, int, int, int, float]] = []
    mask_classes = set(int(key) for key in model.names)
    for result, (origin_x, origin_y) in zip(results, origins):
        candidates = boxes_for_class(result, mask_classes)
        crop_height = result.orig_shape[0]
        # Reject detections on clothing below the face/neck zone.
        candidates = [
            candidate
            for candidate in candidates
            if (candidate[1] + candidate[3]) / 2 <= crop_height * 0.62
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda detection: detection[4])
        x1, y1, x2, y2, score = best
        masks.append(
            (x1 + origin_x, y1 + origin_y, x2 + origin_x, y2 + origin_y, score)
        )
    return masks


def analyze_video(
    input_path: Path,
    output_dir: Path,
    sample_every: int = 3,
    person_confidence: float = 0.28,
    mask_confidence: float = 0.08,
) -> dict[str, Any]:
    person_weights, mask_weights = download_models()
    person_model = YOLO(str(person_weights))
    mask_model = YOLO(str(mask_weights))

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = output_dir / "annotated.mp4"
    writer = imageio.get_writer(
        annotated_path,
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=None,
        ffmpeg_log_level="warning",
    )

    timeline: list[dict[str, Any]] = []
    last_people: list[tuple[int, int, int, int, float]] = []
    last_masks: list[tuple[int, int, int, int, float]] = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % sample_every == 0:
                person_result = person_model.predict(
                    frame,
                    classes=[0],
                    conf=person_confidence,
                    imgsz=640,
                    verbose=False,
                )[0]
                last_people = boxes_for_class(person_result, {0})
                last_masks = detect_masks_in_people(
                    frame,
                    last_people,
                    mask_model,
                    mask_confidence,
                )
                timeline.append(
                    {
                        "time_seconds": round(frame_index / fps, 2),
                        "person_count": len(last_people),
                        "masked_person_count": min(len(last_masks), len(last_people)),
                    }
                )

            annotated = frame.copy()
            for detection in last_people:
                draw_detection(annotated, detection, "person", (75, 220, 120))
            for detection in last_masks:
                draw_detection(annotated, detection, "mask", (255, 190, 60))

            current_people = len(last_people)
            current_masks = min(len(last_masks), current_people)
            current_rate = current_masks / current_people if current_people else 0.0
            cv2.rectangle(annotated, (16, 14), (370, 92), (16, 22, 28), -1)
            cv2.putText(
                annotated,
                f"people {current_people} | masks {current_masks} | rate {current_rate:.2f}",
                (30, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"crowding {crowding_level(current_people)}",
                (30, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (75, 220, 220),
                2,
                cv2.LINE_AA,
            )
            writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            frame_index += 1
    finally:
        capture.release()
        writer.close()

    if not timeline:
        raise ValueError("분석할 수 있는 프레임이 없습니다.")

    # The video describes people entering and remaining in the scene, so its
    # final occupancy is the mode of the last quarter. This suppresses brief
    # duplicate detections and generative-video morphing artifacts.
    final_start = timeline[-1]["time_seconds"] * 0.75
    final_samples = [
        sample for sample in timeline if sample["time_seconds"] >= final_start
    ]
    count_frequency: dict[int, int] = {}
    for sample in final_samples:
        count = int(sample["person_count"])
        count_frequency[count] = count_frequency.get(count, 0) + 1
    final_count = max(count_frequency, key=lambda count: (count_frequency[count], count))

    matching_samples = [
        sample for sample in final_samples if sample["person_count"] == final_count
    ]
    masked_count = round(
        statistics.median(
            int(sample["masked_person_count"]) for sample in matching_samples
        )
    )
    masked_count = min(masked_count, final_count)
    result = VisionContextResult(
        person_count=final_count,
        masked_person_count=masked_count,
        mask_rate=round(masked_count / final_count, 3) if final_count else 0.0,
        crowding_level=crowding_level(final_count),
    )

    payload = {
        "source": {
            "filename": input_path.name,
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "frames": total_frames,
            "duration_seconds": round(total_frames / fps, 2),
            "sample_every_n_frames": sample_every,
        },
        "vision_context": asdict(result),
        "timeline": timeline,
        "notes": [
            "person_count는 영상 마지막 25% 구간의 최빈 동시 인원 수입니다.",
            "mask_rate는 해당 인원 구간의 마스크 탐지 중앙값으로 계산했습니다.",
            "합성 영상의 인물 생성·소멸 오류는 탐지 결과에도 반영될 수 있습니다.",
        ],
    }
    (output_dir / "vision_context.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vision_demo"))
    parser.add_argument("--sample-every", type=int, default=3)
    args = parser.parse_args()
    payload = analyze_video(args.input, args.output_dir, args.sample_every)
    print(json.dumps(payload["vision_context"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
