"""Shared file store for iPhone-to-Mac physiological demo data."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_STATE_PATH = ROOT / "outputs/physiological/latest.json"
DEFAULT_CONTEXT: dict[str, Any] = {
    "device_id": "iphone_demo",
    "heart_rate": 75,
    "spo2": 98,
    "body_temperature": 36.5,
    "stress_index": 20,
    "timestamp": None,
}


def normalize_context(payload: dict[str, Any]) -> dict[str, Any]:
    heart_rate = int(payload["heart_rate"])
    spo2 = int(payload["spo2"])
    body_temperature = float(payload["body_temperature"])
    stress_index = int(payload["stress_index"])

    if not 30 <= heart_rate <= 240:
        raise ValueError("심박수는 30~240 범위여야 합니다.")
    if not 50 <= spo2 <= 100:
        raise ValueError("산소포화도는 50~100 범위여야 합니다.")
    if not 30.0 <= body_temperature <= 45.0:
        raise ValueError("체온은 30~45 범위여야 합니다.")
    if not 0 <= stress_index <= 100:
        raise ValueError("스트레스 지수는 0~100 범위여야 합니다.")

    return {
        "device_id": str(payload.get("device_id") or "iphone_demo"),
        "heart_rate": heart_rate,
        "spo2": spo2,
        "body_temperature": round(body_temperature, 1),
        "stress_index": stress_index,
        "timestamp": payload.get("timestamp")
        or datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def write_context(
    payload: dict[str, Any],
    state_path: Path = DEFAULT_STATE_PATH,
) -> dict[str, Any]:
    context = normalize_context(payload)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(context, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, state_path)
    return context


def read_context(state_path: Path = DEFAULT_STATE_PATH) -> dict[str, Any]:
    if not state_path.exists():
        return dict(DEFAULT_CONTEXT)
    try:
        return normalize_context(
            json.loads(state_path.read_text(encoding="utf-8"))
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return dict(DEFAULT_CONTEXT)
