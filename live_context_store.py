"""Shared latest-context files used by the integrated AIRIS dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LIVE_DIR = ROOT / "outputs/live_context"
VISION_PATH = LIVE_DIR / "vision.json"
FACILITY_PATH = LIVE_DIR / "facility.json"
USER_PROFILE_PATH = LIVE_DIR / "user_profile.json"

DEFAULT_VISION = {
    "person_count": 0,
    "masked_person_count": 0,
    "mask_rate": 0.0,
    "crowding_level": "LOW",
}
DEFAULT_FACILITY = {
    "facility_type": "사무실",
    "risk_level": "LOW",
    "vulnerable_population": False,
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else dict(default)
    except (OSError, json.JSONDecodeError):
        return dict(default)


def write_vision_context(payload: dict[str, Any]) -> None:
    person_count = max(0, int(payload.get("person_count", 0)))
    masked_count = min(
        person_count,
        max(0, int(payload.get("masked_person_count", 0))),
    )
    normalized = {
        "person_count": person_count,
        "masked_person_count": masked_count,
        "mask_rate": round(
            float(payload.get("mask_rate", 0.0)),
            3,
        ),
        "crowding_level": str(payload.get("crowding_level", "LOW")),
    }
    _write_json(VISION_PATH, normalized)


def read_vision_context() -> dict[str, Any]:
    return _read_json(VISION_PATH, DEFAULT_VISION)


def write_facility_context(payload: dict[str, Any]) -> None:
    normalized = {
        "facility_type": str(payload.get("facility_type", "사무실")),
        "risk_level": str(payload.get("risk_level", "LOW")),
        "vulnerable_population": bool(payload.get("vulnerable_population", False)),
    }
    _write_json(FACILITY_PATH, normalized)


def read_facility_context() -> dict[str, Any]:
    return _read_json(FACILITY_PATH, DEFAULT_FACILITY)


def write_user_profile(payload: dict[str, Any]) -> None:
    """Store the latest measured 체형·사용자 특성 (``UserProfile.to_dict()``)."""
    _write_json(USER_PROFILE_PATH, payload)


def read_user_profile() -> dict[str, Any]:
    """Return the latest measured profile, or ``{}`` if nobody was measured yet."""
    return _read_json(USER_PROFILE_PATH, {})
