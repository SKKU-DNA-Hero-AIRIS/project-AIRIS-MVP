"""Context JSON -> numeric feature vector for the ML control policy.

The ML layer never sees a raw sensor reading. Every feature is a normalized
context signal, so the same code path works for synthetic training data and for
live webcam/wearable input.
"""

from __future__ import annotations

import numpy as np

from .models import MultimodalContext


SENSITIVITY_SCALE = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}

FEATURE_NAMES: tuple[str, ...] = (
    "hr_excess",
    "spo2_deficit",
    "fever",
    "stress",
    "pm25_load",
    "thermal_load",
    "humidity_load",
    "crowding",
    "unmasked_rate",
    "facility_sensitivity",
    "vulnerable_population",
    "hr_x_sensitivity",
    "fever_x_vulnerable",
    "pm25_x_crowding",
    "unmasked_x_crowding",
    "activity_plausible",
)

FEATURE_LABELS: dict[str, str] = {
    "hr_excess": "심박수 상승폭",
    "spo2_deficit": "산소포화도 저하폭",
    "fever": "체온 상승폭",
    "stress": "스트레스 지수",
    "pm25_load": "미세먼지 부하",
    "thermal_load": "실내 고온 부하",
    "humidity_load": "습도 부하",
    "crowding": "밀집도",
    "unmasked_rate": "마스크 미착용률",
    "facility_sensitivity": "시설 민감도",
    "vulnerable_population": "취약계층 시설",
    "hr_x_sensitivity": "심박 상승 x 시설 민감도",
    "fever_x_vulnerable": "발열 x 취약계층",
    "pm25_x_crowding": "미세먼지 x 밀집",
    "unmasked_x_crowding": "미착용 x 밀집",
    "activity_plausible": "신체 활동으로 설명 가능한 심박 상승",
}


def _unit(value: float, low: float, span: float) -> float:
    return float(min(1.0, max(0.0, (value - low) / span)))


def extract_features(context: MultimodalContext) -> dict[str, float]:
    """Map a standardized context into named [0, 1] context signals."""
    health = context.health_context
    env = context.environment_context
    vision = context.vision_context
    facility = context.facility_context

    hr_excess = _unit(health.heart_rate, 90.0, 60.0)
    spo2_deficit = _unit(97.0 - health.spo2, 0.0, 12.0)
    fever = _unit(health.body_temperature, 36.8, 2.2)
    stress = _unit(health.stress_index, 0.0, 100.0)
    pm25_load = _unit(env.pm25, 0.0, 150.0)
    thermal_load = _unit(env.temperature, 24.0, 14.0)
    humidity_load = _unit(env.humidity, 40.0, 50.0)
    crowding = _unit(vision.person_count, 0.0, 25.0)
    unmasked_rate = _unit(1.0 - vision.mask_rate, 0.0, 1.0)
    sensitivity = SENSITIVITY_SCALE[facility.sensitivity]
    vulnerable = 1.0 if facility.vulnerable_population else 0.0

    features = {
        "hr_excess": hr_excess,
        "spo2_deficit": spo2_deficit,
        "fever": fever,
        "stress": stress,
        "pm25_load": pm25_load,
        "thermal_load": thermal_load,
        "humidity_load": humidity_load,
        "crowding": crowding,
        "unmasked_rate": unmasked_rate,
        "facility_sensitivity": sensitivity,
        "vulnerable_population": vulnerable,
        "hr_x_sensitivity": hr_excess * sensitivity,
        "fever_x_vulnerable": fever * vulnerable,
        "pm25_x_crowding": pm25_load * crowding,
        "unmasked_x_crowding": unmasked_rate * crowding,
        # Elevated pulse with clean oxygen/temperature in a low-sensitivity
        # facility is the signal that should pull the decision *down*.
        "activity_plausible": (
            hr_excess * (1.0 - spo2_deficit) * (1.0 - fever) * (1.0 - sensitivity)
        ),
    }
    return {name: round(features[name], 6) for name in FEATURE_NAMES}


def feature_vector(context: MultimodalContext) -> np.ndarray:
    features = extract_features(context)
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)
