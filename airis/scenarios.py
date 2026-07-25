from copy import deepcopy

from .models import (
    EnvironmentContext,
    FacilityContext,
    HealthContext,
    MultimodalContext,
    VisionContext,
)


HEALTH_SCENARIOS = {
    "정상": {
        "heart_rate": 75,
        "spo2": 98,
        "body_temperature": 36.5,
        "stress_index": 20,
    },
    "신체 활동": {
        "heart_rate": 120,
        "spo2": 98,
        "body_temperature": 36.8,
        "stress_index": 70,
    },
    "고위험": {
        "heart_rate": 125,
        "spo2": 94,
        "body_temperature": 38.1,
        "stress_index": 85,
    },
}

ENVIRONMENT_SCENARIOS = {
    "쾌적": {
        "temperature": 24.0,
        "humidity": 45,
        "pm25": 12,
        "air_quality": "GOOD",
    },
    "고온다습": {
        "temperature": 32.0,
        "humidity": 80,
        "pm25": 28,
        "air_quality": "MODERATE",
    },
    "대기질 나쁨": {
        "temperature": 30.0,
        "humidity": 70,
        "pm25": 85,
        "air_quality": "BAD",
    },
}

VISION_SCENARIOS = {
    "여유": {"person_count": 3, "mask_rate": 0.67, "crowding_level": "LOW"},
    "보통": {"person_count": 8, "mask_rate": 0.5, "crowding_level": "MEDIUM"},
    "혼잡": {"person_count": 22, "mask_rate": 0.36, "crowding_level": "HIGH"},
}

FACILITIES = {
    "일반 사무실": {
        "facility_type": "office",
        "label": "일반 사무실",
        "sensitivity": "LOW",
        "vulnerable_population": False,
    },
    "병원": {
        "facility_type": "hospital",
        "label": "병원",
        "sensitivity": "HIGH",
        "vulnerable_population": True,
    },
    "어린이집": {
        "facility_type": "daycare",
        "label": "어린이집",
        "sensitivity": "HIGH",
        "vulnerable_population": True,
    },
    "학교": {
        "facility_type": "school",
        "label": "학교",
        "sensitivity": "MEDIUM",
        "vulnerable_population": True,
    },
    "공장": {
        "facility_type": "factory",
        "label": "공장",
        "sensitivity": "MEDIUM",
        "vulnerable_population": False,
    },
}


def build_context(
    health_scenario: str,
    environment_scenario: str,
    vision_scenario: str,
    facility: str,
) -> MultimodalContext:
    """Create a standardized context without sharing mutable scenario state."""
    return MultimodalContext(
        health_context=HealthContext(**deepcopy(HEALTH_SCENARIOS[health_scenario])),
        environment_context=EnvironmentContext(
            **deepcopy(ENVIRONMENT_SCENARIOS[environment_scenario])
        ),
        vision_context=VisionContext(**deepcopy(VISION_SCENARIOS[vision_scenario])),
        facility_context=FacilityContext(**deepcopy(FACILITIES[facility])),
    )
