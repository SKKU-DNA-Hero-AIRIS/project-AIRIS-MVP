"""Build and analyze one multimodal context from the live module outputs."""

from __future__ import annotations

from typing import Any

from airis import analyze_hybrid
from airis.models import (
    EnvironmentContext,
    FacilityContext,
    HealthContext,
    MultimodalContext,
    VisionContext,
)


FACILITY_CODES = {
    "병원": ("hospital", "HIGH"),
    "어린이집": ("daycare", "HIGH"),
    "학교": ("school", "MEDIUM"),
    "공공기관": ("public", "MEDIUM"),
    "사무실": ("office", "LOW"),
    "기타": ("other", "LOW"),
}


def build_live_multimodal_context(
    vision: dict[str, Any],
    health: dict[str, Any],
    environment: dict[str, Any],
    facility: dict[str, Any],
) -> MultimodalContext:
    facility_label = str(facility.get("facility_type", "기타"))
    facility_code, default_sensitivity = FACILITY_CODES.get(
        facility_label,
        FACILITY_CODES["기타"],
    )
    sensitivity = str(facility.get("risk_level", default_sensitivity))
    return MultimodalContext(
        vision_context=VisionContext(
            person_count=int(vision.get("person_count", 0)),
            mask_rate=float(vision.get("mask_rate", 0.0)),
            crowding_level=str(vision.get("crowding_level", "LOW")),
        ),
        health_context=HealthContext(
            heart_rate=int(health.get("heart_rate", 75)),
            spo2=int(health.get("spo2", 98)),
            body_temperature=float(health.get("body_temperature", 36.5)),
            stress_index=int(health.get("stress_index", 20)),
        ),
        environment_context=EnvironmentContext(
            temperature=float(environment.get("temperature", 25.0)),
            humidity=int(environment.get("humidity", 60)),
            pm25=round(float(environment.get("pm25", 20))),
            air_quality=str(environment.get("air_quality", "MODERATE")),
        ),
        facility_context=FacilityContext(
            facility_type=facility_code,
            label=facility_label,
            sensitivity=sensitivity,
            vulnerable_population=bool(
                facility.get("vulnerable_population", False)
            ),
        ),
    )


def analyze_live_context(
    vision: dict[str, Any],
    health: dict[str, Any],
    environment: dict[str, Any],
    facility: dict[str, Any],
):
    context = build_live_multimodal_context(
        vision,
        health,
        environment,
        facility,
    )
    return context, analyze_hybrid(context)
