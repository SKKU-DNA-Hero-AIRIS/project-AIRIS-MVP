"""Current weather and air-quality context from Open-Meteo."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / "outputs/external/latest.json"

LOCATIONS = {
    "성균관대 자연과학캠퍼스": (37.2938, 126.9754),
    "서울": (37.5665, 126.9780),
    "수원": (37.2636, 127.0286),
}
_NOMINATIM_LOCK = threading.Lock()
_NOMINATIM_LAST_REQUEST = 0.0


def air_quality_level(pm25: float) -> str:
    if pm25 <= 15:
        return "GOOD"
    if pm25 <= 35:
        return "MODERATE"
    return "BAD"


def _get_json(base_url: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{base_url}?{urlencode(params)}"
    with urlopen(url, timeout=7) as response:
        return json.loads(response.read().decode("utf-8"))


def _write_cache(payload: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = CACHE_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, CACHE_PATH)


def _read_cache() -> Optional[dict[str, Any]]:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_latest_external_context() -> dict[str, Any]:
    cached = _read_cache()
    if cached is not None:
        return cached
    return {
        "location": "미선택",
        "source": "DEMO",
        "timestamp": None,
        "environment_context": {
            "temperature": 25.0,
            "humidity": 60,
            "pm25": 20.0,
            "pm10": 35.0,
            "precipitation": 0.0,
            "rain": False,
            "air_quality": "MODERATE",
        },
    }


def fetch_external_context(location_name: str) -> dict[str, Any]:
    latitude, longitude = LOCATIONS[location_name]
    return fetch_external_context_by_coordinates(
        latitude,
        longitude,
        location_name,
    )


def fetch_external_context_by_coordinates(
    latitude: float,
    longitude: float,
    location_name: str,
) -> dict[str, Any]:
    try:
        weather = _get_json(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,relative_humidity_2m,"
                    "precipitation,rain,weather_code"
                ),
                "timezone": "Asia/Seoul",
            },
        )
        air = _get_json(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": "pm2_5,pm10",
                "timezone": "Asia/Seoul",
            },
        )
        weather_current = weather["current"]
        air_current = air["current"]
        pm25 = float(air_current["pm2_5"])
        payload = {
            "location": location_name,
            "source": "Open-Meteo",
            "timestamp": weather_current["time"],
            "environment_context": {
                "temperature": round(float(weather_current["temperature_2m"]), 1),
                "humidity": int(weather_current["relative_humidity_2m"]),
                "pm25": round(pm25, 1),
                "pm10": round(float(air_current["pm10"]), 1),
                "precipitation": round(
                    float(weather_current["precipitation"]),
                    1,
                ),
                "rain": float(weather_current["rain"]) > 0,
                "air_quality": air_quality_level(pm25),
            },
        }
        _write_cache(payload)
        return payload
    except (OSError, KeyError, TypeError, ValueError):
        cached = _read_cache()
        if cached is not None:
            cached["source"] = "CACHE"
            return cached
        return {
            "location": location_name,
            "source": "DEMO",
            "timestamp": None,
            "environment_context": {
                "temperature": 25.0,
                "humidity": 60,
                "pm25": 20.0,
                "pm10": 35.0,
                "precipitation": 0.0,
                "rain": False,
                "air_quality": "MODERATE",
            },
        }


def infer_facility_type(
    place_name: str,
    osm_category: str = "",
    osm_type: str = "",
) -> str:
    searchable = f"{place_name} {osm_category} {osm_type}".lower()
    keyword_groups = (
        ("병원", ("병원", "의원", "클리닉", "hospital", "clinic", "doctors")),
        (
            "어린이집",
            ("어린이집", "유치원", "nursery", "kindergarten", "childcare"),
        ),
        (
            "학교",
            (
                "학교",
                "대학교",
                "대학",
                "school",
                "university",
                "college",
                "academy",
            ),
        ),
        ("공공기관", ("시청", "구청", "주민센터", "공공기관", "government", "townhall")),
        ("사무실", ("사무실", "오피스", "office", "company")),
    )
    for facility_type, keywords in keyword_groups:
        if any(keyword in searchable for keyword in keywords):
            return facility_type
    return "기타"


def reverse_geocode(latitude: float, longitude: float) -> dict[str, str]:
    global _NOMINATIM_LAST_REQUEST
    params = {
        "lat": round(latitude, 6),
        "lon": round(longitude, 6),
        "format": "jsonv2",
        "addressdetails": 1,
        "namedetails": 1,
        "extratags": 1,
        "zoom": 18,
        "accept-language": "ko",
    }
    request = Request(
        f"https://nominatim.openstreetmap.org/reverse?{urlencode(params)}",
        headers={
            "User-Agent": (
                "AIRIS-MVP/0.1 "
                "(SKKU research prototype; "
                "https://github.com/SKKU-DNA-Hero-AIRIS/project-AIRIS-MVP)"
            )
        },
    )
    try:
        with _NOMINATIM_LOCK:
            wait_seconds = 1.0 - (time.monotonic() - _NOMINATIM_LAST_REQUEST)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            with urlopen(request, timeout=7) as response:
                result = json.loads(response.read().decode("utf-8"))
            _NOMINATIM_LAST_REQUEST = time.monotonic()
        place_name = (
            result.get("name")
            or result.get("display_name", "").split(",")[0]
            or f"{latitude:.5f}, {longitude:.5f}"
        )
        osm_category = str(result.get("category", ""))
        osm_type = str(result.get("type", ""))
        extra_tags = result.get("extratags") or {}
        operator = str(extra_tags.get("operator", ""))
        operator_type = str(extra_tags.get("operator:type", ""))
        return {
            "name": place_name,
            "display_name": str(result.get("display_name", place_name)),
            "category": osm_category,
            "type": osm_type,
            "facility_type": infer_facility_type(
                f"{place_name} {operator} {operator_type}",
                osm_category,
                osm_type,
            ),
        }
    except (OSError, ValueError, json.JSONDecodeError):
        coordinate_name = f"{latitude:.5f}, {longitude:.5f}"
        return {
            "name": coordinate_name,
            "display_name": coordinate_name,
            "category": "",
            "type": "",
            "facility_type": "기타",
        }


def build_facility_context(
    facility_type: str,
    vulnerable_population: bool,
) -> dict[str, Any]:
    risk_levels = {
        "병원": "HIGH",
        "어린이집": "HIGH",
        "학교": "MEDIUM",
        "공공기관": "MEDIUM",
        "사무실": "LOW",
        "기타": "LOW",
    }
    return {
        "facility_type": facility_type,
        "risk_level": risk_levels[facility_type],
        "vulnerable_population": vulnerable_population,
    }
