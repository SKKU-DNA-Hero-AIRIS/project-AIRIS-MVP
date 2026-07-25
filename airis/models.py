from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Any


LEVELS = ("LOW", "MEDIUM", "HIGH")
LEVEL_SCORE = {level: index for index, level in enumerate(LEVELS)}


@dataclass(frozen=True)
class VisionContext:
    person_count: int
    mask_rate: float
    crowding_level: str


@dataclass(frozen=True)
class HealthContext:
    heart_rate: int
    spo2: int
    body_temperature: float
    stress_index: int


@dataclass(frozen=True)
class EnvironmentContext:
    temperature: float
    humidity: int
    pm25: int
    air_quality: str


@dataclass(frozen=True)
class FacilityContext:
    facility_type: str
    label: str
    sensitivity: str
    vulnerable_population: bool


@dataclass(frozen=True)
class MultimodalContext:
    vision_context: VisionContext
    health_context: HealthContext
    environment_context: EnvironmentContext
    facility_context: FacilityContext
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            object.__setattr__(
                self,
                "timestamp",
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MultimodalContext":
        """Build a context from a schema JSON, ignoring extra metadata keys.

        Tolerating unknown keys is what lets a new data source be added to the
        JSON before the decision layer knows about it (extensibility goal Q3).
        """
        return cls(
            vision_context=_build(VisionContext, payload["vision_context"]),
            health_context=_build(HealthContext, payload["health_context"]),
            environment_context=_build(
                EnvironmentContext, payload["environment_context"]
            ),
            facility_context=_build(FacilityContext, payload["facility_context"]),
            timestamp=payload.get("timestamp", ""),
        )


def _build(model: type, payload: dict[str, Any]) -> Any:
    known = {field.name for field in fields(model)}
    return model(**{key: value for key, value in payload.items() if key in known})
