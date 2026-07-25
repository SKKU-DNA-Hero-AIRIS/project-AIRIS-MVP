from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


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
