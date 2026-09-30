"""체형(anthropometry) + 사용자 특성 -> the user model the gate physics consumes.

The decision layer in ``airis.control`` answers *how hard* the gate should work
(LOW / MEDIUM / HIGH). This module describes *who is standing in it*, so the
same control level can be delivered differently to a 1.15 m child and a 1.85 m
adult.

Two things are kept apart on purpose:

* ``BodyProfile`` — geometry. Measurable from one RGB pose (height, shoulder
  width) or entered by hand. Mass is a *prior*, not a measurement.
* ``UserTraits`` — behaviour and sensitivity. Walking speed and mobility come
  from the gait analyzer; age group and comfort sensitivity are opt-in inputs
  (facility registration, caregiver tag). They are never inferred from a face.

Segment proportions follow Drillis & Contini (1966) as tabulated in Winter,
*Biomechanics and Motor Control of Human Movement*. They are adult averages;
the child and seated cases are documented approximations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np


# Drillis & Contini segment heights / widths as a fraction of stature.
SHOULDER_HEIGHT_RATIO = 0.818
HIP_HEIGHT_RATIO = 0.530
ANKLE_HEIGHT_RATIO = 0.039
SHOULDER_WIDTH_RATIO = 0.259
HIP_WIDTH_RATIO = 0.191
# Not in Drillis & Contini: nose tip sits between eye (0.936H) and chin (0.870H).
NOSE_HEIGHT_RATIO = 0.910
HEAD_HALF_WIDTH_RATIO = 0.045
# Arms hang outside the acromion, so the jet meets the body before the joint.
ARM_CLEARANCE_GAIN = 1.12

BUILD_BMI_PRIOR = {"SLIM": 19.5, "AVERAGE": 22.5, "BROAD": 27.0}
CHILD_BMI_PRIOR = 16.5
CHILD_HEIGHT_M = 1.35

# Share of body surface per gate zone. Adults: rule of nines with the arms
# counted in the upper zone they hang beside. Children carry more area in the
# head (Lund-Browder).
ADULT_AREA_SHARE = {"HEAD": 0.09, "UPPER": 0.55, "LOWER": 0.36}
CHILD_AREA_SHARE = {"HEAD": 0.14, "UPPER": 0.52, "LOWER": 0.34}

# Seated (wheelchair) geometry: the chair, not the pelvis, sets the lower width.
SEAT_HEIGHT_M = 0.55
SEATED_HEAD_SPAN_M = 0.24
WHEELCHAIR_HALF_WIDTH_M = 0.33
SEATED_AREA_SHARE = {"HEAD": 0.10, "UPPER": 0.50, "LOWER": 0.40}

ZONES = ("HEAD", "UPPER", "LOWER")
AGE_GROUPS = ("CHILD", "ADULT", "SENIOR")
MOBILITY_LEVELS = ("NORMAL", "SLOW", "ASSISTED")
COMFORT_LEVELS = ("NORMAL", "SENSITIVE")
CLOTHING_TYPES = ("LIGHT", "NORMAL", "BULKY")
POSTURES = ("STANDING", "SEATED")

USER_PROFILE_DIR = Path(__file__).resolve().parent.parent / "assets" / "user_profiles"


@dataclass(frozen=True)
class BodySegment:
    zone: str
    z_low_m: float
    z_high_m: float
    half_width_m: float
    surface_area_m2: float

    @property
    def span_m(self) -> float:
        return self.z_high_m - self.z_low_m


@dataclass(frozen=True)
class BodyProfile:
    height_m: float
    shoulder_width_m: float
    hip_width_m: float
    mass_kg: float
    posture: str = "STANDING"
    source: str = "MANUAL"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.5 <= self.height_m <= 2.3:
            raise ValueError("height_m must be between 0.5 and 2.3 m.")
        if self.shoulder_width_m <= 0 or self.hip_width_m <= 0:
            raise ValueError("body widths must be positive.")
        if self.mass_kg <= 0:
            raise ValueError("mass_kg must be positive.")
        if self.posture not in POSTURES:
            raise ValueError(f"posture must be one of {POSTURES}.")

    @property
    def build(self) -> str:
        ratio = self.shoulder_width_m / self.height_m
        if ratio < 0.235:
            return "SLIM"
        if ratio > 0.280:
            return "BROAD"
        return "AVERAGE"

    @property
    def is_child_sized(self) -> bool:
        return self.posture == "STANDING" and self.height_m < CHILD_HEIGHT_M

    @property
    def surface_area_m2(self) -> float:
        """Du Bois body surface area. Depends on the mass prior."""
        return 0.007184 * self.mass_kg**0.425 * (self.height_m * 100.0) ** 0.725

    def segments(self) -> tuple[BodySegment, ...]:
        """Split the body into the three zones a nozzle row can be assigned to."""
        area = self.surface_area_m2
        upper_half_width = self.shoulder_width_m * ARM_CLEARANCE_GAIN / 2.0

        if self.posture == "SEATED":
            # ``height_m`` is the top of the head while seated.
            head_low = self.height_m - SEATED_HEAD_SPAN_M
            layout = (
                ("HEAD", head_low, self.height_m, 0.08),
                ("UPPER", SEAT_HEIGHT_M, head_low, upper_half_width),
                ("LOWER", 0.0, SEAT_HEIGHT_M, WHEELCHAIR_HALF_WIDTH_M),
            )
            share = SEATED_AREA_SHARE
        else:
            shoulder = self.height_m * SHOULDER_HEIGHT_RATIO
            hip = self.height_m * HIP_HEIGHT_RATIO
            layout = (
                (
                    "HEAD",
                    shoulder,
                    self.height_m,
                    self.height_m * HEAD_HALF_WIDTH_RATIO,
                ),
                ("UPPER", hip, shoulder, upper_half_width),
                ("LOWER", 0.0, hip, self.hip_width_m / 2.0),
            )
            share = CHILD_AREA_SHARE if self.is_child_sized else ADULT_AREA_SHARE

        return tuple(
            BodySegment(
                zone=zone,
                z_low_m=round(low, 4),
                z_high_m=round(high, 4),
                half_width_m=round(half_width, 4),
                surface_area_m2=round(area * share[zone], 4),
            )
            for zone, low, high, half_width in layout
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["build"] = self.build
        payload["surface_area_m2"] = round(self.surface_area_m2, 3)
        return payload


@dataclass(frozen=True)
class UserTraits:
    walking_speed_mps: float = 1.2
    age_group: str = "ADULT"
    mobility: str = "NORMAL"
    comfort_sensitivity: str = "NORMAL"
    clothing: str = "NORMAL"
    mask_on: bool = False

    def __post_init__(self) -> None:
        if self.walking_speed_mps <= 0:
            raise ValueError("walking_speed_mps must be positive.")
        for value, allowed, name in (
            (self.age_group, AGE_GROUPS, "age_group"),
            (self.mobility, MOBILITY_LEVELS, "mobility"),
            (self.comfort_sensitivity, COMFORT_LEVELS, "comfort_sensitivity"),
            (self.clothing, CLOTHING_TYPES, "clothing"),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}.")

    @property
    def is_vulnerable(self) -> bool:
        """True when the user should get the conservative comfort/safety limits."""
        return (
            self.age_group != "ADULT"
            or self.mobility != "NORMAL"
            or self.comfort_sensitivity == "SENSITIVE"
        )


@dataclass(frozen=True)
class UserProfile:
    body: BodyProfile
    traits: UserTraits
    profile_id: str = "LIVE"
    label: str = "실시간 측정 사용자"

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "label": self.label,
            "body_profile": self.body.to_dict(),
            "user_traits": asdict(self.traits),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UserProfile":
        """Build a profile from JSON, ignoring derived or unknown keys."""
        return cls(
            body=_build(BodyProfile, payload["body_profile"]),
            traits=_build(UserTraits, payload.get("user_traits", {})),
            profile_id=str(payload.get("profile_id", "LIVE")),
            label=str(payload.get("label", "실시간 측정 사용자")),
        )


def _build(model: type, payload: dict[str, Any]) -> Any:
    known = {field.name for field in fields(model)}
    return model(**{key: value for key, value in payload.items() if key in known})


def estimate_mass_kg(height_m: float, shoulder_width_m: float) -> float:
    """Mass prior from height and build. A camera cannot weigh anyone."""
    if height_m < CHILD_HEIGHT_M:
        bmi = CHILD_BMI_PRIOR
    else:
        ratio = shoulder_width_m / height_m
        build = "SLIM" if ratio < 0.235 else "BROAD" if ratio > 0.280 else "AVERAGE"
        bmi = BUILD_BMI_PRIOR[build]
    return round(bmi * height_m**2, 1)


def body_from_measurements(
    height_m: float,
    shoulder_width_m: Optional[float] = None,
    hip_width_m: Optional[float] = None,
    mass_kg: Optional[float] = None,
    *,
    posture: str = "STANDING",
    source: str = "MANUAL",
    confidence: float = 1.0,
) -> BodyProfile:
    """Fill the unmeasured dimensions from anthropometric ratios."""
    shoulder = shoulder_width_m or height_m * SHOULDER_WIDTH_RATIO
    # Hip breadth tracks the same build factor the shoulders show.
    build_factor = float(np.clip(shoulder / (height_m * SHOULDER_WIDTH_RATIO), 0.8, 1.3))
    hip = hip_width_m or height_m * HIP_WIDTH_RATIO * build_factor
    return BodyProfile(
        height_m=round(float(height_m), 3),
        shoulder_width_m=round(float(shoulder), 3),
        hip_width_m=round(float(hip), 3),
        mass_kg=float(mass_kg or estimate_mass_kg(height_m, shoulder)),
        posture=posture,
        source=source,
        confidence=round(float(confidence), 2),
    )


def _visible(points: np.ndarray, *indices: int, threshold: float = 0.35) -> bool:
    return all(points[index, 2] >= threshold for index in indices)


def estimate_body_profile(
    points: np.ndarray,
    meters_per_pixel: float,
) -> Optional[BodyProfile]:
    """Estimate 체형 from one COCO pose (17 x [x, y, confidence]).

    ``meters_per_pixel`` is the same calibration the gait analyzer uses
    (real frame width / frame width in pixels), so it is only valid for a person
    at the calibrated distance. Returns ``None`` when the body is not fully in
    frame; the caller should keep the last good profile.
    """
    if points.shape != (17, 3):
        raise ValueError("COCO pose keypoints must have shape (17, 3).")
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be positive.")
    if not _visible(points, 5, 6, 11, 12, 15, 16):
        return None

    shoulder_center = (points[5, :2] + points[6, :2]) / 2.0
    hip_center = (points[11, :2] + points[12, :2]) / 2.0
    ankle_y = float((points[15, 1] + points[16, 1]) / 2.0)

    # Image y grows downward. Two independent stature estimates: shoulders are
    # robust to head tilt, the nose adds the head when it is visible.
    estimates = [
        (ankle_y - float(shoulder_center[1]))
        * meters_per_pixel
        / (SHOULDER_HEIGHT_RATIO - ANKLE_HEIGHT_RATIO)
    ]
    if _visible(points, 0):
        estimates.append(
            (ankle_y - float(points[0, 1]))
            * meters_per_pixel
            / (NOSE_HEIGHT_RATIO - ANKLE_HEIGHT_RATIO)
        )
    height = float(np.mean(estimates))
    if not 0.5 <= height <= 2.3:
        return None

    shoulder_px = float(np.linalg.norm(points[5, :2] - points[6, :2]))
    torso_px = max(float(np.linalg.norm(shoulder_center - hip_center)), 1.0)
    frontal = shoulder_px / torso_px >= 0.45
    # A side view foreshortens the shoulders, so fall back to the ratio there.
    shoulder_width = shoulder_px * meters_per_pixel if frontal else None

    keypoint_confidence = float(np.mean(points[[5, 6, 11, 12, 15, 16], 2]))
    return body_from_measurements(
        height,
        shoulder_width,
        source="POSE_FRONT" if frontal else "POSE_SIDE",
        confidence=keypoint_confidence * (1.0 if frontal else 0.7),
    )


class BodyProfileTracker:
    """Median of recent single-frame estimates; one pose is too noisy to act on."""

    def __init__(self, window: int = 15, minimum: int = 3) -> None:
        self.window = window
        self.minimum = minimum
        self._heights: list[float] = []
        self._frontal_shoulders: list[float] = []
        self._confidences: list[float] = []

    def reset(self) -> None:
        self._heights.clear()
        self._frontal_shoulders.clear()
        self._confidences.clear()

    def update(
        self, points: np.ndarray, meters_per_pixel: float
    ) -> Optional[BodyProfile]:
        estimate = estimate_body_profile(points, meters_per_pixel)
        if estimate is not None:
            self._heights = (self._heights + [estimate.height_m])[-self.window :]
            self._confidences = (self._confidences + [estimate.confidence])[
                -self.window :
            ]
            if estimate.source == "POSE_FRONT":
                self._frontal_shoulders = (
                    self._frontal_shoulders + [estimate.shoulder_width_m]
                )[-self.window :]
        return self.profile()

    def profile(self) -> Optional[BodyProfile]:
        if len(self._heights) < self.minimum:
            return None
        frontal = len(self._frontal_shoulders) >= self.minimum
        return body_from_measurements(
            float(np.median(self._heights)),
            float(np.median(self._frontal_shoulders)) if frontal else None,
            source="POSE_FRONT" if frontal else "POSE_SIDE",
            confidence=float(np.median(self._confidences)),
        )


def traits_from_gait(
    gait_context: dict[str, Any],
    body: BodyProfile,
    base: UserTraits = UserTraits(),
) -> UserTraits:
    """Update the measurable traits from a ``gait_context``; keep the opt-in ones."""
    speed = gait_context.get("walking_speed_mps") or base.walking_speed_mps
    asymmetry = gait_context.get("gait_asymmetry") or 0.0
    mobility = base.mobility
    if mobility == "NORMAL" and (speed < 0.8 or asymmetry >= 0.2):
        mobility = "SLOW"
    age_group = base.age_group
    if age_group == "ADULT" and body.is_child_sized:
        # Stature only. It selects the conservative limits, not an identity.
        age_group = "CHILD"
    return UserTraits(
        walking_speed_mps=round(float(speed), 2),
        age_group=age_group,
        mobility=mobility,
        comfort_sensitivity=base.comfort_sensitivity,
        clothing=base.clothing,
        mask_on=base.mask_on,
    )


def profile_from_contexts(
    body_context: dict[str, Any],
    gait_context: dict[str, Any],
    base: UserTraits = UserTraits(),
) -> UserProfile:
    """Combine the live ``body_context`` and ``gait_context`` into one profile."""
    body = _build(BodyProfile, body_context)
    return UserProfile(body=body, traits=traits_from_gait(gait_context, body, base))


REFERENCE_PROFILE = UserProfile(
    body=BodyProfile(
        height_m=1.70,
        shoulder_width_m=0.42,
        hip_width_m=0.33,
        mass_kg=65.0,
        source="REFERENCE",
    ),
    traits=UserTraits(walking_speed_mps=1.3),
    profile_id="REF",
    label="설계 기준 성인",
)


def load_user_profiles(
    directory: Path = USER_PROFILE_DIR,
) -> list[UserProfile]:
    """Load the committed persona JSONs in profile-id order."""
    profiles = [
        UserProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.json"))
    ]
    profiles.sort(key=lambda profile: int(profile.profile_id.lstrip("P")))
    return profiles
