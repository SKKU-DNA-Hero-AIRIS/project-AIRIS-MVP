"""Reduced-order physics of one air gate passage.

    nozzle rows (fixed hardware)  +  body segments (per user)
            v
    free-jet decay to the body surface      U_s = v0 * min(1, B d / x)
            v
    particle detachment on the surface      f(U_s) * coverage * (1 - e^(-t/tau))
            v
    removal efficiency / face velocity / lateral load / fan energy

This is *not* CFD. It is a closed-form model built from textbook turbulent
round-jet scaling (Pope, *Turbulent Flows*: centerline decay B ~ 5.8-6.3,
spreading rate S ~ 0.094) plus a log-logistic detachment curve whose constants
are placeholders until they are fitted to chamber experiments. Every constant
that is an assumption lives in ``PhysicsParams`` so it can be recalibrated
without touching the optimizer.

What the model is for: the body changes the answer. A broader torso sits closer
to the nozzles and is hit harder; a short user leaves the top rows blowing at
nothing; a light user is pushed harder by the same momentum flux. Those are the
effects the personalization layer exploits.

Everything is vectorized over candidate setpoints so an optimizer can score
thousands of candidates in one call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .body_profile import BodySegment, UserProfile, UserTraits


GRAVITY = 9.81
# Rows this close above the crown still wash the top of the head.
HEAD_ROW_TOLERANCE_M = 0.02
# Below this a nozzle is treated as shut: it cleans nothing and draws nothing.
ACTIVE_VELOCITY_MPS = 0.5

CLOTHING_ADHESION_GAIN = {"LIGHT": 0.90, "NORMAL": 1.00, "BULKY": 1.25}

# Face-level wind limits borrow the Lawson pedestrian comfort classes
# (business walking 10 m/s, strolling 8 m/s, standing 6 m/s). A gate is not a
# street, so treat them as a starting point for human-subject tests.
FACE_VELOCITY_LIMIT_MPS = {"NORMAL": 10.0, "SLOW": 8.0, "VULNERABLE": 6.0}
# One-side jet thrust as a fraction of body weight. Design assumption.
LATERAL_LOAD_LIMIT = {"NORMAL": 0.020, "VULNERABLE": 0.012}


@dataclass(frozen=True)
class GateGeometry:
    passage_width_m: float = 0.90
    jet_zone_length_m: float = 1.20
    nozzle_diameter_m: float = 0.030
    row_heights_m: tuple[float, ...] = (
        0.15, 0.35, 0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95,
    )
    nozzles_per_row: int = 2
    max_jet_velocity_mps: float = 20.0
    max_hold_seconds: float = 12.0
    vertical_sweep_m: float = 0.12
    fan_efficiency: float = 0.55
    nozzle_loss_coefficient: float = 0.5

    @property
    def nozzle_area_m2(self) -> float:
        return math.pi * self.nozzle_diameter_m**2 / 4.0

    @property
    def row_count(self) -> int:
        return len(self.row_heights_m)


@dataclass(frozen=True)
class PhysicsParams:
    air_density: float = 1.2
    jet_decay_constant: float = 6.0
    jet_spread_rate: float = 0.094
    footprint_gain: float = 2.0
    min_standoff_m: float = 0.06
    coverage_cap: float = 0.90
    detachment_time_constant_s: float = 4.0
    detachment_steepness: float = 3.0
    coarse_critical_velocity_mps: float = 6.0
    fine_critical_velocity_mps: float = 14.0


DEFAULT_GATE = GateGeometry()
DEFAULT_PARAMS = PhysicsParams()


@dataclass(frozen=True)
class ComfortLimits:
    face_velocity_mps: float
    lateral_load_ratio: float
    basis: str


@dataclass(frozen=True)
class GateSetpoints:
    """The four things the optimizer decides for one user."""

    head_velocity_mps: float
    upper_velocity_mps: float
    lower_velocity_mps: float
    hold_seconds: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.head_velocity_mps,
                self.upper_velocity_mps,
                self.lower_velocity_mps,
                self.hold_seconds,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> "GateSetpoints":
        head, upper, lower, hold = (round(float(value), 2) for value in values)
        return cls(head, upper, lower, hold)


@dataclass(frozen=True)
class RowState:
    height_m: float
    zone: Optional[str]
    standoff_m: Optional[float]
    nozzle_velocity_mps: float
    surface_velocity_mps: float


@dataclass(frozen=True)
class SimulationBatch:
    removal_efficiency: np.ndarray
    zone_removal: np.ndarray
    face_velocity_mps: np.ndarray
    lateral_load_ratio: np.ndarray
    power_w: np.ndarray
    energy_wh: np.ndarray
    passage_time_s: np.ndarray


@dataclass(frozen=True)
class SimulationResult:
    removal_efficiency: float
    zone_removal: dict[str, float]
    face_velocity_mps: float
    lateral_load_ratio: float
    power_w: float
    energy_wh: float
    passage_time_s: float
    rows: tuple[RowState, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "removal_efficiency": self.removal_efficiency,
            "zone_removal": self.zone_removal,
            "face_velocity_mps": self.face_velocity_mps,
            "lateral_load_ratio": self.lateral_load_ratio,
            "power_w": self.power_w,
            "energy_wh": self.energy_wh,
            "passage_time_s": self.passage_time_s,
            "rows": [
                {
                    "height_m": row.height_m,
                    "zone": row.zone,
                    "standoff_m": row.standoff_m,
                    "nozzle_velocity_mps": row.nozzle_velocity_mps,
                    "surface_velocity_mps": row.surface_velocity_mps,
                }
                for row in self.rows
            ],
        }


def comfort_limits(traits: UserTraits) -> ComfortLimits:
    if (
        traits.age_group != "ADULT"
        or traits.comfort_sensitivity == "SENSITIVE"
        or traits.mobility == "ASSISTED"
    ):
        face_key, basis = "VULNERABLE", "어린이·고령·민감·보조기구 사용자 기준"
    elif traits.mobility == "SLOW":
        face_key, basis = "SLOW", "보행 약자 기준"
    else:
        face_key, basis = "NORMAL", "일반 성인 기준"
    load_key = "VULNERABLE" if traits.is_vulnerable else "NORMAL"
    return ComfortLimits(
        face_velocity_mps=FACE_VELOCITY_LIMIT_MPS[face_key],
        lateral_load_ratio=LATERAL_LOAD_LIMIT[load_key],
        basis=basis,
    )


def fine_fraction_from_pm(pm25: float, pm10: Optional[float] = None) -> float:
    """Mass share of fine (hard to detach) particles in the deposited load."""
    if pm10 and pm10 > 0:
        return float(np.clip(pm25 / pm10, 0.2, 0.8))
    return 0.4


def _zone_of(height_m: float, segments: tuple[BodySegment, ...]) -> Optional[int]:
    """Index of the segment a nozzle row faces, or None if it clears the head."""
    crown = max(segment.z_high_m for segment in segments)
    for index, segment in enumerate(segments):
        if segment.z_high_m == crown:
            if segment.z_low_m <= height_m <= crown + HEAD_ROW_TOLERANCE_M:
                return index
        elif segment.z_low_m <= height_m < segment.z_high_m:
            return index
    return None


@dataclass(frozen=True)
class GateBodyMap:
    """Everything about one (gate, body) pair that does not depend on setpoints."""

    segments: tuple[BodySegment, ...]
    row_zone: tuple[Optional[int], ...]
    standoff_m: np.ndarray
    decay: np.ndarray
    strip_fraction: np.ndarray
    rows_per_zone: np.ndarray
    zone_decay: np.ndarray
    zone_coverage: np.ndarray
    area_share: np.ndarray


def map_gate_to_body(
    profile: UserProfile,
    gate: GateGeometry = DEFAULT_GATE,
    params: PhysicsParams = DEFAULT_PARAMS,
) -> GateBodyMap:
    """Assign each hardware nozzle row to the body zone it actually reaches."""
    segments = profile.body.segments()
    row_zone = tuple(_zone_of(height, segments) for height in gate.row_heights_m)

    standoff = np.zeros(gate.row_count)
    decay = np.zeros(gate.row_count)
    strip_fraction = np.zeros(gate.row_count)
    for row, zone in enumerate(row_zone):
        if zone is None:
            continue
        segment = segments[zone]
        distance = max(
            gate.passage_width_m / 2.0 - segment.half_width_m,
            params.min_standoff_m,
        )
        standoff[row] = distance
        # Potential core, then 1/x decay of the centerline velocity.
        decay[row] = min(
            1.0, params.jet_decay_constant * gate.nozzle_diameter_m / distance
        )
        half_radius = max(gate.nozzle_diameter_m / 2.0, params.jet_spread_rate * distance)
        strip = 2.0 * params.footprint_gain * half_radius + gate.vertical_sweep_m
        strip_fraction[row] = strip / segment.span_m

    rows_per_zone = np.array(
        [sum(zone == index for zone in row_zone) for index in range(len(segments))],
        dtype=np.float64,
    )
    zone_decay = np.array(
        [
            max((decay[row] for row, zone in enumerate(row_zone) if zone == index), default=0.0)
            for index in range(len(segments))
        ]
    )
    zone_coverage = np.array(
        [
            min(
                params.coverage_cap,
                sum(
                    strip_fraction[row]
                    for row, zone in enumerate(row_zone)
                    if zone == index
                ),
            )
            for index in range(len(segments))
        ]
    )
    areas = np.array([segment.surface_area_m2 for segment in segments])
    return GateBodyMap(
        segments=segments,
        row_zone=row_zone,
        standoff_m=standoff,
        decay=decay,
        strip_fraction=strip_fraction,
        rows_per_zone=rows_per_zone,
        zone_decay=zone_decay,
        zone_coverage=zone_coverage,
        area_share=areas / areas.sum(),
    )


def rows_from_setpoints(
    body_map: GateBodyMap,
    setpoints: np.ndarray,
    gate: GateGeometry = DEFAULT_GATE,
) -> np.ndarray:
    """Zone velocities -> per-row nozzle velocities. Rows above the user shut off.

    Setpoint columns 0-2 line up with ``BodyProfile.segments()``: HEAD, UPPER,
    LOWER.
    """
    setpoints = np.atleast_2d(setpoints)
    rows = np.zeros((setpoints.shape[0], gate.row_count))
    for row, zone in enumerate(body_map.row_zone):
        if zone is not None:
            rows[:, row] = setpoints[:, zone]
    return rows


def uniform_rows(
    velocity_mps: float, count: int = 1, gate: GateGeometry = DEFAULT_GATE
) -> np.ndarray:
    """The fixed-actuation baseline: every row blows the same, user or not."""
    return np.full((count, gate.row_count), float(velocity_mps))


def _detachment(surface_velocity: np.ndarray, critical_velocity: float, steepness: float) -> np.ndarray:
    """Log-logistic share of particles whose adhesion the jet can overcome."""
    scaled = (np.maximum(surface_velocity, 0.0) / critical_velocity) ** steepness
    return scaled / (1.0 + scaled)


def fan_power(
    row_velocities: np.ndarray,
    gate: GateGeometry = DEFAULT_GATE,
    params: PhysicsParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """Electrical power for (N, rows) nozzle velocities, both sides of the gate.

    Volume flow x total pressure / fan efficiency, so power grows with v^3.
    This is a fan law, not something a flow simulation has to discover.
    """
    velocities = np.where(row_velocities >= ACTIVE_VELOCITY_MPS, row_velocities, 0.0)
    flow = gate.nozzle_area_m2 * velocities
    pressure = (
        0.5 * params.air_density * velocities**2 * (1.0 + gate.nozzle_loss_coefficient)
    )
    return 2.0 * gate.nozzles_per_row * (flow * pressure).sum(axis=1) / gate.fan_efficiency


def simulate_rows(
    profile: UserProfile,
    row_velocities: np.ndarray,
    hold_seconds: np.ndarray,
    *,
    gate: GateGeometry = DEFAULT_GATE,
    params: PhysicsParams = DEFAULT_PARAMS,
    fine_fraction: float = 0.4,
    body_map: Optional[GateBodyMap] = None,
) -> SimulationBatch:
    """Simulate N passages given per-row nozzle velocities, shape (N, rows)."""
    body_map = body_map or map_gate_to_body(profile, gate, params)
    velocities = np.clip(
        np.atleast_2d(np.asarray(row_velocities, dtype=np.float64)),
        0.0,
        gate.max_jet_velocity_mps,
    )
    hold = np.clip(
        np.atleast_1d(np.asarray(hold_seconds, dtype=np.float64)),
        0.0,
        gate.max_hold_seconds,
    )
    active = velocities >= ACTIVE_VELOCITY_MPS
    velocities = np.where(active, velocities, 0.0)

    surface = velocities * body_map.decay
    adhesion = CLOTHING_ADHESION_GAIN[profile.traits.clothing]
    detached = (1.0 - fine_fraction) * _detachment(
        surface, params.coarse_critical_velocity_mps * adhesion, params.detachment_steepness
    ) + fine_fraction * _detachment(
        surface, params.fine_critical_velocity_mps * adhesion, params.detachment_steepness
    )

    zone_count = len(body_map.segments)
    zone_removal = np.zeros((velocities.shape[0], zone_count))
    face_velocity = np.zeros(velocities.shape[0])
    thrust = np.zeros(velocities.shape[0])
    for zone in range(zone_count):
        rows = [row for row, mapped in enumerate(body_map.row_zone) if mapped == zone]
        if not rows:
            continue
        fractions = body_map.strip_fraction[rows] * active[:, rows]
        covered = fractions.sum(axis=1)
        # Overlapping footprints do not clean the same patch twice.
        overlap = np.where(
            covered > params.coverage_cap,
            params.coverage_cap / np.maximum(covered, 1e-9),
            1.0,
        )
        zone_removal[:, zone] = (fractions * detached[:, rows]).sum(axis=1) * overlap
        thrust += (
            gate.nozzles_per_row
            * params.air_density
            * gate.nozzle_area_m2
            * (velocities[:, rows] ** 2).sum(axis=1)
        )
        if body_map.segments[zone].zone == "HEAD":
            face_velocity = surface[:, rows].max(axis=1)

    passage_time = gate.jet_zone_length_m / max(profile.traits.walking_speed_mps, 0.3) + hold
    kinetics = 1.0 - np.exp(-passage_time / params.detachment_time_constant_s)
    zone_removal = zone_removal * kinetics[:, None]
    removal = zone_removal @ body_map.area_share

    power = fan_power(velocities, gate, params)

    return SimulationBatch(
        removal_efficiency=removal,
        zone_removal=zone_removal,
        face_velocity_mps=face_velocity,
        lateral_load_ratio=thrust / (profile.body.mass_kg * GRAVITY),
        power_w=power,
        energy_wh=power * passage_time / 3600.0,
        passage_time_s=passage_time,
    )


def simulate(
    profile: UserProfile,
    setpoints: np.ndarray,
    *,
    gate: GateGeometry = DEFAULT_GATE,
    params: PhysicsParams = DEFAULT_PARAMS,
    fine_fraction: float = 0.4,
    body_map: Optional[GateBodyMap] = None,
) -> SimulationBatch:
    """Simulate N personalized candidates, ``setpoints`` shaped (N, 4)."""
    body_map = body_map or map_gate_to_body(profile, gate, params)
    setpoints = np.atleast_2d(np.asarray(setpoints, dtype=np.float64))
    return simulate_rows(
        profile,
        rows_from_setpoints(body_map, setpoints, gate),
        setpoints[:, 3],
        gate=gate,
        params=params,
        fine_fraction=fine_fraction,
        body_map=body_map,
    )


def describe(
    profile: UserProfile,
    row_velocities: np.ndarray,
    hold_seconds: float,
    *,
    gate: GateGeometry = DEFAULT_GATE,
    params: PhysicsParams = DEFAULT_PARAMS,
    fine_fraction: float = 0.4,
) -> SimulationResult:
    """Simulate one passage and return the rounded, row-by-row report."""
    body_map = map_gate_to_body(profile, gate, params)
    velocities = np.clip(
        np.asarray(row_velocities, dtype=np.float64).reshape(-1),
        0.0,
        gate.max_jet_velocity_mps,
    )
    velocities = np.where(velocities >= ACTIVE_VELOCITY_MPS, velocities, 0.0)
    batch = simulate_rows(
        profile,
        velocities[None, :],
        np.array([hold_seconds]),
        gate=gate,
        params=params,
        fine_fraction=fine_fraction,
        body_map=body_map,
    )
    rows = tuple(
        RowState(
            height_m=height,
            zone=None if zone is None else body_map.segments[zone].zone,
            standoff_m=None if zone is None else round(float(body_map.standoff_m[row]), 3),
            nozzle_velocity_mps=round(float(velocities[row]), 2),
            surface_velocity_mps=round(float(velocities[row] * body_map.decay[row]), 2),
        )
        for row, (height, zone) in enumerate(zip(gate.row_heights_m, body_map.row_zone))
    )
    return SimulationResult(
        removal_efficiency=round(float(batch.removal_efficiency[0]), 4),
        zone_removal={
            segment.zone: round(float(batch.zone_removal[0, index]), 4)
            for index, segment in enumerate(body_map.segments)
        },
        face_velocity_mps=round(float(batch.face_velocity_mps[0]), 2),
        lateral_load_ratio=round(float(batch.lateral_load_ratio[0]), 4),
        power_w=round(float(batch.power_w[0]), 1),
        energy_wh=round(float(batch.energy_wh[0]), 3),
        passage_time_s=round(float(batch.passage_time_s[0]), 2),
        rows=rows,
    )
