"""Turn a control level into a gate plan fitted to the user standing in it.

    ControlDecision (LOW / MEDIUM / HIGH + fixed ActuationPlan)     who?  UserProfile
                         v                                                  v
        design-point target: what that fixed plan removes        body map + comfort limits
        on the reference adult                                              v
                         +--------------->  optimizer  <--------------------+
                                               v
                       PersonalizedPlan: zone velocities, hold, row table,
                       predicted vs. fixed-plan metrics, reasons

The control level still comes from the ML + rule layer in ``airis.control``.
This module does not second-guess *how much* cleaning the context asks for; it
decides *how to deliver it* to this particular body, and reports plainly when
the user's safety limits make the requested level unreachable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Optional

from .body_profile import REFERENCE_PROFILE, UserProfile
from .control import BASE_ACTUATION, ActuationPlan
from .gate_optimizer import (
    OptimizationTask,
    optimize_with_physics,
    optimize_with_surrogate,
)
from .gate_physics import (
    ComfortLimits,
    DEFAULT_GATE,
    DEFAULT_PARAMS,
    GateGeometry,
    GateSetpoints,
    PhysicsParams,
    SimulationResult,
    comfort_limits,
    describe,
    map_gate_to_body,
    rows_from_setpoints,
    uniform_rows,
)
from .surrogate import GateSurrogate, load_surrogate


# Queue pressure decides how much a second of holding costs.
TIME_WEIGHT_BY_CROWDING = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 1.0}

# Missing the design-point removal by less than this is logged, not escalated.
# Policy parameter, in absolute removal-efficiency points.
TARGET_TOLERANCE = 0.02
TARGET_STATUSES = ("MET", "WITHIN_TOLERANCE", "SHORTFALL")


@dataclass(frozen=True)
class PersonalizedPlan:
    profile: UserProfile
    control_level: str
    target_removal: float
    target_status: str
    limits: ComfortLimits
    setpoints: GateSetpoints
    personalized: SimulationResult
    baseline: SimulationResult
    baseline_violations: tuple[str, ...]
    energy_saving_pct: float
    objective: float
    method: str
    simulator_calls: int
    reasons: tuple[str, ...]

    @property
    def needs_attendant(self) -> bool:
        """The requested level cannot be delivered safely to this user."""
        return self.target_status == "SHORTFALL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "control_level": self.control_level,
            "target_removal": self.target_removal,
            "target_status": self.target_status,
            "needs_attendant": self.needs_attendant,
            "limits": asdict(self.limits),
            "setpoints": asdict(self.setpoints),
            "personalized": self.personalized.to_dict(),
            "baseline": self.baseline.to_dict(),
            "baseline_violations": list(self.baseline_violations),
            "energy_saving_pct": self.energy_saving_pct,
            "objective": self.objective,
            "method": self.method,
            "simulator_calls": self.simulator_calls,
            "reasons": list(self.reasons),
        }


def baseline_velocity(actuation: ActuationPlan, gate: GateGeometry = DEFAULT_GATE) -> float:
    return actuation.fan_level / 100.0 * gate.max_jet_velocity_mps


@lru_cache(maxsize=256)
def design_point_removal(
    fan_level: int,
    hold_seconds: int,
    fine_fraction: float = 0.4,
    gate: GateGeometry = DEFAULT_GATE,
    params: PhysicsParams = DEFAULT_PARAMS,
) -> float:
    """Removal the fixed plan delivers on the reference adult it was sized for."""
    velocity = fan_level / 100.0 * gate.max_jet_velocity_mps
    return describe(
        REFERENCE_PROFILE,
        uniform_rows(velocity, gate=gate)[0],
        hold_seconds,
        gate=gate,
        params=params,
        fine_fraction=fine_fraction,
    ).removal_efficiency


def _violations(result: SimulationResult, limits: ComfortLimits) -> tuple[str, ...]:
    found = []
    if result.face_velocity_mps > limits.face_velocity_mps + 0.05:
        found.append(
            f"얼굴 도달 풍속 {result.face_velocity_mps:.1f} m/s > "
            f"한계 {limits.face_velocity_mps:.0f} m/s"
        )
    if result.lateral_load_ratio > limits.lateral_load_ratio + 1e-4:
        found.append(
            f"측면 하중비 {result.lateral_load_ratio:.1%} > "
            f"한계 {limits.lateral_load_ratio:.1%}"
        )
    return tuple(found)


def _reasons(
    profile: UserProfile,
    limits: ComfortLimits,
    setpoints: GateSetpoints,
    personalized: SimulationResult,
    baseline: SimulationResult,
    baseline_violations: tuple[str, ...],
    target: float,
    target_status: str,
    gate: GateGeometry,
    params: PhysicsParams,
) -> tuple[str, ...]:
    body = profile.body
    traits = profile.traits
    reasons: list[str] = []

    idle = [row for row in personalized.rows if row.zone is None]
    if idle:
        reasons.append(
            f"머리 끝 {body.height_m:.2f} m → {idle[0].height_m:.2f} m 이상 노즐 "
            f"{len(idle)}열 차단 (고정 제어는 사람 없는 높이에도 송풍)"
        )

    reference_map = map_gate_to_body(REFERENCE_PROFILE, gate, params)
    body_map = map_gate_to_body(profile, gate, params)
    delta = body_map.zone_decay[1] - reference_map.zone_decay[1]
    if abs(delta) >= 0.05:
        direction = "노즐에 가까워" if delta > 0 else "노즐에서 멀어"
        reasons.append(
            f"어깨폭 {body.shoulder_width_m:.2f} m → 몸통이 {direction} "
            f"도달 속도비 {body_map.zone_decay[1]:.2f} "
            f"(기준 성인 {reference_map.zone_decay[1]:.2f})"
        )

    reasons.append(
        f"{limits.basis}: 얼굴 풍속 ≤ {limits.face_velocity_mps:.0f} m/s, "
        f"측면 하중 ≤ 체중의 {limits.lateral_load_ratio:.1%} "
        f"→ 맞춤 제어 {personalized.face_velocity_mps:.1f} m/s, "
        f"{personalized.lateral_load_ratio:.1%}"
    )
    for violation in baseline_violations:
        reasons.append(f"고정 제어 그대로면 {violation}")

    transit = gate.jet_zone_length_m / max(traits.walking_speed_mps, 0.3)
    reasons.append(
        f"보행속도 {traits.walking_speed_mps:.1f} m/s → 통과 노출 {transit:.1f}초 "
        f"+ 정지 {setpoints.hold_seconds:.1f}초"
    )
    if traits.clothing == "BULKY":
        reasons.append("부피 큰 의류 → 입자 부착력 가정 25% 상향")

    if target - baseline.removal_efficiency > TARGET_TOLERANCE:
        follow_up = "" if target_status == "SHORTFALL" else " → 맞춤 제어가 보정"
        reasons.append(
            f"고정 제어는 이 체형에서 제거율 {baseline.removal_efficiency:.1%}로 "
            f"목표 {target:.1%}에 미달{follow_up}"
        )
    energy = f"에너지 {baseline.energy_wh:.3f} → {personalized.energy_wh:.3f} Wh"
    if target_status == "MET":
        reasons.append(
            f"목표 제거율 {target:.1%} 충족 ({personalized.removal_efficiency:.1%}), {energy}"
        )
    elif target_status == "WITHIN_TOLERANCE":
        reasons.append(
            f"안전 한계 때문에 제거율 {personalized.removal_efficiency:.1%}로 목표 {target:.1%}에 "
            f"허용 오차({TARGET_TOLERANCE:.0%}p) 이내 미달, {energy}"
        )
    else:
        reasons.append(
            f"안전 한계 안에서 목표 제거율 {target:.1%} 도달 불가 → "
            f"달성 가능한 최대 {personalized.removal_efficiency:.1%}로 운전, 관리자 확인 필요"
        )
    return tuple(reasons)


def _target_status(removal: float, target: float) -> str:
    if removal >= target - 1e-3:
        return "MET"
    if removal >= target - TARGET_TOLERANCE:
        return "WITHIN_TOLERANCE"
    return "SHORTFALL"


def plan_personalized(
    profile: UserProfile,
    control_level: str,
    *,
    actuation: Optional[ActuationPlan] = None,
    crowding_level: str = "LOW",
    fine_fraction: float = 0.4,
    surrogate: Optional[GateSurrogate] = None,
    use_surrogate: bool = True,
    gate: GateGeometry = DEFAULT_GATE,
    params: PhysicsParams = DEFAULT_PARAMS,
    seed: int = 0,
) -> PersonalizedPlan:
    """Optimize the gate for one user at the requested control level."""
    actuation = actuation or BASE_ACTUATION[control_level]
    fine_fraction = round(float(fine_fraction), 2)
    limits = comfort_limits(profile.traits)

    baseline = describe(
        profile,
        uniform_rows(baseline_velocity(actuation, gate), gate=gate)[0],
        actuation.gate_hold_seconds,
        gate=gate,
        params=params,
        fine_fraction=fine_fraction,
    )
    target = design_point_removal(
        actuation.fan_level, actuation.gate_hold_seconds, fine_fraction, gate, params
    )
    task = OptimizationTask(
        profile=profile,
        target_removal=target,
        reference_energy_wh=max(baseline.energy_wh, 1e-6),
        reference_time_s=baseline.passage_time_s,
        time_weight=TIME_WEIGHT_BY_CROWDING[crowding_level],
        fine_fraction=fine_fraction,
        gate=gate,
        params=params,
    )

    if use_surrogate and surrogate is None:
        surrogate = load_surrogate()
    if use_surrogate and surrogate is not None and surrogate.matches(gate, params):
        result = optimize_with_surrogate(task, surrogate, seed=seed)
    else:
        result = optimize_with_physics(task, seed=seed)

    body_map = map_gate_to_body(profile, gate, params)
    personalized = describe(
        profile,
        rows_from_setpoints(body_map, result.setpoints.as_array(), gate)[0],
        result.setpoints.hold_seconds,
        gate=gate,
        params=params,
        fine_fraction=fine_fraction,
    )
    baseline_violations = _violations(baseline, limits)
    target_status = _target_status(personalized.removal_efficiency, target)
    return PersonalizedPlan(
        profile=profile,
        control_level=control_level,
        target_removal=round(target, 4),
        target_status=target_status,
        limits=limits,
        setpoints=result.setpoints,
        personalized=personalized,
        baseline=baseline,
        baseline_violations=baseline_violations,
        energy_saving_pct=round(
            100.0 * (1.0 - personalized.energy_wh / max(baseline.energy_wh, 1e-6)), 1
        ),
        objective=result.cost,
        method=result.method,
        simulator_calls=result.simulator_calls,
        reasons=_reasons(
            profile,
            limits,
            result.setpoints,
            personalized,
            baseline,
            baseline_violations,
            target,
            target_status,
            gate,
            params,
        ),
    )
