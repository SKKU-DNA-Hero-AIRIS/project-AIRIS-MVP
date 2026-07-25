"""Hybrid decision layer: ML policy arbitrated by deterministic rules.

    context JSON
         v
    feature extraction  ->  ML policy  ->  P(LOW/MEDIUM/HIGH)
         v                                        v
    rule layer  ->  floor / ceiling  ->  arbitration  ->  control level
                                                v
                                        actuation setpoints

Arbitration order is fixed and safety-first: ``max(min(ml, ceiling), floor)``.
A ceiling can calm the model down, but a floor can always overrule the ceiling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .features import extract_features
from .ml_policy import ControlPolicy, FeatureContribution, load_policy
from .models import LEVEL_SCORE, MultimodalContext
from .rules import RuleHit, ceiling_level, evaluate_rules, floor_level


@dataclass(frozen=True)
class ActuationPlan:
    fan_level: int
    recirculation: str
    uv_sterilizer: bool
    gate_hold_seconds: int
    air_shower: bool
    alert: str


@dataclass(frozen=True)
class ControlDecision:
    control_level: str
    decision_source: str
    ml_level: str
    ml_confidence: int
    ml_probabilities: dict[str, float]
    floor_level: str
    ceiling_level: str
    baseline_level: str
    interpretation: str
    reasons: tuple[str, ...]
    rule_hits: tuple[RuleHit, ...]
    drivers: tuple[FeatureContribution, ...]
    actuation: ActuationPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_level": self.control_level,
            "decision_source": self.decision_source,
            "ml": {
                "level": self.ml_level,
                "confidence": self.ml_confidence,
                "probabilities": self.ml_probabilities,
                "drivers": [asdict(driver) for driver in self.drivers],
            },
            "rules": {
                "floor_level": self.floor_level,
                "ceiling_level": self.ceiling_level,
                "fired": [asdict(hit) for hit in self.rule_hits],
            },
            "baseline_level": self.baseline_level,
            "interpretation": self.interpretation,
            "reasons": list(self.reasons),
            "actuation": asdict(self.actuation),
        }


BASE_ACTUATION: dict[str, ActuationPlan] = {
    "LOW": ActuationPlan(
        fan_level=30,
        recirculation="OPEN",
        uv_sterilizer=False,
        gate_hold_seconds=2,
        air_shower=False,
        alert="NONE",
    ),
    "MEDIUM": ActuationPlan(
        fan_level=65,
        recirculation="FILTERED",
        uv_sterilizer=True,
        gate_hold_seconds=5,
        air_shower=False,
        alert="DASHBOARD",
    ),
    "HIGH": ActuationPlan(
        fan_level=100,
        recirculation="CLOSED",
        uv_sterilizer=True,
        gate_hold_seconds=9,
        air_shower=True,
        alert="MANAGER",
    ),
}


def plan_actuation(level: str, context: MultimodalContext) -> ActuationPlan:
    """Map a control level to setpoints, tightened for high-sensitivity sites."""
    plan = BASE_ACTUATION[level]
    if level != "LOW" and context.facility_context.sensitivity == "HIGH":
        return ActuationPlan(
            fan_level=min(100, plan.fan_level + 15),
            recirculation=plan.recirculation,
            uv_sterilizer=True,
            gate_hold_seconds=plan.gate_hold_seconds + 2,
            air_shower=True,
            alert=plan.alert,
        )
    return plan


def baseline_control_level(context: MultimodalContext) -> str:
    """Fixed thresholds applied straight to raw sensor values.

    This is the comparison target from the research question: it has no notion
    of facility or activity context.
    """
    health = context.health_context
    if health.heart_rate > 120 or health.body_temperature >= 37.8 or health.spo2 < 95:
        return "HIGH"
    if context.environment_context.pm25 >= 36 or context.vision_context.person_count >= 8:
        return "MEDIUM"
    return "LOW"


def _interpret(
    context: MultimodalContext, level: str, source: str, ml_level: str
) -> str:
    facility = context.facility_context.label
    if source == "RULE_FLOOR":
        return (
            f"ML은 {ml_level}로 추정했으나 안전 하한 규칙이 {level}로 상향 고정한 상황"
        )
    if source == "RULE_CEILING":
        return (
            f"ML은 {ml_level}로 추정했으나 {facility} 맥락상 과잉 제어로 판단해 "
            f"{level}로 완화한 상황"
        )
    if level == "HIGH":
        return f"{facility}에서 즉각적인 강화 제어가 필요한 고위험 진입 상황"
    if level == "MEDIUM":
        return f"{facility}에서 하나 이상의 주의 신호가 감지된 관찰 필요 상황"
    return f"{facility}의 지표가 안정적인 정상 진입 상황"


def analyze_hybrid(
    context: MultimodalContext, policy: ControlPolicy | None = None
) -> ControlDecision:
    """Run the full ML + rule decision pipeline on one standardized context."""
    policy = policy or load_policy()
    prediction = policy.predict(context)
    hits = evaluate_rules(context)

    floor = floor_level(hits)
    ceiling = ceiling_level(hits)
    capped = min(prediction.level, ceiling, key=lambda level: LEVEL_SCORE[level])
    final = max(capped, floor, key=lambda level: LEVEL_SCORE[level])

    if final == prediction.level:
        source = "ML"
    elif LEVEL_SCORE[final] > LEVEL_SCORE[prediction.level]:
        source = "RULE_FLOOR"
    else:
        source = "RULE_CEILING"

    reasons: list[str] = []
    if source == "RULE_FLOOR":
        reasons.extend(
            hit.message for hit in hits if hit.kind == "FLOOR" and hit.level == final
        )
    elif source == "RULE_CEILING":
        reasons.extend(
            hit.message for hit in hits if hit.kind == "CEILING" and hit.level == final
        )
    # A near-zero feature can still carry a positive logit contribution (its
    # absence argues for the class). True, but noise to an operator — the full
    # driver list stays in the JSON for auditing.
    reasons.extend(
        f"{driver.label} {driver.value:.2f} → ML {prediction.level} 지지 "
        f"(+{driver.contribution:.2f})"
        for driver in prediction.drivers
        if driver.contribution > 0 and driver.value >= 0.05
    )
    for hit in hits:
        if hit.message not in reasons:
            reasons.append(hit.message)
    if not reasons:
        reasons.append("두드러진 위험 신호 없음 — 지표 전반이 기준 범위")

    return ControlDecision(
        control_level=final,
        decision_source=source,
        ml_level=prediction.level,
        ml_confidence=prediction.confidence,
        ml_probabilities=prediction.probabilities,
        floor_level=floor,
        ceiling_level=ceiling,
        baseline_level=baseline_control_level(context),
        interpretation=_interpret(context, final, source, prediction.level),
        reasons=tuple(dict.fromkeys(reasons))[:5],
        rule_hits=hits,
        drivers=prediction.drivers,
        actuation=plan_actuation(final, context),
    )


def decision_report(
    context: MultimodalContext, policy: ControlPolicy | None = None
) -> dict[str, Any]:
    """Context JSON in, decision JSON out — the shape the dashboard consumes."""
    decision = analyze_hybrid(context, policy)
    return {
        "context": context.to_dict(),
        "features": extract_features(context),
        "decision": decision.to_dict(),
    }
