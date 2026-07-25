"""Deterministic rule layer around the ML policy.

Two rule kinds, deliberately separated:

* ``FLOOR`` rules are safety guarantees. They raise the control level and can
  never be argued down by the model.
* ``CEILING`` rules are context-aware damping. They cap the control level when
  the context explains away a scary-looking reading (post-workout pulse) or
  when every signal is nominal.

Keeping both as an explicit, auditable table is what makes the hybrid decision
reviewable by a facility operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .models import LEVEL_SCORE, MultimodalContext


@dataclass(frozen=True)
class Rule:
    rule_id: str
    kind: str
    level: str
    name: str
    predicate: Callable[[MultimodalContext], bool]
    message: Callable[[MultimodalContext], str]


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    kind: str
    level: str
    name: str
    message: str


SAFETY_RULES: tuple[Rule, ...] = (
    Rule(
        rule_id="F1",
        kind="FLOOR",
        level="HIGH",
        name="저산소증",
        predicate=lambda ctx: ctx.health_context.spo2 < 92,
        message=lambda ctx: (
            f"산소포화도 {ctx.health_context.spo2}%로 저산소 임계선(92%) 미만"
        ),
    ),
    Rule(
        rule_id="F2",
        kind="FLOOR",
        level="HIGH",
        name="고감도 시설 발열",
        predicate=lambda ctx: (
            ctx.health_context.body_temperature >= 38.0
            and ctx.facility_context.sensitivity == "HIGH"
        ),
        message=lambda ctx: (
            f"{ctx.facility_context.label}에서 체온 "
            f"{ctx.health_context.body_temperature:.1f}℃ 발열 감지"
        ),
    ),
    Rule(
        rule_id="F3",
        kind="FLOOR",
        level="HIGH",
        name="고열",
        predicate=lambda ctx: ctx.health_context.body_temperature >= 38.5,
        message=lambda ctx: (
            f"체온 {ctx.health_context.body_temperature:.1f}℃로 시설 무관 고열 기준 초과"
        ),
    ),
    Rule(
        rule_id="F4",
        kind="FLOOR",
        level="HIGH",
        name="호흡 부담 동반 심박 상승",
        predicate=lambda ctx: (
            ctx.health_context.spo2 < 95 and ctx.health_context.heart_rate >= 120
        ),
        message=lambda ctx: (
            f"산소포화도 {ctx.health_context.spo2}% + 심박수 "
            f"{ctx.health_context.heart_rate} bpm 동반 이상"
        ),
    ),
    Rule(
        rule_id="F5",
        kind="FLOOR",
        level="HIGH",
        name="취약계층 대기질",
        predicate=lambda ctx: (
            ctx.environment_context.pm25 >= 75
            and ctx.facility_context.vulnerable_population
        ),
        message=lambda ctx: (
            f"PM2.5 {ctx.environment_context.pm25}㎍/㎥ + 취약계층 시설"
        ),
    ),
    Rule(
        rule_id="F6",
        kind="FLOOR",
        level="MEDIUM",
        name="밀집 상태 대기질 저하",
        predicate=lambda ctx: (
            ctx.vision_context.crowding_level == "HIGH"
            and ctx.environment_context.pm25 >= 36
        ),
        message=lambda ctx: (
            f"{ctx.vision_context.person_count}명 밀집 + PM2.5 "
            f"{ctx.environment_context.pm25}㎍/㎥"
        ),
    ),
    Rule(
        rule_id="F7",
        kind="FLOOR",
        level="MEDIUM",
        name="미착용 밀집",
        predicate=lambda ctx: (
            ctx.vision_context.person_count >= 15 and ctx.vision_context.mask_rate < 0.4
        ),
        message=lambda ctx: (
            f"{ctx.vision_context.person_count}명 중 마스크 착용률 "
            f"{ctx.vision_context.mask_rate:.0%}"
        ),
    ),
)

DAMPING_RULES: tuple[Rule, ...] = (
    Rule(
        rule_id="C1",
        kind="CEILING",
        level="MEDIUM",
        name="신체 활동으로 설명 가능",
        predicate=lambda ctx: (
            ctx.health_context.heart_rate >= 110
            and ctx.health_context.spo2 >= 96
            and ctx.health_context.body_temperature < 37.5
            and ctx.facility_context.sensitivity == "LOW"
        ),
        message=lambda ctx: (
            f"{ctx.facility_context.label}의 심박수 "
            f"{ctx.health_context.heart_rate} bpm은 산소·체온 정상으로 운동 후 반응 가능"
        ),
    ),
    Rule(
        rule_id="C2",
        kind="CEILING",
        level="LOW",
        name="전 지표 정상",
        predicate=lambda ctx: (
            ctx.health_context.spo2 >= 97
            and ctx.health_context.body_temperature <= 37.2
            and ctx.health_context.heart_rate < 100
            and ctx.health_context.stress_index < 70
            and ctx.environment_context.pm25 <= 35
            and ctx.vision_context.crowding_level == "LOW"
        ),
        message=lambda ctx: "생체·환경·혼잡 지표가 모두 정상 범위",
    ),
)

ALL_RULES: tuple[Rule, ...] = SAFETY_RULES + DAMPING_RULES


def evaluate_rules(context: MultimodalContext) -> tuple[RuleHit, ...]:
    return tuple(
        RuleHit(
            rule_id=rule.rule_id,
            kind=rule.kind,
            level=rule.level,
            name=rule.name,
            message=rule.message(context),
        )
        for rule in ALL_RULES
        if rule.predicate(context)
    )


def floor_level(hits: tuple[RuleHit, ...]) -> str:
    levels = [hit.level for hit in hits if hit.kind == "FLOOR"]
    return max(levels, key=lambda level: LEVEL_SCORE[level], default="LOW")


def ceiling_level(hits: tuple[RuleHit, ...]) -> str:
    levels = [hit.level for hit in hits if hit.kind == "CEILING"]
    return min(levels, key=lambda level: LEVEL_SCORE[level], default="HIGH")
