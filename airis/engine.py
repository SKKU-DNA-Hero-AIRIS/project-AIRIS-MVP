from dataclasses import dataclass

from .models import MultimodalContext


LEVEL_SCORE = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


@dataclass(frozen=True)
class DecisionResult:
    user_risk: str
    environment_risk: str
    crowding_risk: str
    facility_risk: str
    overall_risk: str
    control_level: str
    confidence: int
    interpretation: str
    reasons: tuple[str, ...]


def _health_risk(context: MultimodalContext) -> tuple[str, list[str]]:
    health = context.health_context
    facility = context.facility_context
    reasons: list[str] = []

    danger_signals = 0
    if health.spo2 < 95:
        danger_signals += 2
        reasons.append(f"산소포화도 {health.spo2}%로 주의 기준(95%) 미만")
    if health.body_temperature >= 38:
        danger_signals += 2
        reasons.append(f"체온 {health.body_temperature:.1f}℃로 발열 가능성")
    if health.heart_rate >= 120:
        danger_signals += 1
        reasons.append(f"심박수 {health.heart_rate} bpm으로 상승")
    if health.stress_index >= 80:
        danger_signals += 1
        reasons.append(f"스트레스 지수 {health.stress_index}로 높음")

    if danger_signals >= 3:
        return "HIGH", reasons
    if danger_signals >= 1:
        # The same elevated pulse is interpreted more conservatively in a
        # sensitive facility. This is the prototype's core research claim.
        if facility.sensitivity == "HIGH" and health.heart_rate >= 120:
            reasons.append(f"{facility.label}은 취약 사용자가 있는 고감도 시설")
            return "HIGH", reasons
        return "MEDIUM", reasons
    return "LOW", ["생체 지표가 정상 범위"]


def _environment_risk(context: MultimodalContext) -> tuple[str, list[str]]:
    env = context.environment_context
    reasons: list[str] = []
    if env.pm25 >= 75:
        reasons.append(f"PM2.5 {env.pm25}㎍/㎥로 대기질 나쁨")
        return "HIGH", reasons
    if env.pm25 >= 36 or env.temperature >= 32 or env.humidity >= 80:
        reasons.append("온습도 또는 미세먼지 환경에 주의 필요")
        return "MEDIUM", reasons
    return "LOW", ["환경 지표가 쾌적한 범위"]


def analyze_context(context: MultimodalContext) -> DecisionResult:
    """Fuse contexts, estimate risks, and return an explainable decision."""
    user_risk, health_reasons = _health_risk(context)
    environment_risk, environment_reasons = _environment_risk(context)
    crowding_risk = context.vision_context.crowding_level
    facility_risk = context.facility_context.sensitivity

    scores = [
        LEVEL_SCORE[user_risk],
        LEVEL_SCORE[environment_risk],
        LEVEL_SCORE[crowding_risk],
    ]
    max_score = max(scores)
    high_count = scores.count(2)

    if user_risk == "HIGH":
        control_level = "HIGH"
    elif high_count >= 2:
        control_level = "HIGH"
    elif max_score >= 1:
        control_level = "MEDIUM"
    else:
        control_level = "LOW"

    reasons = list(health_reasons)
    if environment_risk != "LOW":
        reasons.extend(environment_reasons)
    if crowding_risk == "HIGH":
        reasons.append(f"현재 {context.vision_context.person_count}명으로 혼잡도 높음")

    if control_level == "HIGH":
        interpretation = "즉각적인 강화 제어가 필요한 고위험 진입 상황"
    elif (
        user_risk == "MEDIUM"
        and context.facility_context.facility_type == "office"
    ):
        interpretation = "일시적인 신체 활동 가능성 — 주의 수준으로 제어"
    elif control_level == "MEDIUM":
        interpretation = "하나 이상의 주의 신호가 감지된 관찰 필요 상황"
    else:
        interpretation = "현재 지표가 안정적인 정상 진입 상황"

    confidence = min(96, 72 + 6 * sum(score > 0 for score in scores) + 3 * high_count)
    return DecisionResult(
        user_risk=user_risk,
        environment_risk=environment_risk,
        crowding_risk=crowding_risk,
        facility_risk=facility_risk,
        overall_risk=control_level,
        control_level=control_level,
        confidence=confidence,
        interpretation=interpretation,
        reasons=tuple(reasons[:5]),
    )
