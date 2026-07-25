"""
2단계: 제어 정책 — 위험도를 제어단계로 바꾼다. 여기가 룰의 자리다.

왜 여기는 학습하지 않는가
-------------------------
'위험도 0.3이면 게이트를 올려야 하는가'에는 정답 라벨이 없다. 놓쳤을 때와
막았을 때 중 무엇이 더 나쁜지는 데이터가 아니라 시설이 정하는 가치판단이다.
따라서 1단계(위험 추정)만 지도학습하고, 2단계는 비용을 명시한 룰로 둔다.
이 분리가 "같은 센서값, 다른 판정"이라는 연구질문에 원리적으로 답한다.

임계값을 왜 확률이 아니라 백분위로 잡는가
-----------------------------------------
평시 유병률 12.9% → 유행기 38.2% 로 3배 변한다. 절대 확률 임계값은 그때
같이 이동해 버린다. '평시 점수 분포의 상위 N%'로 잡으면 유병률 변화에
강건하고, "시간당 2차 검사로 보낼 수 있는 인원"이라는 운영 용량과 직접
대응된다.
"""
from __future__ import annotations

from dataclasses import dataclass

from .risk_model import RiskEstimate

LEVELS = ("LOW", "MEDIUM", "HIGH")
_RANK = {lv: i for i, lv in enumerate(LEVELS)}


def _max_level(*levels: str) -> str:
    return max(levels, key=lambda lv: _RANK[lv])


@dataclass(frozen=True)
class FacilityPolicy:
    """시설이 정하는 값. 데이터가 아니라 운영진이 결정한다."""
    facility_type: str
    label: str
    miss_cost_ratio: float      # 놓침 : 오경보 비용비
    medium_at: str              # 평시 분포 백분위 키
    high_at: str
    unmeasured_level: str       # 측정 실패 시 기본 처리
    vulnerable_population: bool


# 비용비가 높을수록 넓게(낮은 백분위에서) 잡는다.
POLICIES: dict[str, FacilityPolicy] = {
    "hospital": FacilityPolicy("hospital", "병원", 20.0, "p90", "p95", "MEDIUM", True),
    "school": FacilityPolicy("school", "학교", 10.0, "p95", "p98", "MEDIUM", True),
    "office": FacilityPolicy("office", "일반 사무실", 5.0, "p95", "p98", "LOW", False),
    "transit": FacilityPolicy("transit", "대중교통 환승장", 5.0, "p95", "p98", "LOW", False),
    "cafe": FacilityPolicy("cafe", "카페", 3.0, "p98", "p98", "LOW", False),
}


@dataclass(frozen=True)
class ControlDecision:
    control_level: str
    personal_risk: str
    environment_risk: str
    crowding_risk: str
    risk_probability: float
    confidence: float
    needs_remeasure: bool
    interpretation: str
    reasons: tuple[str, ...]


# =============================================================================
# 환경·혼잡 축 — 개인 생체와 무관하게 독립적으로 판단한다
# =============================================================================

def _environment_risk(env: dict) -> tuple[str, list[str]]:
    pm25, temp, humidity = env["pm25"], env["temperature"], env.get("humidity", 50)
    if pm25 >= 75:
        return "HIGH", [f"PM2.5 {pm25:.0f}㎍/㎥로 대기질 나쁨 — 환기 강화"]
    if pm25 >= 36 or temp >= 32 or humidity >= 80:
        return "MEDIUM", [f"PM2.5 {pm25:.0f}㎍/㎥ / 온도 {temp:.1f}℃ — 환경 주의"]
    return "LOW", []


# =============================================================================
# 개인 위험 축
# =============================================================================

def _personal_risk(risk: RiskEstimate, policy: FacilityPolicy,
                   thresholds: dict[str, float]) -> tuple[str, list[str]]:
    if not risk.measurable:
        return policy.unmeasured_level, [
            f"측정 채널 0개 — 판정 불가. {policy.label} 기본값 "
            f"{policy.unmeasured_level} 적용 후 재측정 요청"]

    p = risk.probability
    tau_med, tau_high = thresholds[policy.medium_at], thresholds[policy.high_at]
    if p >= tau_high:
        level = "HIGH"
    elif p >= tau_med:
        level = "MEDIUM"
    else:
        level = "LOW"

    reasons = []
    if level != "LOW":
        band = policy.high_at if level == "HIGH" else policy.medium_at
        reasons.append(f"위험도 {p:.2f} ≥ {policy.label} {band} 임계값 "
                       f"{thresholds[band]:.2f} (놓침:오경보 = {policy.miss_cost_ratio:.0f}:1)")
        for name, contrib in risk.top_reasons(2):
            resid = risk.residuals.get(name)
            if resid is not None:
                reasons.append(f"{_RESID_LABEL[name]} 환경보정 잔차 {resid:+.2f}"
                               f" (기여 {contrib:+.2f})")
    else:
        reasons.append(f"위험도 {p:.2f} — {policy.label} 임계값 {tau_med:.2f} 미만")

    if risk.n_valid_channels < 3:
        missing = [_RESID_LABEL[n] for n, v in risk.residuals.items() if v is None]
        reasons.append(f"측정 실패 채널: {', '.join(missing)} — 신뢰도 {risk.confidence:.0%}")
    return level, reasons


_RESID_LABEL = {"temp_resid": "체온", "hr_resid": "심박", "spo2_resid": "SpO2"}


# =============================================================================
# 통합
# =============================================================================

def decide(risk: RiskEstimate, thresholds: dict[str, float], facility: dict,
           environment: dict, vision: dict) -> ControlDecision:
    policy = POLICIES.get(facility["facility_type"], POLICIES["office"])
    personal, personal_reasons = _personal_risk(risk, policy, thresholds)
    env_level, env_reasons = _environment_risk(environment)
    crowding = vision["crowding_level"]

    control = _max_level(personal, env_level, crowding)
    reasons = list(personal_reasons) + env_reasons
    if crowding == "HIGH":
        reasons.append(f"{vision['person_count']}명으로 혼잡 — 접촉 위험 및 측정 품질 저하")

    if not risk.measurable:
        interpretation = "생체 측정 실패 — 위험도 판정 불가, 재측정 필요"
    elif personal == "HIGH":
        interpretation = f"{policy.label} 기준 고위험 개인 — 2차 검사 및 게이트 강화"
    elif personal == "MEDIUM":
        interpretation = f"{policy.label} 기준 주의 대상 — 관찰 및 통과 지연"
    elif control != "LOW":
        interpretation = "개인 생체는 정상 — 환경·혼잡 요인으로 환기 상향"
    else:
        interpretation = "정상 진입 상황"

    return ControlDecision(
        control_level=control, personal_risk=personal, environment_risk=env_level,
        crowding_risk=crowding, risk_probability=risk.probability,
        confidence=risk.confidence, needs_remeasure=not risk.measurable,
        interpretation=interpretation, reasons=tuple(reasons[:5]),
    )
