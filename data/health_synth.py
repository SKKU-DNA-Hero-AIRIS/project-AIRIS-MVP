"""
잠재 상태 기반 합성 생체징후 생성기 (Gaussian copula 상관구조 포함)
=====================================================================

설계 원칙
---------
1. 라벨은 임계값이 아니라 '생성에 사용한 잠재 상태'에서 나온다.
   → 임계값으로 라벨링하면 분류기가 임계값을 역산할 뿐이므로 순환논리가 된다.

2. stress_index는 독립 변수가 아니다. HRV(RMSSD/SDNN)와 심박에서
   결정론적으로 파생된다. 독립 샘플링하면 '심박 55 + 스트레스 90' 같은
   생리학적으로 불가능한 조합이 생긴다.

3. 잔차 상관은 Gaussian copula로, 인과 관계는 구조방정식으로 나눠 표현한다.
   - copula: 측정 잡음 수준의 동시 변동 (예: SDNN-RMSSD +0.80)
   - 구조식: 체온↑ → 심박↑ (Liebermeister), SpO2↓ → 호흡수↑ (보상반응)

4. severity(중증도)가 생리학적 편차 크기와 등급을 '동시에' 결정한다.
   → 등급 1은 정상과 겹치고 등급 3은 명확히 분리된다. 겹침이 문제의
     실제 난이도를 정의하며, 선형분리 가능한 합성 데이터는 무의미하다.

5. 교란변수를 의도적으로 심는다.
   - 정상인데 비정상처럼 보임: 운동 직후, 운동선수 서맥, 추운 환경
   - 비정상인데 정상처럼 보임: 베타차단제 복용, 고령자 발열 둔화

⚠️  의료 사용 금지
    합성 데이터로 학습한 모델은 실제 건강 스크리닝에 사용할 수 없다.
    특히 베타차단제/고령자 케이스의 위음성 분포는 합성으로 재현 불가능하며,
    실제 배포 시 정말 아픈 사람을 놓친다. 방법론 연구·파이프라인 검증 전용.

stress_index 주의
-----------------
아래 구현은 vendor-agnostic '대용(surrogate)' 정의다. Samsung Health,
Garmin, Baevsky SI는 각각 계산식이 다르다. 실제 프로젝트에서는
`derive_stress_index()`를 본인 센서의 정의로 교체해야 한다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 1. 잠재 상태 정의
# =============================================================================

class State(str, Enum):
    NORMAL = "normal"
    POST_EXERCISE = "post_exercise"          # 정상이지만 비정상처럼 보임
    FEBRILE_INFECTION = "febrile_infection"
    HYPOXEMIA = "hypoxemia"
    CARDIAC_STRESS = "cardiac_stress"
    DEHYDRATION = "dehydration"
    ANXIETY_PANIC = "anxiety_panic"


@dataclass(frozen=True)
class StateEffect:
    """severity=1.0 기준 최대 편차. 실제 편차 = 이 값 * severity."""
    pathological: bool
    d_hr: float = 0.0              # bpm
    d_core_temp: float = 0.0       # °C
    d_spo2_deficit: float = 0.0    # %p (양수 = SpO2 하락)
    d_rr: float = 0.0              # breaths/min
    d_sbp: float = 0.0             # mmHg
    d_dbp: float = 0.0             # mmHg
    hrv_mult: float = 1.0          # SDNN/RMSSD 배율 (<1 = 억제)
    lf_hf_mult: float = 1.0
    pi_mult: float = 1.0           # perfusion index 배율
    d_skin_temp: float = 0.0       # 심부온과 독립적인 피부온 변화


STATE_EFFECTS: dict[State, StateEffect] = {
    State.NORMAL: StateEffect(pathological=False),

    # 정상 생리반응. 발열성 감염과 표면적으로 유사 → 최대 교란변수.
    State.POST_EXERCISE: StateEffect(
        pathological=False,
        d_hr=48.0, d_core_temp=0.9, d_rr=9.0, d_sbp=14.0,
        hrv_mult=0.40, lf_hf_mult=2.2, pi_mult=1.5, d_skin_temp=1.6,
    ),

    State.FEBRILE_INFECTION: StateEffect(
        pathological=True,
        d_hr=14.0, d_core_temp=1.9, d_spo2_deficit=1.2, d_rr=5.0,
        hrv_mult=0.62, lf_hf_mult=1.4, pi_mult=1.1, d_skin_temp=0.5,
    ),

    State.HYPOXEMIA: StateEffect(
        pathological=True,
        d_hr=17.0, d_spo2_deficit=7.5, d_rr=10.0,
        hrv_mult=0.70, lf_hf_mult=1.5, pi_mult=0.62,
    ),

    # 쇼크 생리: 심박↑ + 혈압↓ → shock index 상승
    State.CARDIAC_STRESS: StateEffect(
        pathological=True,
        d_hr=34.0, d_spo2_deficit=2.5, d_rr=7.0, d_sbp=-16.0, d_dbp=-9.0,
        hrv_mult=0.45, lf_hf_mult=1.9, pi_mult=0.50, d_skin_temp=-0.9,
    ),

    State.DEHYDRATION: StateEffect(
        pathological=True,
        d_hr=20.0, d_core_temp=0.4, d_rr=3.0, d_sbp=-13.0, d_dbp=-7.0,
        hrv_mult=0.68, lf_hf_mult=1.6, pi_mult=0.58, d_skin_temp=-0.4,
    ),

    # 과호흡 → SpO2는 오히려 소폭 상승. 심박·호흡수만 보면 심장성과 혼동.
    State.ANXIETY_PANIC: StateEffect(
        pathological=True,
        d_hr=27.0, d_spo2_deficit=-0.4, d_rr=11.0, d_sbp=11.0, d_dbp=6.0,
        hrv_mult=0.52, lf_hf_mult=2.8, pi_mult=0.85, d_skin_temp=-0.5,
    ),
}


# 실제 유병률에 근접한 기본값. 병리 상태 합계 = 6%.
# 비정상을 50%로 만들면 확률 보정(calibration)이 망가진다.
DEFAULT_STATE_PROBS: dict[State, float] = {
    State.NORMAL: 0.800,
    State.POST_EXERCISE: 0.140,
    State.FEBRILE_INFECTION: 0.025,
    State.DEHYDRATION: 0.012,
    State.ANXIETY_PANIC: 0.009,
    State.HYPOXEMIA: 0.008,
    State.CARDIAC_STRESS: 0.006,
}


# =============================================================================
# 2. Gaussian copula — 잔차 상관구조
# =============================================================================

COPULA_VARS = [
    "hr", "core_temp", "spo2_def", "rr", "sbp", "dbp",
    "sdnn", "rmssd", "lf_hf", "pi",
]

# 잔차 수준의 동시 변동. 구조적 인과(체온→심박 등)는 별도로 처리하므로
# 여기서는 그보다 약한 '측정/개체 내 변동' 상관만 담는다.
_CORR_PAIRS: dict[tuple[str, str], float] = {
    ("sdnn", "rmssd"): 0.80,      # 정의상 강한 상관
    ("hr", "sdnn"): -0.45,
    ("hr", "rmssd"): -0.42,
    ("rmssd", "lf_hf"): -0.40,
    ("sdnn", "lf_hf"): -0.35,
    ("sbp", "dbp"): 0.65,
    ("hr", "sbp"): 0.15,
    ("hr", "rr"): 0.35,
    ("spo2_def", "rr"): 0.30,
    ("spo2_def", "pi"): -0.25,    # 산소 결핍↑ → 관류지수↓
    ("spo2_def", "hr"): 0.20,
    ("core_temp", "hr"): 0.25,    # Liebermeister 구조식 위에 얹히는 잔차상관
    ("core_temp", "rr"): 0.15,
    ("pi", "sbp"): 0.20,
}


def build_correlation_matrix(
    variables: list[str] = COPULA_VARS,
    pairs: dict[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """지정한 쌍별 상관에서 상관행렬을 만들고, 양정부호가 아니면 보정한다."""
    pairs = _CORR_PAIRS if pairs is None else pairs
    idx = {v: i for i, v in enumerate(variables)}
    n = len(variables)
    R = np.eye(n)
    for (a, b), r in pairs.items():
        if a not in idx or b not in idx:
            raise KeyError(f"알 수 없는 변수: {a} 또는 {b}")
        R[idx[a], idx[b]] = R[idx[b], idx[a]] = r

    # 최근접 양정부호 행렬로 투영 (고유값 clipping + 재정규화)
    eigvals, eigvecs = np.linalg.eigh(R)
    if eigvals.min() < 1e-8:
        eigvals = np.clip(eigvals, 1e-8, None)
        R = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
        np.fill_diagonal(R, 1.0)
    return R


def sample_copula(n: int, R: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """상관된 표준정규 잔차와 그에 대응하는 균등변량을 반환."""
    L = np.linalg.cholesky(R)
    Z = rng.standard_normal((n, R.shape[0])) @ L.T
    U = stats.norm.cdf(Z)
    return {
        "z": {v: Z[:, i] for i, v in enumerate(COPULA_VARS)},
        "u": {v: np.clip(U[:, i], 1e-6, 1 - 1e-6) for i, v in enumerate(COPULA_VARS)},
    }


# =============================================================================
# 3. 개인 기저선 (계층 모형: 개인 랜덤효과)
# =============================================================================

# Beta(2.2, 2.6)의 기댓값. 기저선 회귀식에서 fitness 중심화에 사용.
FITNESS_REF = 2.2 / (2.2 + 2.6)


@dataclass
class PopulationConfig:
    """모집단 구성. 인구 특성을 바꿔 도메인 이동 실험이 가능하다."""
    age_mean: float = 46.0
    age_sd: float = 18.0
    age_min: int = 18
    age_max: int = 92
    female_frac: float = 0.51
    athlete_frac: float = 0.07            # 운동선수 수준 체력 → 서맥
    beta_blocker_frac: float = 0.09       # 심박 반응 둔화
    chronic_frac: float = 0.18
    elderly_threshold: int = 75           # 이 이상 발열 반응 둔화
    state_probs: dict[State, float] = field(
        default_factory=lambda: dict(DEFAULT_STATE_PROBS)
    )


def sample_individuals(n: int, cfg: PopulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    """개인 속성 + 개인별 생리 기저선을 생성."""
    age = np.clip(
        rng.normal(cfg.age_mean, cfg.age_sd, n), cfg.age_min, cfg.age_max
    ).round().astype(int)
    is_female = rng.random(n) < cfg.female_frac
    fitness = np.clip(rng.beta(2.2, 2.6, n) + 0.35 * (rng.random(n) < cfg.athlete_frac), 0, 1)
    is_athlete = fitness > 0.85
    bmi = np.clip(rng.normal(23.8 + 0.02 * (age - 40), 3.6, n), 15.5, 45.0)

    on_beta_blocker = (rng.random(n) < cfg.beta_blocker_frac) & (age > 40)
    chronic = rng.random(n) < cfg.chronic_frac

    # 안정시 심박: 체력↑ → 낮음, 연령↑ → 소폭 상승, 여성 소폭 높음
    # fitness는 반드시 중심화(-FITNESS_REF)해야 한다. 그렇지 않으면 모집단
    # 전체 평균이 -22*E[fitness] ≈ -10bpm 만큼 끌려 내려간다.
    fit_c = fitness - FITNESS_REF
    baseline_hr = (
        72.0
        - 22.0 * fit_c
        + 0.06 * (age - 45)
        + 2.5 * is_female
        + 0.35 * (bmi - 24)
        + rng.normal(0, 4.5, n)
    )
    baseline_hr -= 6.0 * on_beta_blocker          # 약물에 의한 기저 서맥
    baseline_hr = np.clip(baseline_hr, 38, 105)

    baseline_core_temp = np.clip(rng.normal(36.78, 0.22, n) - 0.004 * (age - 45), 35.8, 37.5)

    # HRV: 연령과 강한 음의 관계, 체력과 양의 관계 (로그정규)
    # 로그정규 편향 exp(σ²/2) 도 함께 보정한다.
    _sdnn_sigma = 0.26
    baseline_sdnn = np.exp(
        np.log(50.0) - _sdnn_sigma**2 / 2
        - 0.011 * (age - 45) + 0.45 * fit_c + rng.normal(0, _sdnn_sigma, n)
    )
    baseline_rmssd = baseline_sdnn * np.exp(rng.normal(-0.22, 0.13, n))

    baseline_sbp = np.clip(
        118.0 + 0.42 * (age - 45) + 0.55 * (bmi - 24) - 3.0 * is_female
        + 8.0 * chronic + rng.normal(0, 8.5, n),
        88, 195,
    )
    baseline_dbp = np.clip(
        0.58 * baseline_sbp + 6.0 + rng.normal(0, 5.0, n), 52, 118
    )
    baseline_rr = np.clip(rng.normal(15.0, 1.7, n) + 0.02 * (age - 45), 9, 22)
    baseline_pi = np.exp(np.log(2.1) - 0.006 * (age - 45) + rng.normal(0, 0.35, n))

    return pd.DataFrame({
        "age": age,
        "sex": np.where(is_female, "F", "M"),
        "bmi": bmi.round(1),
        "fitness_level": fitness.round(3),
        "is_athlete": is_athlete,
        "on_beta_blocker": on_beta_blocker,
        "chronic_condition": chronic,
        "is_elderly": age >= cfg.elderly_threshold,
        "baseline_hr": baseline_hr.round(1),
        "baseline_core_temp": baseline_core_temp.round(2),
        "baseline_sdnn": baseline_sdnn.round(1),
        "baseline_rmssd": baseline_rmssd.round(1),
        "baseline_sbp": baseline_sbp.round(1),
        "baseline_dbp": baseline_dbp.round(1),
        "baseline_rr": baseline_rr.round(1),
        "baseline_pi": baseline_pi.round(2),
    })


# =============================================================================
# 4. 맥락 변수 (활동 / 환경 / 일주기)
# =============================================================================

def sample_context(
    n: int, states: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """활동 상태, 환경, 측정 시각. 운동 직후 상태와 정합적으로 생성."""
    is_post_ex = states == State.POST_EXERCISE.value

    activity = np.where(
        is_post_ex,
        "post_exercise",
        rng.choice(["rest", "walking"], size=n, p=[0.72, 0.28]),
    )
    # 운동 직후: 경과시간이 짧을수록 효과 강함 (지수 감쇠)
    minutes_since_activity = np.where(
        is_post_ex,
        rng.gamma(shape=2.0, scale=3.5, size=n),
        rng.gamma(shape=3.0, scale=25.0, size=n),
    )

    hour = rng.integers(0, 24, n)
    ambient_temp = np.clip(rng.normal(22.5, 5.5, n), 2.0, 38.0)
    ambient_humidity = np.clip(rng.normal(52.0, 15.0, n), 15.0, 95.0)
    prior_sleep_hours = np.clip(rng.normal(6.8, 1.4, n), 2.0, 11.0)

    return pd.DataFrame({
        "activity_state": activity,
        "minutes_since_activity": minutes_since_activity.round(1),
        "hour_of_day": hour,
        "ambient_temp": ambient_temp.round(1),
        "ambient_humidity": ambient_humidity.round(1),
        "prior_sleep_hours": prior_sleep_hours.round(1),
    })


# =============================================================================
# 5. 파생 피쳐
# =============================================================================

def derive_stress_index(
    hr: np.ndarray, baseline_hr: np.ndarray,
    rmssd: np.ndarray, baseline_rmssd: np.ndarray,
) -> np.ndarray:
    """
    HRV + 심박에서 결정론적으로 파생되는 0~100 스트레스 지수 (대용 정의).

    ⚠️  실제 프로젝트에서는 사용 센서의 정의로 교체할 것.
        Samsung Health / Garmin / Baevsky SI는 계산식이 서로 다르다.
        핵심은 '독립 샘플링하지 않는다'는 것이다.
    """
    hrv_drop = (baseline_rmssd - rmssd) / np.maximum(baseline_rmssd, 1.0)
    hr_rise = (hr - baseline_hr) / np.maximum(baseline_hr, 1.0)
    logit = -0.55 + 3.4 * hrv_drop + 2.9 * hr_rise
    return np.clip(100.0 / (1.0 + np.exp(-logit)), 0.0, 100.0)


def news2_score(rr, spo2, sbp, hr, temp) -> np.ndarray:
    """
    NEWS2 부분 점수 (의식수준·산소투여 항목 제외 → 최대 15점).
    도메인 지식을 피쳐로 주입하는 가장 효율적인 방법.
    """
    s = np.zeros(len(rr), dtype=int)
    s += np.select([rr <= 8, rr <= 11, rr <= 20, rr <= 24], [3, 1, 0, 2], default=3)
    s += np.select([spo2 <= 91, spo2 <= 93, spo2 <= 95], [3, 2, 1], default=0)
    s += np.select([sbp <= 90, sbp <= 100, sbp <= 110, sbp <= 219], [3, 2, 1, 0], default=3)
    s += np.select([hr <= 40, hr <= 50, hr <= 90, hr <= 110, hr <= 130], [3, 1, 0, 1, 2], default=3)
    s += np.select([temp <= 35.0, temp <= 36.0, temp <= 38.0, temp <= 39.0], [3, 1, 0, 1], default=2)
    return s


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """원시값보다 강력한 파생 피쳐. 특히 개인 기저선 대비 z-score."""
    out = df.copy()

    out["map"] = (out["systolic_bp"] + 2 * out["diastolic_bp"]) / 3
    out["pulse_pressure"] = out["systolic_bp"] - out["diastolic_bp"]
    out["shock_index"] = out["heart_rate"] / out["systolic_bp"]
    out["hr_temp_ratio"] = out["heart_rate"] / out["body_temperature"]
    out["core_skin_gradient"] = out["body_temperature"] - out["skin_temperature"]

    # 개인 기저선 대비 편차 — 단일 최강 피쳐.
    # 안정시 심박 50인 선수의 90과 기저 80인 사람의 90은 전혀 다른 의미다.
    out["hr_z"] = (out["heart_rate"] - out["baseline_hr"]) / 8.0
    out["temp_z"] = (out["body_temperature"] - out["baseline_core_temp"]) / 0.30
    out["sbp_z"] = (out["systolic_bp"] - out["baseline_sbp"]) / 10.0
    out["rr_z"] = (out["respiratory_rate"] - out["baseline_rr"]) / 2.0
    out["rmssd_ratio"] = out["rmssd"] / out["baseline_rmssd"]
    out["sdnn_ratio"] = out["sdnn"] / out["baseline_sdnn"]

    out["news2_partial"] = news2_score(
        out["respiratory_rate"].to_numpy(), out["spo2"].to_numpy(),
        out["systolic_bp"].to_numpy(), out["heart_rate"].to_numpy(),
        out["body_temperature"].to_numpy(),
    )
    return out


# =============================================================================
# 6. 메인 생성기 (단면 데이터)
# =============================================================================

def _severity_to_grade(state: str, sev: float) -> int:
    """
    등급은 severity에서 나온다. severity는 생리학적 편차 크기도 함께
    결정하므로, 등급 1은 정상과 자연스럽게 겹치고 등급 3은 분리된다.
    임계값으로 라벨링하지 않는 것이 핵심.
    """
    if not STATE_EFFECTS[State(state)].pathological:
        return 0
    # 임계값은 severity 분포(하한 0.22 + 0.78*Beta(2,2.2))의 3분위에 맞춤.
    # 하한선 도입 전 값(0.35/0.70)을 그대로 쓰면 등급 1이 0.5%로 붕괴한다.
    if sev < 0.48:
        return 1   # 경계
    if sev < 0.72:
        return 2   # 주의
    return 3       # 이상


PATHOLOGICAL_SEVERITY_FLOOR = 0.22
"""
병리 상태의 severity 하한.

이게 없으면 severity≈0.05인 'febrile_infection' 표본이 생기는데, 이는
발열이 없는 발열성 감염이라는 모순이다. 실측에서 그 결과로 병리 상태의
평균 체온이 37.5°C(발열 기준 38°C 미달)로 나왔다. 하한선을 두면 상태
정의가 유지되면서도 등급 1은 여전히 정상과 겹친다.
"""


def sample_severity(states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """상태 종류에 따라 다른 severity 분포를 적용."""
    n = len(states)
    sev = np.zeros(n)

    is_path = np.array([STATE_EFFECTS[State(s)].pathological for s in states])
    is_post_ex = states == State.POST_EXERCISE.value

    # 병리: 하한선 위에서 경증 우편향
    f = PATHOLOGICAL_SEVERITY_FLOOR
    sev[is_path] = f + (1 - f) * rng.beta(2.0, 2.2, is_path.sum())
    # 운동 직후: 가벼운 산책 ~ 전력질주까지 전 구간
    sev[is_post_ex] = rng.beta(2.0, 2.0, is_post_ex.sum())
    # State.NORMAL 은 0 유지
    return sev


def generate(
    n: int = 20_000,
    cfg: PopulationConfig | None = None,
    seed: int = 42,
    *,
    states: np.ndarray | None = None,
    people: pd.DataFrame | None = None,
    severity: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    단면 데이터 생성.

    states / people / severity 를 주면 그것을 사용한다 (시계열 모드에서
    동일 개인의 상태 경로를 그대로 물리 생성에 반영하기 위한 경로).
    """
    cfg = cfg or PopulationConfig()
    rng = np.random.default_rng(seed)

    # --- 잠재 상태 추출 ---
    if states is None:
        labels = list(cfg.state_probs.keys())
        probs = np.array([cfg.state_probs[s] for s in labels], dtype=float)
        probs = probs / probs.sum()
        states = rng.choice([s.value for s in labels], size=n, p=probs)
    else:
        states = np.asarray(states)
        n = len(states)

    if severity is None:
        severity = sample_severity(states, rng)
    else:
        severity = np.asarray(severity, dtype=float)

    if people is None:
        people = sample_individuals(n, cfg, rng)
    people = people.reset_index(drop=True)

    ctx = sample_context(n, states, rng)

    R = build_correlation_matrix()
    cop = sample_copula(n, R, rng)
    z, u = cop["z"], cop["u"]

    # --- 상태 효과 벡터화 ---
    def eff(attr: str) -> np.ndarray:
        return np.array([getattr(STATE_EFFECTS[State(s)], attr) for s in states])

    # 베타차단제: 심박 반응 둔화 (감염인데 심박 정상 → 위음성 유발)
    hr_blunt = np.where(people["on_beta_blocker"].to_numpy(), 0.45, 1.0)
    # 고령자: 발열 반응 둔화 (감염인데 체온 정상 → 위음성 유발)
    temp_blunt = np.where(people["is_elderly"].to_numpy(), 0.50, 1.0)

    # 개인별 반응 이질성: 같은 감염에도 반응 크기가 사람마다 다르다.
    # severity 하한선만으로는 등급 1이 지나치게 쉬워지므로, 계통별로
    # 독립적인 반응 이득을 곱해 자연스러운 겹침을 만든다.
    gain_hr = np.exp(rng.normal(0, 0.30, n))
    gain_temp = np.exp(rng.normal(0, 0.26, n))
    gain_resp = np.exp(rng.normal(0, 0.30, n))
    gain_bp = np.exp(rng.normal(0, 0.28, n))
    gain_auto = np.exp(rng.normal(0, 0.22, n))   # 자율신경(HRV) 계통

    # 운동 직후 효과는 경과시간에 따라 지수 감쇠
    decay = np.where(
        states == State.POST_EXERCISE.value,
        np.exp(-ctx["minutes_since_activity"].to_numpy() / 8.0),
        1.0,
    )
    sev = severity * decay

    # --- 일주기 리듬: 새벽 최저, 저녁 최고 (진폭 ±0.3°C) ---
    hour = ctx["hour_of_day"].to_numpy()
    circadian = 0.30 * np.cos(2 * np.pi * (hour - 18) / 24)

    # --- 심부 체온 ---
    core_temp = (
        people["baseline_core_temp"].to_numpy()
        + eff("d_core_temp") * sev * temp_blunt * gain_temp
        + circadian
        + 0.22 * z["core_temp"]
    )

    # --- SpO2: 결핍량을 감마 분포로 → 좌편향 + 100% 상한 보장 ---
    spo2_deficit_base = stats.gamma.ppf(u["spo2_def"], a=2.2, scale=0.55)
    spo2 = 100.0 - spo2_deficit_base - eff("d_spo2_deficit") * sev * gain_resp
    spo2 = np.clip(spo2, 70.0, 100.0)

    # --- 호흡수: 구조식으로 저산소 보상반응 반영 ---
    hypoxia_drive = np.maximum(0.0, 94.0 - spo2) * 1.15
    resp_rate = (
        people["baseline_rr"].to_numpy()
        + eff("d_rr") * sev * gain_resp
        + hypoxia_drive
        + 1.5 * z["rr"]
    )
    resp_rate = np.clip(resp_rate, 6.0, 46.0)

    # --- 심박: Liebermeister 법칙(1°C당 약 9bpm)을 구조식으로 ---
    temp_deviation = core_temp - people["baseline_core_temp"].to_numpy() - circadian
    heart_rate = (
        people["baseline_hr"].to_numpy()
        + eff("d_hr") * sev * hr_blunt * gain_hr
        + 9.0 * temp_deviation * hr_blunt        # 발열성 빈맥
        + 0.9 * hypoxia_drive * hr_blunt         # 저산소성 빈맥
        + 4.5 * z["hr"]
    )
    heart_rate = np.clip(heart_rate, 35.0, 205.0)

    # --- 혈압 ---
    systolic = np.clip(
        people["baseline_sbp"].to_numpy() + eff("d_sbp") * sev * gain_bp + 7.0 * z["sbp"], 65, 235
    )
    diastolic = np.clip(
        people["baseline_dbp"].to_numpy() + eff("d_dbp") * sev * gain_bp + 4.5 * z["dbp"], 38, 135
    )
    diastolic = np.minimum(diastolic, systolic - 12.0)   # 생리학적 제약

    # --- HRV: 로그정규 + 상태별 억제 배율 ---
    sdnn = np.clip(
        people["baseline_sdnn"].to_numpy()
        * (eff("hrv_mult") ** (sev * gain_auto))
        * np.exp(0.20 * z["sdnn"]),
        3.0, 260.0,
    )
    rmssd = np.clip(
        people["baseline_rmssd"].to_numpy()
        * (eff("hrv_mult") ** (sev * gain_auto))
        * np.exp(0.22 * z["rmssd"]),
        2.0, 240.0,
    )
    lf_hf = np.clip(
        1.5 * (eff("lf_hf_mult") ** (sev * gain_auto)) * np.exp(0.38 * z["lf_hf"]), 0.15, 22.0
    )
    pnn50 = np.clip(100 / (1 + np.exp(-(rmssd - 30) / 11)) + rng.normal(0, 4, n), 0, 100)

    perfusion_index = np.clip(
        people["baseline_pi"].to_numpy() * (eff("pi_mult") ** (sev * gain_bp)) * np.exp(0.28 * z["pi"]),
        0.08, 12.0,
    )

    # --- 피부온: 환경 온도에 직접 영향 (추운 곳 → 낮음, 정상인데 이상해 보임) ---
    skin_temp = (
        core_temp - 1.6
        + 0.085 * (ctx["ambient_temp"].to_numpy() - 22.0)
        + eff("d_skin_temp") * sev * gain_temp
        + rng.normal(0, 0.28, n)
    )

    # --- stress_index: 파생값. 독립 샘플링하지 않는다. ---
    stress_index = derive_stress_index(
        heart_rate, people["baseline_hr"].to_numpy(),
        rmssd, people["baseline_rmssd"].to_numpy(),
    )

    # --- 측정 잡음 (센서 특성) ---
    heart_rate += rng.normal(0, 1.2, n)
    spo2 = np.clip(spo2 + rng.normal(0, 0.45, n), 70, 100)
    core_temp += rng.normal(0, 0.08, n)

    df = pd.DataFrame({
        # 원본 요청 4개 피쳐
        "heart_rate": heart_rate.round(0).astype(int),
        "spo2": spo2.round(0).astype(int),
        "body_temperature": core_temp.round(1),
        "stress_index": stress_index.round(0).astype(int),
        # 확장 피쳐
        "respiratory_rate": resp_rate.round(0).astype(int),
        "systolic_bp": systolic.round(0).astype(int),
        "diastolic_bp": diastolic.round(0).astype(int),
        "skin_temperature": skin_temp.round(1),
        "sdnn": sdnn.round(1),
        "rmssd": rmssd.round(1),
        "pnn50": pnn50.round(1),
        "lf_hf_ratio": lf_hf.round(2),
        "perfusion_index": perfusion_index.round(2),
    })

    df = pd.concat([df, people, ctx], axis=1)
    df["latent_state"] = states
    df["severity"] = sev.round(3)
    df["grade"] = [_severity_to_grade(s, v) for s, v in zip(states, sev)]
    df["is_abnormal"] = (df["grade"] > 0).astype(int)
    df = add_derived_features(df)

    return df


# =============================================================================
# 7. 시계열 모드 (연속 측정)
# =============================================================================

# 상태 지속성: 대각 성분이 크다 = 상태가 쉽게 바뀌지 않는다
def _transition_matrix(labels: list[State], persistence: float = 0.94) -> np.ndarray:
    n = len(labels)
    base = np.array([DEFAULT_STATE_PROBS[s] for s in labels])
    base = base / base.sum()
    T = np.tile(base, (n, 1)) * (1 - persistence)
    np.fill_diagonal(T, np.diag(T) + persistence)
    return T / T.sum(axis=1, keepdims=True)


def generate_timeseries(
    n_subjects: int = 300,
    n_steps: int = 96,           # 예: 15분 간격 × 24시간
    cfg: PopulationConfig | None = None,
    seed: int = 42,
    ar_rho: float = 0.72,        # 상태 내 severity 자기상관
) -> pd.DataFrame:
    """
    개인별 연속 측정. 상태는 마르코프 전이, severity는 상태 내에서 AR(1) 변동.

    중요: 상태 경로를 먼저 확정한 뒤 그것을 generate()에 주입한다.
    생성 후에 라벨을 덮어쓰면 생체징후와 라벨이 무관해져 데이터가 망가진다.

    롤링 윈도우 통계와 변화율(dHR/dt) 피쳐를 쓰려면 이 모드가 필요하다.
    """
    cfg = cfg or PopulationConfig()
    rng = np.random.default_rng(seed)
    labels = list(cfg.state_probs.keys())
    label_vals = np.array([s.value for s in labels])
    T = _transition_matrix(labels)

    p0 = np.array([cfg.state_probs[s] for s in labels], dtype=float)
    p0 = p0 / p0.sum()

    # --- 1. 상태 경로 (마르코프) ---
    paths = np.empty((n_subjects, n_steps), dtype=int)
    paths[:, 0] = rng.choice(len(labels), size=n_subjects, p=p0)
    for t in range(1, n_steps):
        for k in range(len(labels)):
            m = paths[:, t - 1] == k
            if m.any():
                paths[m, t] = rng.choice(len(labels), size=m.sum(), p=T[k])
    states_flat = label_vals[paths.ravel()]

    # --- 2. severity: 상태 내에서 AR(1) 로 매끄럽게 변동, 상태 전환 시 재추출 ---
    target = sample_severity(states_flat, rng).reshape(n_subjects, n_steps)
    sev = np.empty_like(target)
    sev[:, 0] = target[:, 0]
    for t in range(1, n_steps):
        changed = paths[:, t] != paths[:, t - 1]
        smooth = ar_rho * sev[:, t - 1] + (1 - ar_rho) * target[:, t]
        sev[:, t] = np.where(changed, target[:, t], smooth)
    # 정상 상태는 항상 0
    sev[label_vals[paths] == State.NORMAL.value] = 0.0

    # --- 3. 개인 속성은 피험자당 1회 추출 후 시점 수만큼 복제 ---
    people_once = sample_individuals(n_subjects, cfg, rng)
    people = people_once.loc[people_once.index.repeat(n_steps)].reset_index(drop=True)

    # --- 4. 확정된 상태/severity/개인으로 물리량 생성 ---
    df = generate(
        cfg=cfg,
        seed=int(rng.integers(0, 2**31 - 1)),
        states=states_flat,
        people=people,
        severity=sev.ravel(),
    )

    df["subject_id"] = np.repeat(np.arange(n_subjects), n_steps)
    df["t"] = np.tile(np.arange(n_steps), n_subjects)

    # --- 5. 시계열 전용 파생 피쳐 ---
    g = df.groupby("subject_id", sort=False)
    df["hr_delta"] = g["heart_rate"].diff().fillna(0.0)
    df["temp_delta"] = g["body_temperature"].diff().fillna(0.0)
    df["hr_roll_mean_8"] = g["heart_rate"].transform(
        lambda s: s.rolling(8, min_periods=1).mean()
    )
    df["hr_roll_std_8"] = g["heart_rate"].transform(
        lambda s: s.rolling(8, min_periods=1).std()
    ).fillna(0.0)
    # 개인 롤링 기저선 대비 편차 — 시계열에서 가장 강력한 피쳐
    df["hr_vs_roll"] = df["heart_rate"] - df["hr_roll_mean_8"]

    return df.sort_values(["subject_id", "t"]).reset_index(drop=True)


# =============================================================================
# 8. CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="합성 생체징후 생성기")
    ap.add_argument("-n", "--n-samples", type=int, default=20_000)
    ap.add_argument("-o", "--out", type=str, default="synthetic_vitals.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeseries", action="store_true", help="시계열 모드")
    ap.add_argument("--subjects", type=int, default=300)
    ap.add_argument("--steps", type=int, default=96)
    args = ap.parse_args()

    if args.timeseries:
        df = generate_timeseries(args.subjects, args.steps, seed=args.seed)
    else:
        df = generate(args.n_samples, seed=args.seed)

    df.to_csv(args.out, index=False)
    print(f"{len(df):,} 행 → {args.out}")
    print(df["grade"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
