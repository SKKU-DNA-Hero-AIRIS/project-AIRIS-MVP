"""
에어게이트 멀티모달 컨텍스트 생성기
====================================

카메라(비전) + 적외선(생체) 게이트에서 발생하는 통합 컨텍스트 레코드를 생성한다.

설계 핵심 — 측정 채널을 명시적으로 모델링한다
---------------------------------------------
"적외선으로 측정 가능하다고 가정"한다고 해서 완벽한 값을 생성하면
파이프라인이 실전에서 그대로 깨진다. 이 생성기는 3단계로 분리한다:

    1) 진짜 생리 상태      (health_synth 의 잠재상태 모델)
         ↓  환경이 생리에 실제로 영향
    2) 환경 보정된 진짜 값  (더위 → 심박↑, PM2.5 → 호흡수↑)
         ↓  측정 채널 열화
    3) 게이트가 관측한 값   (편향 + 잡음 + 실패)

_truth 필드에 1~2단계 값을 남겨두므로 측정 오차의 영향을 직접 분석할 수 있다.
배포 시에는 _truth 를 제거한다.

측정 채널의 물리적 한계 (반드시 인지할 것)
------------------------------------------
체온  : 적외선은 **피부 표면온**을 잰다. 심부온이 아니다.
        주변온도 의존성이 지배적 오차원이다. 상용 기기의 심부온 환산식은
        보통 실내 22℃ 기준으로 보정되어 있어, 32℃ 환경에서는 과대 측정한다.
        → environment_context.temperature 가 높으면 위양성이 폭증한다.

심박  : rPPG(원격 광용적맥파)로 RGB 영상에서 추정 가능하다. 단 마스크가
        볼·입 주변 ROI를 가려 사용 가능 영역이 크게 줄고, 보행 중 움직임과
        조명 변화에 취약하다.

SpO2  : ⚠️  통과형 적외선/카메라로는 사실상 측정 불가능하다.
        맥박산소측정은 적색/적외 2파장 흡광비를 광검출기로 재야 한다.
        rPPG 기반 SpO2 추정은 연구 단계이며 오차가 크다. 이 생성기는
        요청대로 값을 만들지만 오차 ±2~3%p, 실패율 높게 설정했다.
        94% 와 97% 를 구분해야 하는 용도에는 부적합하다.

stress_index : HRV(비트간 간격)가 필요하고, 이는 수 초의 통과 시간으로는
        신뢰성 있게 얻을 수 없다. 실패율을 매우 높게 두었다.
        앞선 분석대로 이 채널은 제거를 권한다.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import health_synth as hs


# =============================================================================
# 1. 시설 프로파일 — facility_context 가 모집단을 결정한다
# =============================================================================

@dataclass(frozen=True)
class FacilityProfile:
    facility_type: str
    risk_level: str
    vulnerable_population: bool
    capacity_per_frame: int          # crowding_level 판정 기준
    base_mask_rate: float
    population: dict                 # PopulationConfig 오버라이드
    state_probs: dict | None = None  # 시설별 유병률 조정


FACILITIES: dict[str, FacilityProfile] = {
    # 병원: 고령·만성질환·베타차단제 복용 비율이 높고 유병률도 높다.
    # → 앞선 검증에서 위음성이 집중된 인구가 바로 여기 몰려 있다.
    "hospital": FacilityProfile(
        facility_type="hospital", risk_level="HIGH", vulnerable_population=True,
        capacity_per_frame=12, base_mask_rate=0.72,
        population=dict(age_mean=61.0, age_sd=19.0, beta_blocker_frac=0.26,
                        chronic_frac=0.44, athlete_frac=0.02),
        state_probs={
            hs.State.NORMAL: 0.760, hs.State.POST_EXERCISE: 0.060,
            hs.State.FEBRILE_INFECTION: 0.070, hs.State.DEHYDRATION: 0.040,
            hs.State.ANXIETY_PANIC: 0.030, hs.State.HYPOXEMIA: 0.024,
            hs.State.CARDIAC_STRESS: 0.016,
        },
    ),
    # 학교: 젊고 건강. 유병률 낮음. 등교 시간대에 인원 폭증.
    "school": FacilityProfile(
        facility_type="school", risk_level="MEDIUM", vulnerable_population=True,
        capacity_per_frame=25, base_mask_rate=0.55,
        population=dict(age_mean=17.0, age_sd=3.0, age_min=13, age_max=64,
                        beta_blocker_frac=0.002, chronic_frac=0.09,
                        athlete_frac=0.16),
        state_probs={
            hs.State.NORMAL: 0.845, hs.State.POST_EXERCISE: 0.115,
            hs.State.FEBRILE_INFECTION: 0.020, hs.State.DEHYDRATION: 0.009,
            hs.State.ANXIETY_PANIC: 0.008, hs.State.HYPOXEMIA: 0.002,
            hs.State.CARDIAC_STRESS: 0.001,
        },
    ),
    "office": FacilityProfile(
        facility_type="office", risk_level="LOW", vulnerable_population=False,
        capacity_per_frame=15, base_mask_rate=0.30,
        population=dict(age_mean=39.0, age_sd=11.0, age_min=22, age_max=68,
                        beta_blocker_frac=0.07, chronic_frac=0.15),
    ),
    "cafe": FacilityProfile(
        facility_type="cafe", risk_level="LOW", vulnerable_population=False,
        capacity_per_frame=10, base_mask_rate=0.25,
        population=dict(age_mean=34.0, age_sd=13.0, age_min=18, age_max=75),
    ),
    "transit": FacilityProfile(
        facility_type="transit", risk_level="MEDIUM", vulnerable_population=False,
        capacity_per_frame=40, base_mask_rate=0.48,
        population=dict(age_mean=41.0, age_sd=17.0),
    ),
}


# =============================================================================
# 2. 환경 컨텍스트
# =============================================================================

def sample_environment(ts: datetime, rng: np.random.Generator,
                      season_bias: float = 0.0) -> dict:
    """일주기 + 계절 편향을 반영한 환경. season_bias>0 이면 여름."""
    h = ts.hour + ts.minute / 60.0
    # 기온: 새벽 최저, 15시 최고
    temp = (18.0 + season_bias
            + 7.0 * np.sin(2 * np.pi * (h - 9.0) / 24.0)
            + rng.normal(0, 1.8))
    temp = float(np.clip(temp, -8, 41))
    # 습도: 기온과 역상관
    hum = float(np.clip(72.0 - 1.1 * (temp - 20) + rng.normal(0, 9), 15, 98))
    # PM2.5: 로그정규, 출퇴근 시간 가중
    rush = 1.35 if (7 <= h <= 9 or 18 <= h <= 20) else 1.0
    pm25 = float(np.clip(np.exp(np.log(22 * rush) + rng.normal(0, 0.62)), 2, 300))

    if pm25 <= 15:
        aq = "GOOD"
    elif pm25 <= 35:
        aq = "MODERATE"
    elif pm25 <= 75:
        aq = "BAD"
    else:
        aq = "VERY_BAD"

    return {"temperature": round(temp, 1), "humidity": round(hum, 1),
            "pm25": round(pm25, 1), "air_quality": aq}


def apply_environmental_physiology(df: pd.DataFrame, env: dict,
                                   rng: np.random.Generator) -> pd.DataFrame:
    """
    환경이 '진짜 생리'에 미치는 영향. 측정 오차가 아니라 실제 변화다.

    이게 진짜 교란변수를 만든다. 32℃ 환경에서는 건강한 사람도 심박이 오르고
    피부온이 올라간다 — 발열과 구분해야 하는 대상이 하나 더 늘어나는 것이다.
    """
    out = df.copy()
    t, hum, pm = env["temperature"], env["humidity"], env["pm25"]

    # 열 스트레스: 체감지수(습도 가중)로 계산
    heat = max(0.0, t - 26.0) * (1.0 + 0.010 * max(0.0, hum - 55.0))
    # 취약 인구(고령·만성질환)가 더 민감
    vuln = 1.0 + 0.5 * out["is_elderly"].to_numpy() + 0.3 * out["chronic_condition"].to_numpy()

    out["heart_rate"] = out["heart_rate"] + heat * 1.10 * vuln + rng.normal(0, 1.2, len(out))
    out["skin_temperature"] = out["skin_temperature"] + heat * 0.075
    out["body_temperature"] = out["body_temperature"] + heat * 0.016 * vuln
    out["respiratory_rate"] = out["respiratory_rate"] + heat * 0.14

    # 한랭: 말초 혈관 수축 → 관류지수·피부온 하락
    cold = max(0.0, 14.0 - t)
    out["skin_temperature"] = out["skin_temperature"] - cold * 0.16
    out["perfusion_index"] = out["perfusion_index"] * np.exp(-0.020 * cold)

    # 대기오염: 호흡기 자극. 취약 인구에서 SpO2 소폭 하락, 호흡수 상승
    poll = max(0.0, pm - 35.0) / 50.0
    out["respiratory_rate"] = out["respiratory_rate"] + poll * 1.5 * vuln
    out["spo2"] = out["spo2"] - poll * 0.55 * vuln

    out["heart_rate"] = out["heart_rate"].clip(35, 210)
    out["spo2"] = out["spo2"].clip(70, 100)
    out["respiratory_rate"] = out["respiratory_rate"].clip(6, 50)
    return out


# =============================================================================
# 3. 측정 채널 모델 — 여기가 이 생성기의 핵심
# =============================================================================

@dataclass
class GateSensorConfig:
    """
    통과형 게이트 센서 특성. 전부 조정 가능하게 두었다 —
    센서 스펙을 바꿔가며 파이프라인 민감도를 볼 수 있어야 한다.
    """
    # --- 적외선 체온 ---
    ir_calibration_ambient: float = 22.0   # 심부온 환산식이 보정된 기준 주변온도
    ir_core_offset: float = 1.6            # 피부온 → 심부온 고정 환산 오프셋
    ir_skin_ambient_sensitivity: float = 0.085  # 피부온의 주변온도 민감도 (진단용)
    ir_noise_sd: float = 0.28              # 판독 잡음
    ir_mask_offset: float = 0.16           # 마스크 착용 시 안면 열 축적
    ir_fail_base: float = 0.04             # 판독 실패 기본 확률

    # --- rPPG 심박 ---
    hr_bias: float = -1.2
    hr_noise_sd: float = 5.5               # 정지 상태 기준
    hr_motion_penalty: float = 6.5         # 보행 속도에 따른 추가 잡음
    hr_mask_penalty: float = 3.8           # 마스크가 ROI를 가림
    hr_fail_base: float = 0.11
    hr_fail_mask: float = 0.14             # 마스크 착용 시 실패 증가

    # --- SpO2 (⚠️ 통과형으로는 원리적 한계) ---
    spo2_noise_sd: float = 2.4             # 94 vs 97 구분 불가능한 수준
    spo2_bias: float = 0.8                 # 대체로 과대 측정
    spo2_fail_base: float = 0.42

    # --- stress_index (⚠️ HRV 필요, 통과 시간으로는 불가) ---
    stress_noise_sd: float = 15.0
    stress_fail_base: float = 0.63

    # --- 혼잡 영향 ---
    crowd_fail_coef: float = 0.55          # 혼잡도 → 실패율 증가
    crowd_swap_coef: float = 0.06          # 혼잡도 → 트랙 오배정 확률


def measure(df: pd.DataFrame, env: dict, masked: np.ndarray,
            crowd_ratio: float, walk_speed: np.ndarray,
            cfg: GateSensorConfig, rng: np.random.Generator) -> pd.DataFrame:
    """진짜 생리값 → 게이트가 관측한 값. 실패 시 None."""
    n = len(df)
    out = pd.DataFrame(index=df.index)

    # 혼잡할수록 가림·짧은 체류로 실패 증가
    crowd_pen = cfg.crowd_fail_coef * min(crowd_ratio, 1.5)

    # ---- 체온: 피부온을 재고 고정 오프셋으로 심부온 환산 ----
    # 서로 다른 두 현상이 겹친다. 이중 계산이 아니다.
    #   1) 생리: 더위·추위가 진짜 피부온을 바꾼다
    #            → apply_environmental_physiology 에서 이미 반영됨
    #   2) 측정: 기기의 심부온 환산식이 ir_calibration_ambient(22℃) 기준
    #            고정이라, 주변온도가 다르면 환산 자체가 틀어진다
    # (2)를 빠뜨리면 14~26℃ 구간에서 주변온도 효과가 완전히 사라진다.
    # 생리 모델의 열/한랭 항이 그 구간에서 0이기 때문이다.
    ambient_bias = cfg.ir_skin_ambient_sensitivity * (
        env["temperature"] - cfg.ir_calibration_ambient)
    skin_true = df["skin_temperature"].to_numpy()
    temp_meas = (
        skin_true
        + cfg.ir_core_offset                      # 22℃ 기준으로 보정된 고정 오프셋
        + ambient_bias                            # 환산식 기준과의 불일치
        + cfg.ir_mask_offset * masked
        + rng.normal(0, cfg.ir_noise_sd, n)
    )
    temp_fail = rng.random(n) < (cfg.ir_fail_base + 0.5 * crowd_pen)
    out["body_temperature"] = np.where(temp_fail, np.nan, np.round(temp_meas, 1))

    # ---- 심박: rPPG. 보행 속도와 마스크에 취약 ----
    hr_sd = (cfg.hr_noise_sd
             + cfg.hr_motion_penalty * walk_speed
             + cfg.hr_mask_penalty * masked)
    hr_meas = df["heart_rate"].to_numpy() + cfg.hr_bias + rng.normal(0, 1, n) * hr_sd
    hr_fail = rng.random(n) < (cfg.hr_fail_base + cfg.hr_fail_mask * masked + crowd_pen)
    out["heart_rate"] = np.where(hr_fail, np.nan, np.round(hr_meas))

    # ---- SpO2: 원리적 한계. 오차가 임상 판단 폭보다 크다 ----
    spo2_meas = (df["spo2"].to_numpy() + cfg.spo2_bias
                 + rng.normal(0, cfg.spo2_noise_sd, n))
    spo2_fail = rng.random(n) < (cfg.spo2_fail_base + crowd_pen)
    out["spo2"] = np.where(spo2_fail, np.nan, np.round(np.clip(spo2_meas, 70, 100)))

    # ---- stress_index: HRV 필요. 대부분 실패 ----
    st_meas = df["stress_index"].to_numpy() + rng.normal(0, cfg.stress_noise_sd, n)
    st_fail = rng.random(n) < (cfg.stress_fail_base + 0.7 * masked * 0.3 + crowd_pen)
    out["stress_index"] = np.where(st_fail, np.nan, np.round(np.clip(st_meas, 0, 100)))

    # ---- 품질 메타데이터: 파이프라인이 반드시 다뤄야 하는 정보 ----
    out["_quality"] = [
        {
            "temp_valid": bool(not tf), "hr_valid": bool(not hf),
            "spo2_valid": bool(not sf), "stress_valid": bool(not stf),
            "masked": bool(m), "walk_speed_mps": round(float(w), 2),
            "ir_ambient_bias_c": round(float(ambient_bias), 2),
            "n_valid_channels": int(4 - tf - hf - sf - stf),
        }
        for tf, hf, sf, stf, m, w in zip(
            temp_fail, hr_fail, spo2_fail, st_fail, masked, walk_speed)
    ]

    # ---- 혼잡 시 트랙 오배정: 생체값이 옆 사람에게 붙는다 ----
    swap_p = cfg.crowd_swap_coef * min(crowd_ratio, 1.5)
    if n >= 2 and swap_p > 0:
        for i in range(n - 1):
            if rng.random() < swap_p:
                cols = ["body_temperature", "heart_rate", "spo2", "stress_index"]
                out.loc[out.index[[i, i + 1]], cols] = \
                    out.loc[out.index[[i + 1, i]], cols].to_numpy()
    return out


# =============================================================================
# 4. 게이트 레코드 생성
# =============================================================================

def _crowding_level(count: int, capacity: int) -> str:
    r = count / capacity
    if r < 0.35:
        return "LOW"
    if r < 0.70:
        return "MEDIUM"
    return "HIGH"


def _arrival_intensity(facility: str, ts: datetime) -> float:
    """시설별 시간대 도착 강도 (0~1)."""
    h = ts.hour + ts.minute / 60.0
    if facility == "school":
        return float(np.clip(
            1.25 * np.exp(-((h - 8.1) ** 2) / 0.5)
            + 0.95 * np.exp(-((h - 16.3) ** 2) / 1.1) + 0.04, 0.02, 1.0))
    if facility == "hospital":
        return float(np.clip(
            0.85 * np.exp(-((h - 10.0) ** 2) / 5.0)
            + 0.55 * np.exp(-((h - 14.5) ** 2) / 4.0) + 0.10, 0.05, 1.0))
    if facility == "office":
        return float(np.clip(
            1.15 * np.exp(-((h - 8.7) ** 2) / 0.7)
            + 0.85 * np.exp(-((h - 18.4) ** 2) / 1.0)
            + 0.30 * np.exp(-((h - 12.2) ** 2) / 0.6) + 0.03, 0.02, 1.0))
    if facility == "transit":
        return float(np.clip(
            1.0 * np.exp(-((h - 8.0) ** 2) / 1.2)
            + 1.0 * np.exp(-((h - 18.7) ** 2) / 1.5) + 0.12, 0.03, 1.0))
    return float(np.clip(0.45 + 0.4 * np.sin(2 * np.pi * (h - 10) / 24), 0.05, 1.0))


def generate_gate_records(
    facility_type: str = "hospital",
    start: datetime | None = None,
    n_records: int = 200,
    interval_sec: int = 60,
    sensor: GateSensorConfig | None = None,
    seed: int = 42,
    season_bias: float = 0.0,
    outbreak_multiplier: float = 1.0,     # 유행 시나리오: 감염 상태 확률 배수
    include_truth: bool = True,
) -> list[dict]:
    """에어게이트 통합 컨텍스트 레코드 목록을 생성."""
    prof = FACILITIES[facility_type]
    sensor = sensor or GateSensorConfig()
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 7, 23, 7, 0, 0)

    # 시설 인구 + 유병률 설정
    probs = dict(prof.state_probs or hs.DEFAULT_STATE_PROBS)
    if outbreak_multiplier != 1.0:
        infl = probs[hs.State.FEBRILE_INFECTION] * outbreak_multiplier
        delta = infl - probs[hs.State.FEBRILE_INFECTION]
        probs[hs.State.FEBRILE_INFECTION] = infl
        probs[hs.State.NORMAL] = max(0.01, probs[hs.State.NORMAL] - delta)
    cfg_pop = hs.PopulationConfig(**prof.population, state_probs=probs)

    records = []
    for k in range(n_records):
        ts = start + timedelta(seconds=k * interval_sec)
        env = sample_environment(ts, rng, season_bias)

        # --- 도착 인원 (포아송) ---
        lam = 0.85 + prof.capacity_per_frame * 0.55 * _arrival_intensity(facility_type, ts)
        n_true = int(rng.poisson(lam))
        if n_true == 0:
            continue

        crowd_ratio = n_true / prof.capacity_per_frame
        crowding = _crowding_level(n_true, prof.capacity_per_frame)

        # --- 마스크 (혼잡·위험도에 따라 상승) ---
        mrate = float(np.clip(
            prof.base_mask_rate + 0.10 * (crowding == "HIGH")
            + rng.normal(0, 0.09), 0.0, 1.0))
        masked = rng.random(n_true) < mrate

        # --- 보행 속도: 혼잡하면 느려짐 (rPPG 에 유리) ---
        walk = np.clip(rng.normal(1.30 - 0.35 * min(crowd_ratio, 1.4), 0.22, n_true), 0.15, 2.2)

        # --- 1) 진짜 생리 상태 ---
        truth = hs.generate(n=n_true, cfg=cfg_pop, seed=int(rng.integers(0, 2**31 - 1)))
        # --- 2) 환경이 진짜 생리에 미치는 영향 ---
        truth = apply_environmental_physiology(truth, env, rng)
        # --- 3) 게이트 측정 ---
        meas = measure(truth, env, masked, crowd_ratio, walk, sensor, rng)

        # --- 비전 계수 자체의 오차 (혼잡할수록 과소 계수) ---
        count_err = rng.normal(-0.045 * n_true * min(crowd_ratio, 1.5),
                               0.30 + 0.11 * n_true * min(crowd_ratio, 1.5))
        person_count = int(max(1, round(n_true + count_err)))
        count_conf = float(np.clip(0.97 - 0.20 * min(crowd_ratio, 1.5)
                                   - 0.05 * mrate + rng.normal(0, 0.03), 0.35, 0.995))

        # --- 개인별 레코드 ---
        persons = []
        for i in range(n_true):
            hc = {
                "heart_rate": None if pd.isna(meas["heart_rate"].iloc[i])
                else int(meas["heart_rate"].iloc[i]),
                "spo2": None if pd.isna(meas["spo2"].iloc[i])
                else int(meas["spo2"].iloc[i]),
                "body_temperature": None if pd.isna(meas["body_temperature"].iloc[i])
                else float(meas["body_temperature"].iloc[i]),
                "stress_index": None if pd.isna(meas["stress_index"].iloc[i])
                else int(meas["stress_index"].iloc[i]),
            }
            p = {
                "person_id": f"trk_{k:05d}_{i:02d}",
                "health_context": hc,
                "measurement_quality": meas["_quality"].iloc[i],
            }
            if include_truth:
                tr = truth.iloc[i]
                p["_truth"] = {
                    "latent_state": tr["latent_state"],
                    "grade": int(tr["grade"]),
                    "is_abnormal": int(tr["is_abnormal"]),
                    "true_core_temp": round(float(tr["body_temperature"]), 2),
                    "true_heart_rate": int(round(tr["heart_rate"])),
                    "true_spo2": int(round(tr["spo2"])),
                    "true_respiratory_rate": int(round(tr["respiratory_rate"])),
                    "age": int(tr["age"]),
                    "on_beta_blocker": bool(tr["on_beta_blocker"]),
                    "is_elderly": bool(tr["is_elderly"]),
                }
            persons.append(p)

        # --- 집계 health_context (원본 스키마 호환: 중앙값) ---
        def med(col):
            v = meas[col].dropna()
            return None if len(v) == 0 else (
                round(float(v.median()), 1) if col == "body_temperature"
                else int(round(v.median())))

        agg_health = {"heart_rate": med("heart_rate"), "spo2": med("spo2"),
                      "body_temperature": med("body_temperature"),
                      "stress_index": med("stress_index")}

        # --- 집단 신호: 개인 판정 대신 이쪽이 실제로 작동하는 지표 ---
        tv = meas["body_temperature"].dropna()
        hv = meas["heart_rate"].dropna()
        aggregate = {
            "measured_persons": int(len(persons)),
            "temp_valid_n": int(len(tv)),
            "hr_valid_n": int(len(hv)),
            "febrile_count_measured": int((tv >= 37.5).sum()),
            # 프레임 단위 비율은 표본이 작아 의미가 없다. 시간창 집계용 원자료로만 쓴다.
            "febrile_rate_measured": None if len(tv) < 5
            else round(float((tv >= 37.5).mean()), 4),
            "tachycardia_rate_measured": None if len(hv) < 5
            else round(float((hv >= 100).mean()), 4),
            "temp_median": None if len(tv) == 0 else round(float(tv.median()), 2),
            "temp_p95": None if len(tv) == 0 else round(float(tv.quantile(0.95)), 2),
        }
        if include_truth:
            aggregate["_truth_abnormal_count"] = int(truth["is_abnormal"].sum())
            aggregate["_truth_febrile_count"] = int(
                (truth["latent_state"] == "febrile_infection").sum())

        rec = {
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "gate_id": f"{facility_type}_gate_01",
            "vision_context": {
                "person_count": person_count,
                "mask_rate": round(mrate, 2),
                "crowding_level": crowding,
                "count_confidence": round(count_conf, 3),
                **({"_true_person_count": n_true} if include_truth else {}),
            },
            "health_context": agg_health,
            "environment_context": env,
            "facility_context": {
                "facility_type": prof.facility_type,
                "risk_level": prof.risk_level,
                "vulnerable_population": prof.vulnerable_population,
            },
            "persons": persons,
            "aggregate_signal": aggregate,
        }
        records.append(rec)
    return records


# =============================================================================
# 5. 분석 헬퍼: 측정 오차의 영향을 정량화
# =============================================================================

def measurement_impact(records: list[dict]) -> pd.DataFrame:
    """개인 레코드를 평탄화해 측정 오차 영향을 분석 가능한 표로."""
    rows = []
    for r in records:
        for p in r["persons"]:
            if "_truth" not in p:
                continue
            hc, q, t = p["health_context"], p["measurement_quality"], p["_truth"]
            rows.append({
                "timestamp": r["timestamp"],
                "ambient_temp": r["environment_context"]["temperature"],
                "pm25": r["environment_context"]["pm25"],
                "crowding": r["vision_context"]["crowding_level"],
                "masked": q["masked"],
                "ir_bias": q["ir_ambient_bias_c"],
                "n_valid": q["n_valid_channels"],
                "meas_temp": hc["body_temperature"],
                "true_temp": t["true_core_temp"],
                "meas_hr": hc["heart_rate"],
                "true_hr": t["true_heart_rate"],
                "meas_spo2": hc["spo2"],
                "true_spo2": t["true_spo2"],
                "latent_state": t["latent_state"],
                "is_abnormal": t["is_abnormal"],
                "on_beta_blocker": t["on_beta_blocker"],
                "is_elderly": t["is_elderly"],
            })
    return pd.DataFrame(rows)


def rollup(records: list[dict], window_min: int = 60,
           febrile_threshold: float = 37.5,
           min_n: int = 30) -> pd.DataFrame:
    """
    시간창 집계 — 집단 신호가 실제로 존재하는 층.

    프레임 단위로는 표본이 5~15명이라 비율 추정이 무의미하다.
    앞선 검정력 분석에서 N=300~1000이 필요했으므로 시간 단위로 묶는다.
    Wilson 신뢰구간을 함께 제공해 표본 크기가 결론에 반영되게 한다.
    """
    rows = []
    for r in records:
        ts = pd.Timestamp(r["timestamp"])
        for p in r["persons"]:
            t = p["health_context"]["body_temperature"]
            h = p["health_context"]["heart_rate"]
            rows.append({
                "ts": ts, "temp": t, "hr": h,
                "ambient": r["environment_context"]["temperature"],
                "counted": r["vision_context"]["person_count"],
                "truth_febrile": p.get("_truth", {}).get("latent_state") == "febrile_infection",
                "truth_abnormal": p.get("_truth", {}).get("is_abnormal", np.nan),
            })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["window"] = df["ts"].dt.floor(f"{window_min}min")

    out = []
    for w, g in df.groupby("window"):
        tv = g["temp"].dropna()
        hv = g["hr"].dropna()
        n = len(tv)
        k = int((tv >= febrile_threshold).sum()) if n else 0
        # Wilson 95% 구간
        if n >= min_n:
            phat, z = k / n, 1.96
            denom = 1 + z**2 / n
            centre = (phat + z**2 / (2 * n)) / denom
            half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
            lo, hi = max(0.0, centre - half), min(1.0, centre + half)
        else:
            phat = lo = hi = np.nan

        out.append({
            "window": w,
            "persons_passed": len(g),
            "temp_valid_n": n,
            "temp_coverage": round(n / len(g), 3),
            "ambient_mean": round(float(g["ambient"].mean()), 1),
            "febrile_flagged": k,
            "febrile_rate": None if np.isnan(phat) else round(phat, 4),
            "ci95_low": None if np.isnan(lo) else round(lo, 4),
            "ci95_high": None if np.isnan(hi) else round(hi, 4),
            "temp_median": None if n == 0 else round(float(tv.median()), 2),
            "hr_median": None if len(hv) == 0 else round(float(hv.median()), 1),
            "truth_febrile_rate": round(float(g["truth_febrile"].mean()), 4),
            "truth_abnormal_rate": round(float(g["truth_abnormal"].mean()), 4),
        })
    return pd.DataFrame(out).sort_values("window").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="에어게이트 컨텍스트 생성기")
    ap.add_argument("-f", "--facility", default="hospital", choices=list(FACILITIES))
    ap.add_argument("-n", "--n-records", type=int, default=300)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--season-bias", type=float, default=0.0,
                    help="여름 시나리오는 +14 정도")
    ap.add_argument("--outbreak", type=float, default=1.0,
                    help="감염 유행 배수 (예: 4.0)")
    ap.add_argument("-o", "--out", default="gate_records.jsonl")
    ap.add_argument("--flat-csv", default=None,
                    help="개인 단위로 평탄화한 CSV 도 함께 저장 "
                         "(예: gate_persons_flat.csv). 측정 오차 분석용.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    recs = generate_gate_records(
        facility_type=args.facility, n_records=args.n_records,
        interval_sec=args.interval, season_bias=args.season_bias,
        outbreak_multiplier=args.outbreak, seed=args.seed,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_persons = sum(len(r["persons"]) for r in recs)
    print(f"{len(recs):,} 레코드 / {n_persons:,} 명 → {args.out}")

    if args.flat_csv:
        flat = measurement_impact(recs)
        flat.to_csv(args.flat_csv, index=False)
        print(f"{len(flat):,} 행 → {args.flat_csv}")


if __name__ == "__main__":
    main()
