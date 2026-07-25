"""
게이트 컨텍스트 시나리오 — 룰과 ML이 갈리는 지점

모두 gate_hospital_outbreak.jsonl 에서 실제로 발굴한 레코드다. 손으로
만든 예시가 아니므로 "이런 경우가 있을 수 있다"가 아니라 "데이터에 N명
있었다"고 말할 수 있다. `population` 은 같은 조건을 만족한 실제 인원수다.

각 시나리오의 `_truth` 는 평가용이며 추론 경로에서는 존재하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateScenario:
    key: str
    title: str
    population: int          # 유행 시나리오 5,906명 중 같은 조건에 해당한 인원
    lesson: str
    record: dict[str, Any]

    @property
    def person(self) -> dict:
        return self.record["person"]

    @property
    def truth(self) -> dict:
        return self.record["_truth"]


def _rec(ts, vision, env, facility, health, quality, truth) -> dict:
    return {
        "timestamp": ts, "gate_id": "hospital_gate_01",
        "vision_context": vision, "environment_context": env,
        "facility_context": facility,
        "person": {"health_context": health, "measurement_quality": quality},
        "_truth": truth,
    }


_HOSPITAL = {"facility_type": "hospital", "risk_level": "HIGH", "vulnerable_population": True}


SCENARIOS: list[GateScenario] = [

    GateScenario(
        key="summer_false_positive",
        title="여름철 위양성 — 건강한데 고정 룰이 발동",
        population=976,
        lesson="주변온도 38.1℃에서 피부온이 38.1℃로 읽힌다. 고정 임계값 38℃는 "
               "발동하지만 진짜 심부온은 37.15℃로 완전한 정상이다. 적외선이 재는 "
               "것은 심부온이 아니라 표면온이며, 환경 보정 없이는 여름 내내 "
               "이런 위양성이 쏟아진다.",
        record=_rec(
            "2026-07-24T12:43:30",
            {"person_count": 7, "mask_rate": 0.66, "crowding_level": "MEDIUM"},
            {"temperature": 38.1, "humidity": 38.3, "pm25": 40.9, "air_quality": "BAD"},
            _HOSPITAL,
            {"heart_rate": None, "spo2": None, "body_temperature": 38.1, "stress_index": None},
            {"temp_valid": True, "hr_valid": False, "spo2_valid": False,
             "masked": True, "walk_speed_mps": 1.08, "ir_ambient_bias_c": 1.37,
             "n_valid_channels": 1},
            {"latent_state": "normal", "is_abnormal": 0, "true_core_temp": 37.15,
             "age": 49, "on_beta_blocker": False, "is_elderly": False}),
    ),

    GateScenario(
        key="beta_blocker_fever",
        title="베타차단제 발열자 — 아픈데 고정 룰이 놓침",
        population=5,
        lesson="체온 측정이 실패했고 심박은 베타차단제로 억제되어 110에 그친다. "
               "고정 룰은 신호 0으로 통과시킨다. 잔차 모델은 남은 두 채널의 "
               "환경 대비 편차만으로 위험을 잡아낸다. 현행 룰의 베타차단제 재현율은 "
               "1.8%로, 이 인구에서 사실상 작동하지 않는다.",
        record=_rec(
            "2026-07-24T13:11:00",
            {"person_count": 4, "mask_rate": 0.78, "crowding_level": "LOW"},
            {"temperature": 39.0, "humidity": 53.8, "pm25": 8.8, "air_quality": "GOOD"},
            _HOSPITAL,
            {"heart_rate": 110, "spo2": 95, "body_temperature": None, "stress_index": None},
            {"temp_valid": False, "hr_valid": True, "spo2_valid": True,
             "masked": True, "walk_speed_mps": 1.19, "ir_ambient_bias_c": 1.45,
             "n_valid_channels": 2},
            {"latent_state": "febrile_infection", "is_abnormal": 1, "true_core_temp": 37.97,
             "age": 56, "on_beta_blocker": True, "is_elderly": False}),
    ),

    GateScenario(
        key="elderly_blunted_fever",
        title="고령자 발열 둔화 — 아픈데 고정 룰이 놓침",
        population=16,
        lesson="79세 발열 감염자. 체온·SpO2 측정이 모두 실패하고 심박 117만 남았다. "
               "고정 룰의 심박 임계값 120에 3 모자라 통과된다. 임계값 하나 차이로 "
               "취약 인구를 놓치는 전형이며, 연속적인 위험도 점수는 이 절벽이 없다.",
        record=_rec(
            "2026-07-24T10:43:30",
            {"person_count": 6, "mask_rate": 0.59, "crowding_level": "MEDIUM"},
            {"temperature": 35.7, "humidity": 49.6, "pm25": 22.5, "air_quality": "MODERATE"},
            _HOSPITAL,
            {"heart_rate": 117, "spo2": None, "body_temperature": None, "stress_index": None},
            {"temp_valid": False, "hr_valid": True, "spo2_valid": False,
             "masked": False, "walk_speed_mps": 1.15, "ir_ambient_bias_c": 1.16,
             "n_valid_channels": 1},
            {"latent_state": "febrile_infection", "is_abnormal": 1, "true_core_temp": 38.13,
             "age": 79, "on_beta_blocker": False, "is_elderly": True}),
    ),

    GateScenario(
        key="measurement_failure",
        title="측정 대부분 실패 — 판정 자체를 보류해야 하는 경우",
        population=1008,
        lesson="유행기 아픈 사람 중 1,008명이 측정 채널 1개 이하다. 채널 0개인 "
               "구간의 판별력은 PR-AUC 0.375로 무작위(0.382)와 같다. 즉 정보가 "
               "없다. 이때 필요한 건 더 좋은 모델이 아니라 '판정 불가'를 "
               "출력하고 재측정을 요구하는 것이다.",
        record=_rec(
            "2026-07-24T11:10:30",
            {"person_count": 5, "mask_rate": 0.60, "crowding_level": "MEDIUM"},
            {"temperature": 36.3, "humidity": 58.9, "pm25": 17.5, "air_quality": "MODERATE"},
            _HOSPITAL,
            {"heart_rate": None, "spo2": None, "body_temperature": None, "stress_index": None},
            {"temp_valid": False, "hr_valid": False, "spo2_valid": False,
             "masked": True, "walk_speed_mps": 1.33, "ir_ambient_bias_c": 1.22,
             "n_valid_channels": 0},
            {"latent_state": "febrile_infection", "is_abnormal": 1, "true_core_temp": 36.77,
             "age": 49, "on_beta_blocker": False, "is_elderly": False}),
    ),

    GateScenario(
        key="clear_high_risk",
        title="명확한 고위험 — 룰과 ML이 함께 발동",
        population=110,
        lesson="체온 잔차 +2.51℃, 심박 잔차 +63.3bpm. 이런 경우엔 고정 룰도 "
               "맞는다. 모델의 가치는 이런 사례가 아니라 경계 사례에 있다는 점을 "
               "보여주는 대조군이다.",
        record=_rec(
            "2026-07-24T13:02:00",
            {"person_count": 5, "mask_rate": 0.80, "crowding_level": "MEDIUM"},
            {"temperature": 39.3, "humidity": 48.9, "pm25": 22.0, "air_quality": "MODERATE"},
            _HOSPITAL,
            {"heart_rate": 153, "spo2": None, "body_temperature": 40.3, "stress_index": None},
            {"temp_valid": True, "hr_valid": True, "spo2_valid": False,
             "masked": True, "walk_speed_mps": 1.30, "ir_ambient_bias_c": 1.47,
             "n_valid_channels": 2},
            {"latent_state": "febrile_infection", "is_abnormal": 1, "true_core_temp": 38.18,
             "age": 68, "on_beta_blocker": False, "is_elderly": False}),
    ),

    GateScenario(
        key="masked_crowded",
        title="마스크 + 혼잡 — 측정 열화와 환경 위험이 겹침",
        population=110,
        lesson="마스크가 rPPG ROI 를 가리고 혼잡이 측정 시간을 줄여 SpO2 가 실패했다. "
               "동시에 PM2.5 42.9 로 환경 축도 주의 구간이다. 개인 위험과 환경 위험은 "
               "독립적으로 판정하고 최종 제어에서 합쳐야 근거가 섞이지 않는다.",
        record=_rec(
            "2026-07-24T10:21:00",
            {"person_count": 10, "mask_rate": 0.67, "crowding_level": "HIGH"},
            {"temperature": 36.4, "humidity": 54.0, "pm25": 42.9, "air_quality": "BAD"},
            _HOSPITAL,
            {"heart_rate": 104, "spo2": None, "body_temperature": 38.8, "stress_index": None},
            {"temp_valid": True, "hr_valid": True, "spo2_valid": False,
             "masked": True, "walk_speed_mps": 0.85, "ir_ambient_bias_c": 1.22,
             "n_valid_channels": 2},
            {"latent_state": "febrile_infection", "is_abnormal": 1, "true_core_temp": 38.02,
             "age": 61, "on_beta_blocker": True, "is_elderly": False}),
    ),
]

BY_KEY = {s.key: s for s in SCENARIOS}


def legacy_rule_signals(health: dict) -> int:
    """현행 engine.py 의 _health_risk 점수. 결측은 '미발동'으로 처리된다."""
    temp = health.get("body_temperature")
    spo2 = health.get("spo2")
    hr = health.get("heart_rate")
    return (2 * (spo2 is not None and spo2 < 95)
            + 2 * (temp is not None and temp >= 38)
            + 1 * (hr is not None and hr >= 120))


def legacy_rule_level(health: dict) -> str:
    sig = legacy_rule_signals(health)
    return "HIGH" if sig >= 3 else "MEDIUM" if sig >= 1 else "LOW"
