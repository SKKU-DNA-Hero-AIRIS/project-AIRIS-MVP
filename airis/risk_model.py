"""
1단계: 위험도 추정 — 학습된 모델로 p(병리상태)를 낸다.

맥락(시설·혼잡도·절대환경값)은 여기 들어오지 않는다. 학습셋에서 시나리오와
유병률이 얽혀 모델이 생리 대신 '시나리오 정체'를 학습하는 지름길이 생기기
때문이다 (절제실험: PR-AUC 0.629 → 0.594). 환경은 '잔차화'로만,
즉 효과를 제거하는 방향으로만 쓴다. 맥락의 의사결정 반영은 policy.py 담당.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "risk_model.json"

# 게이트에서 실제로 측정되는 채널만 쓴다. stress_index 는 HRV 가 필요해
# 수 초의 통과 시간으로는 얻을 수 없고, 생성 데이터에서도 전량 결측이다.
_SOURCE = {"temp_resid": "body_temperature", "hr_resid": "heart_rate", "spo2_resid": "spo2"}
_CAL_KEY = {"temp_resid": "meas_temp", "hr_resid": "meas_hr", "spo2_resid": "meas_spo2"}


@dataclass(frozen=True)
class RiskEstimate:
    probability: float
    confidence: float                  # 측정 성공 채널 비율 — 위험도와 독립
    n_valid_channels: int
    residuals: dict[str, float | None]
    contributions: dict[str, float]    # 계수×표준화값 — 설명 근거

    @property
    def measurable(self) -> bool:
        return self.n_valid_channels > 0

    def top_reasons(self, k: int = 3) -> list[tuple[str, float]]:
        items = [(n, v) for n, v in self.contributions.items() if abs(v) > 1e-9]
        return sorted(items, key=lambda x: -abs(x[1]))[:k]


class RiskModel:
    """risk_model.json 을 읽어 추론한다. 학습은 data/train_risk_model.py."""

    def __init__(self, spec: dict):
        self.features: list[str] = spec["features"]
        self.coef: dict[str, float] = spec["coef"]
        self.intercept: float = spec["intercept"]
        self.mean: dict[str, float] = spec["standardize"]["mean"]
        self.std: dict[str, float] = spec["standardize"]["std"]
        self.impute: dict[str, float] = spec["impute"]
        self.env_cal: dict = spec["env_calibration"]
        self.thresholds: dict[str, float] = spec["thresholds"]
        self.metrics: dict = spec.get("metrics", {})

    @classmethod
    def load(cls, path: Path | str = MODEL_PATH) -> "RiskModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- 환경 잔차화 -------------------------------------------------------
    def _residual(self, resid_name: str, measured: float | None, env: dict) -> float | None:
        """측정값에서 '환경이 설명하는 몫'을 뺀다. 이게 임계값의 진짜 자리다."""
        if measured is None:
            return None
        cal = self.env_cal[_CAL_KEY[resid_name]]
        expected = cal["intercept"] + (
            cal["slopes"]["ambient_temp"] * env["temperature"]
            + cal["slopes"]["humidity"] * env["humidity"]
            + cal["slopes"]["pm25"] * env["pm25"])
        return measured - expected

    def estimate(self, health: dict, environment: dict) -> RiskEstimate:
        residuals = {name: self._residual(name, health.get(src), environment)
                     for name, src in _SOURCE.items()}
        n_valid = sum(v is not None for v in residuals.values())

        values: dict[str, float] = {}
        for name, resid in residuals.items():
            values[name] = self.impute[name] if resid is None else resid
            values[name + "_missing"] = 1.0 if resid is None else 0.0

        contributions, z = {}, self.intercept
        for feat in self.features:
            std = self.std.get(feat, 1.0) or 1.0
            scaled = (values[feat] - self.mean.get(feat, 0.0)) / std
            term = self.coef[feat] * scaled
            contributions[feat] = term
            z += term

        return RiskEstimate(
            probability=1.0 / (1.0 + math.exp(-z)),
            confidence=n_valid / len(_SOURCE),
            n_valid_channels=n_valid,
            residuals=residuals,
            contributions=contributions,
        )
