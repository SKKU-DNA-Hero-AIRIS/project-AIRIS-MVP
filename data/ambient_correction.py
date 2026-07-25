"""
주변온도 보정 — 에어게이트 체온 신호의 필수 전처리
=================================================

실측 결과: 고정 임계값 37.5℃ 방식은 작동하지 않는다.
주변온도 일주기 드리프트(07시 0.37 → 14시 0.77)가 유행 신호보다 크기 때문이다.

  고정 임계값 37.5℃  : 평시 0.619 → 유행 0.691   배수 1.12   → 탐지 불가
  주변온도 잔차화     : 평시 0.050 → 유행 0.130   배수 2.60   → 전 시간대 유의

배포 시에는 평시 캘리브레이션 기간(2~4주)에 회귀계수를 학습하고,
계절이 바뀌면 재학습해야 한다.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class AmbientCalibration:
    """평시 데이터로 학습한 주변온도 → 판독체온 회귀."""
    intercept: float
    slope: float
    resid_threshold: float      # 평시 잔차 분포의 상위 분위
    r_value: float
    n_train: int
    quantile: float

    def residual(self, meas_temp, ambient_temp):
        return np.asarray(meas_temp) - (self.intercept + self.slope * np.asarray(ambient_temp))

    def flag(self, meas_temp, ambient_temp):
        return self.residual(meas_temp, ambient_temp) > self.resid_threshold

    def summary(self) -> str:
        return (f"측정체온 = {self.intercept:.2f} + {self.slope:.4f}×주변온도 "
                f"(r={self.r_value:.3f}, n={self.n_train:,})\n"
                f"잔차 임계값({self.quantile:.0%} 분위) = {self.resid_threshold:+.2f}℃\n"
                f"주변온도 15℃ 변동 시 판독 드리프트 = {self.slope*15:.2f}℃")


def fit_ambient_calibration(df: pd.DataFrame, quantile: float = 0.95,
                            temp_col: str = "meas_temp",
                            ambient_col: str = "ambient_temp") -> AmbientCalibration:
    """평시 데이터로 캘리브레이션. 유행기 데이터를 넣으면 임계값이 오염된다."""
    d = df[[temp_col, ambient_col]].dropna()
    if len(d) < 200:
        raise ValueError(f"캘리브레이션 표본 부족: {len(d)} < 200")
    sl, ic, r, _, _ = stats.linregress(d[ambient_col], d[temp_col])
    resid = d[temp_col] - (ic + sl * d[ambient_col])
    return AmbientCalibration(intercept=float(ic), slope=float(sl),
                              resid_threshold=float(resid.quantile(quantile)),
                              r_value=float(r), n_train=len(d), quantile=quantile)


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> float:
    """평시 대비 현재 창의 이상률 상승 유의성 (단측)."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    p = (x1 + x2) / (n1 + n2)
    denom = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return float((x2 / n2 - x1 / n1) / denom) if denom > 0 else float("nan")


def monitor(df: pd.DataFrame, cal: AmbientCalibration, baseline_rate: float,
            baseline_n: int, window_min: int = 60, min_n: int = 30) -> pd.DataFrame:
    """시간창별 보정 이상률 + 평시 대비 유의성."""
    d = df.dropna(subset=["meas_temp"]).copy()
    d["flag"] = cal.flag(d["meas_temp"], d["ambient_temp"])
    d["window"] = pd.to_datetime(d["timestamp"]).dt.floor(f"{window_min}min")

    rows = []
    for w, g in d.groupby("window"):
        n, k = len(g), int(g["flag"].sum())
        z = two_proportion_z(int(baseline_rate * baseline_n), baseline_n, k, n)
        rows.append({
            "window": w, "n": n, "flagged": k,
            "rate": round(k / n, 4) if n else None,
            "ambient_mean": round(float(g["ambient_temp"].mean()), 1),
            "z_vs_baseline": None if np.isnan(z) else round(z, 2),
            "alert": bool(n >= min_n and z > 2.33),
            "truth_febrile_rate": round(float(
                (g["latent_state"] == "febrile_infection").mean()), 4)
            if "latent_state" in g else None,
        })
    return pd.DataFrame(rows).sort_values("window").reset_index(drop=True)
