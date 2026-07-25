"""
위험도 추정 모델 학습 — 평시 학습 → 유행 테스트
================================================

절제실험 결과 (docs/risk-strategy.md 참조)로 확정된 설계:

  · 특징은 '환경 잔차화된 생체신호'만 쓴다.
  · 시설·혼잡도·절대환경값은 위험 모델에 넣지 않는다.
    학습셋에서 시나리오와 유병률이 얽혀 있어 모델이 생리 대신
    '시나리오 정체'를 학습하는 지름길이 생긴다 (PR-AUC 0.619 → 0.594).
    맥락은 2단계 정책(policy.py)에서 비용으로 반영한다.

  · 임계값은 확률이 아니라 '평시 점수 분포의 백분위'로 잡는다.
    유병률이 평시와 유행기에 2~3배 달라지므로 절대 확률 임계값은 이동한다.
    백분위 임계값은 "통행자의 상위 N%를 2차 검사로 보낸다"는 운영 용량과
    직접 대응되고 유병률 변화에 강건하다.

출력: risk_model.json — airis/risk_model.py 가 읽는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

DATA = Path(__file__).parent
PATHOLOGICAL = {"febrile_infection", "hypoxemia", "cardiac_stress",
                "dehydration", "anxiety_panic"}
CROWDING_ORD = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

ENV_REGRESSORS = ["ambient_temp", "humidity", "pm25"]
RESIDUALIZE = {"meas_temp": "temp_resid", "meas_hr": "hr_resid", "meas_spo2": "spo2_resid"}
FEATURES = ["temp_resid", "hr_resid", "spo2_resid"]

TRAIN_FILES = ["gate_school_spring.jsonl", "gate_hospital_summer.jsonl"]
TEST_FILE = "gate_hospital_outbreak.jsonl"


# =============================================================================
# 1. 로딩
# =============================================================================

def load_persons(*paths: Path) -> pd.DataFrame:
    """레코드 단위 맥락을 개인 행에 붙여 평면화한다."""
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                env, fac, vis = (rec["environment_context"], rec["facility_context"],
                                 rec["vision_context"])
                for p in rec["persons"]:
                    h, q, t = p["health_context"], p["measurement_quality"], p["_truth"]
                    rows.append({
                        "timestamp": rec["timestamp"], "scenario": path.stem,
                        "meas_temp": h["body_temperature"], "meas_hr": h["heart_rate"],
                        "meas_spo2": h["spo2"],
                        "masked": int(q["masked"]), "n_valid": q["n_valid_channels"],
                        "ambient_temp": env["temperature"], "humidity": env["humidity"],
                        "pm25": env["pm25"],
                        "facility_type": fac["facility_type"],
                        "crowding": CROWDING_ORD[vis["crowding_level"]],
                        # ---- 라벨·층화용. 추론 경로에서는 존재하지 않는다 ----
                        "y": int(t["latent_state"] in PATHOLOGICAL),
                        "true_temp": t["true_core_temp"], "true_hr": t["true_heart_rate"],
                        "true_spo2": t["true_spo2"], "true_rr": t["true_respiratory_rate"],
                        "on_beta_blocker": bool(t["on_beta_blocker"]),
                        "is_elderly": bool(t["is_elderly"]),
                    })
    return pd.DataFrame(rows)


# =============================================================================
# 2. 환경 잔차화 — 평시 '정상군'에서만 학습
# =============================================================================

def fit_env_calibration(train: pd.DataFrame) -> dict:
    """
    각 생체신호를 관측 가능한 환경변수에 선형 회귀한다.
    병리 표본이 섞이면 기준선이 위로 끌려가 임계값이 오염된다.
    """
    cal = {}
    base = train[train.y == 0]
    for col in RESIDUALIZE:
        d = base[[col] + ENV_REGRESSORS].dropna()
        X = np.c_[np.ones(len(d)), d[ENV_REGRESSORS].to_numpy()]
        coef, *_ = np.linalg.lstsq(X, d[col].to_numpy(), rcond=None)
        cal[col] = {"intercept": float(coef[0]),
                    "slopes": dict(zip(ENV_REGRESSORS, map(float, coef[1:]))),
                    "n_train": int(len(d))}
    return cal


def apply_env_calibration(df: pd.DataFrame, cal: dict) -> pd.DataFrame:
    out = df.copy()
    for col, resid_col in RESIDUALIZE.items():
        c = cal[col]
        pred = c["intercept"] + sum(c["slopes"][r] * out[r] for r in ENV_REGRESSORS)
        out[resid_col] = out[col] - pred
    return out


# =============================================================================
# 3. 학습
# =============================================================================

def build_matrix(df: pd.DataFrame, cols: list[str], impute: dict | None) -> tuple[pd.DataFrame, dict]:
    """결측은 평시 중앙값으로 대체하고, 결측 여부 자체를 특징으로 남긴다."""
    X = df[cols].copy()
    if impute is None:
        impute = {c: float(X[c].median()) for c in cols}
    for c, med in impute.items():
        X[c + "_missing"] = X[c].isna().astype(int)
        X[c] = X[c].fillna(med)
    return X.astype(float), impute


def fit_model(train: pd.DataFrame, cols: list[str]):
    X, impute = build_matrix(train, cols, None)
    mu, sd = X.mean(), X.std().replace(0, 1)
    model = LogisticRegression(max_iter=2000)
    model.fit((X - mu) / sd, train.y)
    return model, list(X.columns), impute, mu, sd


def score(df: pd.DataFrame, model, cols, feat_cols, impute, mu, sd) -> np.ndarray:
    X, _ = build_matrix(df, cols, impute)
    return model.predict_proba((X[feat_cols] - mu) / sd)[:, 1]


def rule_baseline(df: pd.DataFrame) -> np.ndarray:
    """현행 engine.py 의 _health_risk. stress_index 는 측정 채널이 없어 항상 미발동."""
    return (2 * (df.meas_spo2.fillna(99) < 95).to_numpy()
            + 2 * (df.meas_temp.fillna(36.5) >= 38).to_numpy()
            + 1 * (df.meas_hr.fillna(75) >= 120).to_numpy())


def summarize(y: np.ndarray, pred: np.ndarray, strat: dict | None = None) -> str:
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = (f"재현율 {rec:6.1%} | 위양성률 {fp / (fp + tn) if fp + tn else 0:6.1%} "
           f"| 정밀도 {tp / (tp + fp) if tp + fp else 0:6.1%}")
    if strat:
        for name, mask in strat.items():
            m = mask & (y == 1)
            out += f"\n      └ {name:10s} n={int(m.sum()):4d} 재현율 {(pred & m).sum() / max(m.sum(), 1):6.1%}"
    return out


def main() -> None:
    train = load_persons(*[DATA / f for f in TRAIN_FILES])
    test = load_persons(DATA / TEST_FILE)
    cal = fit_env_calibration(train)
    train, test = apply_env_calibration(train, cal), apply_env_calibration(test, cal)
    y = test.y.to_numpy()
    strat = {"베타차단제": test.on_beta_blocker.to_numpy(),
             "고령자": test.is_elderly.to_numpy(),
             "일반": (~test.on_beta_blocker & ~test.is_elderly).to_numpy()}

    print(f"학습 {len(train):,}명 (병리 {train.y.mean():.1%}) "
          f"→ 테스트 {len(test):,}명 (병리 {test.y.mean():.1%})")
    print(f"무작위 기준선 PR-AUC = {test.y.mean():.3f}\n" + "=" * 74)

    # ---- 3중 비교 + 성능 상한 ----
    print("A. 현행 고정 룰")
    print("   HIGH(>=3)     ", summarize(y, rule_baseline(test) >= 3, strat))
    print("   MEDIUM+(>=1)  ", summarize(y, rule_baseline(test) >= 1))

    arms = {"B  측정 원값": ["meas_temp", "meas_hr", "meas_spo2"],
            "C  측정 잔차 (채택)": FEATURES,
            "★ 진짜값 (측정 완벽 가정)": ["true_temp", "true_hr", "true_spo2"],
            "★ 진짜값 + 호흡수": ["true_temp", "true_hr", "true_spo2", "true_rr"]}
    print()
    for name, cols in arms.items():
        m, fc, imp, mu, sd = fit_model(train, cols)
        p = score(test, m, cols, fc, imp, mu, sd)
        print(f"{name:26s} PR-AUC {average_precision_score(y, p):.3f}")

    # ---- 채택 모델 재학습 및 임계값 산출 ----
    model, feat_cols, impute, mu, sd = fit_model(train, FEATURES)
    p_train = score(train, model, FEATURES, feat_cols, impute, mu, sd)
    p_test = score(test, model, FEATURES, feat_cols, impute, mu, sd)

    print("\n" + "=" * 74)
    print("학습된 가중치 (표준화 계수 — 현행 룰의 +2/+2/+1 을 대체한다):")
    for c, w in sorted(zip(feat_cols, model.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {c:22s} {w:+.3f}")

    # 임계값 = 평시 점수 분포의 백분위. 유병률 변화에 강건하다.
    quantiles = {"p90": 0.90, "p95": 0.95, "p98": 0.98}
    taus = {k: float(np.quantile(p_train, q)) for k, q in quantiles.items()}
    print("\n평시 분포 백분위 임계값 → 유행기 실제 성능:")
    for k, tau in taus.items():
        print(f"  {k} (τ={tau:.3f})  {summarize(y, p_test >= tau)}")
    print("\n  상위 p90 임계값의 층화 성능:")
    print("     ", summarize(y, p_test >= taus["p90"], strat))

    out = {
        "features": feat_cols,
        "coef": {c: float(w) for c, w in zip(feat_cols, model.coef_[0])},
        "intercept": float(model.intercept_[0]),
        "standardize": {"mean": {k: float(v) for k, v in mu.items()},
                        "std": {k: float(v) for k, v in sd.items()}},
        "impute": impute,
        "env_calibration": cal,
        "thresholds": taus,
        "metrics": {"pr_auc_test": float(average_precision_score(y, p_test)),
                    "train_prevalence": float(train.y.mean()),
                    "test_prevalence": float(test.y.mean())},
        "trained_on": TRAIN_FILES, "tested_on": TEST_FILE,
    }
    (DATA / "risk_model.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\n→ {(DATA / 'risk_model.json').name} 저장 완료")


if __name__ == "__main__":
    main()
