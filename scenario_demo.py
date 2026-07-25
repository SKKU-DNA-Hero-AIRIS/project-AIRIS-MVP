"""
시나리오 데모 — 현행 고정 룰 vs 2단 구조(ML 위험도 + 룰 정책)

    python scenario_demo.py            전체 시나리오 비교
    python scenario_demo.py --facility 같은 사람을 시설별로 다르게 판정하는 데모
"""
from __future__ import annotations

import argparse

from airis.gate_scenarios import SCENARIOS, legacy_rule_level, legacy_rule_signals
from airis.policy import POLICIES, decide
from airis.risk_model import RiskModel

VERDICT = {True: "정답", False: "오판"}


def _correct(level: str, is_abnormal: int) -> bool:
    """병리 상태면 LOW 가 아니어야 하고, 정상이면 LOW 여야 한다."""
    return (level != "LOW") == bool(is_abnormal)


def compare_all(model: RiskModel) -> None:
    for s in SCENARIOS:
        health = s.person["health_context"]
        risk = model.estimate(health, s.record["environment_context"])
        d = decide(risk, model.thresholds, s.record["facility_context"],
                   s.record["environment_context"], s.record["vision_context"])
        rule = legacy_rule_level(health)
        truth = s.truth

        print("\n" + "=" * 78)
        print(f"[{s.key}] {s.title}   (해당 인원 {s.population:,}명)")
        print(f"  진실: {truth['latent_state']} (심부온 {truth['true_core_temp']}℃, "
              f"{truth['age']}세{', 베타차단제' if truth['on_beta_blocker'] else ''}"
              f"{', 고령' if truth['is_elderly'] else ''})")
        print(f"  측정: 체온 {health['body_temperature']} / 심박 {health['heart_rate']} / "
              f"SpO2 {health['spo2']}  (주변 {s.record['environment_context']['temperature']}℃, "
              f"유효채널 {risk.n_valid_channels}/3)")
        print(f"  ── 현행 고정 룰  → {rule:6s} (신호 {legacy_rule_signals(health)}점)"
              f"   {VERDICT[_correct(rule, truth['is_abnormal'])]}")
        print(f"  ── 2단 구조      → {d.control_level:6s} (위험도 {d.risk_probability:.3f}, "
              f"개인 {d.personal_risk}, 신뢰도 {d.confidence:.0%})"
              f"   {VERDICT[_correct(d.personal_risk, truth['is_abnormal'])]}")
        print(f"     {d.interpretation}")
        for r in d.reasons:
            print(f"       · {r}")
        print(f"  교훈: {s.lesson}")


def compare_facilities(model: RiskModel) -> None:
    """같은 생체값이 시설에 따라 다른 제어를 낳는가 — 연구질문 Q2."""
    print("같은 사람, 다른 시설 — 비용비가 임계값을 바꾼다\n")
    header = f"{'시나리오':34s} {'위험도':>7s} " + "".join(
        f"{POLICIES[f].label:>8s}" for f in ("hospital", "school", "office", "cafe"))
    print(header + "\n" + "-" * len(header))
    for s in SCENARIOS:
        risk = model.estimate(s.person["health_context"], s.record["environment_context"])
        cells = ""
        for ftype in ("hospital", "school", "office", "cafe"):
            fac = {"facility_type": ftype, "risk_level": "HIGH",
                   "vulnerable_population": POLICIES[ftype].vulnerable_population}
            d = decide(risk, model.thresholds, fac, s.record["environment_context"],
                       s.record["vision_context"])
            cells += f"{d.personal_risk:>8s}"
        print(f"{s.title[:32]:34s} {risk.probability:7.3f} {cells}")

    print("\n임계값 (평시 점수 분포 백분위):")
    for k, v in model.thresholds.items():
        print(f"  {k} = {v:.3f}")
    print("\n시설 정책:")
    for f in ("hospital", "school", "office", "cafe"):
        p = POLICIES[f]
        print(f"  {p.label:12s} 놓침:오경보 {p.miss_cost_ratio:4.0f}:1 | "
              f"MEDIUM≥{p.medium_at} HIGH≥{p.high_at} | 측정실패시 {p.unmeasured_level}")


def main() -> None:
    ap = argparse.ArgumentParser(description="게이트 제어 시나리오 데모")
    ap.add_argument("--facility", action="store_true", help="시설별 판정 비교")
    args = ap.parse_args()

    model = RiskModel.load()
    if args.facility:
        compare_facilities(model)
    else:
        compare_all(model)


if __name__ == "__main__":
    main()
