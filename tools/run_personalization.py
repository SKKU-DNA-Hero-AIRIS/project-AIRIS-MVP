"""Run every user persona through the personalized gate optimizer.

    python tools/run_personalization.py

For each persona x control level this compares the fixed actuation table
(``airis.control.BASE_ACTUATION``) against the plan fitted to that body, prints
the comparison, and writes assets/reports/personalization_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airis.body_profile import load_user_profiles  # noqa: E402
from airis.gate_physics import SimulationResult  # noqa: E402
from airis.models import LEVELS  # noqa: E402
from airis.personalization import PersonalizedPlan, plan_personalized  # noqa: E402
from airis.surrogate import load_surrogate  # noqa: E402


DEFAULT_REPORT_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "reports"
    / "personalization_report.json"
)


def _metrics(result: SimulationResult) -> dict[str, float]:
    return {
        "removal_efficiency": result.removal_efficiency,
        "zone_removal": result.zone_removal,
        "face_velocity_mps": result.face_velocity_mps,
        "lateral_load_ratio": result.lateral_load_ratio,
        "energy_wh": result.energy_wh,
        "passage_time_s": result.passage_time_s,
    }


def summarize(plan: PersonalizedPlan) -> dict:
    return {
        "profile_id": plan.profile.profile_id,
        "label": plan.profile.label,
        "control_level": plan.control_level,
        "target_removal": plan.target_removal,
        "target_status": plan.target_status,
        "needs_attendant": plan.needs_attendant,
        "limits": {
            "face_velocity_mps": plan.limits.face_velocity_mps,
            "lateral_load_ratio": plan.limits.lateral_load_ratio,
            "basis": plan.limits.basis,
        },
        "setpoints": {
            "head_velocity_mps": plan.setpoints.head_velocity_mps,
            "upper_velocity_mps": plan.setpoints.upper_velocity_mps,
            "lower_velocity_mps": plan.setpoints.lower_velocity_mps,
            "hold_seconds": plan.setpoints.hold_seconds,
        },
        "idle_rows": sum(row.zone is None for row in plan.personalized.rows),
        "fixed": {**_metrics(plan.baseline), "violations": list(plan.baseline_violations)},
        "personalized": _metrics(plan.personalized),
        "energy_saving_pct": plan.energy_saving_pct,
        "method": plan.method,
        "simulator_calls": plan.simulator_calls,
        "reasons": list(plan.reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--physics-only",
        action="store_true",
        help="skip the surrogate and score every candidate with the simulator",
    )
    args = parser.parse_args()

    surrogate = None if args.physics_only else load_surrogate()
    report = []
    rows = []
    for profile in load_user_profiles():
        for level in LEVELS:
            plan = plan_personalized(
                profile,
                level,
                surrogate=surrogate,
                use_surrogate=not args.physics_only,
            )
            report.append(summarize(plan))
            fixed, fitted = plan.baseline, plan.personalized
            rows.append(
                (
                    profile.profile_id,
                    level,
                    f"{plan.target_removal:.0%}",
                    f"{fixed.removal_efficiency:.0%}",
                    f"{fixed.face_velocity_mps:.1f}",
                    f"{fixed.energy_wh:.3f}",
                    "위반" if plan.baseline_violations else "-",
                    f"{fitted.removal_efficiency:.0%}",
                    f"{fitted.face_velocity_mps:.1f}",
                    f"{fitted.energy_wh:.3f}",
                    f"{fitted.passage_time_s:.1f}",
                    f"{plan.energy_saving_pct:+.0f}%",
                    plan.target_status,
                )
            )

    headers = (
        "ID", "레벨", "목표", "고정η", "고정 얼굴", "고정 Wh", "고정 안전",
        "맞춤η", "맞춤 얼굴", "맞춤 Wh", "통과(s)", "에너지 절감", "목표 상태",
    )
    widths = [
        max(len(str(row[i])) for row in (*rows, headers)) for i in range(len(headers))
    ]
    line = "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))

    print()
    for profile in load_user_profiles():
        entry = next(
            item
            for item in report
            if item["profile_id"] == profile.profile_id
            and item["control_level"] == "HIGH"
        )
        print(f"[{entry['profile_id']}] {entry['label']} — HIGH")
        for reason in entry["reasons"]:
            print(f"    - {reason}")

    violating = sum(bool(item["fixed"]["violations"]) for item in report)
    met = sum(item["target_status"] != "SHORTFALL" for item in report)
    print()
    print(f"고정 제어가 사용자 안전 한계를 넘는 경우 : {violating}/{len(report)}")
    print(f"맞춤 제어가 한계 내에서 목표(허용 오차 포함)를 지킨 경우 : {met}/{len(report)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "simulator": "airis.gate_physics reduced-order model (not CFD, not field data)",
                "surrogate_metrics": surrogate.metrics if surrogate else None,
                "summary": {
                    "cases": len(report),
                    "fixed_plan_violates_user_limits": violating,
                    "personalized_within_target_tolerance": met,
                },
                "cases": report,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote report -> {args.output}")


if __name__ == "__main__":
    main()
