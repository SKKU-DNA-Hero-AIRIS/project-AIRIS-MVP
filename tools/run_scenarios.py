"""Run every context sample through the hybrid decision layer.

    python tools/run_scenarios.py

Prints a comparison table and writes assets/reports/scenario_decisions.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airis.control import analyze_hybrid  # noqa: E402
from airis.ml_policy import load_policy  # noqa: E402
from airis.scenarios import load_context_samples  # noqa: E402


DEFAULT_REPORT_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "reports" / "scenario_decisions.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    policy = load_policy()
    rows = []
    report = []

    for payload, context in load_context_samples():
        decision = analyze_hybrid(context, policy)
        rows.append(
            (
                payload["scenario_id"],
                payload["scenario_label"],
                decision.baseline_level,
                f"{decision.ml_level} ({decision.ml_confidence}%)",
                f"{decision.floor_level}/{decision.ceiling_level}",
                decision.control_level,
                decision.decision_source,
            )
        )
        report.append(
            {
                "scenario_id": payload["scenario_id"],
                "scenario_label": payload["scenario_label"],
                "context": context.to_dict(),
                "decision": decision.to_dict(),
            }
        )

    headers = ("ID", "시나리오", "고정규칙", "ML", "하한/상한", "최종", "결정 주체")
    widths = [
        max(len(str(row[i])) for row in (*rows, headers)) for i in range(len(headers))
    ]
    line = "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))

    print()
    for entry in report:
        decision = entry["decision"]
        print(f"[{entry['scenario_id']}] {decision['interpretation']}")
        for reason in decision["reasons"]:
            print(f"    - {reason}")
        actuation = decision["actuation"]
        print(
            f"    => fan {actuation['fan_level']}% / "
            f"recirc {actuation['recirculation']} / "
            f"uv {actuation['uv_sterilizer']} / "
            f"hold {actuation['gate_hold_seconds']}s / "
            f"alert {actuation['alert']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"policy_metrics": policy.metrics, "scenarios": report},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote report -> {args.output}")


if __name__ == "__main__":
    main()
