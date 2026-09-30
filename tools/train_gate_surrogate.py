"""Train the gate surrogate and write assets/policy/gate_surrogate.json.

    python tools/train_gate_surrogate.py

The simulator generates the training set, so this is deterministic: the same
seed reproduces the same weights. Besides regression accuracy the script
reports what actually matters — whether optimizing on the surrogate lands on
the same plan as optimizing on the physics model, and how many simulator calls
it takes to get there.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airis.body_profile import load_user_profiles  # noqa: E402
from airis.models import LEVELS  # noqa: E402
from airis.personalization import plan_personalized  # noqa: E402
from airis.surrogate import (  # noqa: E402
    DEFAULT_SURROGATE_PATH,
    TARGET_NAMES,
    build_training_set,
    evaluate_surrogate,
    fit_surrogate,
)


def optimization_agreement(surrogate) -> dict[str, float]:
    """Compare surrogate-assisted plans against physics-only plans per persona."""
    removal_gap, objective_gap, physics_calls, fused_calls = [], [], [], []
    same_status = 0
    cases = 0
    for profile in load_user_profiles():
        for level in LEVELS:
            reference = plan_personalized(profile, level, use_surrogate=False)
            fused = plan_personalized(profile, level, surrogate=surrogate)
            cases += 1
            same_status += reference.target_status == fused.target_status
            removal_gap.append(
                abs(
                    fused.personalized.removal_efficiency
                    - reference.personalized.removal_efficiency
                )
            )
            # Both objectives are scored by the physics model, so this is the
            # price of searching on the surrogate, not the surrogate's own error.
            objective_gap.append(
                (fused.objective - reference.objective) / reference.objective
            )
            physics_calls.append(reference.simulator_calls)
            fused_calls.append(fused.simulator_calls)
    return {
        "cases": cases,
        "same_target_status_rate": round(same_status / cases, 4),
        "mean_abs_removal_gap": round(float(np.mean(removal_gap)), 4),
        "max_abs_removal_gap": round(float(np.max(removal_gap)), 4),
        "mean_relative_objective_gap": round(float(np.mean(objective_gap)), 4),
        "max_relative_objective_gap": round(float(np.max(objective_gap)), 4),
        "physics_only_simulator_calls": int(np.mean(physics_calls)),
        "surrogate_assisted_simulator_calls": int(np.mean(fused_calls)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-profiles", type=int, default=1500)
    parser.add_argument("--holdout-profiles", type=int, default=300)
    parser.add_argument("--setpoints-per-profile", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output", type=Path, default=DEFAULT_SURROGATE_PATH)
    args = parser.parse_args()

    started = time.perf_counter()
    train_x, train_y = build_training_set(
        args.train_profiles, args.setpoints_per_profile, args.seed
    )
    holdout_x, holdout_y = build_training_set(
        args.holdout_profiles, args.setpoints_per_profile, args.seed + 1
    )
    simulated = time.perf_counter() - started

    started = time.perf_counter()
    surrogate = fit_surrogate(train_x, train_y, seed=args.seed, epochs=args.epochs)
    trained = time.perf_counter() - started

    metrics = {
        "train_samples": int(len(train_x)),
        "holdout_samples": int(len(holdout_x)),
        "holdout_profiles_are_unseen": True,
        "reference": "airis.gate_physics reduced-order simulator (not CFD, not field data)",
        **evaluate_surrogate(surrogate, holdout_x, holdout_y),
    }
    surrogate = replace(surrogate, metrics=metrics)
    metrics["optimization_agreement"] = optimization_agreement(surrogate)
    saved = replace(surrogate, metrics=metrics).save(args.output)

    print(f"saved surrogate -> {saved}")
    print(f"simulated {len(train_x) + len(holdout_x)} passages in {simulated:.1f}s, "
          f"trained in {trained:.1f}s")
    print(f"{'output':22} {'R2':>8} {'MAE':>10}")
    for name in TARGET_NAMES:
        print(f"{name:22} {metrics['holdout_r2'][name]:8.4f} "
              f"{metrics['holdout_mae'][name]:10.5f}")
    agreement = metrics["optimization_agreement"]
    print(f"plans with the same target status : "
          f"{agreement['same_target_status_rate']:.0%} of {agreement['cases']}")
    print(f"removal gap vs physics-only       : mean {agreement['mean_abs_removal_gap']:.4f}, "
          f"max {agreement['max_abs_removal_gap']:.4f}")
    print(f"objective gap vs physics-only     : mean {agreement['mean_relative_objective_gap']:+.1%}, "
          f"max {agreement['max_relative_objective_gap']:+.1%}")
    print(f"simulator calls per plan          : "
          f"{agreement['physics_only_simulator_calls']} physics-only -> "
          f"{agreement['surrogate_assisted_simulator_calls']} surrogate-assisted")


if __name__ == "__main__":
    main()
