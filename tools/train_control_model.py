"""Train the ML control policy and write assets/policy/control_policy.json.

    python tools/train_control_model.py

Deterministic: the same seed reproduces the same weights, so the committed
policy JSON is auditable against this script.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airis.control import analyze_hybrid, baseline_control_level  # noqa: E402
from airis.dataset import build_dataset, class_distribution, label_context  # noqa: E402
from airis.ml_policy import DEFAULT_POLICY_PATH, fit_policy, predict_batch  # noqa: E402
from airis.models import LEVELS  # noqa: E402


def confusion_matrix(actual: np.ndarray, predicted: np.ndarray) -> list[list[int]]:
    size = len(LEVELS)
    matrix = np.zeros((size, size), dtype=int)
    for true_label, predicted_label in zip(actual, predicted):
        matrix[true_label][predicted_label] += 1
    return matrix.tolist()


def macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for index in range(len(LEVELS)):
        true_positive = int(((predicted == index) & (actual == index)).sum())
        false_positive = int(((predicted == index) & (actual != index)).sum())
        false_negative = int(((predicted != index) & (actual == index)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--holdout-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path, default=DEFAULT_POLICY_PATH)
    args = parser.parse_args()

    train_features, train_labels, _ = build_dataset(args.train_size, args.seed)
    holdout_features, _, holdout_contexts = build_dataset(
        args.holdout_size, args.seed + 1
    )
    # Evaluate against the unnoised rubric: boundary disagreement is training
    # noise, not ground truth.
    holdout_labels = np.array([label_context(ctx) for ctx in holdout_contexts])

    policy = fit_policy(train_features, train_labels, seed=args.seed)

    ml_predicted = predict_batch(policy, holdout_features)
    ml_accuracy = float((ml_predicted == holdout_labels).mean())

    level_index = {level: index for index, level in enumerate(LEVELS)}
    hybrid_predicted = np.array(
        [
            level_index[analyze_hybrid(ctx, policy).control_level]
            for ctx in holdout_contexts
        ]
    )
    baseline_predicted = np.array(
        [level_index[baseline_control_level(ctx)] for ctx in holdout_contexts]
    )

    # The rubric is a reference risk score, not field ground truth. "Fidelity"
    # measures how closely a controller reproduces it; the damping rules are
    # *supposed* to disagree with it in specific contexts, so a lower hybrid
    # fidelity is expected and is reported separately as an intervention rate.
    metrics = {
        "train_size": args.train_size,
        "holdout_size": args.holdout_size,
        "train_class_distribution": class_distribution(train_labels),
        "reference": "expert rubric (airis.dataset.RUBRIC_WEIGHTS), noise-free",
        "ml_rubric_fidelity": round(ml_accuracy, 4),
        "ml_macro_f1": round(macro_f1(holdout_labels, ml_predicted), 4),
        "hybrid_rubric_fidelity": round(
            float((hybrid_predicted == holdout_labels).mean()), 4
        ),
        "flat_baseline_fidelity": round(
            float((baseline_predicted == holdout_labels).mean()), 4
        ),
        "flat_baseline_macro_f1": round(macro_f1(holdout_labels, baseline_predicted), 4),
        "rule_intervention_rate": round(
            float((hybrid_predicted != ml_predicted).mean()), 4
        ),
        "floor_raise_rate": round(float((hybrid_predicted > ml_predicted).mean()), 4),
        "ceiling_damp_rate": round(float((hybrid_predicted < ml_predicted).mean()), 4),
        "ml_below_reference_rate": round(
            float((ml_predicted < holdout_labels).mean()), 4
        ),
        "hybrid_below_reference_rate": round(
            float((hybrid_predicted < holdout_labels).mean()), 4
        ),
        "confusion_matrix": {
            "order": list(LEVELS),
            "ml": confusion_matrix(holdout_labels, ml_predicted),
            "hybrid": confusion_matrix(holdout_labels, hybrid_predicted),
        },
    }

    saved = replace(policy, metrics=metrics).save(args.output)

    print(f"saved policy -> {saved}")
    print(f"train class distribution   : {metrics['train_class_distribution']}")
    print(f"flat baseline fidelity     : {metrics['flat_baseline_fidelity']:.3f}")
    print(f"ML rubric fidelity         : {metrics['ml_rubric_fidelity']:.3f}")
    print(f"ML macro F1                : {metrics['ml_macro_f1']:.3f}")
    print(f"hybrid rubric fidelity     : {metrics['hybrid_rubric_fidelity']:.3f}")
    print(f"rule intervention rate     : {metrics['rule_intervention_rate']:.3f}")
    print(f"  floor raise / ceiling damp: "
          f"{metrics['floor_raise_rate']:.3f} / {metrics['ceiling_damp_rate']:.3f}")
    print(f"ML below reference rate    : {metrics['ml_below_reference_rate']:.3f}")
    print(f"hybrid below reference rate: {metrics['hybrid_below_reference_rate']:.3f}")


if __name__ == "__main__":
    main()
