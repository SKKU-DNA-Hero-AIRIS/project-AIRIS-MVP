"""ML side of the decision layer: a multinomial softmax regression policy.

A linear model is a deliberate choice over a stronger learner. It is small
enough to ship as JSON weights, it is deterministic, and every prediction
decomposes into per-feature logit contributions, so the dashboard can show
*why* the model asked for a control level instead of just asserting it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_LABELS, FEATURE_NAMES, feature_vector
from .models import LEVELS, MultimodalContext


DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "policy" / "control_policy.json"
)


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    label: str
    value: float
    contribution: float


@dataclass(frozen=True)
class MLPrediction:
    level: str
    probabilities: dict[str, float]
    confidence: int
    drivers: tuple[FeatureContribution, ...]


@dataclass(frozen=True)
class ControlPolicy:
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]
    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    metrics: dict[str, Any]
    trained_at: str
    seed: int

    def _standardize(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.mean) / self.scale

    def logits(self, standardized: np.ndarray) -> np.ndarray:
        return self.weights @ standardized + self.bias

    def predict(self, context: MultimodalContext) -> MLPrediction:
        raw = feature_vector(context)
        standardized = self._standardize(raw)
        probabilities = softmax(self.logits(standardized))
        index = int(np.argmax(probabilities))

        # Contribution relative to the average class weight: this is the part of
        # the feature that pushed the decision toward the winning class rather
        # than the shared baseline.
        relative = self.weights[index] - self.weights.mean(axis=0)
        contributions = relative * standardized
        order = np.argsort(-np.abs(contributions))[:4]
        drivers = tuple(
            FeatureContribution(
                feature=self.feature_names[i],
                label=FEATURE_LABELS.get(self.feature_names[i], self.feature_names[i]),
                value=round(float(raw[i]), 3),
                contribution=round(float(contributions[i]), 3),
            )
            for i in order
            if abs(contributions[i]) >= 0.05
        )
        return MLPrediction(
            level=self.classes[index],
            probabilities={
                name: round(float(probabilities[i]), 4)
                for i, name in enumerate(self.classes)
            },
            confidence=int(round(float(probabilities[index]) * 100)),
            drivers=drivers,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "multinomial_softmax_regression",
            "trained_at": self.trained_at,
            "seed": self.seed,
            "classes": list(self.classes),
            "feature_names": list(self.feature_names),
            "standardization": {
                "mean": [round(v, 6) for v in self.mean.tolist()],
                "scale": [round(v, 6) for v in self.scale.tolist()],
            },
            "weights": [[round(v, 6) for v in row] for row in self.weights.tolist()],
            "bias": [round(v, 6) for v in self.bias.tolist()],
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ControlPolicy":
        feature_names = tuple(payload["feature_names"])
        if feature_names != FEATURE_NAMES:
            raise ValueError(
                "policy feature layout does not match airis.features.FEATURE_NAMES; "
                "retrain with tools/train_control_model.py"
            )
        return cls(
            feature_names=feature_names,
            classes=tuple(payload["classes"]),
            weights=np.array(payload["weights"], dtype=np.float64),
            bias=np.array(payload["bias"], dtype=np.float64),
            mean=np.array(payload["standardization"]["mean"], dtype=np.float64),
            scale=np.array(payload["standardization"]["scale"], dtype=np.float64),
            metrics=payload.get("metrics", {}),
            trained_at=payload.get("trained_at", ""),
            seed=int(payload.get("seed", 0)),
        )

    def save(self, path: Path = DEFAULT_POLICY_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path


_CACHE: dict[Path, ControlPolicy] = {}


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> ControlPolicy:
    """Load the trained policy, caching per path (weights are read-only)."""
    resolved = Path(path).resolve()
    if resolved not in _CACHE:
        _CACHE[resolved] = ControlPolicy.from_dict(
            json.loads(resolved.read_text(encoding="utf-8"))
        )
    return _CACHE[resolved]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def fit_policy(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    metrics: dict[str, Any] | None = None,
    epochs: int = 4000,
    learning_rate: float = 0.35,
    l2: float = 1e-3,
) -> ControlPolicy:
    """Full-batch gradient descent on the cross-entropy loss."""
    mean = features.mean(axis=0)
    scale = np.where(features.std(axis=0) < 1e-8, 1.0, features.std(axis=0))
    standardized = (features - mean) / scale

    num_classes = len(LEVELS)
    onehot = np.eye(num_classes)[labels]
    weights = np.zeros((num_classes, features.shape[1]))
    bias = np.zeros(num_classes)

    for _ in range(epochs):
        probabilities = softmax(standardized @ weights.T + bias)
        error = probabilities - onehot
        weights -= learning_rate * (
            error.T @ standardized / len(standardized) + l2 * weights
        )
        bias -= learning_rate * error.mean(axis=0)

    return ControlPolicy(
        feature_names=FEATURE_NAMES,
        classes=LEVELS,
        weights=weights,
        bias=bias,
        mean=mean,
        scale=scale,
        metrics=metrics or {},
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed=seed,
    )


def predict_batch(policy: ControlPolicy, features: np.ndarray) -> np.ndarray:
    standardized = (features - policy.mean) / policy.scale
    return np.argmax(softmax(standardized @ policy.weights.T + policy.bias), axis=1)
