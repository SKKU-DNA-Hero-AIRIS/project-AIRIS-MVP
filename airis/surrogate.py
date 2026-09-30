"""AI side of the physics-AI fusion: a neural surrogate of the gate simulator.

The reduced-order model in ``gate_physics`` is cheap, so today the surrogate
buys little wall-clock time. It is here because the simulator is the part that
will be replaced: once ``simulate`` is backed by CFD runs or chamber
measurements, each evaluation costs minutes, and a gate has well under a second
to decide. The surrogate is trained offline on simulator output and answers in
microseconds; the optimizer then spends its few real simulator calls only on
verifying the candidates the surrogate shortlisted.

Fusion, concretely:

* inputs are not raw setpoints alone — jet theory supplies closed-form features
  (surface velocity ratio, row coverage, exposure time) the network would
  otherwise have to rediscover;
* the network only learns what needs a flow simulation (particle removal per
  zone); fan energy, face velocity and lateral load are closed-form and are
  never handed to it;
* the network is never trusted for the final answer — see
  ``gate_optimizer.optimize_with_surrogate``, which re-scores every shortlisted
  candidate with the physics model.

Like ``ml_policy``, the model is plain NumPy and ships as JSON weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .body_profile import CLOTHING_TYPES, UserProfile, UserTraits, body_from_measurements
from .gate_physics import (
    CLOTHING_ADHESION_GAIN,
    DEFAULT_GATE,
    DEFAULT_PARAMS,
    GateBodyMap,
    GateGeometry,
    PhysicsParams,
    map_gate_to_body,
    simulate,
)


DEFAULT_SURROGATE_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "policy" / "gate_surrogate.json"
)

FEATURE_NAMES: tuple[str, ...] = (
    "height_m",
    "shoulder_width_m",
    "hip_width_m",
    "mass_kg",
    "walking_speed_mps",
    "clothing_adhesion",
    "seated",
    "fine_fraction",
    "decay_head",
    "decay_upper",
    "decay_lower",
    "coverage_head",
    "coverage_upper",
    "coverage_lower",
    "rows_head",
    "rows_upper",
    "rows_lower",
    "v_head",
    "v_upper",
    "v_lower",
    "hold_s",
    "surface_v_head",
    "surface_v_upper",
    "surface_v_lower",
    "exposure_s",
)

# Only the particle-flow interaction is learned. Fan energy, face velocity and
# lateral load are closed-form in the setpoints and stay analytic.
TARGET_NAMES: tuple[str, ...] = (
    "removal_efficiency",
    "removal_head",
    "removal_upper",
    "removal_lower",
)


def gate_signature(gate: GateGeometry, params: PhysicsParams) -> dict[str, Any]:
    """What the surrogate was trained against; a mismatch means retrain."""
    return {
        "passage_width_m": gate.passage_width_m,
        "jet_zone_length_m": gate.jet_zone_length_m,
        "nozzle_diameter_m": gate.nozzle_diameter_m,
        "row_heights_m": list(gate.row_heights_m),
        "nozzles_per_row": gate.nozzles_per_row,
        "max_jet_velocity_mps": gate.max_jet_velocity_mps,
        "max_hold_seconds": gate.max_hold_seconds,
        "vertical_sweep_m": gate.vertical_sweep_m,
        "jet_decay_constant": params.jet_decay_constant,
        "detachment_time_constant_s": params.detachment_time_constant_s,
        "coarse_critical_velocity_mps": params.coarse_critical_velocity_mps,
        "fine_critical_velocity_mps": params.fine_critical_velocity_mps,
    }


def build_features(
    profile: UserProfile,
    setpoints: np.ndarray,
    body_map: GateBodyMap,
    *,
    gate: GateGeometry = DEFAULT_GATE,
    fine_fraction: float = 0.4,
) -> np.ndarray:
    """(N, 4) setpoints -> (N, len(FEATURE_NAMES)) physics-informed features."""
    setpoints = np.atleast_2d(np.asarray(setpoints, dtype=np.float64))
    body = profile.body
    static = np.array(
        [
            body.height_m,
            body.shoulder_width_m,
            body.hip_width_m,
            body.mass_kg,
            profile.traits.walking_speed_mps,
            CLOTHING_ADHESION_GAIN[profile.traits.clothing],
            1.0 if body.posture == "SEATED" else 0.0,
            fine_fraction,
            *body_map.zone_decay,
            *body_map.zone_coverage,
            *body_map.rows_per_zone,
        ]
    )
    exposure = (
        gate.jet_zone_length_m / max(profile.traits.walking_speed_mps, 0.3)
        + setpoints[:, 3:4]
    )
    return np.hstack(
        [
            np.broadcast_to(static, (setpoints.shape[0], static.size)),
            setpoints,
            setpoints[:, :3] * body_map.zone_decay,
            exposure,
        ]
    )


@dataclass(frozen=True)
class GateSurrogate:
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    signature: dict[str, Any]
    metrics: dict[str, Any]
    trained_at: str
    seed: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return (N, len(TARGET_NAMES)) predictions in ``TARGET_NAMES`` order."""
        activation = (np.atleast_2d(features) - self.x_mean) / self.x_scale
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            activation = np.tanh(activation @ weight + bias)
        output = activation @ self.weights[-1] + self.biases[-1]
        return output * self.y_scale + self.y_mean

    def matches(self, gate: GateGeometry, params: PhysicsParams) -> bool:
        return self.signature == gate_signature(gate, params)

    @property
    def removal_mae(self) -> float:
        return float(self.metrics.get("holdout_mae", {}).get("removal_efficiency", 0.02))

    def to_dict(self) -> dict[str, Any]:
        def rounded(array: np.ndarray) -> list[Any]:
            return np.round(array, 5).tolist()

        return {
            "model": "mlp_tanh_gate_surrogate",
            "trained_at": self.trained_at,
            "seed": self.seed,
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "gate_signature": self.signature,
            "metrics": self.metrics,
            "standardization": {
                "x_mean": rounded(self.x_mean),
                "x_scale": rounded(self.x_scale),
                "y_mean": rounded(self.y_mean),
                "y_scale": rounded(self.y_scale),
            },
            "layers": [
                {"weights": rounded(weight), "bias": rounded(bias)}
                for weight, bias in zip(self.weights, self.biases)
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GateSurrogate":
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError(
                "surrogate feature layout does not match airis.surrogate.FEATURE_NAMES; "
                "retrain with tools/train_gate_surrogate.py"
            )
        standardization = payload["standardization"]
        return cls(
            weights=tuple(np.array(layer["weights"]) for layer in payload["layers"]),
            biases=tuple(np.array(layer["bias"]) for layer in payload["layers"]),
            x_mean=np.array(standardization["x_mean"]),
            x_scale=np.array(standardization["x_scale"]),
            y_mean=np.array(standardization["y_mean"]),
            y_scale=np.array(standardization["y_scale"]),
            signature=payload["gate_signature"],
            metrics=payload.get("metrics", {}),
            trained_at=payload.get("trained_at", ""),
            seed=int(payload.get("seed", 0)),
        )

    def save(self, path: Path = DEFAULT_SURROGATE_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        layers = payload.pop("layers")
        # Metadata stays readable; the weight matrices go one layer per line.
        head = json.dumps(payload, indent=2, ensure_ascii=False)
        body = ",\n".join(
            "    " + json.dumps(layer, separators=(",", ":")) for layer in layers
        )
        path.write_text(
            head[:-2] + ',\n  "layers": [\n' + body + "\n  ]\n}\n",
            encoding="utf-8",
        )
        return path


_CACHE: dict[Path, GateSurrogate] = {}


def load_surrogate(path: Path = DEFAULT_SURROGATE_PATH) -> Optional[GateSurrogate]:
    """Load the trained surrogate, or ``None`` if it has not been trained yet."""
    resolved = Path(path).resolve()
    if resolved not in _CACHE:
        if not resolved.exists():
            return None
        _CACHE[resolved] = GateSurrogate.from_dict(
            json.loads(resolved.read_text(encoding="utf-8"))
        )
    return _CACHE[resolved]


def sample_profile(rng: np.random.Generator) -> UserProfile:
    """Draw one synthetic user spanning children to large adults, some seated."""
    seated = rng.random() < 0.10
    height = rng.uniform(1.15, 1.45) if seated else rng.uniform(1.00, 2.00)
    shoulder = (1.70 if seated else height) * rng.uniform(0.215, 0.310)
    bmi = rng.uniform(14.5, 32.0)
    mass = bmi * (1.70 if seated else height) ** 2
    body = body_from_measurements(
        height,
        shoulder,
        mass_kg=round(float(mass), 1),
        posture="SEATED" if seated else "STANDING",
        source="SYNTHETIC",
    )
    traits = UserTraits(
        walking_speed_mps=round(float(rng.uniform(0.4, 1.8)), 2),
        clothing=CLOTHING_TYPES[int(rng.integers(len(CLOTHING_TYPES)))],
    )
    return UserProfile(body=body, traits=traits, profile_id="SYN", label="synthetic")


def sample_setpoints(
    rng: np.random.Generator, count: int, gate: GateGeometry = DEFAULT_GATE
) -> np.ndarray:
    setpoints = rng.uniform(
        0.0,
        [gate.max_jet_velocity_mps] * 3 + [gate.max_hold_seconds],
        size=(count, 4),
    )
    # Shut zones are a real operating point the optimizer visits.
    shut = rng.random((count, 3)) < 0.08
    setpoints[:, :3] = np.where(shut, 0.0, setpoints[:, :3])
    return setpoints


def build_training_set(
    profile_count: int,
    setpoints_per_profile: int,
    seed: int,
    *,
    gate: GateGeometry = DEFAULT_GATE,
    params: PhysicsParams = DEFAULT_PARAMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the simulator over random users x setpoints -> (features, targets)."""
    rng = np.random.default_rng(seed)
    features, targets = [], []
    for _ in range(profile_count):
        profile = sample_profile(rng)
        fine_fraction = float(rng.uniform(0.2, 0.8))
        body_map = map_gate_to_body(profile, gate, params)
        setpoints = sample_setpoints(rng, setpoints_per_profile, gate)
        batch = simulate(
            profile,
            setpoints,
            gate=gate,
            params=params,
            fine_fraction=fine_fraction,
            body_map=body_map,
        )
        features.append(
            build_features(
                profile, setpoints, body_map, gate=gate, fine_fraction=fine_fraction
            )
        )
        targets.append(
            np.column_stack([batch.removal_efficiency, batch.zone_removal])
        )
    return np.vstack(features), np.vstack(targets)


def fit_surrogate(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    gate: GateGeometry = DEFAULT_GATE,
    params: PhysicsParams = DEFAULT_PARAMS,
    hidden: tuple[int, ...] = (64, 64),
    epochs: int = 120,
    batch_size: int = 512,
    learning_rate: float = 3e-3,
) -> GateSurrogate:
    """Train the MLP with Adam on a mean-squared error in standardized units."""
    rng = np.random.default_rng(seed)
    x_mean, x_scale = features.mean(axis=0), features.std(axis=0)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
    y_mean, y_scale = targets.mean(axis=0), targets.std(axis=0)
    y_scale = np.where(y_scale < 1e-8, 1.0, y_scale)
    inputs = (features - x_mean) / x_scale
    outputs = (targets - y_mean) / y_scale

    sizes = (inputs.shape[1], *hidden, outputs.shape[1])
    weights = [
        rng.normal(0.0, np.sqrt(1.0 / fan_in), size=(fan_in, fan_out))
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:])
    ]
    biases = [np.zeros(fan_out) for fan_out in sizes[1:]]
    moments = [np.zeros_like(p) for p in (*weights, *biases)]
    velocities = [np.zeros_like(p) for p in (*weights, *biases)]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    step = 0
    steps_per_epoch = int(np.ceil(len(inputs) / batch_size))
    total_steps = epochs * steps_per_epoch
    for _ in range(epochs):
        order = rng.permutation(len(inputs))
        for start in range(0, len(inputs), batch_size):
            index = order[start : start + batch_size]
            activations = [inputs[index]]
            for weight, bias in zip(weights[:-1], biases[:-1]):
                activations.append(np.tanh(activations[-1] @ weight + bias))
            prediction = activations[-1] @ weights[-1] + biases[-1]

            delta = 2.0 * (prediction - outputs[index]) / len(index)
            weight_grads: list[np.ndarray] = [np.empty(0)] * len(weights)
            bias_grads: list[np.ndarray] = [np.empty(0)] * len(biases)
            for layer in reversed(range(len(weights))):
                weight_grads[layer] = activations[layer].T @ delta
                bias_grads[layer] = delta.sum(axis=0)
                if layer:
                    delta = (delta @ weights[layer].T) * (1.0 - activations[layer] ** 2)

            step += 1
            # Cosine decay keeps the last epochs from bouncing around the minimum.
            rate = learning_rate * 0.5 * (1.0 + np.cos(np.pi * step / total_steps))
            parameters = (*weights, *biases)
            for slot, gradient in enumerate((*weight_grads, *bias_grads)):
                moments[slot] = beta1 * moments[slot] + (1 - beta1) * gradient
                velocities[slot] = beta2 * velocities[slot] + (1 - beta2) * gradient**2
                corrected_m = moments[slot] / (1 - beta1**step)
                corrected_v = velocities[slot] / (1 - beta2**step)
                parameters[slot][...] -= rate * corrected_m / (np.sqrt(corrected_v) + epsilon)

    return GateSurrogate(
        weights=tuple(weights),
        biases=tuple(biases),
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        signature=gate_signature(gate, params),
        metrics={},
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed=seed,
    )


def evaluate_surrogate(
    surrogate: GateSurrogate, features: np.ndarray, targets: np.ndarray
) -> dict[str, dict[str, float]]:
    """Per-output R^2 and MAE on a held-out set, in physical units."""
    prediction = surrogate.predict(features)
    residual = prediction - targets
    total = ((targets - targets.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - (residual**2).sum(axis=0) / np.maximum(total, 1e-12)
    mae = np.abs(residual).mean(axis=0)
    return {
        "holdout_r2": {name: round(float(r2[i]), 4) for i, name in enumerate(TARGET_NAMES)},
        "holdout_mae": {name: round(float(mae[i]), 5) for i, name in enumerate(TARGET_NAMES)},
    }
