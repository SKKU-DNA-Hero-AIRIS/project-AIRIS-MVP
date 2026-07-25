"""Synthetic labeled dataset for training the ML control policy.

No logged operator decisions exist yet, so labels come from an expert-defined
risk rubric (``RUBRIC_WEIGHTS``) plus boundary disagreement noise. The ML layer
is therefore not discovering new knowledge — it learns a smooth, calibrated
interpolation of the rubric across the ~10^6 context combinations nobody
enumerated by hand, and it stays swappable: replace ``label_context`` with real
operator labels and the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES, extract_features
from .models import (
    EnvironmentContext,
    FacilityContext,
    HealthContext,
    LEVELS,
    MultimodalContext,
    VisionContext,
)
from .scenarios import FACILITIES


RUBRIC_WEIGHTS: dict[str, float] = {
    "spo2_deficit": 2.6,
    "fever": 2.2,
    "hr_excess": 1.0,
    "stress": 0.6,
    "pm25_load": 1.5,
    "thermal_load": 0.7,
    "humidity_load": 0.4,
    "crowding": 1.2,
    "unmasked_rate": 0.3,
    "facility_sensitivity": 0.45,
    "vulnerable_population": 0.15,
    "hr_x_sensitivity": 1.4,
    "fever_x_vulnerable": 1.6,
    "pm25_x_crowding": 0.8,
    "unmasked_x_crowding": 0.5,
    "activity_plausible": -1.5,
}

MEDIUM_THRESHOLD = 1.35
HIGH_THRESHOLD = 3.00

# Simultaneously elevated modalities compound rather than add. This term is
# deliberately *not* part of FEATURE_NAMES, so the linear policy has to
# approximate it — the residual is where the rule layer earns its place.
COMPOUNDING_THRESHOLD = 0.45
COMPOUNDING_WEIGHT = 1.2

# Operators disagree near the thresholds, not in the obvious cases.
BOUNDARY_MARGIN = 0.25
BOUNDARY_FLIP_RATE = 0.35

HEALTH_MODES = (
    ("resting", 0.42, (78, 9), (98.0, 0.8), (36.6, 0.2), (28, 10)),
    ("active", 0.24, (128, 14), (97.5, 1.0), (37.1, 0.3), (70, 12)),
    ("febrile", 0.16, (105, 12), (96.0, 1.4), (38.3, 0.5), (62, 14)),
    ("hypoxic", 0.10, (112, 15), (91.0, 2.0), (37.2, 0.5), (70, 15)),
    ("critical", 0.08, (130, 14), (89.0, 2.2), (38.6, 0.6), (86, 8)),
)

ENVIRONMENT_MODES = (
    ("clean", 0.45, (23.5, 2.0), (45, 8), (14, 6)),
    ("warm_humid", 0.25, (32.0, 2.5), (78, 8), (30, 12)),
    ("dusty", 0.20, (28.0, 4.0), (62, 12), (85, 25)),
    ("severe", 0.10, (33.0, 3.0), (75, 10), (135, 30)),
)

OCCUPANCY_MODES = ((0.40, 0, 3), (0.35, 4, 7), (0.25, 8, 28))


def crowding_level(person_count: int) -> str:
    """Mirror of the vision module's occupancy banding for a single gate view."""
    if person_count >= 8:
        return "HIGH"
    if person_count >= 4:
        return "MEDIUM"
    return "LOW"


def _pick(rng: np.random.Generator, modes: tuple) -> tuple:
    weights = np.array([mode[1] for mode in modes])
    index = rng.choice(len(modes), p=weights / weights.sum())
    return modes[index]


def _normal(rng: np.random.Generator, spec: tuple[float, float]) -> float:
    return float(rng.normal(spec[0], spec[1]))


def sample_context(rng: np.random.Generator) -> MultimodalContext:
    _, _, hr, spo2, temp, stress = _pick(rng, HEALTH_MODES)
    health = HealthContext(
        heart_rate=int(np.clip(_normal(rng, hr), 50, 190)),
        spo2=int(np.clip(_normal(rng, spo2), 80, 100)),
        body_temperature=round(float(np.clip(_normal(rng, temp), 35.5, 40.5)), 1),
        stress_index=int(np.clip(_normal(rng, stress), 0, 100)),
    )

    _, _, env_temp, humidity, pm25 = _pick(rng, ENVIRONMENT_MODES)
    pm25_value = int(np.clip(_normal(rng, pm25), 3, 220))
    environment = EnvironmentContext(
        temperature=round(float(np.clip(_normal(rng, env_temp), 16, 42)), 1),
        humidity=int(np.clip(_normal(rng, humidity), 20, 95)),
        pm25=pm25_value,
        air_quality="BAD" if pm25_value >= 75 else "MODERATE" if pm25_value >= 36 else "GOOD",
    )

    weights = np.array([mode[0] for mode in OCCUPANCY_MODES])
    low, high = OCCUPANCY_MODES[rng.choice(len(OCCUPANCY_MODES), p=weights / weights.sum())][1:]
    person_count = int(rng.integers(low, high + 1))
    mask_rate = 0.0 if person_count == 0 else round(float(rng.beta(2.2, 2.2)), 3)
    vision = VisionContext(
        person_count=person_count,
        mask_rate=mask_rate,
        crowding_level=crowding_level(person_count),
    )

    facility = FacilityContext(**FACILITIES[str(rng.choice(list(FACILITIES)))])
    return MultimodalContext(
        vision_context=vision,
        health_context=health,
        environment_context=environment,
        facility_context=facility,
    )


def active_modality_count(features: dict[str, float]) -> int:
    signals = (
        max(features["spo2_deficit"], features["fever"], features["hr_excess"]),
        max(features["pm25_load"], features["thermal_load"]),
        features["crowding"],
    )
    return sum(signal >= COMPOUNDING_THRESHOLD for signal in signals)


def rubric_score(features: dict[str, float]) -> float:
    linear = sum(weight * features[name] for name, weight in RUBRIC_WEIGHTS.items())
    compounding = max(0, active_modality_count(features) - 1)
    return linear + COMPOUNDING_WEIGHT * compounding


def label_context(
    context: MultimodalContext, rng: np.random.Generator | None = None
) -> int:
    """Return the rubric label index, optionally with boundary disagreement."""
    score = rubric_score(extract_features(context))
    label = 2 if score >= HIGH_THRESHOLD else 1 if score >= MEDIUM_THRESHOLD else 0
    if rng is None:
        return label

    for threshold, lower, upper in (
        (MEDIUM_THRESHOLD, 0, 1),
        (HIGH_THRESHOLD, 1, 2),
    ):
        if abs(score - threshold) < BOUNDARY_MARGIN and rng.random() < BOUNDARY_FLIP_RATE:
            return lower if label == upper else upper
    return label


def build_dataset(
    size: int, seed: int, *, noisy_labels: bool = True
) -> tuple[np.ndarray, np.ndarray, list[MultimodalContext]]:
    """Sample contexts and label them with the rubric."""
    rng = np.random.default_rng(seed)
    features = np.empty((size, len(FEATURE_NAMES)))
    labels = np.empty(size, dtype=int)
    contexts: list[MultimodalContext] = []
    for row in range(size):
        context = sample_context(rng)
        extracted = extract_features(context)
        features[row] = [extracted[name] for name in FEATURE_NAMES]
        labels[row] = label_context(context, rng if noisy_labels else None)
        contexts.append(context)
    return features, labels, contexts


def class_distribution(labels: np.ndarray) -> dict[str, int]:
    return {level: int((labels == index).sum()) for index, level in enumerate(LEVELS)}
