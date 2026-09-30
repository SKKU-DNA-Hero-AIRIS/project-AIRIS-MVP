"""Find the gate setpoints that fit one user.

    minimize    energy + time + face-level draught + zone neglect
    subject to  removal efficiency >= target        (from the control level)
                face velocity      <= user limit    (hard, by construction)
                lateral load       <= user limit    (hard, by construction)

The two safety limits are closed-form in the setpoints, so every candidate is
*projected* into the safe set before it is scored — the search never proposes
an unsafe plan and then hopes a penalty discourages it. Only the removal target
is handled as a constraint, with Deb's feasibility rules: if the target cannot
be met inside the safe set, the optimizer returns the cleanest safe plan and
says so.

Two drivers share the same differential-evolution core:

* ``optimize_with_physics`` scores every candidate with the simulator.
* ``optimize_with_surrogate`` searches on the neural surrogate, then spends a
  handful of simulator calls verifying and polishing the shortlist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .body_profile import UserProfile
from .gate_physics import (
    ACTIVE_VELOCITY_MPS,
    ComfortLimits,
    DEFAULT_GATE,
    DEFAULT_PARAMS,
    GRAVITY,
    GateBodyMap,
    GateGeometry,
    GateSetpoints,
    PhysicsParams,
    comfort_limits,
    fan_power,
    map_gate_to_body,
    rows_from_setpoints,
    simulate,
)
from .surrogate import GateSurrogate, build_features


FEASIBILITY_TOLERANCE = 1e-4

Evaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class OptimizationTask:
    profile: UserProfile
    target_removal: float
    reference_energy_wh: float
    reference_time_s: float
    time_weight: float = 0.3
    comfort_weight: float = 0.2
    # A plan that meets the whole-body target by abandoning one zone is not a
    # clean user. Each zone is pulled toward this share of the target.
    zone_floor_share: float = 0.6
    balance_weight: float = 2.0
    fine_fraction: float = 0.4
    gate: GateGeometry = DEFAULT_GATE
    params: PhysicsParams = DEFAULT_PARAMS

    @property
    def limits(self) -> ComfortLimits:
        return comfort_limits(self.profile.traits)


@dataclass(frozen=True)
class OptimizationResult:
    setpoints: GateSetpoints
    cost: float
    removal_efficiency: float
    target_met: bool
    method: str
    simulator_calls: int
    surrogate_calls: int = 0


class _Problem:
    """Bounds, safety projection and scoring for one task."""

    def __init__(self, task: OptimizationTask) -> None:
        self.task = task
        self.body_map: GateBodyMap = map_gate_to_body(task.profile, task.gate, task.params)
        limits = task.limits
        gate = task.gate

        velocity_cap = np.full(3, gate.max_jet_velocity_mps)
        for zone, segment in enumerate(self.body_map.segments):
            if self.body_map.rows_per_zone[zone] == 0:
                # No nozzle row faces this zone; do not let the search pretend.
                velocity_cap[zone] = 0.0
            elif segment.zone == "HEAD":
                velocity_cap[zone] = min(
                    velocity_cap[zone],
                    limits.face_velocity_mps / self.body_map.zone_decay[zone],
                )
        self.lower = np.zeros(4)
        self.upper = np.array([*velocity_cap, gate.max_hold_seconds])

        # One-side thrust = sum over rows of n * rho * A * v^2.
        self._thrust_per_v2 = (
            gate.nozzles_per_row
            * task.params.air_density
            * gate.nozzle_area_m2
            * self.body_map.rows_per_zone
        )
        self._thrust_limit = (
            limits.lateral_load_ratio * task.profile.body.mass_kg * GRAVITY
        )
        self.simulator_calls = 0
        self.surrogate_calls = 0

    def project(self, candidates: np.ndarray) -> np.ndarray:
        """Clip to the bounds, then scale velocities down to the load limit."""
        projected = np.clip(candidates, self.lower, self.upper)
        thrust = (projected[:, :3] ** 2) @ self._thrust_per_v2
        scale = np.sqrt(
            np.minimum(1.0, self._thrust_limit / np.maximum(thrust, 1e-12))
        )
        projected[:, :3] *= scale[:, None]
        return projected

    def _score(
        self,
        removal: np.ndarray,
        zone_removal: np.ndarray,
        face_velocity: np.ndarray,
        energy: np.ndarray,
        hold: np.ndarray,
        margin: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        task = self.task
        transit = task.gate.jet_zone_length_m / max(
            task.profile.traits.walking_speed_mps, 0.3
        )
        neglect = np.maximum(
            0.0, task.zone_floor_share * task.target_removal - zone_removal
        ).sum(axis=1)
        cost = (
            energy / task.reference_energy_wh
            + task.time_weight * (transit + hold) / task.reference_time_s
            + task.comfort_weight
            * (face_velocity / task.limits.face_velocity_mps) ** 2
            + task.balance_weight * neglect / max(task.target_removal, 1e-6)
        )
        shortfall = np.maximum(0.0, task.target_removal + margin - removal)
        shortfall = np.where(shortfall < FEASIBILITY_TOLERANCE, 0.0, shortfall)
        return cost, shortfall

    def physics(self, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.simulator_calls += len(candidates)
        batch = simulate(
            self.task.profile,
            candidates,
            gate=self.task.gate,
            params=self.task.params,
            fine_fraction=self.task.fine_fraction,
            body_map=self.body_map,
        )
        return self._score(
            batch.removal_efficiency,
            batch.zone_removal,
            batch.face_velocity_mps,
            batch.energy_wh,
            candidates[:, 3],
            0.0,
        )

    def surrogate(self, model: GateSurrogate, margin: float) -> Evaluator:
        task = self.task
        transit = task.gate.jet_zone_length_m / max(
            task.profile.traits.walking_speed_mps, 0.3
        )
        head_decay = self.body_map.zone_decay[0]

        def evaluate(candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            self.surrogate_calls += len(candidates)
            # Learned: removal (whole body, then per zone), per TARGET_NAMES.
            removal = model.predict(
                build_features(
                    task.profile,
                    candidates,
                    self.body_map,
                    gate=task.gate,
                    fine_fraction=task.fine_fraction,
                )
            )
            # Analytic: fan law and jet decay need no simulation.
            rows = rows_from_setpoints(self.body_map, candidates, task.gate)
            energy = (
                fan_power(rows, task.gate, task.params)
                * (transit + candidates[:, 3])
                / 3600.0
            )
            head = candidates[:, 0]
            face_velocity = np.where(head >= ACTIVE_VELOCITY_MPS, head * head_decay, 0.0)
            return self._score(
                removal[:, 0],
                removal[:, 1:4],
                face_velocity,
                energy,
                candidates[:, 3],
                margin,
            )

        return evaluate

    def finalize(self, candidate: np.ndarray) -> tuple[GateSetpoints, float]:
        """Round to actuator resolution without leaving the safe set.

        Velocities round down and the hold rounds up, so the rounded plan is
        never less safe than the one the search scored.
        """
        velocities = np.floor(candidate[:3] * 100.0) / 100.0
        velocities = np.where(velocities < ACTIVE_VELOCITY_MPS, 0.0, velocities)
        hold = min(np.ceil(candidate[3] * 10.0) / 10.0, self.upper[3])
        setpoints = GateSetpoints.from_array(np.array([*velocities, hold]))
        self.simulator_calls += 1
        batch = simulate(
            self.task.profile,
            setpoints.as_array(),
            gate=self.task.gate,
            params=self.task.params,
            fine_fraction=self.task.fine_fraction,
            body_map=self.body_map,
        )
        return setpoints, float(batch.removal_efficiency[0])


def _better(
    cost: np.ndarray, shortfall: np.ndarray, other_cost: np.ndarray, other_shortfall: np.ndarray
) -> np.ndarray:
    """Deb's rules: less constraint violation wins, then lower cost."""
    return (shortfall < other_shortfall) | (
        (shortfall <= other_shortfall) & (cost < other_cost)
    )


def _rank(cost: np.ndarray, shortfall: np.ndarray) -> np.ndarray:
    return np.lexsort((cost, shortfall))


def _differential_evolution(
    problem: _Problem,
    evaluate: Evaluator,
    rng: np.random.Generator,
    *,
    population: int,
    generations: int,
    seeds: Optional[np.ndarray] = None,
    spread: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rand/1/bin DE. Returns the final population sorted best-first."""
    span = problem.upper - problem.lower
    members = problem.lower + rng.random((population, 4)) * span
    if seeds is not None:
        # Start around known-good points instead of the whole box.
        picks = seeds[rng.integers(len(seeds), size=population)]
        members = picks + rng.normal(0.0, 0.08 * spread, size=(population, 4)) * span
        members[: len(seeds)] = seeds[:population]
    members = problem.project(members)
    cost, shortfall = evaluate(members)

    for _ in range(generations):
        a, b, c = (rng.permutation(population) for _ in range(3))
        mutant = members[a] + 0.6 * (members[b] - members[c])
        cross = rng.random((population, 4)) < 0.9
        cross[np.arange(population), rng.integers(4, size=population)] = True
        trial = problem.project(np.where(cross, mutant, members))
        trial_cost, trial_shortfall = evaluate(trial)
        improved = _better(trial_cost, trial_shortfall, cost, shortfall)
        members[improved] = trial[improved]
        cost[improved] = trial_cost[improved]
        shortfall[improved] = trial_shortfall[improved]

    order = _rank(cost, shortfall)
    return members[order], cost[order], shortfall[order]


def _result(
    problem: _Problem, candidate: np.ndarray, cost: float, method: str
) -> OptimizationResult:
    setpoints, removal = problem.finalize(candidate)
    return OptimizationResult(
        setpoints=setpoints,
        cost=round(float(cost), 4),
        removal_efficiency=round(removal, 4),
        target_met=removal >= problem.task.target_removal - 1e-3,
        method=method,
        simulator_calls=problem.simulator_calls,
        surrogate_calls=problem.surrogate_calls,
    )


def optimize_with_physics(
    task: OptimizationTask,
    *,
    seed: int = 0,
    population: int = 40,
    generations: int = 70,
) -> OptimizationResult:
    """Reference solution: every candidate is scored by the simulator."""
    problem = _Problem(task)
    members, cost, _ = _differential_evolution(
        problem,
        problem.physics,
        np.random.default_rng(seed),
        population=population,
        generations=generations,
    )
    return _result(problem, members[0], cost[0], "PHYSICS")


def optimize_with_surrogate(
    task: OptimizationTask,
    surrogate: GateSurrogate,
    *,
    seed: int = 0,
    population: int = 40,
    generations: int = 70,
    shortlist: int = 3,
    polish_population: int = 12,
    polish_generations: int = 10,
) -> OptimizationResult:
    """Search on the surrogate, then verify and polish with the simulator.

    The surrogate is run at three safety margins on the removal target so the
    shortlist brackets its own error. Nothing the network says is final: the
    returned plan is always one the physics model scored.
    """
    problem = _Problem(task)
    rng = np.random.default_rng(seed)

    candidates = []
    for margin in (0.0, 1.5 * surrogate.removal_mae, 4.0 * surrogate.removal_mae):
        members, _, _ = _differential_evolution(
            problem,
            problem.surrogate(surrogate, margin),
            rng,
            population=population,
            generations=generations,
        )
        candidates.append(members[:shortlist])
    shortlisted = np.vstack(candidates)

    members, cost, _ = _differential_evolution(
        problem,
        problem.physics,
        rng,
        population=max(polish_population, len(shortlisted)),
        generations=polish_generations,
        seeds=shortlisted,
        spread=0.5,
    )
    return _result(problem, members[0], cost[0], "SURROGATE+PHYSICS")
