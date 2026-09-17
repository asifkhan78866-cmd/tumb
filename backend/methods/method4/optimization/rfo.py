"""Red Fox Optimization (Połap & Woźniak, 2021).

A population metaheuristic modelled on red fox hunting and territory:

1. **Global search (exploration).** Each fox moves toward the best-known
   position by a random fraction of the distance to it:
   ``x' = x + α · sign(best − x) · |best − x|`` with ``α ~ U(0, 1)``,
   applied per coordinate. A move is kept only if it improves fitness.
2. **Local search (traversing through the habitat).** With probability
   ``1 − μ`` (``μ ~ U(0, 1)``, the chance the fox is noticed and stays put) a
   fox circles its prey: a random observation angle ``φ₀ ∈ (0, 2π)`` gives a
   radius ``r = a · sin(φ₀)/φ₀`` (``a ~ U(0, 0.2)``), and a vector of random
   angles moves it on a hypersphere of that radius. Kept only if better.
3. **Reproduction and leaving the herd.** The worst ``replace_fraction`` of
   foxes are removed. The two best define a habitat: centre ``(b₁ + b₂)/2``,
   diameter ``‖b₁ − b₂‖``. With probability ``κ > 0.45`` a replacement fox is
   born anywhere in the search space (a fox leaving the herd); otherwise it is
   born inside the habitat around the centre.

Parameters live in the same normalised ``[0, 1]`` cube as the SFLA search
(``backend.methods.method1.optimization.sfla.SearchSpace``), so mixed
float/int/choice spaces need no special handling. Pure Python, seeded: the same
seed and objective always give the same evaluation sequence.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from backend.methods.method1.optimization.sfla import Parameter, SearchSpace

__all__ = ["Parameter", "SearchSpace", "RFOResult", "RedFoxOptimizer"]


@dataclass
class RFOResult:
    best_parameters: dict[str, Any]
    best_fitness: float
    best_position: list[float]
    objective: str
    seed: int
    population: int
    iterations: int
    replace_fraction: float
    evaluations: int
    duration_s: float
    search_space: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    evaluation_log: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "algorithm": "Red Fox Optimization (RFO)",
            "reference": "Połap & Woźniak (2021), Expert Systems with Applications 166:114107",
            "objective": self.objective,
            "seed": self.seed,
            "population": self.population,
            "iterations": self.iterations,
            "replace_fraction": self.replace_fraction,
            "evaluations": self.evaluations,
            "duration_s": round(self.duration_s, 3),
            "search_space": self.search_space,
            "best_parameters": self.best_parameters,
            "best_fitness": self.best_fitness,
            "best_position": self.best_position,
            "history": self.history,
            "evaluation_log": self.evaluation_log,
            "notes": self.notes,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


@dataclass
class _Fox:
    position: list[float]
    fitness: float = -math.inf
    params: dict[str, Any] = field(default_factory=dict)


class RedFoxOptimizer:
    """Maximising Red Fox Optimization over a :class:`SearchSpace`."""

    def __init__(
        self,
        space: SearchSpace,
        objective: Callable[[dict[str, Any]], float],
        *,
        population: int = 8,
        iterations: int = 5,
        replace_fraction: float = 0.25,
        seed: int = 42,
        objective_name: str = "objective",
    ):
        if population < 3:
            raise ValueError("population must be at least 3 (two parents plus one fox to replace).")
        if not 0 < replace_fraction < 1:
            raise ValueError("replace_fraction must be in (0, 1).")
        self.space = space
        self.objective = objective
        self.population = population
        self.iterations = iterations
        self.replace_fraction = replace_fraction
        self.seed = seed
        self.objective_name = objective_name
        self._rng = random.Random(seed)
        self._evaluations = 0
        self._log: list[dict] = []
        self._cache: dict[tuple, float] = {}

    # -- evaluation -------------------------------------------------------- #
    def _evaluate(self, fox: _Fox, phase: str) -> None:
        fox.position = [min(1.0, max(0.0, u)) for u in fox.position]
        params = self.space.decode(fox.position)
        key = tuple(sorted((k, repr(v)) for k, v in params.items()))
        if key in self._cache:
            fox.fitness, fox.params = self._cache[key], params
            return
        fitness = float(self.objective(params))
        self._cache[key] = fitness
        self._evaluations += 1
        fox.fitness, fox.params = fitness, params
        self._log.append({"evaluation": self._evaluations, "phase": phase,
                          "parameters": params, "fitness": fitness})

    def _random_fox(self) -> _Fox:
        return _Fox([self._rng.random() for _ in range(len(self.space))])

    # -- moves ------------------------------------------------------------- #
    def _global_move(self, fox: _Fox, best: _Fox) -> _Fox:
        alpha = self._rng.random()
        return _Fox([x + alpha * (b - x) for x, b in zip(fox.position, best.position)])

    def _local_move(self, fox: _Fox) -> _Fox:
        a = self._rng.uniform(0.0, 0.2)
        phi0 = self._rng.uniform(0.0, 2 * math.pi)
        r = a * math.sin(phi0) / phi0 if phi0 > 1e-9 else self._rng.random()
        n = len(fox.position)
        angles = [self._rng.uniform(0.0, 2 * math.pi) for _ in range(max(1, n - 1))]
        # Hyperspherical coordinates: x_k = r·(Π_{j<k} sin φ_j)·cos φ_k, last uses sin.
        delta, prod = [], 1.0
        for k in range(n):
            if k < n - 1:
                delta.append(r * prod * math.cos(angles[k]))
                prod *= math.sin(angles[k])
            else:
                delta.append(r * prod)
        return _Fox([x + d for x, d in zip(fox.position, delta)])

    # -- main loop --------------------------------------------------------- #
    def run(self) -> RFOResult:
        started = time.perf_counter()
        foxes = [self._random_fox() for _ in range(self.population)]
        for f in foxes:
            self._evaluate(f, "init")
        foxes.sort(key=lambda f: -f.fitness)
        history: list[dict] = []

        for iteration in range(1, self.iterations + 1):
            best = foxes[0]
            # 1. Global search.
            for i in range(1, len(foxes)):
                cand = self._global_move(foxes[i], best)
                self._evaluate(cand, "global")
                if cand.fitness > foxes[i].fitness:
                    foxes[i] = cand
            # 2. Local search (the fox is noticed and stays put with probability μ).
            for i in range(len(foxes)):
                if self._rng.random() > 0.75:
                    continue
                cand = self._local_move(foxes[i])
                self._evaluate(cand, "local")
                if cand.fitness > foxes[i].fitness:
                    foxes[i] = cand
            foxes.sort(key=lambda f: -f.fitness)
            # 3. Reproduction / leaving the herd.
            n_replace = max(1, int(round(self.replace_fraction * len(foxes))))
            b1, b2 = foxes[0], foxes[1]
            centre = [(p + q) / 2 for p, q in zip(b1.position, b2.position)]
            diameter = math.sqrt(sum((p - q) ** 2 for p, q in zip(b1.position, b2.position)))
            for i in range(len(foxes) - n_replace, len(foxes)):
                kappa = self._rng.random()
                if kappa > 0.45:
                    child = self._random_fox()
                    phase = "leave_herd"
                else:
                    child = _Fox([c + self._rng.uniform(-0.5, 0.5) * diameter for c in centre])
                    phase = "reproduce"
                self._evaluate(child, phase)
                foxes[i] = child
            foxes.sort(key=lambda f: -f.fitness)
            history.append({
                "iteration": iteration,
                "best_fitness": foxes[0].fitness,
                "mean_fitness": sum(f.fitness for f in foxes) / len(foxes),
                "worst_fitness": foxes[-1].fitness,
                "evaluations": self._evaluations,
            })

        best = foxes[0]
        return RFOResult(
            best_parameters=best.params or self.space.decode(best.position),
            best_fitness=best.fitness,
            best_position=list(best.position),
            objective=self.objective_name,
            seed=self.seed,
            population=self.population,
            iterations=self.iterations,
            replace_fraction=self.replace_fraction,
            evaluations=self._evaluations,
            duration_s=time.perf_counter() - started,
            search_space=self.space.to_dict(),
            history=history,
            evaluation_log=self._log,
        )
