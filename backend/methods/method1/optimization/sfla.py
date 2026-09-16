"""Shuffled Frog Leaping Algorithm (Eusuff & Lansey, 2003).

A real implementation, not a label. SFLA is a memetic population search:

1. Generate a population of *frogs*, each a point in the search space.
2. Evaluate fitness and sort the population best-to-worst.
3. Deal the sorted frogs round-robin into ``m`` *memeplexes*, so every memeplex
   gets a mix of strong and weak frogs.
4. Inside each memeplex, run local search for ``local_iterations`` steps. Each
   step moves the memeplex's **worst** frog toward the memeplex's best:
   ``X_w' = X_w + rand * (X_b - X_w)`` bounded by a maximum step. If that does
   not improve fitness, retry toward the **global** best. If that also fails,
   replace the worst frog with a fresh random one — this is what keeps SFLA from
   collapsing onto a local optimum.
5. Shuffle all memeplexes back together and repeat for ``iterations`` rounds.

The module is pure Python (stdlib ``random`` only) and the objective is injected
by the caller, which makes it independently testable: the same ``seed`` and the
same objective always produce the same evaluation sequence and the same result,
with no torch, no GPU and no dataset involved.

Search-space parameters may be continuous (``float``), integral (``int``) or
categorical (an explicit list of choices); the leaping rule operates in a
normalised [0, 1] cube and each parameter decodes its own coordinate, so mixed
spaces need no special handling from the caller.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

__all__ = ["Parameter", "SearchSpace", "SFLAResult", "SFLA"]


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Parameter:
    """One dimension of the search space.

    ``kind``:
        * ``"float"``  — continuous in ``[low, high]`` (optionally ``log`` scaled)
        * ``"int"``    — integer in ``[low, high]``
        * ``"choice"`` — one of ``choices``
    """

    name: str
    kind: str = "float"
    low: float = 0.0
    high: float = 1.0
    log: bool = False
    choices: tuple[Any, ...] = ()

    def decode(self, u: float) -> Any:
        """Map a coordinate in [0, 1] to a concrete parameter value."""
        u = min(1.0, max(0.0, float(u)))
        if self.kind == "choice":
            if not self.choices:
                raise ValueError(f"Parameter {self.name!r} is a choice with no choices.")
            idx = min(int(u * len(self.choices)), len(self.choices) - 1)
            return self.choices[idx]
        if self.kind == "int":
            return int(round(self.low + u * (self.high - self.low)))
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return float(math.exp(lo + u * (hi - lo)))
        return float(self.low + u * (self.high - self.low))

    def to_dict(self) -> dict:
        d = {"name": self.name, "kind": self.kind}
        if self.kind == "choice":
            d["choices"] = list(self.choices)
        else:
            d.update({"low": self.low, "high": self.high, "log": self.log})
        return d


class SearchSpace:
    """An ordered collection of :class:`Parameter` objects."""

    def __init__(self, parameters: Sequence[Parameter]):
        if not parameters:
            raise ValueError("Search space must contain at least one parameter.")
        self.parameters = tuple(parameters)

    def __len__(self) -> int:
        return len(self.parameters)

    def decode(self, position: Sequence[float]) -> dict[str, Any]:
        return {p.name: p.decode(u) for p, u in zip(self.parameters, position)}

    def to_dict(self) -> list[dict]:
        return [p.to_dict() for p in self.parameters]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class SFLAResult:
    best_parameters: dict[str, Any]
    best_fitness: float
    best_position: list[float]
    objective: str
    seed: int
    population: int
    memeplexes: int
    frogs_per_memeplex: int
    iterations: int
    local_iterations: int
    max_step: float
    evaluations: int
    duration_s: float
    search_space: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    evaluation_log: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "algorithm": "Shuffled Frog Leaping Algorithm (SFLA)",
            "reference": "Eusuff & Lansey (2003), Journal of Water Resources Planning and Management",
            "objective": self.objective,
            "seed": self.seed,
            "population": self.population,
            "memeplexes": self.memeplexes,
            "frogs_per_memeplex": self.frogs_per_memeplex,
            "iterations": self.iterations,
            "local_iterations": self.local_iterations,
            "max_step": self.max_step,
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


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #
@dataclass
class _Frog:
    position: list[float]
    fitness: float = -math.inf
    params: dict[str, Any] = field(default_factory=dict)


class SFLA:
    """Maximising Shuffled Frog Leaping optimiser.

    Parameters
    ----------
    space : the search space.
    objective : ``(params: dict) -> float``; higher is better. It must never read
        the held-out test set — see ``run_sfla.py`` for how the split is enforced.
    population : total number of frogs (rounded up to memeplexes * per-memeplex).
    memeplexes : number of memeplexes the population is dealt into.
    iterations : number of global shuffle rounds.
    local_iterations : local-search steps inside each memeplex per round.
    max_step : maximum movement per coordinate in the normalised cube.
    seed : makes the whole run reproducible.
    """

    def __init__(
        self,
        space: SearchSpace,
        objective: Callable[[dict[str, Any]], float],
        *,
        population: int = 20,
        memeplexes: int = 4,
        iterations: int = 10,
        local_iterations: int = 5,
        max_step: float = 0.5,
        seed: int = 42,
        objective_name: str = "objective",
        log_evaluations: bool = True,
    ):
        if memeplexes < 1:
            raise ValueError("memeplexes must be >= 1")
        if population < memeplexes * 2:
            raise ValueError(
                f"population ({population}) must be at least 2 frogs per memeplex "
                f"({memeplexes * 2}); local search needs a best and a worst frog."
            )
        self.space = space
        self.objective = objective
        self.memeplexes = memeplexes
        self.frogs_per_memeplex = population // memeplexes
        self.population = self.frogs_per_memeplex * memeplexes
        self.iterations = iterations
        self.local_iterations = local_iterations
        self.max_step = max_step
        self.seed = seed
        self.objective_name = objective_name
        self.log_evaluations = log_evaluations

        self._rng = random.Random(seed)
        self._evaluations = 0
        self._eval_log: list[dict] = []
        self._cache: dict[tuple, float] = {}

    # -- evaluation -------------------------------------------------------- #
    def _evaluate(self, frog: _Frog) -> None:
        params = self.space.decode(frog.position)
        key = tuple(sorted((k, repr(v)) for k, v in params.items()))
        if key in self._cache:
            frog.fitness, frog.params = self._cache[key], params
            return
        fitness = float(self.objective(params))
        self._cache[key] = fitness
        self._evaluations += 1
        frog.fitness, frog.params = fitness, params
        if self.log_evaluations:
            self._eval_log.append(
                {"evaluation": self._evaluations, "parameters": params, "fitness": fitness}
            )

    def _random_frog(self) -> _Frog:
        return _Frog([self._rng.random() for _ in range(len(self.space))])

    # -- leaping ----------------------------------------------------------- #
    def _leap(self, worst: _Frog, target: _Frog) -> _Frog:
        """Move ``worst`` toward ``target`` by a random bounded step."""
        position = []
        for w, t in zip(worst.position, target.position):
            step = self._rng.random() * (t - w)
            step = max(-self.max_step, min(self.max_step, step))
            position.append(min(1.0, max(0.0, w + step)))
        return _Frog(position)

    # -- main loop --------------------------------------------------------- #
    def run(self) -> SFLAResult:
        started = time.perf_counter()

        frogs = [self._random_frog() for _ in range(self.population)]
        for f in frogs:
            self._evaluate(f)
        frogs.sort(key=lambda f: -f.fitness)
        best = frogs[0]
        history: list[dict] = []

        for iteration in range(1, self.iterations + 1):
            # Deal round-robin: memeplex j receives frogs j, j+m, j+2m, ...
            memeplexes = [frogs[j :: self.memeplexes] for j in range(self.memeplexes)]

            for memeplex in memeplexes:
                for _ in range(self.local_iterations):
                    memeplex.sort(key=lambda f: -f.fitness)
                    local_best, local_worst = memeplex[0], memeplex[-1]

                    candidate = self._leap(local_worst, local_best)
                    self._evaluate(candidate)

                    if candidate.fitness <= local_worst.fitness:
                        # Step 2: try leaping toward the global best instead.
                        candidate = self._leap(local_worst, best)
                        self._evaluate(candidate)

                    if candidate.fitness <= local_worst.fitness:
                        # Step 3: censorship — replace with a random frog.
                        candidate = self._random_frog()
                        self._evaluate(candidate)

                    memeplex[-1] = candidate

            frogs = [f for memeplex in memeplexes for f in memeplex]
            frogs.sort(key=lambda f: -f.fitness)
            if frogs[0].fitness > best.fitness:
                best = frogs[0]

            history.append(
                {
                    "iteration": iteration,
                    "best_fitness": best.fitness,
                    "iteration_best": frogs[0].fitness,
                    "iteration_mean": sum(f.fitness for f in frogs) / len(frogs),
                    "iteration_worst": frogs[-1].fitness,
                    "evaluations": self._evaluations,
                }
            )

        return SFLAResult(
            best_parameters=best.params or self.space.decode(best.position),
            best_fitness=best.fitness,
            best_position=list(best.position),
            objective=self.objective_name,
            seed=self.seed,
            population=self.population,
            memeplexes=self.memeplexes,
            frogs_per_memeplex=self.frogs_per_memeplex,
            iterations=self.iterations,
            local_iterations=self.local_iterations,
            max_step=self.max_step,
            evaluations=self._evaluations,
            duration_s=time.perf_counter() - started,
            search_space=self.space.to_dict(),
            history=history,
            evaluation_log=self._eval_log,
        )
