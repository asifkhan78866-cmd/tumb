"""7. SFLA reproducibility with a fixed seed."""
from __future__ import annotations

import math

import pytest

from backend.methods.method1.optimization.sfla import SFLA, Parameter, SearchSpace

SPACE = SearchSpace(
    [
        Parameter("lr", "float", 1e-4, 1e-1, log=True),
        Parameter("dropout", "float", 0.0, 0.9),
        Parameter("hidden", "choice", choices=(32, 64, 128)),
        Parameter("steps", "int", 1, 5),
    ]
)


def objective(params: dict) -> float:
    """Deterministic analytic surface with a known optimum."""
    return -(
        (math.log10(params["lr"]) + 2.0) ** 2
        + 4 * (params["dropout"] - 0.3) ** 2
        + 0.0005 * (params["hidden"] - 64) ** 2
        + 0.1 * (params["steps"] - 3) ** 2
    )


def _run(seed: int, **kwargs):
    return SFLA(
        SPACE, objective, population=16, memeplexes=4, iterations=4,
        local_iterations=3, seed=seed, **kwargs
    ).run()


def test_same_seed_reproduces_the_result_exactly():
    a, b = _run(42), _run(42)
    assert a.best_fitness == b.best_fitness
    assert a.best_parameters == b.best_parameters
    assert a.best_position == b.best_position
    assert a.evaluations == b.evaluations
    assert [h["best_fitness"] for h in a.history] == [h["best_fitness"] for h in b.history]


def test_same_seed_reproduces_the_whole_evaluation_sequence():
    a, b = _run(7), _run(7)
    seq_a = [(e["parameters"], e["fitness"]) for e in a.evaluation_log]
    seq_b = [(e["parameters"], e["fitness"]) for e in b.evaluation_log]
    assert seq_a == seq_b and len(seq_a) > 0


def test_different_seeds_explore_differently():
    assert _run(1).best_position != _run(2).best_position


def test_search_actually_improves_over_the_initial_population():
    result = _run(3)
    first = result.history[0]["best_fitness"]
    last = result.history[-1]["best_fitness"]
    assert last >= first
    # and it should beat the mean of the random starting population
    assert last > result.history[0]["iteration_mean"]


def test_best_fitness_is_monotonic_non_decreasing():
    history = [h["best_fitness"] for h in _run(11).history]
    assert all(b >= a for a, b in zip(history, history[1:]))


def test_result_records_every_required_setting():
    d = _run(5).to_dict()
    for key in ("seed", "population", "memeplexes", "frogs_per_memeplex", "iterations",
                "local_iterations", "objective", "best_parameters", "best_fitness",
                "search_space", "history", "evaluations"):
        assert key in d, f"missing {key}"
    assert d["population"] == 16 and d["memeplexes"] == 4 and d["frogs_per_memeplex"] == 4


def test_parameters_decode_within_their_declared_bounds():
    best = _run(13).best_parameters
    assert 1e-4 <= best["lr"] <= 1e-1
    assert 0.0 <= best["dropout"] <= 0.9
    assert best["hidden"] in (32, 64, 128)
    assert 1 <= best["steps"] <= 5 and isinstance(best["steps"], int)


def test_population_must_allow_local_search():
    with pytest.raises(ValueError, match="at least 2 frogs"):
        SFLA(SPACE, objective, population=4, memeplexes=4)


def test_result_is_json_serialisable(tmp_path):
    import json

    path = _run(4).save(tmp_path / "sfla.json")
    loaded = json.loads(path.read_text())
    assert loaded["seed"] == 4 and "best_parameters" in loaded
