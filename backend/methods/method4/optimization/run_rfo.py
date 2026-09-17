"""Red Fox Optimization of ZFNet hyper-parameters.

    python -m backend.methods.method4.optimization.run_rfo --population 8 --iterations 4

**Fitness** is validation macro-F1 after a short proxy training run of
``--proxy-epochs`` epochs from the same seed. **The test set is never loaded**:
data comes from ``load_split_tensors(load_test=False)``.

Writes ``backend/logs/method4/rfo_results.json`` (seed, population, iterations,
search space, every evaluation, convergence history, best parameters), which
``method4.training.train_classifier`` then uses. ``--dry-run`` optimises an
analytic objective instead, for tests and quick checks.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone

import torch

from backend import config
from backend.methods.common.cnn_training import predict_probs, train_epoch
from backend.methods.common.runcard import RunCard, write_run_card
from backend.methods.method4.optimization.rfo import Parameter, RedFoxOptimizer, SearchSpace
from backend.methods.registry import METHOD4
from backend.training.common import seed_everything
from backend.utils.metrics import classification_metrics

SEARCH_SPACE = SearchSpace([
    Parameter("lr", "float", 3e-5, 3e-3, log=True),
    Parameter("weight_decay", "float", 1e-5, 5e-3, log=True),
    Parameter("dropout", "float", 0.2, 0.7),
    Parameter("fc_units", "choice", choices=(1024, 2048, 4096)),
    Parameter("batch_size", "choice", choices=(32, 64)),
])
OBJECTIVE_NAME = "validation macro-F1 after a short proxy training run (test split untouched)"


def _dry_run_objective(p: dict) -> float:
    return float(-((math.log10(p["lr"]) + 3.3) ** 2) - 3 * (p["dropout"] - 0.45) ** 2
                 - 0.2 * (math.log10(p["weight_decay"]) + 3.5) ** 2 - 1e-8 * (p["fc_units"] - 2048) ** 2)


def build_objective(args):
    from backend.methods.method4.training.train_classifier import (
        CLASS_NAMES, build_model, load_split_tensors, make_criterion, make_optimizer,
    )
    from backend.methods.method4.transforms import to_input

    split, tensors = load_split_tensors(args.data_root, args.val_frac, load_test=False)
    assert "x_test" not in tensors
    device = config.DEVICE
    y_val = tensors["y_val"].numpy().tolist()
    print(f"[rfo] optimising on train={len(split.train)} val={len(split.val)} (test set not loaded)")

    def objective(params: dict) -> float:
        seed_everything(config.SEED)
        t0 = time.perf_counter()
        model = build_model(params, device)
        opt = make_optimizer(model, params)
        crit = make_criterion(split.train, device)
        gen = torch.Generator().manual_seed(config.SEED)
        for _ in range(args.proxy_epochs):
            train_epoch(model, tensors["x_train"], tensors["y_train"], int(params["batch_size"]),
                        opt, crit, device, to_input, gen)
        probs, _, _ = predict_probs(model, tensors["x_val"], 128, device, to_input)
        m = classification_metrics(y_val, probs.argmax(1).tolist(), len(CLASS_NAMES))
        f1 = m["f1"] if math.isfinite(m["f1"]) else 0.0
        print(f"[rfo]   f1={f1:.4f} acc={m['accuracy']:.4f} ({time.perf_counter() - t0:.0f}s) <- {params}",
              flush=True)
        del model, opt
        if device.type == "mps":
            torch.mps.empty_cache()
        return float(f1)

    return objective, split


def main() -> int:
    ap = argparse.ArgumentParser(description="Red Fox Optimization for ZFNet")
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--replace-fraction", type=float, default=0.25)
    ap.add_argument("--proxy-epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    split = None
    if args.dry_run:
        objective, objective_name = _dry_run_objective, "DRY RUN — analytic objective, no model was trained"
    else:
        objective, split = build_objective(args)
        objective_name = OBJECTIVE_NAME
    print(f"[rfo] device={config.DEVICE} population={args.population} iterations={args.iterations} "
          f"proxy_epochs={args.proxy_epochs} seed={args.seed}")

    started = time.perf_counter()
    result = RedFoxOptimizer(SEARCH_SPACE, objective, population=args.population,
                             iterations=args.iterations, replace_fraction=args.replace_fraction,
                             seed=args.seed, objective_name=objective_name).run()
    result.notes = (list(split.warnings) if split else []) + [
        "The test set was never loaded during optimisation.",
        f"Fitness is maximised. Objective: {objective_name}",
    ]
    out = config.LOGS_DIR / "method4" / ("rfo_dry_run.json" if args.dry_run else "rfo_results.json")
    result.save(out)
    saved = json.loads(out.read_text())
    saved.update({
        "best_validation_score": result.best_fitness,
        "fitness_history": [h["best_fitness"] for h in result.history],
        "proxy_epochs": args.proxy_epochs,
        "device": str(config.DEVICE),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"path": str(split.ds_meta["root"]), "train": len(split.train),
                    "validation": len(split.val), "split": split.split_kind} if split else None,
    })
    out.write_text(json.dumps(saved, indent=2, default=str))

    print(f"\n[rfo] best fitness {result.best_fitness:.4f} after {result.evaluations} evaluations "
          f"in {result.duration_s:.0f}s")
    for k, v in result.best_parameters.items():
        print(f"[rfo]   {k:14s} = {v}")
    print(f"[rfo] result -> {out}")

    if not args.dry_run:
        write_run_card(
            config.LOGS_DIR / METHOD4.run_card_filenames["rfo"],
            RunCard(method_id=METHOD4.method_id, stage="rfo", dataset=split.ds_meta,
                    split_strategy=split.summary, random_seed=args.seed,
                    architecture="ZFNet", model_config={"search_space": SEARCH_SPACE.to_dict()},
                    optimizer={"name": "Red Fox Optimization", "population": args.population,
                               "iterations": args.iterations, "replace_fraction": args.replace_fraction},
                    epochs=args.proxy_epochs,
                    metrics={"best_fitness": result.best_fitness, "best_parameters": result.best_parameters,
                             "evaluations": result.evaluations},
                    checkpoint_path=str(out), train_duration_s=round(time.perf_counter() - started, 1),
                    device=str(config.DEVICE), warnings=result.notes),
            root=config.ROOT_DIR,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
