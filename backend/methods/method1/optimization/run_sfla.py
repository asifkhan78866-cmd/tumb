"""Run the SFLA hyper-parameter search for Method 1's classifier.

    python -m backend.methods.method1.optimization.run_sfla --population 12 --iterations 10

**Optimisation target.** Classifier hyper-parameters: learning rate, weight
decay, dropout, ConvLSTM hidden width, number of recurrent refinement steps,
convolution base width and batch size.

**Fitness.** Macro-F1 on the *validation* split after a short proxy training run
of ``--proxy-epochs`` epochs. Macro-F1 rather than accuracy because accuracy
would reward ignoring the weakest class.

**The test set is never loaded by this script.** Data is prepared by the same
function the training script uses (``prepare_data(load_test=False)``), so the
validation split SFLA optimises is exactly the one final training early-stops on,
and the dataset's ``Testing/`` folder is neither decoded nor preprocessed.

The result — seed, population, memeplexes, iterations, objective, search space,
full evaluation log, convergence history and best parameter set — is written to
``backend/logs/method1/sfla_results.json``. Training picks it up automatically
when ``SFLA_ENABLED=true``.

With ``--dry-run`` the search runs against a cheap analytic objective instead of
training anything, which is how the reproducibility test exercises it.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone

import torch

from backend import config
from backend.methods.common.runcard import RunCard, write_run_card
from backend.methods.method1 import config as m1
from backend.methods.method1.optimization.sfla import SFLA, Parameter, SearchSpace
from backend.training.common import seed_everything

SEARCH_SPACE = SearchSpace(
    [
        Parameter("lr", "float", 1e-4, 1e-2, log=True),
        Parameter("weight_decay", "float", 1e-6, 1e-3, log=True),
        Parameter("dropout", "float", 0.1, 0.6),
        Parameter("lstm_hidden", "choice", choices=(32, 64, 96, 128)),
        Parameter("lstm_steps", "int", 1, 5),
        Parameter("base", "choice", choices=(16, 32, 48)),
        Parameter("batch_size", "choice", choices=(8, 16, 32)),
    ]
)

OBJECTIVE_NAME = "validation macro-F1 after a short proxy training run (test split untouched)"


def _dry_run_objective(params: dict) -> float:
    """Cheap analytic stand-in used by ``--dry-run`` and the reproducibility test.

    Smooth, deterministic, and shaped like a plausible hyper-parameter response
    surface so the search has something real to climb.
    """
    lr_term = -((math.log10(params["lr"]) + 3.0) ** 2)
    wd_term = -0.25 * ((math.log10(params["weight_decay"]) + 4.5) ** 2)
    drop_term = -4.0 * (params["dropout"] - 0.35) ** 2
    hidden_term = -0.0002 * (params["lstm_hidden"] - 64) ** 2
    steps_term = -0.05 * (params["lstm_steps"] - 3) ** 2
    return float(lr_term + wd_term + drop_term + hidden_term + steps_term)


def build_training_objective(args):
    """Real objective: train briefly on `train`, score macro-F1 on `val`."""
    from backend.methods.method1.training.train_classifier import (
        build_criterion,
        build_model,
        make_loader,
        prepare_data,
        run_epoch,
    )

    data = prepare_data(args, load_test=False)
    assert "test" not in data.arrays, "SFLA must never load the test set"
    print(f"[sfla] optimising on train={len(data.train)} val={len(data.val)} "
          f"(test set not loaded) | leak-free={data.summary['leak_free']}")

    device = config.DEVICE
    train_ds = data.dataset("train", augment=True)
    val_ds = data.dataset("val")

    def objective(params: dict) -> float:
        seed_everything(config.SFLA_SEED)  # same init and data order for every candidate
        t0 = time.perf_counter()
        bs = int(params["batch_size"])
        train_loader = make_loader(train_ds, bs, True, args.workers)
        val_loader = make_loader(val_ds, bs, False, args.workers)
        model = build_model(params, device)
        criterion, _ = build_criterion(data.train, device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)

        for _ in range(args.proxy_epochs):
            run_epoch(model, train_loader, criterion, optimizer, scaler, True, device)
        _, val_m = run_epoch(model, val_loader, criterion, optimizer, scaler, False, device)
        f1 = float(val_m["f1"])
        if not math.isfinite(f1):
            f1 = 0.0
        print(f"[sfla]   f1={f1:.4f} acc={val_m['accuracy']:.4f} "
              f"({time.perf_counter() - t0:.0f}s) <- {params}", flush=True)
        return f1

    return objective, data


def main() -> int:
    ap = argparse.ArgumentParser(description="SFLA hyper-parameter search for Method 1")
    ap.add_argument("--population", type=int, default=config.SFLA_POPULATION)
    ap.add_argument("--memeplexes", type=int, default=config.SFLA_MEMEPLEXES)
    ap.add_argument("--iterations", type=int, default=config.SFLA_ITERATIONS)
    ap.add_argument("--local-iterations", type=int, default=config.SFLA_LOCAL_ITERATIONS)
    ap.add_argument("--max-step", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=config.SFLA_SEED)
    ap.add_argument("--proxy-epochs", type=int, default=3,
                    help="epochs per fitness evaluation (keep small; this runs many times)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--transform", choices=["v1", "v2"], default="v2")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ignore-official-split", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="optimise a cheap analytic objective instead of training (no data needed)")
    args = ap.parse_args()

    started = time.perf_counter()
    warnings: list[str] = []
    data = None

    if args.dry_run:
        objective = _dry_run_objective
        objective_name = "DRY RUN — analytic objective, no model was trained"
        warnings.append(
            "Dry run: the reported best parameters come from an analytic test "
            "objective, not from training. Do not quote them as a tuning result."
        )
    else:
        try:
            objective, data = build_training_objective(args)
        except Exception as exc:
            print(f"\nERROR: could not build the optimisation objective: {exc}")
            print("Hint: use --dry-run to exercise the optimiser without a dataset.")
            return 1
        warnings = list(data.warnings)
        objective_name = OBJECTIVE_NAME

    print(f"[sfla] device={config.DEVICE} population={args.population} "
          f"memeplexes={args.memeplexes} iterations={args.iterations} "
          f"local={args.local_iterations} proxy_epochs={args.proxy_epochs} seed={args.seed}")

    optimiser = SFLA(
        SEARCH_SPACE, objective,
        population=args.population, memeplexes=args.memeplexes,
        iterations=args.iterations, local_iterations=args.local_iterations,
        max_step=args.max_step, seed=args.seed, objective_name=objective_name,
    )
    result = optimiser.run()
    result.notes = warnings + [
        "The test set was never loaded during optimisation.",
        f"Fitness is maximised. Objective: {objective_name}",
    ]
    result.save(m1.SFLA_RESULT_PATH)

    # Context the SFLAResult itself does not carry.
    saved = json.loads(m1.SFLA_RESULT_PATH.read_text())
    saved.update({
        "best_validation_score": result.best_fitness,
        "fitness_history": [h["best_fitness"] for h in result.history],
        "proxy_epochs": args.proxy_epochs,
        "device": str(config.DEVICE),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "path": str(data.ds_meta["root"]) if data else None,
            "train": len(data.train) if data else None,
            "validation": len(data.val) if data else None,
            "preprocessing": data.spec.to_dict() if data else None,
        },
    })
    m1.SFLA_RESULT_PATH.write_text(json.dumps(saved, indent=2, default=str))

    print(f"\n[sfla] best fitness {result.best_fitness:.4f} after "
          f"{result.evaluations} evaluations in {result.duration_s:.1f}s")
    for k, v in result.best_parameters.items():
        print(f"[sfla]   {k:14s} = {v}")
    print(f"[sfla] result -> {m1.SFLA_RESULT_PATH}")
    if not config.SFLA_ENABLED:
        print("[sfla] NOTE: SFLA_ENABLED is false, so training will ignore these "
              "parameters. Set SFLA_ENABLED=true in .env to apply them.")

    write_run_card(
        m1.run_card_path("sfla"),
        RunCard(
            method_id=m1.METHOD_ID, stage="sfla",
            dataset=data.ds_meta if data else {"note": "dry run — no dataset"},
            split_strategy=data.summary if data else {"note": "dry run"},
            random_seed=args.seed,
            preprocessing=data.spec.to_dict() if data else {},
            image_size=data.spec.image_size if data else 0,
            architecture=m1.CLS_ARCHITECTURE,
            model_config={"search_space": SEARCH_SPACE.to_dict()},
            optimizer={"name": "SFLA", "population": result.population,
                       "memeplexes": result.memeplexes, "iterations": result.iterations,
                       "local_iterations": result.local_iterations, "max_step": result.max_step},
            epochs=args.proxy_epochs,
            metrics={"best_fitness": result.best_fitness,
                     "best_parameters": result.best_parameters,
                     "evaluations": result.evaluations},
            checkpoint_path=str(m1.SFLA_RESULT_PATH),
            train_duration_s=round(time.perf_counter() - started, 1),
            device=str(config.DEVICE), warnings=result.notes,
        ),
        root=config.ROOT_DIR,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
