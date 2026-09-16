"""Run the SFLA hyper-parameter search for Method 1's classifier.

    python -m backend.methods.method1.optimization.run_sfla --iterations 10

**Optimisation target.** Classifier hyper-parameters: learning rate, weight
decay, dropout, ConvLSTM hidden width, number of recurrent refinement steps,
convolution base width and batch size.

**Fitness.** Macro-F1 on the *validation* split after a short proxy training run
of ``--proxy-epochs`` epochs. Macro-F1 rather than accuracy because the class
balance is uneven and accuracy would reward ignoring the weakest class.

**The held-out test split is never loaded by this script.** Data is split three
ways with the same patient-grouped, seeded splitter the training scripts use,
and only the train and validation parts are opened. That is what makes the final
test number an honest estimate of a model whose hyper-parameters were tuned.

The result — seed, population, memeplexes, iterations, objective, full
evaluation log, convergence history and best parameter set — is written to
``backend/logs/method1_sfla_result.json``. Training picks it up automatically
when ``SFLA_ENABLED=true``.

With ``--dry-run`` the search runs against a cheap analytic objective instead of
training anything, which is how the reproducibility test exercises it.
"""
from __future__ import annotations

import argparse
import math
import time

import torch
from torch.utils.data import DataLoader

from backend import config
from backend.methods.common.runcard import RunCard, write_run_card
from backend.methods.common.splits import group_train_val_test_split, split_summary
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
    from backend.methods.method1.datasets import (
        ClassificationDataset,
        DatasetError,
        classification_group_of,
        discover_classification_samples,
    )
    from backend.methods.method1.models import ConvLSTMClassifier
    from backend.methods.method1.training.train_classifier import resolve_spec, run_epoch
    from backend.utils.metrics import classification_metrics  # noqa: F401  (used via run_epoch)

    warnings: list[str] = []
    samples, ds_meta = discover_classification_samples(args.data_root or m1.BRI_PATH)
    warnings.extend(ds_meta.get("warnings", []))
    spec, _ = resolve_spec(args.transform, warnings)

    train_s, val_s, _test_s = group_train_val_test_split(
        samples, classification_group_of, args.val_frac, args.test_frac, seed=config.SEED
    )
    # _test_s is deliberately discarded here: the optimiser must not be able to
    # read it even by accident.
    del _test_s
    summary = split_summary({"train": train_s, "val": val_s}, classification_group_of)
    print(f"[sfla] optimising on train={len(train_s)} val={len(val_s)} "
          f"(held-out test excluded) | leak-free={summary['leak_free']}")

    device = config.DEVICE

    def objective(params: dict) -> float:
        seed_everything(config.SFLA_SEED)  # same init for every candidate
        bs = int(params["batch_size"])
        train_loader = DataLoader(
            ClassificationDataset(train_s, spec, augment=True),
            batch_size=bs, shuffle=True, num_workers=args.workers, drop_last=True,
        )
        val_loader = DataLoader(
            ClassificationDataset(val_s, spec, augment=False),
            batch_size=bs, shuffle=False, num_workers=args.workers,
        )
        model = ConvLSTMClassifier(
            in_channels=1, num_classes=m1.NUM_CLASSES,
            base=int(params["base"]), lstm_hidden=int(params["lstm_hidden"]),
            lstm_steps=int(params["lstm_steps"]), dropout=float(params["dropout"]),
        ).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)

        for _ in range(args.proxy_epochs):
            run_epoch(model, train_loader, criterion, optimizer, scaler, True, device)
        _, val_m = run_epoch(model, val_loader, criterion, optimizer, scaler, False, device)
        print(f"[sfla]   f1={val_m['f1']:.4f} <- {params}")
        return float(val_m["f1"])

    return objective, spec, ds_meta, summary, warnings


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
    ap.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--transform", choices=["v1", "v2"], default="v2")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="optimise a cheap analytic objective instead of training (no data needed)")
    args = ap.parse_args()

    started = time.perf_counter()
    warnings: list[str] = []
    spec = ds_meta = summary = None

    if args.dry_run:
        objective = _dry_run_objective
        objective_name = "DRY RUN — analytic objective, no model was trained"
        warnings.append(
            "Dry run: the reported best parameters come from an analytic test "
            "objective, not from training. Do not quote them as a tuning result."
        )
    else:
        try:
            objective, spec, ds_meta, summary, warnings = build_training_objective(args)
        except Exception as exc:
            print(f"\nERROR: could not build the optimisation objective: {exc}")
            print("Hint: use --dry-run to exercise the optimiser without a dataset.")
            return 1
        objective_name = OBJECTIVE_NAME

    print(f"[sfla] population={args.population} memeplexes={args.memeplexes} "
          f"iterations={args.iterations} local={args.local_iterations} seed={args.seed}")

    optimiser = SFLA(
        SEARCH_SPACE, objective,
        population=args.population, memeplexes=args.memeplexes,
        iterations=args.iterations, local_iterations=args.local_iterations,
        max_step=args.max_step, seed=args.seed, objective_name=objective_name,
    )
    result = optimiser.run()
    result.notes = warnings + [
        "The held-out test split was never loaded during optimisation.",
        f"Fitness is maximised. Objective: {objective_name}",
    ]
    result.save(m1.SFLA_RESULT_PATH)

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
            dataset=ds_meta or {"note": "dry run — no dataset"},
            split_strategy=summary or {"note": "dry run"},
            random_seed=args.seed,
            preprocessing=spec.to_dict() if spec else {},
            image_size=spec.image_size if spec else 0,
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
