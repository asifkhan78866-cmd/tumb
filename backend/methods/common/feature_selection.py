"""Metaheuristic feature selection over a fixed feature matrix, with an SVM wrapper.

Shared by the two methods whose architecture includes a selection stage: Red Fox
over ZFNet's features, and SFLA over the ConvLSTM's. Only the search space and
the objective live here — which metaheuristic walks that space is the caller's
choice, and both are already implemented in their own methods.

The architecture this implements is the classic wrapper scheme: a trained CNN
provides a fixed feature vector per image, a metaheuristic searches for the
subset of those features that classifies best, and a support vector machine is
the classifier whose accuracy the search is judged by.

* **What is searched** — one binary coordinate per feature (keep / drop) plus
  the SVM's own ``C``, ``gamma`` and kernel. Both metaheuristics already work in
  a normalised ``[0, 1]`` cube, so a feature is kept when its coordinate lands
  in the upper half of the cube: no separate binary machinery is needed.
* **What the fitness is** — macro-F1 on the *validation* split of an SVM fitted
  on the *training* split, minus a small penalty per feature kept. The penalty
  breaks ties towards the smaller subset; without it the search drifts towards
  keeping everything, which is not a selection at all.
* **What never enters it** — the test split. The objective below is given two
  matrices and neither of them can be the test set: the caller loads the test
  features only after the search has finished.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from backend.methods.method1.optimization.sfla import Parameter, SearchSpace

__all__ = ["SVM_PARAMETERS", "feature_space", "mask_of", "build_svm", "make_objective",
           "SelectionObjective", "fitted_pipeline", "load_selection_bundle"]

# The SVM's own hyper-parameters, searched alongside the feature mask.
SVM_PARAMETERS: tuple[Parameter, ...] = (
    Parameter("C", "float", 0.1, 100.0, log=True),
    Parameter("gamma", "float", 1e-4, 1e-1, log=True),
    Parameter("kernel", "choice", choices=("rbf", "linear")),
)


def feature_space(n_features: int, extra: Sequence[Parameter] = SVM_PARAMETERS) -> SearchSpace:
    """``n_features`` keep/drop coordinates followed by the SVM hyper-parameters."""
    if n_features < 1:
        raise ValueError("n_features must be at least 1.")
    width = len(str(n_features - 1))
    keep = [Parameter(f"f{i:0{width}d}", "choice", choices=(0, 1)) for i in range(n_features)]
    return SearchSpace([*keep, *extra])


def mask_of(params: dict[str, Any], n_features: int) -> np.ndarray:
    """The boolean keep-mask encoded in a decoded position."""
    width = len(str(n_features - 1))
    return np.array([bool(params[f"f{i:0{width}d}"]) for i in range(n_features)], dtype=bool)


def build_svm(params: dict[str, Any], *, seed: int = 42):
    """An SVM configured by a decoded position. ``gamma`` is ignored by ``linear``."""
    from sklearn.svm import SVC

    return SVC(
        C=float(params["C"]),
        gamma=float(params["gamma"]),
        kernel=str(params["kernel"]),
        class_weight="balanced",
        cache_size=500,
        random_state=seed,
    )


@dataclass
class SelectionObjective:
    """Validation-only objective for the feature search.

    ``calls`` and ``best`` are bookkeeping for the caller's report; the search
    itself only ever sees the returned float.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    sparsity_penalty: float = 0.02
    min_features: int = 8
    seed: int = 42
    calls: int = 0
    best: dict | None = None

    def __call__(self, params: dict[str, Any]) -> float:
        from sklearn.metrics import f1_score

        n = self.x_train.shape[1]
        mask = mask_of(params, n)
        kept = int(mask.sum())
        if kept < self.min_features:
            # Not a candidate: an empty or near-empty subset cannot be fitted, and
            # reporting a low score rather than raising keeps the search running.
            return 0.0
        svm = build_svm(params, seed=self.seed)
        svm.fit(self.x_train[:, mask], self.y_train)
        f1 = float(f1_score(self.y_val, svm.predict(self.x_val[:, mask]), average="macro"))
        fitness = f1 - self.sparsity_penalty * (kept / n)
        self.calls += 1
        if self.best is None or fitness > self.best["fitness"]:
            self.best = {"fitness": fitness, "val_macro_f1": f1, "features_kept": kept,
                         "C": float(params["C"]), "gamma": float(params["gamma"]),
                         "kernel": str(params["kernel"])}
        return fitness


def make_objective(x_train, y_train, x_val, y_val, **kwargs) -> SelectionObjective:
    return SelectionObjective(x_train, y_train, x_val, y_val, **kwargs)


def fitted_pipeline(params: dict[str, Any], mask: np.ndarray, x_train: np.ndarray,
                    y_train: np.ndarray, *, seed: int = 42, probability: bool = True):
    """The final classifier: the selected columns, fitted on the training split.

    An SVM has no natural probability output. When one is asked for, the fitted
    margins are turned into probabilities by Platt scaling, cross-fitted on the
    training split (``CalibratedClassifierCV``) so the sigmoid is not fitted on
    the same margins it calibrates.
    """
    svm = build_svm(params, seed=seed)
    if not probability:
        return svm.fit(x_train[:, mask], y_train)

    from sklearn.calibration import CalibratedClassifierCV

    calibrated = CalibratedClassifierCV(svm, method="sigmoid", cv=5, ensemble=False)
    return calibrated.fit(x_train[:, mask], y_train)


def load_selection_bundle(path, *, method_id: str, extractor_version: str,
                          class_names: Sequence[str], extractor_name: str,
                          rebuild_command: str) -> tuple[dict | None, list[str]]:
    """Load a selection artefact for serving, or explain why it is not used.

    The artefact is only meaningful for the exact network that produced the
    features it selected: its indices point into that network's feature vector.
    So a bundle is refused — loudly — when it was built from a different
    checkpoint, for different classes, or for another method, and silently
    ignored when training recorded that the network's own head scored higher on
    validation. A broken file never takes the method down with it.
    """
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return None, []
    try:
        import joblib

        bundle = joblib.load(path)
        if str(bundle.get("method_id")) != method_id:
            return None, [f"{path.name} belongs to {bundle.get('method_id')}, not {method_id}; "
                          f"it is ignored."]
        if str(bundle.get(f"{extractor_name}_model_version")) != str(extractor_version):
            return None, [
                f"The feature-selection SVM in {path.name} was built from "
                f"{extractor_name} {bundle.get(f'{extractor_name}_model_version')}, but the loaded "
                f"checkpoint is {extractor_version}; it is ignored and the network's own softmax "
                f"layer is used. Rebuild it with `{rebuild_command}`."
            ]
        if list(bundle.get("class_names") or []) != list(class_names):
            return None, [f"{path.name} was built for different classes; it is ignored."]
        if not bundle.get("serve"):
            return None, []  # built, but the softmax head scored higher on validation
        return bundle, []
    except Exception as exc:  # a broken artefact must not break serving
        return None, [f"The feature-selection SVM could not be loaded: {exc}"]
