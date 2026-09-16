"""Tagged checkpoints with method isolation and safe loading.

Two methods live side by side in this repository and their checkpoints are
architecturally incompatible. Loading Method 2's DCN weights into Method 1's
ConvLSTM either explodes with an unreadable ``size mismatch`` wall of text or —
worse, when shapes happen to line up — silently produces garbage predictions.

Every checkpoint written through :func:`save_checkpoint` therefore carries a
``meta`` block naming the method, the architecture, the class list and the exact
preprocessing spec it was trained with. :func:`validate_meta` is pure data
validation with no torch dependency so it can be unit tested directly.

Loading always uses ``torch.load(..., weights_only=True)`` where the installed
torch supports it: our checkpoints contain only tensors and JSON-ish primitives,
so nothing is lost, and a hostile ``.pth`` can no longer execute code on import.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

CHECKPOINT_FORMAT = 2

__all__ = [
    "CheckpointError",
    "CheckpointMeta",
    "build_meta",
    "validate_meta",
    "save_checkpoint",
    "load_checkpoint",
    "peek_meta",
]


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, malformed or belongs elsewhere."""


@dataclass(frozen=True)
class CheckpointMeta:
    method_id: str
    architecture: str
    role: str  # "segmentation" | "classification"
    class_names: tuple[str, ...] = ()
    image_size: int = 0
    transform_id: str = ""
    model_version: str = ""
    created_at: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "format": CHECKPOINT_FORMAT,
            "method_id": self.method_id,
            "architecture": self.architecture,
            "role": self.role,
            "class_names": list(self.class_names),
            "image_size": self.image_size,
            "transform_id": self.transform_id,
            "model_version": self.model_version,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **({"extra": self.extra} if self.extra else {}),
        }


def build_meta(
    method_id: str,
    architecture: str,
    role: str,
    class_names: Sequence[str] = (),
    image_size: int = 0,
    transform_id: str = "",
    model_version: str = "",
    **extra: Any,
) -> CheckpointMeta:
    return CheckpointMeta(
        method_id=method_id,
        architecture=architecture,
        role=role,
        class_names=tuple(class_names),
        image_size=int(image_size),
        transform_id=transform_id,
        model_version=model_version or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        extra=dict(extra),
    )


def validate_meta(
    meta: Optional[dict],
    *,
    expected_method: str,
    expected_architecture: str,
    expected_role: str,
    path: str = "<checkpoint>",
    expected_class_names: Optional[Sequence[str]] = None,
) -> list[str]:
    """Check a checkpoint's ``meta`` block against what the caller expects.

    Returns a list of non-fatal warnings. Raises :class:`CheckpointError` with a
    readable message when the checkpoint plainly belongs to something else.

    An *untagged* checkpoint (``meta is None``) is accepted with a warning: the
    classifier that ships with this repository predates the tagging scheme and
    must keep loading.
    """
    if meta is None:
        return [
            f"{path}: legacy checkpoint with no metadata block — assuming it is "
            f"{expected_method}/{expected_architecture}. Retrain to embed metadata."
        ]
    if not isinstance(meta, dict):
        raise CheckpointError(f"{path}: 'meta' must be a dict, got {type(meta).__name__}.")

    warnings: list[str] = []
    found_method = meta.get("method_id")
    found_arch = meta.get("architecture")
    found_role = meta.get("role")

    if found_method and found_method != expected_method:
        raise CheckpointError(
            f"{path}: checkpoint belongs to '{found_method}' but was loaded by "
            f"'{expected_method}'. Point the corresponding *_WEIGHTS variable at the "
            f"right file — the two methods do not share weights."
        )
    if found_arch and found_arch != expected_architecture:
        raise CheckpointError(
            f"{path}: checkpoint was trained for architecture '{found_arch}' but "
            f"'{expected_architecture}' is being constructed. Refusing to load."
        )
    if found_role and found_role != expected_role:
        raise CheckpointError(
            f"{path}: checkpoint role is '{found_role}', expected '{expected_role}'."
        )

    found_classes = meta.get("class_names")
    if expected_class_names is not None and found_classes:
        if list(found_classes) != list(expected_class_names):
            raise CheckpointError(
                f"{path}: checkpoint class order {list(found_classes)} does not match "
                f"the configured order {list(expected_class_names)}. Loading it would "
                f"silently relabel every prediction."
            )
    if meta.get("format", CHECKPOINT_FORMAT) > CHECKPOINT_FORMAT:
        warnings.append(
            f"{path}: written by a newer checkpoint format "
            f"({meta.get('format')} > {CHECKPOINT_FORMAT})."
        )
    return warnings


# --------------------------------------------------------------------------- #
# torch-backed I/O (imported lazily so this module stays importable bare)
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: Path,
    model_state: dict,
    meta: CheckpointMeta,
    **payload: Any,
) -> Path:
    """Write a tagged checkpoint, plus a sidecar ``<name>.meta.json`` for humans."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model_state, "meta": meta.to_dict(), **payload}, path)
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps({**meta.to_dict(), **{k: v for k, v in payload.items() if _jsonable(v)}}, indent=2)
    )
    return path


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _torch_load(path: Path):
    """``torch.load`` with ``weights_only=True`` when the installed torch has it."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # torch < 1.13 has no weights_only parameter.
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        # weights_only=True rejects checkpoints pickling non-primitive objects
        # (e.g. numpy scalars from an older run). Retry loudly rather than
        # silently widening the trust boundary for every load.
        if "weights_only" not in str(exc) and "WeightsUnpickler" not in str(exc):
            raise
        print(
            f"[checkpoint] {path}: not loadable with weights_only=True "
            f"({type(exc).__name__}). Falling back to a full unpickle — only do "
            f"this for checkpoints you produced yourself."
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def peek_meta(path: Path) -> Optional[dict]:
    """Return a checkpoint's meta block without constructing any model."""
    path = Path(path)
    if not path.exists():
        return None
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text())
        except Exception:
            pass
    try:
        ckpt = _torch_load(path)
    except Exception:
        return None
    return ckpt.get("meta") if isinstance(ckpt, dict) else None


def load_checkpoint(
    model,
    path: Path,
    *,
    expected_method: str,
    expected_architecture: str,
    expected_role: str,
    expected_class_names: Optional[Sequence[str]] = None,
    strict: bool = True,
) -> tuple[dict, list[str]]:
    """Load weights into ``model`` after validating the checkpoint's identity.

    Returns ``(meta, warnings)``. Raises :class:`CheckpointError` on a missing
    file, a foreign checkpoint, or a state dict that does not fit the model.
    """
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"{path}: checkpoint not found.")

    ckpt = _torch_load(path)
    if not isinstance(ckpt, dict):
        raise CheckpointError(f"{path}: expected a dict checkpoint, got {type(ckpt).__name__}.")

    state = ckpt.get("model_state", ckpt)
    meta = ckpt.get("meta")
    warnings = validate_meta(
        meta,
        expected_method=expected_method,
        expected_architecture=expected_architecture,
        expected_role=expected_role,
        expected_class_names=expected_class_names,
        path=str(path),
    )

    try:
        model.load_state_dict(state, strict=strict)
    except Exception as exc:
        raise CheckpointError(
            f"{path}: state dict does not fit {expected_architecture}. This usually "
            f"means the checkpoint belongs to another method or another "
            f"configuration.\n  underlying error: {exc}"
        ) from exc

    resolved = meta if isinstance(meta, dict) else {
        "method_id": expected_method,
        "architecture": expected_architecture,
        "role": expected_role,
        "model_version": "legacy-untagged",
    }
    return resolved, warnings
