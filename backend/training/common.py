"""Shared training utilities: seeding, early stopping, split helpers."""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    ``mode='max'`` for metrics like Dice/accuracy, ``mode='min'`` for loss.
    """

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Return True if this is a new best value."""
        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def train_val_split(samples, val_frac: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    val_idx = set(idx[:n_val])
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val
