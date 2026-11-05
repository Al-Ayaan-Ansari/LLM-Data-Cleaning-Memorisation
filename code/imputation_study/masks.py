"""Decide which cells to hide.

A mask is a True/False grid the same shape as the table; True = hidden.

    MCAR       every cell is equally likely to be hidden
    MAR        whether a cell is hidden depends on *other* values in its row:
               a few "driver" columns (never hidden themselves) decide it
    MNAR       whether a cell is hidden depends on *its own* value: larger
               values are hidden more often, like high earners not reporting
               their income
    PIMA_REAL  copy the real missing-value patterns of the Pima dataset

`rate` is the share of all cells that ends up hidden. Every mechanism hides
an exact number of cells, so a 20% mask really is 20% - minus the odd cell
revealed again because its row had every cell hidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MECHANISMS = ("MCAR", "MAR", "MNAR", "PIMA_REAL")


def make_mask(data: pd.DataFrame, mechanism: str, rate: float | None, seed: int,
              categorical=(), patterns: np.ndarray | None = None,
              strength: float = 2.0) -> np.ndarray:
    """`strength`: for MAR and MNAR, a cell whose deciding value is one standard
    deviation higher is about e**strength (~7x) more likely to be hidden."""
    rng = np.random.default_rng(seed)
    z = _standardised(data, categorical)
    n, d = z.shape
    mask = np.zeros((n, d), dtype=bool)

    if mechanism == "MCAR":
        mask.flat[rng.choice(n * d, size=round(rate * n * d), replace=False)] = True
    elif mechanism == "MAR":
        drivers = rng.choice(d, size=max(1, round(0.3 * d)), replace=False)
        others = [j for j in range(d) if j not in drivers]
        col_rate = min(rate * d / len(others), 0.9)
        for j in others:
            score = z[:, drivers] @ rng.normal(size=len(drivers))
            mask[:, j] = _pick(_z(score), col_rate, strength, rng)
    elif mechanism == "MNAR":
        for j in range(d):
            mask[:, j] = _pick(z[:, j], rate, strength, rng)
    elif mechanism == "PIMA_REAL":
        mask = patterns[rng.integers(len(patterns), size=n)].copy()
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")

    # a row with every cell hidden gives an imputer nothing to go on: reveal one cell
    for i in np.flatnonzero(mask.all(axis=1)):
        mask[i, rng.integers(d)] = False
    return mask


def _pick(z: np.ndarray, rate: float, strength: float, rng) -> np.ndarray:
    """Choose exactly round(rate * n) entries, each with probability proportional
    to exp(strength * z), without replacement (the Gumbel top-k trick)."""
    keys = strength * z + rng.gumbel(size=len(z))
    out = np.zeros(len(z), dtype=bool)
    out[np.argsort(-keys)[:round(rate * len(z))]] = True
    return out


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def _standardised(data: pd.DataFrame, categorical) -> np.ndarray:
    """Numbers as z-scores; categories as their sorted position (so 'Male' > 'Female'
    the same way in every version of a dataset)."""
    cols = []
    for c in data.columns:
        x = np.unique(data[c].astype(str), return_inverse=True)[1] if c in categorical else data[c]
        cols.append(_z(np.asarray(x, dtype=float)))
    return np.column_stack(cols)
