"""Make the 3 x 3 = 9 versions of a dataset.

origin - where the rows come from:
    real       the published table
    perturbed  the same rows, shuffled, with every number nudged by one step
               in its last digit (5.1 -> 5.0 or 5.2), so no row matches the
               published text any more (the perturbation of Bordt et al. [4])
    synthetic  a twin: brand-new rows sampled from a model fitted to the real
               table (see twins.py), so nothing can be recalled
naming - what the columns are called:
    real       the original names
    hidden     f1, f2, f3, ...
    shuffled   every column wears another column's name
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .datasets import Dataset
from .twins import DEFAULT, synthetic_twin

ORIGINS = ("real", "perturbed", "synthetic")
NAMINGS = ("real", "hidden", "shuffled")


@dataclass
class Origin:
    name: str
    data: pd.DataFrame          # complete table, original column names
    target: pd.Series
    row_map: np.ndarray | None  # row i here is row row_map[i] of the real table; None for the twin


def make_origin(ds: Dataset, origin: str, seed: int, twin: str = DEFAULT) -> Origin:
    if origin == "real":
        return Origin("real", ds.data, ds.target, np.arange(len(ds.data)))
    if origin == "perturbed":
        return _perturb(ds, seed)
    if origin == "synthetic":
        data, target = synthetic_twin(ds, seed, twin)
        return Origin("synthetic", data, target, None)
    raise ValueError(f"unknown origin {origin!r}")


def decimals(col: pd.Series) -> int:
    """How many decimal places a column is recorded with (5.1 -> 1, 148 -> 0)."""
    for d in range(7):
        if np.allclose(col, col.round(d), rtol=0, atol=1e-9):
            return d
    return 6


def _perturb(ds: Dataset, seed: int) -> Origin:
    rng = np.random.default_rng(seed)
    row_map = rng.permutation(len(ds.data))
    data = ds.data.iloc[row_map].reset_index(drop=True)
    for c in data.columns:
        if c in ds.categorical:
            continue
        d = decimals(ds.data[c])
        nudged = data[c] + rng.choice([-1, 1], size=len(data)) * 10.0 ** -d
        data[c] = nudged.clip(ds.data[c].min(), ds.data[c].max()).round(d)
    target = ds.target.iloc[row_map].reset_index(drop=True)
    return Origin("perturbed", data, target, row_map)


def column_names(columns: list[str], naming: str, seed: int) -> dict[str, str]:
    """Map each original column name to the name it gets in this version."""
    if naming == "real":
        return {c: c for c in columns}
    if naming == "hidden":
        return {c: f"f{i}" for i, c in enumerate(columns, 1)}
    if naming == "shuffled":
        rng = np.random.default_rng(seed)
        while True:  # redraw until no column keeps its own name
            perm = rng.permutation(len(columns))
            if (perm != np.arange(len(columns))).all():
                return {c: columns[p] for c, p in zip(columns, perm)}
    raise ValueError(f"unknown naming {naming!r}")
