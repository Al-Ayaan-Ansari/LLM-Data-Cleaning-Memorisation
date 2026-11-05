"""Score filled-in values against the true ones, on hidden cells only.

    nrmse    root-mean-square error divided by the column's standard
             deviation, so columns in different units can be averaged
             (0 = perfect; 1 = about as far off as a typical value is
             from the column mean)
    nmae     mean absolute error divided by the same standard deviation
    nmdae    median absolute error divided by a robust spread (IQR / 1.349,
             which equals the SD for bell-shaped data): the typical miss.
             A few wild answers or impossible values in the data barely move it
    cat_acc  share of hidden categories filled in exactly right

Each score is computed per column, then averaged over columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hidden_cells(truth: pd.DataFrame, filled: pd.DataFrame, mask: np.ndarray,
                 rows: np.ndarray | None = None) -> pd.DataFrame:
    """One line per scored cell: row, column, true value, filled-in value."""
    if rows is not None:
        keep = np.zeros(len(mask), dtype=bool)
        keep[rows] = True
        mask = mask & keep[:, None]
    r, c = np.nonzero(mask)
    return pd.DataFrame({"row": r, "column": truth.columns[c],
                         "truth": truth.to_numpy(dtype=object)[r, c],
                         "guess": filled.to_numpy(dtype=object)[r, c]})


def score(truth: pd.DataFrame, filled: pd.DataFrame, mask: np.ndarray,
          categorical=(), rows: np.ndarray | None = None) -> dict:
    cells = hidden_cells(truth, filled, mask, rows)
    nrmse, nmae, nmdae, acc = [], [], [], []
    for c, g in cells.groupby("column", sort=False):
        if c in categorical:
            acc.append(np.mean(g.truth.astype(str) == g.guess.astype(str)))
        else:
            x = truth[c]
            sd = x.std() or 1.0
            spread = (x.quantile(0.75) - x.quantile(0.25)) / 1.349 or sd
            err = np.abs(g.guess.astype(float) - g.truth.astype(float))
            nrmse.append(np.sqrt(np.mean(err ** 2)) / sd)
            nmae.append(err.mean() / sd)
            nmdae.append(err.median() / spread)

    mean = lambda v: round(float(np.mean(v)), 5) if v else ""
    return {"eval_rows": int(cells.row.nunique()), "eval_cells": len(cells), "nrmse": mean(nrmse),
            "nmae": mean(nmae), "nmdae": mean(nmdae), "cat_acc": mean(acc)}
