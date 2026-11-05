"""Classical imputers.

They fill gaps from patterns in the numbers alone and never look at column
names, so their result is identical for all three namings of a version.

    mean / median  each column's average (categories: most common value)
    knn            average of the 5 most similar rows
    mice           each column predicted from the others by linear
                   regression, round and round until stable (MICE)
    missforest     the same loop with random forests instead (MissForest)

Categories are coded 0, 1, 2, ... for the maths and decoded afterwards.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge

METHODS = ("mean", "median", "knn", "mice", "missforest")


def impute(method: str, masked: pd.DataFrame, categorical=(), seed: int = 0) -> pd.DataFrame:
    codes = {c: sorted(masked[c].dropna().unique()) for c in categorical}
    x = pd.DataFrame({c: masked[c].map({v: i for i, v in enumerate(codes[c])}) if c in codes
                      else masked[c].astype(float) for c in masked.columns})

    if method in ("mean", "median"):
        filled = x.fillna(x.mean() if method == "mean" else x.median())
        for c in codes:
            filled[c] = x[c].fillna(x[c].mode().iloc[0])
    else:
        mu, sd = x.mean(), x.std().replace(0, 1)
        imputer = {
            "knn": lambda: KNNImputer(n_neighbors=5),
            "mice": lambda: IterativeImputer(BayesianRidge(), max_iter=10, random_state=seed),
            "missforest": lambda: IterativeImputer(
                RandomForestRegressor(n_estimators=100, max_features="sqrt", n_jobs=-1,
                                      random_state=seed),
                max_iter=5, random_state=seed),
        }[method]()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            z = imputer.fit_transform(((x - mu) / sd).to_numpy())
        filled = pd.DataFrame(z, columns=x.columns, index=x.index) * sd + mu

    out = masked.copy()
    for c in masked.columns:
        values = filled[c]
        if c in codes:
            idx = values.round().clip(0, len(codes[c]) - 1).astype(int)
            values = pd.Series(np.asarray(codes[c], dtype=object)[idx], index=values.index)
        out[c] = masked[c].where(masked[c].notna(), values)
    return out
