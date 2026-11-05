"""How sure are we? Bootstrap intervals and p-values for the effects in analysis.py.

Works from the saved per-cell predictions (<out>_cells.csv), so nothing is re-run.
The 50 scored rows of a run are a sample; a different sample would give a
somewhat different score. The bootstrap redraws the scored rows with
replacement n_boot times and recomputes every effect on each redraw:

    rows of the real table   shared by the real and perturbed versions (matched
                             through source_row), every naming and every method,
                             so they are redrawn together (a paired comparison)
    rows of the twin         redrawn on their own

    low, high   2.5th and 97.5th percentile of the redraws: a 95% interval
    p           twice the share of redraws on the far side of zero (two-sided)
    p_holm      p after Holm's correction over all rows of the same effect, so
                the chance of even one false alarm stays at 5%

This covers which rows happened to be scored. It does not cover run-to-run
noise of the LLM itself (the same run can differ by ~0.1); that needs repeated
runs or seeds.
"""
from __future__ import annotations

import warnings
import zlib

import numpy as np
import pandas as pd

from .datasets import load
from .run import KEY
from .twins import DEFAULT
from .versions import make_origin

SPACE = ["dataset", "seed", "mechanism", "rate"]   # one mask, so one set of scored rows per origin
EFFECTS = ["d_mem", "d_pert", "d_sem", "d_mislead"]


def scales(dataset: str, seed: int, origin: str, twin: str = DEFAULT) -> pd.DataFrame:
    """Per numeric column of that version's table: sd (nrmse, nmae) and robust spread (nmdae),
    exactly as metrics.score computes them."""
    ds = load(dataset)
    data = make_origin(ds, origin, seed, twin).data
    num = data[[c for c in data.columns if c not in ds.categorical]]
    sd = num.std().replace(0, 1.0)
    spread = (num.quantile(0.75) - num.quantile(0.25)) / 1.349
    return pd.DataFrame({"sd": sd, "spread": spread.where(spread != 0, sd)})


def _errors(cells: pd.DataFrame, rows: np.ndarray, sc: pd.Series) -> np.ndarray:
    """|guess - truth| / scale: one line per scored row (in the order of `rows`), one column
    per numeric column, NaN where nothing was hidden."""
    c = cells[cells.column.isin(sc.index)]
    err = (pd.to_numeric(c.guess) - pd.to_numeric(c.truth)).abs().to_numpy() / sc[c.column].to_numpy()
    m = pd.DataFrame({"r": c.rowid.to_numpy(), "col": c.column.to_numpy(), "e": err})
    return m.pivot(index="r", columns="col", values="e").reindex(index=rows, columns=sc.index).to_numpy(float)


def _score(e: np.ndarray, idx: np.ndarray, metric: str) -> np.ndarray:
    """The metric on each redraw; idx is (redraws, rows). Per column, then averaged over columns."""
    a = e[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # a column with no cell in a redraw is skipped
        if metric == "nrmse":
            per_col = np.sqrt(np.nanmean(a ** 2, axis=1))
        elif metric == "nmae":
            per_col = np.nanmean(a, axis=1)
        elif metric == "nmdae":
            per_col = np.nanmedian(a, axis=1)
        else:
            raise ValueError(f"no bootstrap for {metric!r}")
        return np.nanmean(per_col, axis=1)


def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN stays NaN)."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    adj = np.maximum.accumulate((len(ok) - np.arange(len(ok))) * p[order])
    out[order] = np.minimum(adj, 1.0)
    return out


def bootstrap(cells: pd.DataFrame, reference: str = "mice", metric: str = "nrmse",
              n_boot: int = 2000, seed: int = 0, twin: str = DEFAULT) -> pd.DataFrame:
    """One row per (dataset, seed, mechanism, rate, LLM, prompt, effect) with the estimate
    (equal to analysis.effects), a 95% interval and p-values. `cells` holds the saved cells
    of the LLM runs and of the reference method (e.g. both *_cells.csv files concatenated)."""
    c = cells.fillna({"prompt": "", "rate": ""}).astype({k: str for k in KEY})
    c = c[c.method.str.startswith("llm:") | (c.method == reference)]
    rowid = np.where(c.origin == "synthetic", c.row, c.source_row)   # real + perturbed: original row
    c = c.assign(rowid=pd.to_numeric(pd.Series(rowid), errors="raise").astype(int).astype(str).to_numpy())
    out = []
    for space, cs in c.groupby(SPACE):
        dataset, s, mech, rate = space
        rng = np.random.default_rng([seed, zlib.crc32("|".join(space).encode())])
        draws, sc = {}, {}
        for origin in ["real", "synthetic"]:
            ids = np.sort(cs[cs.origin == origin].rowid.unique())
            n = len(ids)
            if n:
                draws[origin] = (ids, np.vstack([np.arange(n), rng.integers(0, n, (n_boot, n))]))
        by = "spread" if metric == "nmdae" else "sd"
        runs = {k: g for k, g in cs.groupby(["origin", "naming", "method", "prompt"])}

        def e(origin, naming, method, prompt):
            if (origin, naming, method, prompt) not in runs:
                return np.full(n_boot + 1, np.nan)
            if origin not in sc:
                sc[origin] = scales(dataset, int(s), origin, twin)[by]
            ids, idx = draws["synthetic" if origin == "synthetic" else "real"]
            return _score(_errors(runs[origin, naming, method, prompt], ids, sc[origin]), idx, metric)

        ref = {o: e(o, "real", reference, "") for o in ["real", "perturbed", "synthetic"]}
        for method, prompt in sorted({(m, p) for _, _, m, p in runs if m.startswith("llm:")}):
            llm = lambda o, n: e(o, n, method, prompt)
            base = llm("real", "real")
            values = {"d_mem": (llm("synthetic", "real") - base) - (ref["synthetic"] - ref["real"]),
                      "d_pert": (llm("perturbed", "real") - base) - (ref["perturbed"] - ref["real"]),
                      "d_sem": llm("real", "hidden") - base,
                      "d_mislead": llm("real", "shuffled") - base}
            for effect, v in values.items():
                row = dict(dataset=dataset, seed=s, mechanism=mech, rate=rate, method=method,
                           prompt=prompt, effect=effect, estimate=v[0], low=np.nan, high=np.nan, p=np.nan)
                if not np.isnan(v[0]):
                    boot = v[1:]
                    tail = min(np.sum(boot <= 0), np.sum(boot >= 0))
                    row.update(low=np.nanpercentile(boot, 2.5), high=np.nanpercentile(boot, 97.5),
                               p=min(1.0, 2 * (tail + 1) / (n_boot + 1)))
                out.append(row)
    res = pd.DataFrame(out)
    res["p_holm"] = np.nan
    for effect, g in res.groupby("effect"):
        res.loc[g.index, "p_holm"] = holm(g.p.to_numpy())
    return res
