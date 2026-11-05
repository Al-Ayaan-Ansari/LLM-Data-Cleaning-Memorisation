"""The three effects from the slides, computed from a results CSV.

E is a method's error (NRMSE by default; any score column via `metric`, e.g. the
robust nmdae); "ref" is a classical reference method (MICE by default).

    d_mem   = [E_llm(twin) - E_llm(real)] - [E_ref(twin) - E_ref(real)]       (1)
              extra error the LLM suffers when there is nothing to recall
    d_pert  = the same with the perturbed copy instead of the twin
    d_sem   = E_llm(hidden names) - E_llm(real names)                         (2)
              (the classical bracket of (2) is exactly zero: classical methods
              never see names, so it drops out)
    d_mislead = E_llm(shuffled names) - E_llm(real names)
    D       = E(MNAR) - E(MCAR)                                               (3)
              how much a method worsens under MNAR; fingerprint = D on the twin
              minus D on the real table, LLM minus reference

Positive d_mem / d_pert: the LLM leaned on the original rows (memory).
Positive d_sem: the LLM leaned on the column names (meaning).
Only runs scored on the same cells are compared, so classical and LLM runs
must share --max-eval-rows.
"""
from __future__ import annotations

import pandas as pd

CELL = ["dataset", "seed", "mechanism", "rate"]


def _read(results, metric: str) -> pd.DataFrame:
    r = pd.read_csv(results, keep_default_na=False) if not isinstance(results, pd.DataFrame) else results
    r = r.copy()
    r[metric] = pd.to_numeric(r[metric], errors="coerce")
    for c in CELL + ["prompt"]:
        r[c] = r[c].astype(str)
    return r


def effects(results, reference: str = "mice", metric: str = "nrmse") -> pd.DataFrame:
    """One row per (dataset, seed, mechanism, rate, LLM, prompt) with equations (1) and (2)."""
    r = _read(results, metric)
    llm = r[r.method.str.startswith("llm:")]
    e = llm.pivot_table(index=CELL + ["method", "prompt"], columns=["origin", "naming"],
                        values=metric, aggfunc="first")
    e.columns = [f"{o}/{n}" for o, n in e.columns]
    ref = r[(r.method == reference) & (r.naming == "real")].pivot_table(
        index=CELL, columns="origin", values=metric, aggfunc="first").add_prefix("ref_")
    out = e.reset_index().merge(ref.reset_index(), on=CELL, how="left")

    get = lambda c: out[c] if c in out else float("nan")
    out["d_mem"] = (get("synthetic/real") - get("real/real")) - (get("ref_synthetic") - get("ref_real"))
    out["d_pert"] = (get("perturbed/real") - get("real/real")) - (get("ref_perturbed") - get("ref_real"))
    out["d_sem"] = get("real/hidden") - get("real/real")
    out["d_mislead"] = get("real/shuffled") - get("real/real")
    return out


def mnar_fingerprint(results, reference: str = "mice", rate: str = "0.2",
                     metric: str = "nrmse") -> pd.DataFrame:
    """Equation (3): D = E(MNAR) - E(MCAR) per method and origin (real names),
    plus fingerprint = [D_llm(twin) - D_llm(real)] - [D_ref(twin) - D_ref(real)]."""
    r = _read(results, metric)
    r = r[(r.naming == "real") & (r.rate == rate) & r.mechanism.isin(["MCAR", "MNAR"])]
    r = r[r.method.str.startswith("llm:") | (r.method == reference)]
    e = r.pivot_table(index=["dataset", "seed", "method", "prompt", "origin"], columns="mechanism",
                      values=metric, aggfunc="first")
    d = (e["MNAR"] - e["MCAR"]).unstack("origin")
    d.columns = [f"D_{o}" for o in d.columns]
    d = d.reset_index()
    ref = d[d.method == reference].set_index(["dataset", "seed"])
    d = d[d.method != reference].join(ref[["D_real", "D_synthetic"]].add_prefix("ref_"),
                                      on=["dataset", "seed"])
    d["fingerprint"] = (d.D_synthetic - d.D_real) - (d.ref_D_synthetic - d.ref_D_real)
    return d
