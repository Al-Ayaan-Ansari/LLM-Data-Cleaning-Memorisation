"""RQ4: does a synthetic twin rank the classical imputers in the same order as the real table?

Each of the five classical imputers fills 20% MCAR gaps, once on the real table and
once on its twin; every row with a gap is scored by NRMSE, averaged over mask seeds
0, 1 and 2. The two orders of the imputers are compared by

    kendall_tau   1 = same order, 0 = unrelated, -1 = reversed
    same_best     do the real table and the twin pick the same best imputer?
    mean_abs_gap  average absolute change in NRMSE between real table and twin

This is the rule of the side analysis run on 6 Nov (extended.rq4_ranking), rebuilt
here, with two additions for testing the neural generators fairly:

    --twin-seeds  train the generator several times (one twin per training seed);
                  the 6 Nov analysis used one twin per table (seed 0)
    --min-steps   train CTGAN for at least this many steps; SDV's default of 300
                  epochs is only 300 steps on our tables of 150-569 rows
                  (0 = the default 300 epochs; several values allowed)

Twins are trained first (in parallel, saved under data/twins and reused), then the
imputers run in parallel; each worker uses --threads cores.

    python -m imputation_study.rq4 --models ctgan --twin-seeds 0 1 2 3 4 --min-steps 0 3000 \\
        --out results/rq4_ctgan_runs.csv
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (0, 1, 2)
RATE = 0.2
SIZE = {"diabd": 4, "breast_cancer": 3, "wine": 2, "pima": 1, "iris": 0}   # longest jobs first


def errors(data: pd.DataFrame, categorical, seeds=SEEDS) -> dict[str, float]:
    """NRMSE of each classical imputer at 20% MCAR, every row with a gap scored, mean over seeds."""
    from . import classical
    from .masks import make_mask
    from .metrics import score
    out = {}
    for m in classical.METHODS:
        e = []
        for s in seeds:
            mask = make_mask(data, "MCAR", RATE, s, categorical)
            filled = classical.impute(m, data.mask(mask), categorical, s)
            e.append(score(data, filled, mask, categorical)["nrmse"])
        out[m] = float(np.mean(e))
    return out


def compare(real: dict[str, float], twin: dict[str, float]) -> dict:
    """Agreement between the imputers' errors on the real table and on the twin."""
    from scipy import stats
    methods = list(real)
    r, t = [real[m] for m in methods], [twin[m] for m in methods]
    best_real, best_twin = min(real, key=real.get), min(twin, key=twin.get)
    return {"kendall_tau": round(float(stats.kendalltau(r, t).statistic), 3),
            "best_real": best_real, "best_twin": best_twin, "same_best": best_real == best_twin,
            "mean_abs_gap": round(float(np.mean(np.abs(np.subtract(t, r)))), 3)}


def _train(job):
    from .datasets import load
    from .twins import synthetic_twin
    name, model, epochs, seed = job
    t = time.time()
    synthetic_twin(load(name), seed, model, epochs)
    return job, time.time() - t


def _errors(job):
    from .datasets import load
    from .twins import synthetic_twin
    name, model, epochs, seed = job
    ds = load(name)
    data = ds.data if model == "real" else synthetic_twin(ds, seed, model, epochs)[0]
    return job, errors(data, ds.categorical)


def _pool(fn, jobs, workers, label):
    out = {}
    with ProcessPoolExecutor(workers) as ex:
        futures = [ex.submit(fn, j) for j in sorted(jobs, key=lambda j: -SIZE.get(j[0], 0))]
        for i, f in enumerate(as_completed(futures), 1):
            job, res = f.result()
            out[job] = res
            print(f"  {label} {i}/{len(jobs)}: {job}", flush=True)
    return out


def main(argv=None):
    from .datasets import LOADERS, load
    from .twins import EPOCHS, MODELS, TWINS, ctgan_epochs

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--twin-seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--min-steps", nargs="+", type=int, default=[0])
    ap.add_argument("--datasets", nargs="+", default=list(LOADERS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2 - 1))
    ap.add_argument("--threads", type=int, default=2, help="cores per worker")
    ap.add_argument("--out", default="results/rq4_runs.csv")
    args = ap.parse_args(argv)

    # Set before the workers start, so their numerical libraries read it at import.
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        os.environ[k] = str(args.threads)

    rows_of = {n: len(load(n).data) for n in args.datasets}
    twins = sorted({(n, m, ctgan_epochs(rows_of[n], s) if m == "ctgan" and s else EPOCHS, seed)
                    for n in args.datasets for m in args.models for s in args.min_steps
                    for seed in args.twin_seeds})
    t0 = time.time()

    def cached(job):
        n, m, e, seed = job
        tag = f"_e{e}" if m in ("ctgan", "tvae") and e != EPOCHS else ""
        return (TWINS / f"{n}_{m}{tag}_seed{seed}.csv").exists()

    todo = [j for j in twins if not cached(j)]
    print(f"{len(twins)} twins ({len(todo)} to train), {args.workers} workers x {args.threads} cores", flush=True)
    trained = _pool(_train, todo, args.workers, "trained") if todo else {}

    jobs = [(n, "real", 0, 0) for n in args.datasets] + twins
    err = _pool(_errors, jobs, args.workers, "scored")

    out = []
    for n, m, e, seed in twins:
        real, twin = err[(n, "real", 0, 0)], err[(n, m, e, seed)]
        steps = e * max(rows_of[n] // 500, 1) if m == "ctgan" else None
        out.append({"dataset": n, "generator": m, "epochs": e if m in ("ctgan", "tvae") else None,
                    "ctgan_steps": steps, "twin_seed": seed, **compare(real, twin),
                    "train_seconds": round(trained.get((n, m, e, seed), np.nan), 1),
                    **{f"real_{k}": round(v, 4) for k, v in real.items()},
                    **{f"twin_{k}": round(v, 4) for k, v in twin.items()}})
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} rows) in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
