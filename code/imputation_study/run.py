"""Run the benchmark grid and append one CSV line per run.

Every scored cell is also saved, one line each, to <out>_cells.csv (row, column,
true value, guess), so any metric can be recomputed later without re-running.

Runs already in the output file are skipped, so a run that stopped halfway
(for example a Colab session that timed out) carries on where it left off.

Examples
  python -m imputation_study.run --datasets iris wine --methods mean knn mice
  python -m imputation_study.run --datasets iris --methods llm:qwen2.5:7b --prompts zero few --max-eval-rows 50
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from . import classical, llm, twins
from .datasets import LOADERS, load
from .masks import MECHANISMS, make_mask
from .metrics import hidden_cells, score
from .versions import NAMINGS, ORIGINS, column_names, make_origin

KEY = ["dataset", "seed", "mechanism", "rate", "origin", "naming", "method", "prompt"]
FIELDS = KEY + ["twin", "famous", "rows", "hidden_share", "eval_rows", "eval_cells", "nrmse", "nmae",
                "nmdae", "cat_acc", "calls", "invalid", "prompt_tokens", "output_tokens", "seconds"]


def eval_rows(mask: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Rows whose hidden cells get scored: every row with a gap, or a fixed random
    sample of them (LLMs are slow, so they get a sample - and so does everyone
    else, so all methods are scored on exactly the same cells)."""
    rows = np.flatnonzero(mask.any(axis=1))
    if max_rows and len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_rows, replace=False))
    return rows


def jobs(methods, prompts):
    for m in methods:
        if m.startswith("llm:"):
            yield from ((m, p) for p in prompts)
        else:
            yield m, ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(LOADERS))
    ap.add_argument("--origins", nargs="+", default=list(ORIGINS))
    ap.add_argument("--namings", nargs="+", default=list(NAMINGS))
    ap.add_argument("--mechanisms", nargs="+", default=list(MECHANISMS))
    ap.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--methods", nargs="+", default=list(classical.METHODS),
                    help="classical names and/or llm:<ollama model>, e.g. llm:qwen2.5:7b")
    ap.add_argument("--prompts", nargs="+", default=list(llm.PROMPTS))
    ap.add_argument("--twin", default=twins.DEFAULT, choices=twins.MODELS, help="twin generator")
    ap.add_argument("--max-eval-rows", type=int, default=0, help="0 = score every row with a gap")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests to Ollama")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama server")
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args(argv)

    out = Path(args.out)
    cells_out = out.with_name(out.stem + "_cells.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open(newline="") as f:
            done = {tuple(r[k] for k in KEY) for r in csv.DictReader(f)}

    def record(row, cells):
        new = not out.exists()
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        cells.to_csv(cells_out, mode="a", header=not cells_out.exists(), index=False)

    for name in args.datasets:
        ds = load(name)
        for seed in args.seeds:
            origins = [make_origin(ds, o, seed, args.twin) for o in args.origins]
            for mech in args.mechanisms:
                if mech == "PIMA_REAL" and ds.patterns is None:
                    continue
                for rate in [None] if mech == "PIMA_REAL" else args.rates:
                    base = make_mask(ds.data, mech, rate, seed, ds.categorical, ds.patterns)
                    base_rows = eval_rows(base, args.max_eval_rows, seed)
                    for o in origins:
                        if o.row_map is None:  # the twin has new rows: same mechanism, its own cells
                            mask = make_mask(o.data, mech, rate, seed, ds.categorical, ds.patterns)
                            rows = eval_rows(mask, args.max_eval_rows, seed)
                        else:                  # the very same cells as the real table, wherever rows moved
                            mask = base[o.row_map]
                            rows = np.flatnonzero(np.isin(o.row_map, base_rows))
                        name_blind = {}        # classical results don't depend on column names
                        for naming in args.namings:
                            names = column_names(list(ds.data.columns), naming, seed)
                            original = {v: k for k, v in names.items()}
                            truth = o.data.rename(columns=names)
                            masked = truth.mask(mask)
                            cats = [names[c] for c in ds.categorical]
                            for method, prompt in jobs(args.methods, args.prompts):
                                key = dict(dataset=name, seed=str(seed), mechanism=mech,
                                           rate="" if rate is None else f"{rate:g}", origin=o.name,
                                           naming=naming, method=method, prompt=prompt)
                                if tuple(key.values()) in done:
                                    continue
                                if method in name_blind:
                                    result, cells = name_blind[method]
                                else:
                                    t0 = time.time()
                                    if method.startswith("llm:"):
                                        client = llm.Ollama(method[4:], args.host, seed=seed)
                                        filled, cost = llm.impute(masked, cats, client, prompt, rows,
                                                                  workers=args.workers, seed=seed)
                                    else:
                                        filled, cost = classical.impute(method, masked, cats, seed), {}
                                    result = {**score(truth, filled, mask, cats, rows), **cost,
                                              "seconds": round(time.time() - t0, 2)}
                                    cells = hidden_cells(truth, filled, mask, rows)
                                    cells["column"] = cells.column.map(original)
                                    cells.insert(1, "source_row", o.row_map[cells.row] if o.row_map is not None else "")
                                    if not method.startswith("llm:"):
                                        name_blind[method] = result, cells
                                record({**key, "twin": args.twin if o.row_map is None else "",
                                        "famous": ds.famous, "rows": len(truth),
                                        "hidden_share": round(float(mask.mean()), 4), **result},
                                       cells.assign(**key)[KEY + list(cells.columns)])
                                print(f"{name:13} s{seed} {mech:9} {key['rate']:4} {o.name:9} "
                                      f"{naming:8} {method:18} {prompt:4} nrmse={result['nrmse']} "
                                      f"nmdae={result['nmdae']} acc={result['cat_acc']} ({result['seconds']}s)",
                                      flush=True)


if __name__ == "__main__":
    main()
