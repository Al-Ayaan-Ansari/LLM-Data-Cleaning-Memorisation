"""Memorisation test: has a model seen a dataset's CSV file before?

Uses the tests of Bordt et al. [4] (package tabmemcheck). The model is shown
part of the file exactly as it circulates online and asked to continue it;
only a model that has seen the file can continue it word for word.

    header   the first rows, cut mid-row: how many characters of the
             continuation does it reproduce before its first mistake?
             (best of 4 cut points, at most 500)
    row      10 consecutive rows: does it write the next row exactly? (25 tries)
    feature  a row with its most distinctive value removed: does it fill in
             exactly that value? (25 tries)

Scores near 0 mean no sign of memory. For scale, Bordt et al. found that GPT-4
completes 125 of 136 Iris rows from memory.

    python -m imputation_study.memorisation --models qwen2.5:7b --out results/memorisation.csv
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import time
import zipfile
from importlib import resources
from pathlib import Path

import numpy as np
import requests

from .datasets import LOADERS, RAW, _fetch, load

BUNDLED = {"iris": "iris.csv", "pima": "openml-diabetes.csv", "wine": "uci-wine.csv"}
WDBC_URL = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
WDBC_SHA256 = "d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a"
WDBC_HEADER = "id,diagnosis," + ",".join(
    f"{f}_{s}" for s in ("mean", "se", "worst")
    for f in ("radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity",
              "concave points", "symmetry", "fractal_dimension"))
FIELDS = ["model", "dataset", "famous", "header_chars", "row_exact", "row_tries", "feature",
          "feature_exact", "feature_tries", "seconds"]


def csv_file(name: str) -> Path:
    """The dataset's CSV exactly as it circulates online: tabmemcheck's copies of
    Iris, Pima and Wine, the UCI Breast Cancer file (with the usual Kaggle
    header), and DiaBD as published on Mendeley."""
    if name in BUNDLED:
        return Path(str(resources.files("tabmemcheck.resources.csv") / BUNDLED[name]))
    if name == "diabd":
        return _fetch("DiaBD.csv")
    if name == "breast_cancer":
        path = RAW / "breast_cancer_uci.csv"
        if not path.exists():
            archive = zipfile.ZipFile(io.BytesIO(requests.get(WDBC_URL, timeout=120).content))
            data = archive.read("wdbc.data")
            if hashlib.sha256(data).hexdigest() != WDBC_SHA256:
                raise ValueError("wdbc.data: checksum mismatch, refusing to use it")
            RAW.mkdir(parents=True, exist_ok=True)
            path.write_bytes(WDBC_HEADER.encode() + b"\n" + data)
        return path
    raise ValueError(f"no CSV for {name!r}")


def ollama_llm(model: str, host: str = "http://localhost:11434"):
    """A tabmemcheck model that talks to Ollama through its OpenAI-compatible API."""
    from openai import OpenAI
    from tabmemcheck.llm import OpenAILLM
    return OpenAILLM(OpenAI(base_url=f"{host}/v1", api_key="ollama"), model, chat_mode=True)


def matching_prefix(a: str, b: str) -> int:
    """Number of leading characters two strings share."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def test_dataset(llm, name: str, queries: int = 25, seed: int = 0) -> dict:
    import tabmemcheck as tm
    path = str(csv_file(name))
    rng = np.random.default_rng(seed)
    t0 = time.time()
    # Cap answer length: asked to continue a CSV, a chat model keeps writing rows until
    # the limit (tabmemcheck's default is 1000 tokens). 400 fits the longest row we test
    # (Breast Cancer, ~300 tokens because Qwen writes digits one token each).
    tm.config.max_tokens = 400
    _, completion, response = tm.header_test(path, llm, verbose=False, rng=rng)
    rows, answers = tm.row_completion_test(path, llm, num_queries=queries, print_levenshtein=False, rng=rng)
    from tabmemcheck.analysis import find_most_unique_feature
    feature = find_most_unique_feature(path)[0]
    tm.config.max_tokens = 100
    values, guesses = tm.feature_completion_test(path, llm, feature_name=feature, num_queries=queries, rng=rng)
    return {"header_chars": matching_prefix(completion, response),
            "row_exact": sum(r.strip() == a.strip() for r, a in zip(rows, answers)), "row_tries": len(rows),
            "feature": feature, "feature_exact": sum(str(v).strip() == str(g).strip() for v, g in zip(values, guesses)),
            "feature_tries": len(values), "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="Ollama model names, e.g. qwen2.5:7b")
    ap.add_argument("--datasets", nargs="+", default=list(LOADERS))
    ap.add_argument("--queries", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--out", default="results/memorisation.csv")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open(newline="") as f:
            done = {(r["model"], r["dataset"]) for r in csv.DictReader(f)}
    for model in args.models:
        llm = ollama_llm(model, args.host)
        for name in args.datasets:
            if (model, name) in done:
                continue
            report = io.StringIO()  # tabmemcheck prints coloured reports; keep them in a log instead
            with contextlib.redirect_stdout(report):
                result = test_dataset(llm, name, args.queries, args.seed)
            with out.with_suffix(".log").open("a", encoding="utf-8") as log:
                log.write(f"===== {model} / {name}\n{report.getvalue()}\n")
            row = {"model": model, "dataset": name, "famous": load(name).famous, **result}
            new = not out.exists()
            with out.open("a", newline="") as f:
                w = csv.DictWriter(f, FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            print(row, flush=True)


if __name__ == "__main__":
    main()
