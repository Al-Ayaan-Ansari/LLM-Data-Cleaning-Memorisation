"""Synthetic twins: brand-new rows that follow the statistics of a real table.

    copula        one Gaussian copula for the whole table: learns each column's
                  shape and the correlations between columns
    class_copula  one copula per class (per species, diabetic yes/no, ...), so
                  groups that sit apart in the real data stay apart in the twin
    ctgan, tvae   SDV's neural-network generators (a GAN and a variational
                  autoencoder), trained for 300 epochs

A twin has as many rows as the real table (class_copula: as many per class).
Each twin is saved as CSV and reused, so it is made once and every experiment
uses the identical file.

    python -m imputation_study.twins            # compare the generators (see compare())
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import Dataset

MODELS = ("copula", "class_copula", "ctgan", "tvae")
DEFAULT = "class_copula"
TWINS = Path(__file__).resolve().parent.parent / "data" / "twins"
TARGET = "__target__"


def synthetic_twin(ds: Dataset, seed: int, model: str = DEFAULT) -> tuple[pd.DataFrame, pd.Series]:
    path = TWINS / f"{ds.name}_{model}_seed{seed}.csv"
    if not path.exists():
        if model == "class_copula":
            parts = [_sample(_fit("copula", rows, ds.categorical, seed), len(rows), ds.data)
                     .assign(**{TARGET: label})
                     for label, rows in ds.data.groupby(ds.target.to_numpy(), sort=True)]
            twin = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            table = ds.data.assign(**{TARGET: ds.target})
            twin = _sample(_fit(model, table, ds.categorical + [TARGET], seed), len(table), ds.data)
        TWINS.mkdir(parents=True, exist_ok=True)
        twin.to_csv(path, index=False)

    twin = pd.read_csv(path, dtype={c: str for c in ds.categorical + [TARGET]})
    target = twin.pop(TARGET)
    return twin.astype({c: float for c in twin.columns if c not in ds.categorical}), target


def _fit(model: str, table: pd.DataFrame, categorical: list[str], seed: int):
    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

    metadata = Metadata.detect_from_dataframes({"t": table}, infer_keys=None)
    for cols, sdtype in [([c for c in table if c not in categorical], "numerical"),
                         ([c for c in table if c in categorical], "categorical")]:
        if cols:
            metadata.update_columns(cols, table_name="t", sdtype=sdtype)
    if model == "copula":
        synth = GaussianCopulaSynthesizer(metadata)
    else:
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)
        synth = {"ctgan": CTGANSynthesizer, "tvae": TVAESynthesizer}[model](metadata, epochs=300)
    synth.fit({"t": table})
    synth._set_random_state(seed)  # SDV otherwise samples with one fixed seed
    return synth


def _sample(synth, n: int, real: pd.DataFrame) -> pd.DataFrame:
    """n new rows, none an exact copy of a real row. With coarse data such as
    Iris (4 columns, 1 decimal) a sampled row can match a real one by chance."""
    seen = real.drop_duplicates()
    parts, have = [], 0
    while have < n:
        s = synth.sample(2 * n)["t"]
        s = s.merge(seen, on=list(real.columns), how="left", indicator=True)
        s = s[s.pop("_merge") == "left_only"]
        parts.append(s)
        have += len(s)
    return pd.concat(parts).head(n).reset_index(drop=True)


def fidelity(ds: Dataset, twin: pd.DataFrame) -> dict:
    """How closely the twin's numeric columns match the real table's."""
    num = [c for c in ds.data if c not in ds.categorical]
    r, t = ds.data[num], twin[num]
    upper = np.triu_indices(len(num), 1)
    return {"mean_shift_sd": float(((t.mean() - r.mean()).abs() / r.std()).mean()),
            "spread_kept": float((t.std() / r.std()).mean()),
            "corr_error": float(np.abs(t.corr().to_numpy() - r.corr().to_numpy())[upper].mean()),
            "copied_rows": len(pd.merge(twin, ds.data, how="inner").drop_duplicates())}


def compare(names=None, models=MODELS, seeds=(0,), methods=("mean", "knn", "mice")) -> pd.DataFrame:
    """For every dataset x generator: fidelity, plus how much harder the twin is
    than the real table for classical imputers (NRMSE at 20% MCAR, all rows).
    A good twin has a difficulty gap near 0 for every method."""
    from . import classical
    from .datasets import LOADERS, load
    from .masks import make_mask
    from .metrics import score

    out = []
    for name in names or LOADERS:
        ds = load(name)
        for seed in seeds:
            real_mask = make_mask(ds.data, "MCAR", 0.2, seed, ds.categorical)
            real = {m: score(ds.data, classical.impute(m, ds.data.mask(real_mask), ds.categorical, seed),
                             real_mask, ds.categorical)["nrmse"] for m in methods}
            for model in models:
                twin, _ = synthetic_twin(ds, seed, model)
                mask = make_mask(twin, "MCAR", 0.2, seed, ds.categorical)
                row = {"dataset": name, "seed": seed, "twin": model, **fidelity(ds, twin)}
                for m in methods:
                    e = score(twin, classical.impute(m, twin.mask(mask), ds.categorical, seed),
                              mask, ds.categorical)["nrmse"]
                    row[f"gap_{m}"] = e - real[m]
                out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)
    print(compare().round(3).to_string(index=False))
