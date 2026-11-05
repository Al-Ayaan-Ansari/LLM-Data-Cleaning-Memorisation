"""Load the benchmark datasets.

Every table is returned complete (no missing values), so when we hide a cell
later we still know its true value and can score whoever fills it in.
The class label is kept apart in `target`: it is never hidden and never shown
to the imputers. It is only used later, to check whether the cleaned data
still trains a good classifier.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn import datasets as skd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# file name -> (download URL, expected SHA-256 or None)
DOWNLOADS = {
    "pima_openml.arff": (
        "https://api.openml.org/data/v1/download/37/diabetes.arff",
        "4eddd5b2b64679e8888348e306520a393d6a28e1ddc9643cfb76fc5d912d6d40",
    ),
    "DiaBD.csv": (
        "https://data.mendeley.com/public-files/datasets/m8cgwxs9s6/files/"
        "4c109a9f-2462-4dce-b93c-5789168c5401/file_downloaded",
        "d9f868c9ce8eefe545294156ed1626f60fdb33f507f9ca021fd34fa4b575741a",
    ),
}

PIMA_COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                "BMI", "DiabetesPedigreeFunction", "Age"]
# In Pima a 0 in these columns is an impossible reading, i.e. a real missing value.
PIMA_ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

DIABD_FLAGS = ["family_diabetes", "hypertensive", "family_hypertension",
               "cardiovascular_disease", "stroke"]


@dataclass
class Dataset:
    name: str
    data: pd.DataFrame            # complete feature table
    target: pd.Series             # class label (as strings)
    categorical: list[str] = field(default_factory=list)  # filled as categories, not numbers
    famous: bool = True           # likely seen by LLMs during training
    patterns: np.ndarray | None = None  # real missing-value patterns (Pima only), True = missing


def _fetch(filename: str) -> Path:
    path = RAW / filename
    if not path.exists():
        url, sha256 = DOWNLOADS[filename]
        content = requests.get(url, timeout=120).content
        if sha256 and hashlib.sha256(content).hexdigest() != sha256:
            raise ValueError(f"{filename}: checksum mismatch, refusing to use it")
        RAW.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return path


def _from_sklearn(name: str, loader) -> Dataset:
    bunch = loader(as_frame=True)
    target = pd.Series(np.asarray(bunch.target_names)[bunch.target], name="target")
    return Dataset(name, bunch.data.astype(float), target.astype(str))


def load_pima() -> Dataset:
    """Pima Indians Diabetes, kept to the 392 rows with no hidden zeros.

    The other 376 rows contain real missing values, so their true values are
    unknown. We keep their missing patterns to reuse as a realistic mask.
    """
    lines = _fetch("pima_openml.arff").read_text().splitlines()
    rows = [line.strip().split(",") for line in lines if line[:1].isdigit()]
    data = pd.DataFrame(rows, columns=PIMA_COLUMNS + ["Outcome"])
    target = (data.pop("Outcome") == "tested_positive").astype(int).astype(str)
    data = data.astype(float)

    zero = data[PIMA_ZERO_IS_MISSING] == 0
    patterns = np.zeros(data.shape, dtype=bool)
    patterns[:, [PIMA_COLUMNS.index(c) for c in PIMA_ZERO_IS_MISSING]] = zero.to_numpy()

    complete = ~zero.any(axis=1).to_numpy()
    return Dataset("pima", data[complete].reset_index(drop=True),
                   target[complete].reset_index(drop=True), patterns=patterns)


def load_diabd() -> Dataset:
    """DiaBD (Bangladesh, 2025): released after most LLMs were trained."""
    data = pd.read_csv(_fetch("DiaBD.csv"))
    target = data.pop("diabetic").astype(str)
    categorical = ["gender"] + DIABD_FLAGS
    for c in data.columns:
        data[c] = data[c].astype(str) if c in categorical else data[c].astype(float)
    return Dataset("diabd", data, target, categorical, famous=False)


LOADERS = {
    "iris": lambda: _from_sklearn("iris", skd.load_iris),
    "wine": lambda: _from_sklearn("wine", skd.load_wine),
    "breast_cancer": lambda: _from_sklearn("breast_cancer", skd.load_breast_cancer),
    "pima": load_pima,
    "diabd": load_diabd,
}


def load(name: str) -> Dataset:
    return LOADERS[name]()
