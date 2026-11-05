"""LLM imputer: write a row out as text, ask a local model (through Ollama)
to fill the "?" cells, and read its JSON answer back into the table.

The model only ever receives the masked table - never the true values.

Prompt styles
    zero  the column list (with the range of values seen) and the row
    few   the same, plus 4 example rows from the same table
          (the rows with the fewest hidden cells)

A prompt looks like this (Pima, real names, zero-shot):

    The table has these columns:
    - Glucose: number, seen from 56 to 198
    - BMI: number, seen from 18.2 to 67.1
    ...
    Row with missing values (?):
    Glucose = 148; BMI = ?; Age = 50; ...
    Estimate the missing values. Answer in JSON with the keys: "BMI".

Ollama is told the exact JSON shape to return ("structured output"), so
answers almost always parse. Anything still unusable is counted as
`invalid` and replaced by the column mean (or most common category).
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

PROMPTS = ("zero", "few")

SYSTEM = ("You fill in missing values in a data table. Use the other values in the row "
          "and what you know about the columns. Reply with JSON only.")


@dataclass
class Ollama:
    model: str
    host: str = "http://localhost:11434"
    seed: int = 0
    num_ctx: int = 8192

    def ask(self, prompt: str, schema: dict) -> tuple[str, dict]:
        body = {
            "model": self.model, "stream": False, "format": schema,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx},
        }
        for attempt in range(3):
            try:
                r = requests.post(f"{self.host}/api/chat", json=body, timeout=600)
                r.raise_for_status()
                out = r.json()
                return out["message"]["content"], {"prompt_tokens": out.get("prompt_eval_count", 0),
                                                   "output_tokens": out.get("eval_count", 0)}
            except requests.RequestException:
                if attempt == 2:
                    return "", {}
                time.sleep(5)


def fmt(v) -> str:
    return v if isinstance(v, str) else f"{v:g}"


def serialise(row: pd.Series) -> str:
    return "; ".join(f"{c} = {'?' if pd.isna(v) else fmt(v)}" for c, v in row.items())


def describe(masked: pd.DataFrame, categorical) -> str:
    lines = []
    for c in masked.columns:
        seen = masked[c].dropna()
        if c in categorical:
            lines.append(f"- {c}: one of {', '.join(sorted(seen.astype(str).unique()))}")
        else:
            lines.append(f"- {c}: number, seen from {fmt(seen.min())} to {fmt(seen.max())}")
    return "\n".join(lines)


def build_prompt(header: str, row: pd.Series, missing: list[str]) -> str:
    keys = ", ".join(json.dumps(c) for c in missing)
    return (f"{header}\n\nRow with missing values (?):\n{serialise(row)}\n\n"
            f"Estimate the missing values. Answer in JSON with the keys: {keys}.")


def schema(missing: list[str], options: dict[str, list[str]]) -> dict:
    props = {c: {"type": "string", "enum": options[c]} if c in options else {"type": "number"}
             for c in missing}
    return {"type": "object", "properties": props, "required": missing}


def parse(text: str, missing: list[str], options: dict[str, list[str]]) -> dict:
    """Return {column: value or None}; None marks an unusable answer."""
    try:
        answer = json.loads(text)
    except json.JSONDecodeError:
        answer = {}
    if not isinstance(answer, dict):
        answer = {}
    out = {}
    for c in missing:
        v = answer.get(c)
        if c in options:
            out[c] = str(v) if v is not None and str(v) in options[c] else None
        else:
            try:
                out[c] = float(v) if math.isfinite(float(v)) else None
            except (TypeError, ValueError):
                out[c] = None
    return out


def impute(masked: pd.DataFrame, categorical, client: Ollama, prompt: str = "zero",
           rows=None, n_examples: int = 4, workers: int = 4, seed: int = 0):
    """Fill the hidden cells of `rows` (default: every row with a gap).
    Returns the filled table and a dict of costs."""
    if rows is None:
        rows = np.flatnonzero(masked.isna().any(axis=1))
    options = {c: sorted(masked[c].dropna().astype(str).unique()) for c in categorical}
    fallback = {c: masked[c].mode().iloc[0] if c in options else masked[c].mean()
                for c in masked.columns}

    header = "The table has these columns:\n" + describe(masked, categorical)
    if prompt == "few":
        gaps = masked.isna().sum(axis=1) + np.random.default_rng(seed).random(len(masked))
        examples = masked.iloc[np.argsort(gaps.to_numpy())[:n_examples]]
        header += "\n\nExample rows from the same table:\n" + "\n".join(
            serialise(r) for _, r in examples.iterrows())
    elif prompt != "zero":
        raise ValueError(f"unknown prompt style {prompt!r}")

    def one(i):
        row = masked.iloc[i]
        missing = [c for c in masked.columns if pd.isna(row[c])]
        text, info = client.ask(build_prompt(header, row, missing), schema(missing, options))
        return i, parse(text, missing, options), info

    filled = masked.copy()
    cost = {"calls": 0, "invalid": 0, "prompt_tokens": 0, "output_tokens": 0}
    with ThreadPoolExecutor(workers) as pool:
        for i, answer, info in pool.map(one, rows):
            cost["calls"] += 1
            cost["prompt_tokens"] += info.get("prompt_tokens", 0)
            cost["output_tokens"] += info.get("output_tokens", 0)
            for c, v in answer.items():
                if v is None:
                    cost["invalid"] += 1
                    v = fallback[c]
                filled.iat[i, masked.columns.get_loc(c)] = v
    return filled, cost
