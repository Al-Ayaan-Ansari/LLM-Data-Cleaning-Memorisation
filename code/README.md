# Code: Disentangling Semantic Knowledge and Memorisation in LLM-Based Data Cleaning

Minor project, Department of Computer Engineering, Jamia Millia Islamia.
Al Ayaan Ansari (23BCS034), Mohd Adnan Khan (23BCS042). Supervisor: Dr. Faiyaz Ahmad.

The package `imputation_study` hides cells in five tabular datasets, fills them with classical imputers and with local LLMs (Ollama), and measures how the LLM's error changes when the rows are replaced by a synthetic twin (memory) or the column names are hidden or shuffled (meaning). See the project page for the research design and results.

```
pip install -r requirements.txt          # plus tabmemcheck for the memorisation test, and Ollama for LLM runs
python -m pytest -q tests                # 29 tests
python -m imputation_study.run --help    # the experiment runner
```

| Module | Purpose |
|---|---|
| `datasets.py` | load the five datasets from their sources, verify checksums |
| `twins.py` | synthetic twins (per-class Gaussian copula by default) |
| `versions.py` | the nine versions: origin (real / perturbed / twin) × naming (real / hidden / shuffled) |
| `masks.py` | MCAR, MAR, MNAR and the real Pima pattern at exact rates |
| `classical.py` | mean, median, k-NN, MICE, MissForest-style imputation |
| `llm.py` | zero- and few-shot prompts, JSON-schema answers through Ollama |
| `metrics.py` | NRMSE, NMAE, robust nmdae, category accuracy |
| `run.py` | resumable experiment runner; saves every scored cell |
| `analysis.py` | effect formulas (Δmem, Δpert, Δsem, Δmislead, MNAR fingerprint) |
| `stats.py` | bootstrap 95% intervals and Holm-corrected p-values |
| `memorisation.py` | Bordt et al.'s header / row / feature tests on local models |

Datasets are downloaded from their original sources on first use and are not redistributed here.
