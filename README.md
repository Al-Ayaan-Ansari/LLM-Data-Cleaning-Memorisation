# Disentangling Semantic Knowledge and Memorisation in LLM-Based Data Cleaning: A Benchmark on Synthetic vs. Real Tabular Data

Minor project, Department of Computer Engineering, Faculty of Engineering & Technology, Jamia Millia Islamia, New Delhi.

**Al Ayaan Ansari** (23BCS034) · **Mohd Adnan Khan** (23BCS042) · Supervisor: **Dr. Faiyaz Ahmad**

**Project page:** https://al-ayaan-ansari.github.io/LLM-Data-Cleaning-Memorisation/

When a large language model fills in missing values in a table, is it using what the columns mean, or recalling a dataset it saw during training? This benchmark changes one thing at a time (where the rows come from: real, perturbed or a synthetic twin; and what the columns are called: real, hidden or shuffled) and measures how five open models (Qwen2.5 3B/7B/14B, Llama 3.1 8B, Gemma 2 9B) respond on five datasets (Iris, Wine, Breast Cancer, Pima, and DiaBD 2025 as an unseen control), against classical imputers.

**Main result:** removing the column names hurts significantly in 20 of 50 tests, while replacing every record with a synthetic twin never makes the models significantly worse, even on Iris, which some of them partly memorised (up to 60% of rows recited). The models remain less accurate than MICE.

**Side analyses** (classical imputers only, mask seeds 0–2): the per-class synthetic twin ranks the five classical imputers like the real table on 4 of 5 datasets (RQ4; not on DiaBD); simulated MNAR gives the hidden values away (leakage AUC 0.901, and 0.999 for the percentile rule), so a "missing means large" trick beats MICE (RQ5); and for classifiers trained on the filled-in tables, the choice of imputer changes accuracy by less than 2 points in 14 of 15 cases.

| Folder / page | Contents |
|---|---|
| `FINAL_PRESENTATION.html` / `.pdf` | Final presentation (arrow keys to move, `N` for speaker notes) |
| `results.html` | All measured results with 95% bootstrap intervals |
| `method.html` | Research design: datasets, nine versions, masks, metrics, effect formulas, statistics |
| `code.html`, `code/` | The Python pipeline (`imputation_study`) and its tests |
| `data/` | Summary tables behind every reported number (CSV), including the side analyses (`rq4_*`, `rq5_masks`, `downstream_*`) |
