import numpy as np
import pandas as pd
import pytest

from imputation_study import analysis, classical, llm, run, twins
from imputation_study.datasets import load
from imputation_study.masks import make_mask
from imputation_study.metrics import hidden_cells, score
from imputation_study.versions import column_names, decimals, make_origin


@pytest.fixture(scope="module")
def iris():
    return load("iris")


@pytest.mark.parametrize("mech", ["MCAR", "MAR", "MNAR"])
@pytest.mark.parametrize("rate", [0.1, 0.3])
def test_mask_hides_the_requested_share(iris, mech, rate):
    mask = make_mask(iris.data, mech, rate, seed=0)
    assert mask.shape == iris.data.shape
    assert abs(mask.mean() - rate) < 0.015
    assert not mask.all(axis=1).any()


def test_mnar_hides_larger_values(iris):
    mask = make_mask(iris.data, "MNAR", 0.3, seed=0)
    x = iris.data.to_numpy()
    assert all(x[mask[:, j], j].mean() > x[~mask[:, j], j].mean() for j in range(x.shape[1]))


def test_mar_drivers_are_never_hidden(iris):
    mask = make_mask(iris.data, "MAR", 0.2, seed=0)
    assert (mask.sum(axis=0) == 0).sum() >= 1


def test_masks_are_reproducible(iris):
    assert (make_mask(iris.data, "MAR", 0.2, 3) == make_mask(iris.data, "MAR", 0.2, 3)).all()


def test_pima_real_patterns():
    pima = load("pima")
    assert pima.data.shape == (392, 8) and (pima.data[["Glucose", "Insulin"]] > 0).all().all()
    mask = make_mask(pima.data, "PIMA_REAL", None, 0, patterns=pima.patterns)
    assert 0.05 < mask.mean() < 0.15  # Pima's own missing share is about 10.6%


def test_perturbed_keeps_the_same_cells(iris):
    pert = make_origin(iris, "perturbed", seed=0)
    base = make_mask(iris.data, "MNAR", 0.2, 0)
    mask = base[pert.row_map]
    # row i of the perturbed table is row row_map[i] of the real one, give or take one step
    step = 10.0 ** -max(decimals(iris.data[c]) for c in iris.data)
    diff = np.abs(pert.data.to_numpy() - iris.data.to_numpy()[pert.row_map])
    assert diff.max() <= step + 1e-9 and (diff > 0).mean() > 0.9
    assert mask.sum() == base.sum()
    assert (pert.target.to_numpy() == iris.target.to_numpy()[pert.row_map]).all()


def test_shuffled_names_never_keep_their_own(iris):
    cols = list(iris.data.columns)
    for seed in range(20):
        names = column_names(cols, "shuffled", seed)
        assert sorted(names.values()) == sorted(cols)
        assert all(names[c] != c for c in cols)
    assert list(column_names(cols, "hidden", 0).values()) == ["f1", "f2", "f3", "f4"]


@pytest.mark.parametrize("method", classical.METHODS)
def test_classical_fills_every_gap_and_keeps_the_rest(iris, method):
    mask = make_mask(iris.data, "MCAR", 0.2, 0)
    masked = iris.data.mask(mask)
    filled = classical.impute(method, masked, seed=0)
    assert not filled.isna().any().any()
    assert (filled.to_numpy()[~mask] == iris.data.to_numpy()[~mask]).all()
    assert score(iris.data, filled, mask)["nrmse"] < 1.2


def test_classical_handles_categories():
    df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6], "g": ["a", "b", "a", "b", "a", "b"]})
    masked = df.copy()
    masked.loc[[0, 3], "g"] = np.nan
    for method in classical.METHODS:
        out = classical.impute(method, masked, ["g"])
        assert set(out["g"]) <= {"a", "b"}


def test_score_is_zero_for_a_perfect_fill(iris):
    mask = make_mask(iris.data, "MCAR", 0.2, 0)
    s = score(iris.data, iris.data, mask)
    assert s["nrmse"] == 0 and s["eval_cells"] == mask.sum()


def test_prompt_shows_question_marks_not_true_values():
    df = pd.DataFrame({"Glucose": [148.0, 85.0], "BMI": [33.6, 26.6], "sex": ["F", "M"]})
    masked = df.mask(np.array([[False, True, False], [False, False, True]]))
    header = "The table has these columns:\n" + llm.describe(masked, ["sex"])
    prompt = llm.build_prompt(header, masked.iloc[0], ["BMI"])
    assert "BMI = ?" in prompt and "33.6" not in prompt
    assert '"BMI"' in prompt


def test_parse_accepts_good_answers_and_flags_bad_ones():
    opts = {"sex": ["F", "M"]}
    assert llm.parse('{"BMI": 30.5, "sex": "M"}', ["BMI", "sex"], opts) == {"BMI": 30.5, "sex": "M"}
    assert llm.parse('{"BMI": "thirty", "sex": "X"}', ["BMI", "sex"], opts) == {"BMI": None, "sex": None}
    assert llm.parse("not json", ["BMI"], opts) == {"BMI": None}


def test_llm_impute_with_a_fake_model(iris):
    class Fake:
        def ask(self, prompt, schema):
            return "{" + ", ".join(f'"{k}": 1.0' for k in schema["required"]) + "}", {"prompt_tokens": 10}

    mask = make_mask(iris.data, "MCAR", 0.2, 0)
    rows = np.flatnonzero(mask.any(axis=1))[:5]
    filled, cost = llm.impute(iris.data.mask(mask), [], Fake(), "few", rows, workers=2)
    assert cost["calls"] == 5 and cost["invalid"] == 0
    assert (filled.to_numpy()[rows][mask[rows]] == 1.0).all()


def test_effects_match_the_slide_formulas():
    rows = []
    def add(method, origin, naming, nrmse, mech="MCAR", prompt=""):
        rows.append(dict(dataset="d", seed=0, mechanism=mech, rate=0.2, origin=origin,
                         naming=naming, method=method, prompt=prompt, nrmse=nrmse))
    for origin, e in [("real", 0.5), ("perturbed", 0.55), ("synthetic", 0.7)]:
        add("mice", origin, "real", e)
        add("mice", origin, "real", e + 0.4, mech="MNAR")
    llm_e = {("real", "real"): 0.3, ("real", "hidden"): 0.6, ("real", "shuffled"): 0.9,
             ("perturbed", "real"): 0.45, ("synthetic", "real"): 0.8}
    for (origin, naming), e in llm_e.items():
        add("llm:m", origin, naming, e, prompt="zero")
    add("llm:m", "real", "real", 0.35, mech="MNAR", prompt="zero")
    add("llm:m", "synthetic", "real", 1.4, mech="MNAR", prompt="zero")

    e = analysis.effects(pd.DataFrame(rows)).iloc[0]
    assert e.d_mem == pytest.approx((0.8 - 0.3) - (0.7 - 0.5))
    assert e.d_pert == pytest.approx((0.45 - 0.3) - (0.55 - 0.5))
    assert e.d_sem == pytest.approx(0.3) and e.d_mislead == pytest.approx(0.6)

    f = analysis.mnar_fingerprint(pd.DataFrame(rows)).iloc[0]
    # LLM: D_real = 0.05, D_twin = 0.6; MICE: D = 0.4 on both
    assert f.D_real == pytest.approx(0.05) and f.fingerprint == pytest.approx(0.55)


def test_robust_error_ignores_one_wild_guess(iris):
    mask = make_mask(iris.data, "MCAR", 0.2, 0)
    filled = iris.data.copy()
    r, c = np.argwhere(mask)[0]
    filled.iat[r, c] = 1000.0                      # one absurd answer, everything else perfect
    s = score(iris.data, filled, mask)
    assert s["nrmse"] > 10 and s["nmdae"] == 0
    assert len(hidden_cells(iris.data, filled, mask)) == mask.sum() == s["eval_cells"]


@pytest.fixture
def tmp_twins(tmp_path, monkeypatch):
    monkeypatch.setattr(twins, "TWINS", tmp_path / "twins")
    return tmp_path


def test_class_twin_keeps_class_sizes_and_structure(iris, tmp_twins):
    twin, target = twins.synthetic_twin(iris, 0, "class_copula")
    assert twin.shape == iris.data.shape
    assert target.value_counts().to_dict() == iris.target.value_counts().to_dict()
    assert twins.fidelity(iris, twin)["copied_rows"] == 0
    # species clusters survive: each species' twin rows sit near that species' real mean petal length
    for s in iris.target.unique():
        real = iris.data["petal length (cm)"][iris.target == s].mean()
        assert abs(twin["petal length (cm)"][target == s].mean() - real) < 0.3


def test_run_end_to_end_saves_cells_and_resumes(iris, tmp_twins):
    out = tmp_twins / "res.csv"
    argv = ["--datasets", "iris", "--namings", "real", "hidden", "--mechanisms", "MCAR",
            "--rates", "0.2", "--methods", "mean", "--max-eval-rows", "10", "--out", str(out)]
    run.main(argv)
    res = pd.read_csv(out, keep_default_na=False)
    cells = pd.read_csv(tmp_twins / "res_cells.csv", keep_default_na=False)
    assert len(res) == 6 and res.eval_rows.max() == 10
    assert len(cells) == res.eval_cells.sum()
    assert set(cells.column) <= set(iris.data.columns)          # original names, whatever the naming
    for origin, tol in [("real", 1e-9), ("perturbed", 0.1 + 1e-9)]:
        c = cells[(cells.origin == origin)]
        src = pd.to_numeric(c.source_row).astype(int).to_numpy()
        true = [iris.data.at[i, col] for i, col in zip(src, c.column)]
        assert np.allclose(c.truth.astype(float), true, atol=tol)
    # the real and perturbed copies are scored on the same original cells
    key = lambda o: set(zip(cells[cells.origin == o].source_row, cells[cells.origin == o].column))
    assert key("real") == key("perturbed")
    assert (res[res.origin == "synthetic"].twin == twins.DEFAULT).all()
    run.main(argv)                                               # second time: nothing new
    assert len(pd.read_csv(out)) == 6


def test_bootstrap_matches_effects_and_finds_a_planted_effect(iris, tmp_twins):
    from imputation_study import stats
    out = tmp_twins / "res.csv"
    run.main(["--datasets", "iris", "--mechanisms", "MCAR", "--rates", "0.2", "--methods", "mean", "mice",
              "--max-eval-rows", "20", "--out", str(out)])
    res = pd.read_csv(out, keep_default_na=False)
    cells = pd.read_csv(tmp_twins / "res_cells.csv", keep_default_na=False)
    fake = lambda d: d.assign(method=d.method.replace("mean", "llm:fake"),
                              prompt=np.where(d.method == "mean", "zero", d.prompt))
    res, cells = fake(res), fake(cells)
    for metric in ["nrmse", "nmdae"]:
        b = stats.bootstrap(cells, metric=metric, n_boot=200).set_index("effect")
        e = analysis.effects(res, metric=metric).iloc[0]
        for effect in stats.EFFECTS:                     # same numbers as the slide formulas
            assert b.at[effect, "estimate"] == pytest.approx(e[effect], abs=1e-4)
        assert b.at["d_sem", "low"] == b.at["d_sem", "high"] == 0   # identical runs: no doubt at all
    # plant an effect: the "LLM" is 2 SD off on every cell once the names are hidden
    sd = iris.data.std()
    hidden = (cells.method == "llm:fake") & (cells.naming == "hidden")
    cells.loc[hidden, "guess"] = cells.guess[hidden].astype(float) + 2 * cells.column[hidden].map(sd)
    b = stats.bootstrap(cells, n_boot=200).set_index("effect")
    assert b.at["d_sem", "low"] > 0.5 and b.at["d_sem", "p"] < 0.01
    assert b.at["d_mem", "low"] < b.at["d_mem", "estimate"] < b.at["d_mem", "high"]


def test_holm():
    from imputation_study.stats import holm
    assert np.allclose(holm([0.01, 0.04, 0.03, np.nan])[:3], [0.03, 0.06, 0.06])
    assert np.isnan(holm([0.01, np.nan])[1])


def test_memorisation_helpers():
    from imputation_study import memorisation
    assert memorisation.matching_prefix("5.1,3.5,1.4", "5.1,3.6,1.4") == 6   # "5.1,3."
    assert memorisation.matching_prefix("abc", "abc") == 3
    assert memorisation.WDBC_HEADER.count(",") == 31            # id, diagnosis + 30 features
    for name in memorisation.BUNDLED:
        path = memorisation.csv_file(name)
        assert path.exists() and path.read_text().count("\n") > 100
