"""Evaluation core of the agent team: splits, pipelines, candidates, comparison, final evaluation."""

import math

import numpy as np
import pandas as pd
import pytest

from ml_agent.team import diagnostics, evaluation
from ml_agent.team.catalog import FAMILIES, make_pipeline, unavailable_reason
from ml_agent.team.preprocess import coerce_features, infer_roles
from ml_agent.team.runs import RunStore
from ml_agent.team.schemas import ColumnRoles, ProblemSpec
from ml_agent.team.splits import cv_splits, make_splits
from tests.conftest import make_churn, make_housing


def make_spec(**overrides) -> ProblemSpec:
    fields = {
        "dataset": "churn",
        "target": "churned",
        "task": "classification",
        "primary_metric": "roc_auc",
        "secondary_metrics": ["accuracy", "log_loss"],
        "positive_class": "yes",
        "success_criteria": "Beat the baseline clearly",
        "min_primary_score": None,
        "cv_folds": 3,
        "test_size": 0.2,
        "group_column": None,
        "threshold_objective": "f1",
        "excluded_columns": ["customer_id", "cancel_reason"],
        "constraints": [],
        "assumptions": [],
    }
    return ProblemSpec(**{**fields, **overrides})


@pytest.fixture
def churn_run(ws):
    ws.save_dataset(make_churn(), "churn")
    run = RunStore.create(ws)
    run.save_spec(make_spec(), budget_seconds=900)
    run.create_splits()
    return run


@pytest.fixture
def housing_run(ws):
    ws.save_dataset(make_housing(), "housing")
    run = RunStore.create(ws)
    spec = make_spec(
        dataset="housing",
        target="price",
        task="regression",
        primary_metric="mae",
        secondary_metrics=["r2"],
        positive_class=None,
        threshold_objective=None,
        excluded_columns=[],
    )
    run.save_spec(spec, budget_seconds=900)
    run.create_splits()
    return run


# ---------------------------------------------------------------------------- splits


def test_splits_are_disjoint_stratified_and_reproducible():
    y = pd.Series(["a"] * 300 + ["b"] * 100)
    first = make_splits(y, "classification", test_size=0.25, n_folds=5, seed=1)
    second = make_splits(y, "classification", test_size=0.25, n_folds=5, seed=1)
    pd.testing.assert_frame_equal(first, second)
    test = first[first["subset"] == "test"]
    train = first[first["subset"] == "train"]
    assert len(test) == 100 and len(train) == 300
    assert (y[test.index] == "b").mean() == pytest.approx(0.25, abs=0.02)
    assert (test["tune_fold"] == -1).all() and (train["sel_fold_0"] >= 0).all()
    assert sorted(train["sel_fold_0"].unique()) == [0, 1, 2, 3, 4]
    # The two selection repeats and the tuning folds are different partitions.
    assert not train["sel_fold_0"].equals(train["sel_fold_1"])
    for fold in range(5):
        in_fold = y[train.index[train["sel_fold_0"] == fold]]
        assert (in_fold == "b").mean() == pytest.approx(0.25, abs=0.05)


def test_group_splits_keep_groups_together():
    rng = np.random.default_rng(0)
    groups = pd.Series(np.repeat(np.arange(80), 5))
    y = pd.Series(rng.choice(["x", "y"], len(groups)))
    frame = make_splits(y, "classification", test_size=0.2, n_folds=4, groups=groups)
    by_group = frame.groupby(groups)
    assert (by_group["subset"].nunique() == 1).all()
    assert (by_group["sel_fold_0"].nunique() == 1).all()
    assert (by_group["tune_fold"].nunique() == 1).all()


def test_cv_splits_cover_every_row_once():
    folds = np.array([0, 1, 2, 0, 1, 2, 2])
    splits = cv_splits(folds)
    assert len(splits) == 3
    assert sorted(np.concatenate([val for _, val in splits]).tolist()) == list(range(7))


def test_tiny_classes_are_rejected():
    with pytest.raises(ValueError, match="fewer than 2"):
        make_splits(pd.Series(["a"] * 20 + ["b"]), "classification", test_size=0.2, n_folds=3)


# -------------------------------------------------------------- roles and pipelines


def test_infer_roles_and_coerce():
    roles = infer_roles(make_churn(), "churned", ["cancel_reason"])
    assert roles.numeric == ["tenure", "tickets"]
    assert roles.categorical == ["plan"]
    dropped = {d.column: d.reason for d in roles.dropped}
    assert set(dropped) == {"customer_id", "signup_date", "constant", "cancel_reason"}
    X = coerce_features(make_churn(), roles)
    assert list(X.columns) == ["tenure", "tickets", "plan"]
    assert X["tenure"].dtype == float and X["plan"].map(type).eq(str).all()


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_every_family_fits_mixed_data_with_missing_values(family, task):
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], n).astype(object),
            "many": [f"v{i % 40}" for i in range(n)],
        }
    )
    df.loc[::6, "num"] = np.nan
    df.loc[::7, "cat"] = np.nan
    roles = ColumnRoles(numeric=["num"], categorical=["cat"], high_cardinality=["many"], dropped=[])
    X = coerce_features(df, roles)
    y = rng.integers(0, 2, n) if task == "classification" else rng.normal(size=n)
    model = make_pipeline(family, task, roles).fit(X, y)
    assert len(model.predict(X)) == n


def test_unavailable_reasons():
    assert "20000" in unavailable_reason("svm", "classification", 50_000, "roc_auc")
    assert "predict_proba" in unavailable_reason("svm", "classification", 100, "log_loss")
    assert unavailable_reason("hist_gb", "classification", 50_000, "roc_auc") is None
    assert "unknown" in unavailable_reason("deep_net", "classification", 10, "roc_auc")


# ----------------------------------------------------------------------- run store


def test_run_store_split_and_roles(churn_run):
    meta = churn_run.meta
    assert meta["classes"] == ["no", "yes"]
    X, y = churn_run.xy_train()
    assert len(X) == 480 and set(np.unique(y)) == {0, 1}
    assert churn_run.roles.features == ["tenure", "tickets", "plan"]
    assert len(churn_run.selection_cv()) == 6 and len(churn_run.tune_cv()) == 3


def test_register_features_validates_prepare(churn_run):
    churn_run.features_path.write_text(
        "import pandas as pd\n"
        "def prepare(df):\n"
        "    df = df.copy()\n"
        "    df['signup_month'] = pd.to_datetime(df['signup_date']).dt.month\n"
        "    return df\n"
    )
    before = churn_run.features_hash
    result = churn_run.register_features()
    assert result["columns_added"] == ["signup_month"]
    assert "signup_month" in churn_run.roles.numeric
    assert churn_run.features_hash != before

    churn_run.features_path.write_text(
        "def prepare(df):\n    df = df.copy()\n    df['leak'] = (df['churned'] == 'yes').astype(int)\n    return df\n"
    )
    with pytest.raises(ValueError, match="must not use the target"):
        churn_run.register_features()

    churn_run.features_path.write_text("def prepare(df):\n    return df[df['tenure'] > 5]\n")
    with pytest.raises(ValueError, match="one row per input row"):
        churn_run.register_features()


def test_set_roles_validation(churn_run):
    roles = churn_run.roles
    with pytest.raises(ValueError, match="not numeric"):
        churn_run.set_roles(
            roles.model_copy(update={"numeric": [*roles.numeric, "plan"], "categorical": []})
        )
    with pytest.raises(ValueError, match="target"):
        churn_run.set_roles(roles.model_copy(update={"numeric": [*roles.numeric, "churned"]}))


def test_binary_only_metric_rejected_for_multiclass(ws):
    df = make_housing()
    df["band"] = pd.cut(df["price"], 3, labels=["low", "mid", "high"]).astype(str)
    ws.save_dataset(df.drop(columns=["price"]), "bands")
    run = RunStore.create(ws)
    with pytest.raises(ValueError, match="binary target"):
        run.save_spec(
            make_spec(dataset="bands", target="band", positive_class=None, excluded_columns=[]),
            budget_seconds=60,
        )


# ------------------------------------------------------------------- candidates


def test_all_candidates_share_identical_folds(churn_run, monkeypatch):
    seen = []
    original = evaluation.cross_validate

    def spy(estimator, X, y, *, cv, **kwargs):
        seen.append([(tuple(tr), tuple(va)) for tr, va in cv])
        return original(estimator, X, y, cv=cv, **kwargs)

    monkeypatch.setattr(evaluation, "cross_validate", spy)
    evaluation.tune_family(churn_run, "dummy", budget_seconds=5)
    evaluation.tune_family(churn_run, "linear", budget_seconds=3, max_iter=3)
    evaluation.flaml_competitor(churn_run, time_budget=5)
    assert len(seen) == 3 and seen[0] == seen[1] == seen[2]


def test_tuned_candidate_record(churn_run):
    record = evaluation.tune_family(
        churn_run, "hist_gb", budget_seconds=5, max_iter=3, notes="first try"
    )
    assert record["id"] == "c01-hist_gb"
    assert record["tuning"]["configurations_tried"] >= 2
    assert len(record["cv"]["roc_auc"]["folds"]) == 6
    assert set(record["cv"]) == {"roc_auc", "accuracy", "log_loss"}
    assert record["cv"]["roc_auc"]["mean"] > 0.75
    oof = churn_run.candidate_oof(record["id"])
    assert len(oof) == 480 and {"p_0", "p_1"} <= set(oof.columns)
    assert churn_run.meta["compute_seconds"] > 0


def test_svm_skips_probability_metrics(churn_run):
    record = evaluation.tune_family(churn_run, "svm", budget_seconds=3, max_iter=2)
    assert "log_loss" not in record["cv"] and record["has_proba"] is False


def test_corrected_ttest_matches_hand_computation():
    diffs = np.array([0.02, 0.01, 0.03, 0.00, 0.02, 0.01])
    t, p = evaluation.corrected_ttest(diffs, test_train_ratio=0.5)
    expected_t = diffs.mean() / math.sqrt((1 / 6 + 0.5) * diffs.var(ddof=1))
    assert t == pytest.approx(expected_t)
    from scipy import stats

    assert p == pytest.approx(2 * stats.t.sf(abs(expected_t), df=5))
    assert evaluation.corrected_ttest(np.zeros(4), 0.5) == (0.0, 1.0)


def _fake_candidate(run, cid, family, complexity, folds):
    record = {
        "id": cid,
        "family": family,
        "source": "sklearn",
        "complexity": complexity,
        "features_hash": run.features_hash,
        "cv": {
            "roc_auc": {
                "mean": float(np.mean(folds)),
                "std": float(np.std(folds, ddof=1)),
                "folds": folds,
            }
        },
        "train": {"roc_auc": 1.0},
        "has_proba": True,
        "threshold": None,
    }
    run.save_candidate(record, estimator=None)


def test_one_se_rule_prefers_simpler_model(churn_run):
    _fake_candidate(churn_run, "c01-hist_gb", "hist_gb", 3, [0.80, 0.82, 0.81, 0.83, 0.80, 0.82])
    _fake_candidate(churn_run, "c02-linear", "linear", 1, [0.80, 0.81, 0.80, 0.82, 0.80, 0.81])
    _fake_candidate(churn_run, "c03-dummy", "dummy", 0, [0.5] * 6)
    result = evaluation.compare(churn_run)
    assert result["best"] == "c01-hist_gb"
    assert result["one_se_recommendation"] == "c02-linear"
    rows = {r["id"]: r for r in result["comparisons"]}
    assert rows["c03-dummy"]["within_1se_of_best"] is False
    assert rows["c03-dummy"]["p_value_vs_best"] < 0.001
    board = evaluation.leaderboard(churn_run)
    assert [r["id"] for r in board["candidates"]] == ["c01-hist_gb", "c02-linear", "c03-dummy"]


def test_stale_candidates_are_excluded_after_feature_change(churn_run):
    evaluation.tune_family(churn_run, "dummy", budget_seconds=5)
    churn_run.features_path.write_text(
        "def prepare(df):\n    return df.assign(t2=df['tenure'] * 2)\n"
    )
    churn_run.register_features()
    board = evaluation.leaderboard(churn_run)
    assert board["candidates"] == [] and board["stale_candidates"] == 1


def test_ensemble_and_threshold(churn_run):
    a = evaluation.tune_family(churn_run, "linear", budget_seconds=3, max_iter=2)
    b = evaluation.tune_family(churn_run, "random_forest", budget_seconds=3, max_iter=2)
    ensemble = evaluation.build_ensemble(churn_run, [a["id"], b["id"]], kind="voting")
    assert ensemble["family"] == "ensemble_voting" and ensemble["complexity"] == 5
    result = evaluation.tune_threshold(churn_run, b["id"], "f2")
    assert 0 < result["threshold"] < 1
    assert churn_run.candidate(b["id"])["threshold"] == pytest.approx(result["threshold"], rel=1e-3)
    assert (
        result["selection_cv_at_tuned_threshold"]["recall"]
        >= result["selection_cv_at_default_threshold"]["recall"]
    )


def test_budget_is_enforced(churn_run):
    churn_run.update_meta(compute_seconds=churn_run.meta["budget_seconds"])
    with pytest.raises(RuntimeError, match="budget is used up"):
        evaluation.tune_family(churn_run, "linear", budget_seconds=5)


# ----------------------------------------------------------------- final evaluation


def test_final_evaluation_reads_test_once(churn_run, monkeypatch):
    record = evaluation.tune_family(churn_run, "hist_gb", budget_seconds=3, max_iter=2)
    evaluation.tune_threshold(churn_run, record["id"])
    calls = []
    original = RunStore.load_test

    def counting(self, champion_id):
        calls.append(champion_id)
        return original(self, champion_id)

    monkeypatch.setattr(RunStore, "load_test", counting)
    result = evaluation.final_evaluation(churn_run, record["id"])
    again = evaluation.final_evaluation(churn_run, record["id"])
    assert calls == [record["id"]] and again == result
    assert result["rows"] == {"train": 480, "test": 120}
    assert result["beats_baseline"] is True and result["test"]["baseline"] == 0.5
    assert result["threshold"] == churn_run.candidate(record["id"])["threshold"]
    assert result["cv_consistency"]["within_2_std_of_cv"] in (True, False)
    assert {r["feature"] for r in result["permutation_importance"]} == {"tenure", "tickets", "plan"}
    assert (churn_run.root / "champion" / "model.joblib").exists()

    other = evaluation.tune_family(churn_run, "linear", budget_seconds=2, max_iter=2)
    with pytest.raises(RuntimeError, match="already used"):
        evaluation.final_evaluation(churn_run, other["id"])


def test_regression_run(housing_run):
    record = evaluation.tune_family(housing_run, "hist_gb", budget_seconds=5, max_iter=3)
    dummy = evaluation.tune_family(housing_run, "dummy", budget_seconds=5)
    assert record["cv"]["mae"]["mean"] < dummy["cv"]["mae"]["mean"]
    assert evaluation.compare(housing_run)["best"] == record["id"]
    final = evaluation.final_evaluation(housing_run, record["id"])
    assert final["lower_is_better"] is True and final["beats_baseline"] is True


# ------------------------------------------------------------------- diagnostics


def test_diagnostics(churn_run):
    assoc = diagnostics.feature_associations(churn_run)
    assert {r["feature"] for r in assoc["top_features"]} >= {"tenure", "tickets", "plan"}
    baselines = diagnostics.quick_baselines(churn_run)
    assert baselines["results"][0]["roc_auc"] == 0.5
    best = evaluation.tune_family(churn_run, "hist_gb", budget_seconds=3, max_iter=2)
    evaluation.tune_family(churn_run, "dummy", budget_seconds=3)
    assert diagnostics.check_overfitting(churn_run)["candidates"][0]["gap"] is not None
    leakage = diagnostics.check_leakage(churn_run)
    assert leakage["suspicious"] == []  # no feature alone explains the target
    assert diagnostics.check_success(churn_run, best["id"])["beats_dummy"] is True
    assert diagnostics.rescore(churn_run, best["id"])["reproducible"] is True


def test_leak_is_detected(ws):
    df = make_churn()
    ws.save_dataset(df.assign(leaky=np.where(df["churned"] == "yes", 1.0, 0.0)), "churn")
    run = RunStore.create(ws)
    run.save_spec(make_spec(), budget_seconds=600)
    run.create_splits()
    evaluation.tune_family(run, "hist_gb", budget_seconds=3, max_iter=2)
    assert "leaky" in diagnostics.check_leakage(run)["suspicious"]


def test_export_bundle_matches_in_process_predictions(churn_run, tmp_path):
    import subprocess
    import sys

    import joblib

    from ml_agent.team.export import export_champion
    from ml_agent.team.preprocess import coerce_features
    from ml_agent.team.runs import load_prepare

    churn_run.features_path.write_text(
        "import pandas as pd\n"
        "def prepare(df):\n"
        "    df = df.copy()\n"
        "    df['signup_month'] = pd.to_datetime(df['signup_date']).dt.month\n"
        "    return df\n"
    )
    churn_run.register_features()
    record = evaluation.tune_family(churn_run, "random_forest", budget_seconds=3, max_iter=2)
    evaluation.tune_threshold(churn_run, record["id"])
    evaluation.final_evaluation(churn_run, record["id"])
    bundle = export_champion(churn_run, name="churn")
    assert {"features.py", "predict.py", "model.joblib", "MODEL_CARD.md"} <= set(bundle["files"])

    fresh = make_churn(n=40, seed=11).drop(columns=["churned"])
    source, target = tmp_path / "fresh.csv", tmp_path / "scored.csv"
    fresh.to_csv(source, index=False)
    proc = subprocess.run(
        [sys.executable, "predict.py", str(source), str(target)],
        cwd=bundle["bundle_path"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    scored = pd.read_csv(target)

    model = joblib.load(churn_run.root / "champion" / "model.joblib")
    X = coerce_features(load_prepare(churn_run.features_path)(pd.read_csv(source)), churn_run.roles)
    expected = np.asarray(churn_run.classes, dtype=object)[model.predict(X)]
    assert scored["prediction"].tolist() == expected.tolist()
    assert np.allclose(scored["proba_yes"], model.predict_proba(X)[:, 1])
