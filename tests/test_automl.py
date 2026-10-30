import json
import subprocess
import sys

import pandas as pd
import pytest

from ml_agent import automl
from tests.conftest import make_churn, make_housing

FAST = {"time_budget": 5, "estimators": ["lgbm", "rf"]}


@pytest.fixture
def churn_experiment(churn_ws):
    record = automl.train(churn_ws, "churn", "churned", drop_columns=["customer_id"], **FAST)
    return churn_ws, record


def test_classification_experiment(churn_experiment):
    ws, record = churn_experiment
    assert record["experiment_id"] == "exp-001"
    assert record["task"] == "classification"
    assert record["metric"] == "roc_auc"
    assert record["positive_class"] == "yes"
    assert record["features"] == ["plan", "tenure", "tickets"]
    assert record["rows"] == {"train": 480, "test": 120, "dropped_missing_target": 0}
    assert record["test"]["baseline"] == 0.5
    assert record["test"]["beats_baseline"] is True
    assert record["test"]["roc_auc"] > 0.75
    assert {row["estimator"] for row in record["leaderboard"]} <= {"lgbm", "rf"}
    assert [p["threshold"] for p in record["operating_points"]][0] == 0.1
    assert record["test_metrics"]["confusion_matrix"]["labels"] == ["no", "yes"]
    exp_dir = ws.experiment_dir("exp-001")
    for name in ("model.joblib", "metadata.json", "test.parquet", "MODEL_CARD.md", "flaml.log"):
        assert (exp_dir / name).exists(), name
    assert "Decision thresholds" in (exp_dir / "MODEL_CARD.md").read_text()


def test_regression_experiment(ws):
    ws.save_dataset(make_housing(), "housing")
    record = automl.train(ws, "housing", "price", metric="mae", **FAST)
    assert record["task"] == "regression"
    assert record["lower_is_better"] is True
    assert record["test"]["mae"] < record["test"]["baseline"]
    assert record["test_metrics"]["r2"] > 0.5
    assert "operating_points" not in record


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"target": "nope"}, KeyError),
        ({"target": "churned", "metric": "rmse"}, ValueError),
        ({"target": "churned", "drop_columns": ["churned"]}, ValueError),
        ({"target": "churned", "estimators": ["deep_net"]}, ValueError),
        ({"target": "churned", "test_size": 0.9}, ValueError),
    ],
)
def test_invalid_training_requests(churn_ws, kwargs, error):
    with pytest.raises(error):
        automl.train(churn_ws, "churn", time_budget=5, **kwargs)
    assert churn_ws.experiment_ids() == []


def test_single_example_class_is_rejected(ws):
    df = pd.DataFrame({"x": range(30), "y": ["a"] * 15 + ["b"] * 14 + ["c"]})
    ws.save_dataset(df, "tiny")
    with pytest.raises(ValueError, match="fewer than 2 rows"):
        automl.train(ws, "tiny", "y", time_budget=5)


def test_explain_ranks_signal_above_noise(churn_experiment):
    ws, _ = churn_experiment
    result = automl.explain(ws, "exp-001", n_repeats=2)
    ranked = [row["feature"] for row in result["top_features"]]
    assert ranked[0] in {"tickets", "tenure"}
    assert ws.load_experiment("exp-001")["feature_importance"]
    assert "permutation importance" in (ws.experiment_dir("exp-001") / "MODEL_CARD.md").read_text()


def test_predict_with_threshold_and_scoring_against_labels(churn_experiment, tmp_path):
    ws, _ = churn_experiment
    fresh = make_churn(n=50, seed=9)
    path = tmp_path / "fresh.csv"
    fresh.to_csv(path, index=False)

    default = automl.predict(ws, "exp-001", str(path))
    low = automl.predict(ws, "exp-001", str(path), output="low.csv", threshold=0.05)
    assert default["rows_scored"] == 50
    assert set(default["columns_added"]) == {"prediction", "proba_no", "proba_yes"}
    assert low["prediction_distribution"].get("yes", 0) >= default["prediction_distribution"].get(
        "yes", 0
    )
    assert "roc_auc" in default["metrics_against_labels_in_input"]
    scored = pd.read_csv(low["output_path"])
    assert ((scored["proba_yes"] >= 0.05) == (scored["prediction"] == "yes")).all()


def test_predict_errors(churn_experiment, tmp_path):
    ws, _ = churn_experiment
    path = tmp_path / "partial.csv"
    make_churn(n=10).drop(columns=["tickets"]).to_csv(path, index=False)
    with pytest.raises(KeyError, match="tickets"):
        automl.predict(ws, "exp-001", str(path))
    with pytest.raises(ValueError, match="inside the workspace"):
        automl.predict(ws, "exp-001", "churn", output="../../escape.csv")
    with pytest.raises(ValueError, match="threshold"):
        automl.predict(ws, "exp-001", "churn", threshold=1.5)


def test_export_bundle_runs_standalone(churn_experiment, tmp_path):
    ws, _ = churn_experiment
    bundle = automl.export(ws, "exp-001", name="Churn Model", threshold=0.3)
    assert sorted(bundle["files"]) == sorted(
        ["MODEL_CARD.md", "metadata.json", "model.joblib", "predict.py", "requirements.txt"]
    )
    assert (
        json.loads((ws.exports_dir / "churn_model" / "metadata.json").read_text())[
            "decision_threshold"
        ]
        == 0.3
    )

    source, target = tmp_path / "in.csv", tmp_path / "out.csv"
    make_churn(n=20, seed=3).to_csv(source, index=False)
    proc = subprocess.run(
        [sys.executable, "predict.py", str(source), str(target)],
        cwd=bundle["bundle_path"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = pd.read_csv(target)
    assert len(out) == 20
    assert ((out["proba_yes"] >= 0.3) == (out["prediction"] == "yes")).all()
