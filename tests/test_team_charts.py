import pytest

from ml_agent.team import evaluation
from ml_agent.team.charts import CHARTS, make_chart
from tests.test_team_core import churn_run, housing_run  # noqa: F401  (fixtures)

BINARY = [
    "leaderboard",
    "paired_differences",
    "roc",
    "precision_recall",
    "confusion_matrix",
    "calibration",
    "feature_importance",
    "partial_dependence",
    "learning_curve",
    "target_distribution",
]


@pytest.fixture
def finished_churn_run(churn_run):  # noqa: F811
    evaluation.tune_family(churn_run, "dummy", budget_seconds=3)
    evaluation.tune_family(churn_run, "linear", budget_seconds=3, max_iter=2)
    best = evaluation.tune_family(churn_run, "hist_gb", budget_seconds=3, max_iter=2)
    evaluation.final_evaluation(churn_run, best["id"])
    return churn_run


def test_every_binary_chart_renders(finished_churn_run):
    for kind in BINARY:
        result = make_chart(finished_churn_run, kind)
        path = finished_churn_run.root / result["relative_path"]
        assert path.stat().st_size > 5_000, kind
        assert result["facts"], kind
    roc = make_chart(finished_churn_run, "roc")["facts"]
    assert len(roc["curves"]) == 3 and all(0.4 < c["auc"] <= 1 for c in roc["curves"])


def test_regression_charts_and_validation(housing_run):  # noqa: F811
    best = evaluation.tune_family(housing_run, "hist_gb", budget_seconds=3, max_iter=2)
    evaluation.final_evaluation(housing_run, best["id"])
    for kind in ("prediction_error", "target_distribution", "feature_importance", "leaderboard"):
        assert make_chart(housing_run, kind)["facts"]
    with pytest.raises(ValueError, match="binary"):
        make_chart(housing_run, "roc")
    with pytest.raises(ValueError, match="Unknown chart"):
        make_chart(housing_run, "pie")
    assert set(BINARY) | {"prediction_error"} == set(CHARTS)


def test_test_set_charts_need_final_evaluation(churn_run):  # noqa: F811
    evaluation.tune_family(churn_run, "linear", budget_seconds=3, max_iter=2)
    with pytest.raises(ValueError, match="final"):
        make_chart(churn_run, "confusion_matrix")
