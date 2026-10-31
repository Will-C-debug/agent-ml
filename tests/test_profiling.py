import pandas as pd
import pytest

from ml_agent.profiling import infer_task, profile_dataframe
from tests.conftest import make_churn, make_housing


@pytest.mark.parametrize(
    ("values", "task"),
    [
        (["yes", "no", "yes"], "classification"),
        ([True, False, True], "classification"),
        ([0, 1, 1, 0], "classification"),
        ([0.0, 1.0, 1.0, 0.0], "classification"),
        (list(range(100)), "regression"),
        ([1.5, 2.25, 3.0], "regression"),
    ],
)
def test_infer_task(values, task):
    assert infer_task(pd.Series(values)) == task


def test_profile_flags_data_quality_issues():
    profile = profile_dataframe(make_churn(), target="churned")
    kinds = {c["name"]: c["kind"] for c in profile["column_profiles"]}
    assert kinds["customer_id"] == "id_like"
    assert kinds["signup_date"] == "date_string"
    assert kinds["constant"] == "constant"
    assert kinds["tenure"] == "numeric"
    target = profile["target"]
    assert target["inferred_task"] == "classification"
    assert target["n_classes"] == 2
    # cancel_reason is only filled in for churned customers: a leak via missingness.
    assert target["possible_leakage"] == ["cancel_reason"]
    assert any("leakage" in w for w in profile["warnings"])


def test_profile_regression_target_and_column_subset():
    profile = profile_dataframe(make_housing(), target="price", columns=["size_m2"])
    assert [c["name"] for c in profile["column_profiles"]] == ["size_m2"]
    assert profile["target"]["inferred_task"] == "regression"
    assert "size_m2" in profile["target"]["top_numeric_correlations"]
    assert "possible_leakage" not in profile["target"]


def test_profile_unknown_target():
    with pytest.raises(KeyError):
        profile_dataframe(make_housing(), target="nope")
