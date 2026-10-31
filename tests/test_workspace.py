import pandas as pd
import pytest

from ml_agent.workspace import slugify


def test_import_and_reload(ws, churn_csv):
    info = ws.import_file(str(churn_csv), name="My Churn")
    assert info["name"] == "my_churn"
    assert info["rows"] == 600
    assert ws.dataset_names() == ["my_churn"]
    assert len(ws.load_dataset("my_churn")) == 600


def test_mixed_type_columns_are_saved(ws):
    df = pd.DataFrame({"mixed": [1, "a", None, 2.5], "y": [0, 1, 0, 1]})
    ws.save_dataset(df, "mixed")
    assert ws.load_dataset("mixed")["mixed"].tolist()[:2] == ["1", "a"]


def test_unknown_dataset_lists_available(ws):
    with pytest.raises(KeyError, match="Available datasets"):
        ws.load_dataset("missing")


def test_outputs_must_stay_in_workspace(ws, tmp_path):
    with pytest.raises(ValueError, match="inside the workspace"):
        ws.resolve_output("../../escape.csv", ws.predictions_dir)
    assert ws.resolve_output("ok.csv", ws.predictions_dir).parent == ws.predictions_dir


def test_experiment_ids_increment(ws):
    assert ws.new_experiment_dir()[0] == "exp-001"
    assert ws.new_experiment_dir()[0] == "exp-002"


def test_slugify():
    assert slugify(" Churn Data v2.csv ") == "churn_data_v2_csv"
    with pytest.raises(ValueError):
        slugify("///")
