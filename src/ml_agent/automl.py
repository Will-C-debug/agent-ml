"""AutoML training, evaluation, explanation, prediction and export, built on FLAML.

Every training run follows the same protocol so results are comparable:

1. Rows with a missing target are dropped and a stratified (for classification) hold-out test
   split is carved off *before* the search starts.
2. FLAML searches estimators and hyperparameters on the training split only, using its own
   internal validation, within a wall-clock budget, then refits the best configuration.
3. The refit model is scored once on the untouched test split, next to a naive baseline
   (majority class / mean), so every reported number is out-of-sample.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn import metrics as skm
from sklearn.model_selection import train_test_split

from ml_agent.profiling import infer_task, round_sig
from ml_agent.workspace import Workspace, now_iso, read_table, slugify, write_parquet

TASKS = ("classification", "regression")
CLASSIFICATION_METRICS = (
    "accuracy",
    "roc_auc",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "log_loss",
    "f1",
    "micro_f1",
    "macro_f1",
    "ap",
)
REGRESSION_METRICS = ("r2", "rmse", "mae", "mse", "mape")
LOWER_IS_BETTER = {"log_loss", "rmse", "mae", "mse", "mape"}
# Estimators that run in-process on a single machine (FLAML also ships Spark variants).
LOCAL_ESTIMATORS = (
    "lgbm",
    "xgboost",
    "xgb_limitdepth",
    "rf",
    "extra_tree",
    "histgb",
    "catboost",
    "lrl1",
    "lrl2",
    "kneighbor",
    "sgd",
    "svc",
    "enet",
    "lassolars",
)
AVAILABLE_ESTIMATORS = tuple(
    e for e in LOCAL_ESTIMATORS if e != "catboost" or find_spec("catboost")
)
MAX_EXPLAIN_ROWS = 5000


# --------------------------------------------------------------------------- metrics


def default_metric(task: str, n_classes: int) -> str:
    """FLAML's own defaults: ROC AUC for binary, log loss for multiclass, R² for regression."""
    if task == "regression":
        return "r2"
    return "roc_auc" if n_classes == 2 else "log_loss"


def score(
    metric: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    classes: list[Any] | None = None,
) -> float | None:
    """Compute one FLAML-named metric on held-out data (``None`` if undefined for this data)."""
    try:
        if metric == "accuracy":
            return skm.accuracy_score(y_true, y_pred)
        if metric == "f1":
            return skm.f1_score(y_true, y_pred, pos_label=classes[1])
        if metric == "macro_f1":
            return skm.f1_score(y_true, y_pred, average="macro")
        if metric == "micro_f1":
            return skm.f1_score(y_true, y_pred, average="micro")
        if metric in ("roc_auc", "roc_auc_ovr", "roc_auc_ovo"):
            if len(classes) == 2:
                return skm.roc_auc_score(y_true == classes[1], proba[:, 1])
            mode = "ovo" if metric == "roc_auc_ovo" else "ovr"
            return skm.roc_auc_score(y_true, proba, multi_class=mode, labels=classes)
        if metric == "ap":
            if len(classes) == 2:
                return skm.average_precision_score(y_true == classes[1], proba[:, 1])
            return None
        if metric == "log_loss":
            return skm.log_loss(y_true, proba, labels=classes)
        if metric == "r2":
            return skm.r2_score(y_true, y_pred)
        if metric == "rmse":
            return skm.root_mean_squared_error(y_true, y_pred)
        if metric == "mse":
            return skm.mean_squared_error(y_true, y_pred)
        if metric == "mae":
            return skm.mean_absolute_error(y_true, y_pred)
        if metric == "mape":
            return (
                None if (y_true == 0).any() else skm.mean_absolute_percentage_error(y_true, y_pred)
            )
    except ValueError:
        # e.g. ROC AUC when the test split contains a single class.
        return None
    raise ValueError(f"Unknown metric {metric!r}")


def evaluate(
    task: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    classes: list[Any] | None = None,
) -> dict[str, Any]:
    """Standard held-out metrics for a task."""
    if task == "regression":
        names = list(REGRESSION_METRICS)
    else:
        names = ["accuracy", "macro_f1", "log_loss"]
        names += ["roc_auc", "f1", "ap"] if len(classes) == 2 else ["roc_auc_ovr"]
    result = {name: round_sig(score(name, y_true, y_pred, proba, classes)) for name in names}
    if task == "classification":
        result["balanced_accuracy"] = round_sig(skm.balanced_accuracy_score(y_true, y_pred))
        result["confusion_matrix"] = {
            "labels": [str(c) for c in classes],
            "matrix": skm.confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        }
        report = skm.classification_report(
            y_true, y_pred, labels=classes, output_dict=True, zero_division=0
        )
        result["per_class"] = {
            str(c): {
                k: round_sig(report[str(c)][k])
                for k in ("precision", "recall", "f1-score", "support")
            }
            for c in classes[:20]
            if str(c) in report
        }
    return result


def _predict(model: AutoML, X: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray | None]:
    y_pred = np.asarray(model.predict(X))
    proba = np.asarray(model.predict_proba(X)) if task == "classification" else None
    return y_pred, proba


def _baseline(task: str, y_train: pd.Series, n_rows: int, classes: list[Any] | None):
    """Predictions of a model that ignores the features: majority class or training mean."""
    if task == "regression":
        return np.full(n_rows, y_train.mean()), None
    priors = y_train.value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy()
    return np.full(n_rows, y_train.mode().iloc[0]), np.tile(priors, (n_rows, 1))


def _is_better(metric: str, value: float | None, baseline: float | None) -> bool | None:
    if value is None or baseline is None:
        return None
    return value < baseline if metric in LOWER_IS_BETTER else value > baseline


def operating_points(y_true: pd.Series, proba_positive: np.ndarray, positive: Any) -> list[dict]:
    """Precision/recall trade-off of a binary classifier at several decision thresholds."""
    actual = np.asarray(y_true) == positive
    rows = []
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        flagged = proba_positive >= threshold
        hits = int((flagged & actual).sum())
        precision = hits / flagged.sum() if flagged.any() else None
        recall = hits / actual.sum() if actual.any() else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
        rows.append(
            {
                "threshold": threshold,
                "precision": round_sig(precision, 3),
                "recall": round_sig(recall, 3),
                "f1": round_sig(f1, 3),
                "flagged_pct": round_sig(100 * flagged.mean(), 3),
            }
        )
    return rows


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return round_sig(obj, 6)


# -------------------------------------------------------------------------- training


def train(
    ws: Workspace,
    dataset: str,
    target: str,
    *,
    task: str = "auto",
    metric: str = "auto",
    time_budget: int = 60,
    estimators: list[str] | None = None,
    drop_columns: list[str] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    max_time_budget: int = 900,
    notes: str | None = None,
) -> dict[str, Any]:
    """Run one AutoML experiment and persist the model, test split and metadata."""
    df = ws.load_dataset(dataset)
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not in dataset {dataset!r}: {list(df.columns)}")
    drop_columns = list(drop_columns or [])
    unknown = [c for c in drop_columns if c not in df.columns]
    if unknown:
        raise KeyError(f"drop_columns not found in dataset: {unknown}")
    if target in drop_columns:
        raise ValueError("The target column cannot also be dropped")
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5")
    if time_budget < 5:
        raise ValueError("time_budget must be at least 5 seconds")
    requested_budget = time_budget
    time_budget = min(time_budget, max_time_budget)

    n_missing_target = int(df[target].isna().sum())
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target, *drop_columns])
    y = df[target]
    if X.shape[1] == 0:
        raise ValueError("No feature columns left after dropping the target and drop_columns")

    task = infer_task(y) if task == "auto" else task
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS} or 'auto'")
    if task == "classification":
        counts = y.value_counts()
        if len(counts) < 2:
            raise ValueError("Classification needs at least two classes in the target")
        if (counts < 2).any():
            raise ValueError(
                f"Classes with fewer than 2 rows cannot be split: {list(counts[counts < 2].index)}. "
                "Merge or drop them first."
            )
        n_classes = len(counts)
    else:
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(f"Regression needs a numeric target; {target!r} is {y.dtype}")
        n_classes = 0
    metric = default_metric(task, n_classes) if metric == "auto" else metric
    allowed = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
    if metric not in allowed:
        raise ValueError(f"metric for {task} must be one of {allowed}")
    if metric in ("f1", "ap") and n_classes != 2:
        raise ValueError(f"{metric!r} is for binary targets; use macro_f1 or roc_auc_ovr")
    if estimators:
        bad = [e for e in estimators if e not in AVAILABLE_ESTIMATORS]
        if bad:
            raise ValueError(f"Unknown estimators {bad}. Choose from {AVAILABLE_ESTIMATORS}")

    stratify = y if task == "classification" else None
    min_test = n_classes if task == "classification" else 1
    if len(y) * test_size < max(min_test, 1):
        raise ValueError("Dataset is too small for a held-out test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    experiment_id, exp_dir = ws.new_experiment_dir()
    automl = AutoML()
    started = time.time()
    try:
        automl.fit(
            X_train,
            y_train,
            task=task,
            metric=metric,
            time_budget=time_budget,
            estimator_list=estimators or "auto",
            seed=seed,
            n_jobs=-1,
            verbose=0,
            log_file_name=str(exp_dir / "flaml.log"),
        )
    except BaseException:
        shutil.rmtree(exp_dir, ignore_errors=True)
        raise
    train_seconds = time.time() - started
    if automl.model is None:
        shutil.rmtree(exp_dir, ignore_errors=True)
        raise RuntimeError(
            "AutoML did not finish a single trial within the time budget; increase time_budget."
        )

    classes = list(automl.classes_) if task == "classification" else None
    y_pred, proba = _predict(automl, X_test, task)
    test_metrics = evaluate(task, y_test, y_pred, proba, classes)
    base_pred, base_proba = _baseline(task, y_train, len(y_test), classes)
    baseline_metrics = evaluate(task, y_test, base_pred, base_proba, classes)
    for key in ("confusion_matrix", "per_class"):
        baseline_metrics.pop(key, None)
    primary_test = round_sig(score(metric, y_test, y_pred, proba, classes))
    primary_baseline = round_sig(score(metric, y_test, base_pred, base_proba, classes))

    best_loss = float(automl.best_loss)
    validation_score = best_loss if metric in LOWER_IS_BETTER else 1 - best_loss
    leaderboard = sorted(
        (
            {
                "estimator": name,
                f"validation_{metric}": round_sig(loss if metric in LOWER_IS_BETTER else 1 - loss),
            }
            for name, loss in automl.best_loss_per_estimator.items()
            if loss is not None and np.isfinite(loss)
        ),
        key=lambda row: row[f"validation_{metric}"],
        reverse=metric not in LOWER_IS_BETTER,
    )

    joblib.dump(automl, exp_dir / "model.joblib")
    test_frame = X_test.copy()
    test_frame[target] = y_test
    write_parquet(test_frame, exp_dir / "test.parquet")

    record = {
        "experiment_id": experiment_id,
        "created_at": now_iso(),
        "dataset": slugify(dataset),
        "target": target,
        "task": task,
        "metric": metric,
        "lower_is_better": metric in LOWER_IS_BETTER,
        "classes": [str(c) for c in classes] if classes else None,
        "positive_class": str(classes[1]) if classes and len(classes) == 2 else None,
        "features": list(X.columns),
        "feature_dtypes": {c: str(t) for c, t in X.dtypes.items()},
        "dropped_columns": drop_columns,
        "rows": {
            "train": len(X_train),
            "test": len(X_test),
            "dropped_missing_target": n_missing_target,
        },
        "time_budget": {"requested": requested_budget, "used": time_budget},
        "search": {
            "seconds": round(train_seconds, 1),
            "trials": len(automl.config_history),
            "seconds_to_best": round_sig(automl.time_to_find_best_model),
            "estimators": estimators or "auto",
            "seed": seed,
        },
        "best_estimator": automl.best_estimator,
        "best_config": _jsonable(automl.best_config),
        "validation": {metric: round_sig(validation_score)},
        "leaderboard": leaderboard,
        "test": {
            metric: primary_test,
            "baseline": primary_baseline,
            "beats_baseline": _is_better(metric, primary_test, primary_baseline),
        },
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "notes": notes,
    }
    if classes and len(classes) == 2:
        record["operating_points"] = operating_points(y_test, proba[:, 1], classes[1])
    ws.save_experiment(experiment_id, record)
    write_model_card(ws, experiment_id)
    return record


# ---------------------------------------------------------------------- inspection


def summarize_experiment(record: dict[str, Any]) -> dict[str, Any]:
    """The headline numbers of an experiment, for leaderboards."""
    metric = record["metric"]
    return {
        "experiment_id": record["experiment_id"],
        "dataset": record["dataset"],
        "target": record["target"],
        "task": record["task"],
        "metric": metric,
        "best_estimator": record["best_estimator"],
        "validation": record["validation"][metric],
        "test": record["test"][metric],
        "baseline": record["test"]["baseline"],
        "n_features": len(record["features"]),
        "budget_s": record["time_budget"]["used"],
        "notes": record.get("notes"),
    }


def load_model(ws: Workspace, experiment_id: str) -> tuple[AutoML, dict[str, Any]]:
    exp_dir = ws.experiment_dir(experiment_id)
    return joblib.load(exp_dir / "model.joblib"), ws.load_experiment(experiment_id)


def explain(
    ws: Workspace, experiment_id: str, *, n_repeats: int = 3, top_k: int = 15, seed: int = 0
) -> dict[str, Any]:
    """Permutation importance on the held-out test split, in units of the experiment's metric.

    Each feature is shuffled ``n_repeats`` times; its importance is how much the test score
    degrades. Model-agnostic, so it works for every estimator FLAML may pick.
    """
    model, record = load_model(ws, experiment_id)
    test = pd.read_parquet(ws.experiment_dir(experiment_id) / "test.parquet")
    if len(test) > MAX_EXPLAIN_ROWS:
        test = test.sample(MAX_EXPLAIN_ROWS, random_state=seed)
    target, task, metric = record["target"], record["task"], record["metric"]
    X, y = test[record["features"]], test[target]
    classes = list(model.classes_) if task == "classification" else None
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0

    def oriented_score(frame: pd.DataFrame) -> float:
        y_pred, proba = _predict(model, frame, task)
        value = score(metric, y, y_pred, proba, classes)
        if value is None:
            raise ValueError(f"Metric {metric!r} is undefined on this test split")
        return sign * value

    reference = oriented_score(X)
    rng = np.random.default_rng(seed)
    importances = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            # Keep the column's dtype so the model's preprocessing sees the same types.
            shuffled[col] = pd.Series(X[col].iloc[rng.permutation(len(X))].array, index=X.index)
            drops.append(reference - oriented_score(shuffled))
        importances.append(
            {
                "feature": col,
                "importance": round_sig(np.mean(drops)),
                "std": round_sig(np.std(drops)),
            }
        )
    importances.sort(key=lambda row: row["importance"] or 0, reverse=True)
    unhelpful = [row["feature"] for row in importances if (row["importance"] or 0) <= 0]

    result = {
        "experiment_id": experiment_id,
        "method": f"permutation importance on {len(X)} held-out rows, {n_repeats} repeats",
        "metric": metric,
        "test_score": round_sig(sign * reference),
        "top_features": importances[:top_k],
        "features_with_no_measurable_effect": unhelpful,
    }
    record["feature_importance"] = importances
    ws.save_experiment(experiment_id, record)
    write_model_card(ws, experiment_id)
    return result


# ---------------------------------------------------------------------- prediction


def predict_frame(
    model: AutoML, record: dict[str, Any], df: pd.DataFrame, threshold: float | None = None
) -> pd.DataFrame:
    """Append prediction (and class-probability) columns to ``df``.

    ``threshold`` (binary classification only) predicts the positive class when its probability
    is at least that value, instead of the model's default 0.5 cut-off.
    """
    if threshold is not None:
        if record["task"] != "classification" or len(model.classes_) != 2:
            raise ValueError("threshold only applies to binary classification models")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
    missing = [c for c in record["features"] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Input is missing feature columns the model needs: {missing}. If these were "
            "engineered during training, apply the same transformation to this data first "
            "(run_python + save_dataset), then predict on the resulting dataset."
        )
    X = df[record["features"]]
    out = df.copy()
    out["prediction"] = model.predict(X)
    if record["task"] == "classification":
        proba = model.predict_proba(X)
        for i, cls in enumerate(model.classes_):
            out[f"proba_{cls}"] = proba[:, i]
        if threshold is not None:
            negative, positive = model.classes_
            out["prediction"] = np.where(proba[:, 1] >= threshold, positive, negative)
    return out


def predict(
    ws: Workspace,
    experiment_id: str,
    source: str,
    output: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Score a dataset name or local file and write the predictions as CSV inside the workspace."""
    model, record = load_model(ws, experiment_id)
    if ws.has_dataset(source):
        df, source_label = ws.load_dataset(source), f"dataset:{slugify(source)}"
    else:
        path = ws.resolve_input(source)
        df, source_label = read_table(path), str(path)
    scored = predict_frame(model, record, df, threshold)

    default_name = f"{experiment_id}_{Path(source).stem}_predictions.csv"
    out_path = ws.resolve_output(output or default_name, ws.predictions_dir)
    scored.to_csv(out_path, index=False)

    summary: dict[str, Any] = {
        "experiment_id": experiment_id,
        "input": source_label,
        "rows_scored": len(scored),
        "output_path": str(out_path),
        "columns_added": [c for c in scored.columns if c not in df.columns],
    }
    if threshold is not None:
        summary["threshold"] = threshold
    if record["task"] == "classification":
        dist = scored["prediction"].astype(str).value_counts(normalize=True).head(20)
        summary["prediction_distribution"] = {k: round_sig(v, 3) for k, v in dist.items()}
    else:
        desc = scored["prediction"].describe()
        summary["prediction_stats"] = {
            k: round_sig(desc[k]) for k in ("mean", "std", "min", "50%", "max")
        }
    if record["target"] in df.columns:
        mask = df[record["target"]].notna()
        y_pred = scored.loc[mask, "prediction"].to_numpy()
        proba_cols = (
            [f"proba_{c}" for c in model.classes_] if record["task"] == "classification" else []
        )
        proba = scored.loc[mask, proba_cols].to_numpy() if proba_cols else None
        classes = list(model.classes_) if proba_cols else None
        summary["metrics_against_labels_in_input"] = {
            record["metric"]: round_sig(
                score(record["metric"], df.loc[mask, record["target"]], y_pred, proba, classes)
            ),
            "warning": "Only meaningful if these rows were not used for training.",
        }
    return summary


# -------------------------------------------------------------------------- export

_PREDICT_SCRIPT = '''\
"""Batch inference for {experiment_id} ({task} on '{target}', best estimator: {estimator}).

Usage:
    pip install -r requirements.txt
    python predict.py input.csv [output.csv]

The input must contain these feature columns (extra columns are kept in the output):
    {features}
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "metadata.json").read_text())
MODEL = joblib.load(HERE / "model.joblib")
READERS = {{".csv": pd.read_csv, ".parquet": pd.read_parquet, ".json": pd.read_json}}


def predict(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in META["features"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing required columns: {{missing}}")
    X = df[META["features"]]
    out = df.copy()
    out["prediction"] = MODEL.predict(X)
    if META["task"] == "classification":
        proba = MODEL.predict_proba(X)
        for i, cls in enumerate(MODEL.classes_):
            out[f"proba_{{cls}}"] = proba[:, i]
        if META.get("decision_threshold") is not None:
            negative, positive = MODEL.classes_
            out["prediction"] = [positive if p >= META["decision_threshold"] else negative for p in proba[:, 1]]
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    source = Path(sys.argv[1])
    reader = READERS.get(source.suffix.lower(), pd.read_csv)
    result = predict(reader(source))
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_name(source.stem + "_predictions.csv")
    result.to_csv(target, index=False)
    print(f"Wrote {{len(result)}} predictions to {{target}}")
'''


def _pinned_requirements() -> str:
    lines = []
    for dist in (
        "flaml",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "catboost",
        "pandas",
        "pyarrow",
        "joblib",
    ):
        try:
            lines.append(f"{dist}=={importlib_metadata.version(dist)}")
        except importlib_metadata.PackageNotFoundError:
            continue
    return "\n".join(lines) + "\n"


def export(
    ws: Workspace, experiment_id: str, name: str | None = None, threshold: float | None = None
) -> dict[str, Any]:
    """Copy a trained model into a self-contained bundle with an inference script."""
    exp_dir = ws.experiment_dir(experiment_id)
    record = ws.load_experiment(experiment_id)
    if threshold is not None:
        if record["task"] != "classification" or len(record["classes"]) != 2:
            raise ValueError("threshold only applies to binary classification models")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
    bundle = ws.resolve_output(slugify(name or experiment_id), ws.exports_dir)
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    for filename in ("model.joblib", "MODEL_CARD.md"):
        shutil.copy2(exp_dir / filename, bundle / filename)
    (bundle / "metadata.json").write_text(
        json.dumps({**record, "decision_threshold": threshold}, indent=2, default=str)
    )
    (bundle / "predict.py").write_text(
        _PREDICT_SCRIPT.format(
            experiment_id=experiment_id,
            task=record["task"],
            target=record["target"],
            estimator=record["best_estimator"],
            features=", ".join(record["features"]),
        )
    )
    (bundle / "requirements.txt").write_text(_pinned_requirements())
    return {
        "experiment_id": experiment_id,
        "bundle_path": str(bundle),
        "decision_threshold": threshold,
        "files": sorted(p.name for p in bundle.iterdir()),
        "usage": f"cd {bundle} && {Path(sys.executable).name} predict.py <input.csv> [output.csv]",
    }


def write_model_card(ws: Workspace, experiment_id: str) -> Path:
    """Render MODEL_CARD.md for an experiment from its metadata."""
    record = ws.load_experiment(experiment_id)
    metric = record["metric"]
    lines = [
        f"# Model card: {experiment_id}",
        "",
        f"- **Created:** {record['created_at']}",
        f"- **Dataset:** `{record['dataset']}` → target `{record['target']}` ({record['task']})",
        f"- **Rows:** {record['rows']['train']} train / {record['rows']['test']} held-out test"
        + (
            f" ({record['rows']['dropped_missing_target']} dropped for missing target)"
            if record["rows"]["dropped_missing_target"]
            else ""
        ),
        f"- **Features ({len(record['features'])}):** {', '.join(record['features'])}",
    ]
    if record["dropped_columns"]:
        lines.append(f"- **Excluded columns:** {', '.join(record['dropped_columns'])}")
    if record.get("positive_class"):
        lines.append(f"- **Positive class:** `{record['positive_class']}`")
    lines += [
        "",
        "## Model",
        "",
        f"FLAML AutoML searched for {record['search']['seconds']}s "
        f"({record['search']['trials']} trials, optimizing `{metric}`). "
        f"Best estimator: **{record['best_estimator']}**.",
        "",
        "```json",
        json.dumps(record["best_config"], indent=2),
        "```",
        "",
        "## Performance",
        "",
        f"| | {metric} |",
        "|---|---|",
        f"| Validation (inside search) | {record['validation'][metric]} |",
        f"| Held-out test | {record['test'][metric]} |",
        f"| Naive baseline on test | {record['test']['baseline']} |",
        "",
        "All held-out test metrics:",
        "",
    ]
    for key, value in record["test_metrics"].items():
        if key not in ("confusion_matrix", "per_class"):
            lines.append(f"- {key}: {value} (baseline {record['baseline_metrics'].get(key)})")
    if record.get("operating_points"):
        lines += [
            "",
            f"## Decision thresholds (held-out test, positive class `{record['positive_class']}`)",
            "",
            "| threshold | precision | recall | f1 | flagged % |",
            "|---|---|---|---|---|",
        ]
        for row in record["operating_points"]:
            lines.append(
                f"| {row['threshold']} | {row['precision']} | {row['recall']} | {row['f1']} "
                f"| {row['flagged_pct']} |"
            )
    if record.get("feature_importance"):
        lines += ["", "## Most important features (permutation importance)", ""]
        for row in record["feature_importance"][:10]:
            lines.append(f"- {row['feature']}: {row['importance']} ± {row['std']}")
    if record.get("notes"):
        lines += ["", "## Notes", "", record["notes"]]
    path = ws.experiment_dir(experiment_id) / "MODEL_CARD.md"
    path.write_text("\n".join(lines) + "\n")
    return path
