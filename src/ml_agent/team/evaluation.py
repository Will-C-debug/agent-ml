"""Model search and selection with one protocol for every candidate.

- Hyperparameters are tuned on the run's *tuning* folds.
- Every candidate (scikit-learn family, FLAML's winner, ensembles) is then scored with frozen
  hyperparameters on the *same selection folds*, so per-fold scores can be compared pairwise.
- ``compare`` ranks candidates with a Nadeau-Bengio corrected resampled t-test and recommends the
  simplest model within one (corrected) standard error of the best.
- ``final_evaluation`` refits the approved champion on all training rows and scores the locked test
  set, which happens once per run.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import (
    FixedThresholdClassifier,
    RandomizedSearchCV,
    TunedThresholdClassifierCV,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from ml_agent.automl import LOWER_IS_BETTER, _baseline, _jsonable, evaluate, score
from ml_agent.profiling import round_sig
from ml_agent.team.catalog import (
    FAMILIES,
    FLAML_RECIPES,
    PROBA_METRICS,
    make_pipeline,
    search_space,
    unavailable_reason,
)
from ml_agent.team.preprocess import build_preprocessor
from ml_agent.team.runs import RunStore
from ml_agent.workspace import now_iso

# FLAML-style metric names (used across ml_agent) -> scikit-learn scorer names.
SCORERS = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "roc_auc_ovr": "roc_auc_ovr",
    "roc_auc_ovo": "roc_auc_ovo",
    "log_loss": "neg_log_loss",
    "f1": "f1",
    "micro_f1": "f1_micro",
    "macro_f1": "f1_macro",
    "ap": "average_precision",
    "r2": "r2",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "mape": "neg_mean_absolute_percentage_error",
}
# Metrics that depend on the decision threshold (rank metrics like roc_auc do not).
THRESHOLD_DEPENDENT = {"accuracy", "f1", "micro_f1", "macro_f1"}
THRESHOLD_SCORERS = {
    "f1": "f1",
    "f2": make_scorer(fbeta_score, beta=2),
    "f0.5": make_scorer(fbeta_score, beta=0.5),
    "balanced_accuracy": "balanced_accuracy",
}
DECISION_METRICS = {
    "f1": "f1",
    "f2": THRESHOLD_SCORERS["f2"],
    "f0.5": THRESHOLD_SCORERS["f0.5"],
    "precision": "precision",
    "recall": "recall",
    "balanced_accuracy": "balanced_accuracy",
}


def natural(metric: str, value: float) -> float:
    """Convert a scikit-learn score (negated for losses) back to the metric's natural scale."""
    return -value if SCORERS[metric].startswith("neg_") else value


def better(metric: str, a: float, b: float) -> bool:
    return a < b if metric in LOWER_IS_BETTER else a > b


def _sort_key(metric: str):
    sign = 1 if metric in LOWER_IS_BETTER else -1
    return lambda record: sign * record["cv"][metric]["mean"]


def _seed(run: RunStore) -> int:
    return int(run.meta.get("seed", 42))


# --------------------------------------------------------------------------- scoring


def _oof_predictions(
    estimators: list, test_indices: list, X: pd.DataFrame, y: np.ndarray, task: str
) -> pd.DataFrame:
    """Out-of-fold predictions from one repeat of the selection folds (each row once)."""
    frames = []
    for est, idx in zip(estimators, test_indices, strict=True):
        part = pd.DataFrame({"row": idx, "y": y[idx]})
        X_val = X.iloc[idx]
        if task == "regression":
            part["pred"] = est.predict(X_val)
        elif hasattr(est, "predict_proba"):
            proba = est.predict_proba(X_val)
            for k in range(proba.shape[1]):
                part[f"p_{k}"] = proba[:, k]
        else:
            decision = np.asarray(est.decision_function(X_val))
            if decision.ndim == 1:
                part["decision"] = decision
            else:
                for k in range(decision.shape[1]):
                    part[f"d_{k}"] = decision[:, k]
        frames.append(part)
    return pd.concat(frames).sort_values("row").reset_index(drop=True)


def score_candidate(
    run: RunStore,
    estimator: Any,
    *,
    family: str,
    source: str,
    params: dict[str, Any],
    complexity: int,
    tuning: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Score a configured estimator on the shared selection folds and save it as a candidate."""
    spec = run.spec
    X, y = run.xy_train()
    has_proba = spec.task == "regression" or hasattr(estimator, "predict_proba")
    metrics = list(dict.fromkeys([spec.primary_metric, *spec.secondary_metrics]))
    if not has_proba:
        metrics = [m for m in metrics if m not in PROBA_METRICS]
    run.require_budget()

    started = time.time()
    cv = cross_validate(
        estimator,
        X,
        y,
        cv=run.selection_cv(),
        scoring={m: SCORERS[m] for m in metrics},
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
        n_jobs=-1,
        error_score="raise",
    )
    elapsed = time.time() - started
    run.charge(elapsed)
    n_first = len(run.first_repeat_cv())
    oof = _oof_predictions(
        cv["estimator"][:n_first], cv["indices"]["test"][:n_first], X, y, spec.task
    )

    scores = {}
    for m in metrics:
        folds = [natural(m, v) for v in cv[f"test_{m}"]]
        scores[m] = {
            "mean": round_sig(float(np.mean(folds)), 6),
            "std": round_sig(float(np.std(folds, ddof=1)), 6),
            "folds": [round_sig(float(v), 6) for v in folds],
        }
    primary = spec.primary_metric
    record = {
        "id": run.new_candidate_id(family),
        "family": family,
        "source": source,
        "complexity": complexity,
        "params": _jsonable(params),
        "iteration": run.meta.get("iteration", 0),
        "features_hash": run.features_hash,
        "created_at": now_iso(),
        "cv": scores,
        "train": {
            primary: round_sig(
                float(np.mean([natural(primary, v) for v in cv[f"train_{primary}"]])), 6
            )
        },
        "fit_seconds": round_sig(float(np.mean(cv["fit_time"]))),
        "cv_seconds": round(elapsed, 1),
        "has_proba": has_proba,
        "tuning": tuning,
        "threshold": None,
        "notes": notes,
    }
    run.save_candidate(record, estimator, oof)
    return record


def tune_family(
    run: RunStore,
    family: str,
    *,
    budget_seconds: float,
    max_iter: int = 40,
    notes: str | None = None,
) -> dict[str, Any]:
    """Random-search one family on the tuning folds, then score the winner as a candidate."""
    spec = run.spec
    X, y = run.xy_train()
    reason = unavailable_reason(family, spec.task, len(X), spec.primary_metric)
    if reason:
        raise ValueError(f"{family}: {reason}")
    budget_seconds = min(budget_seconds, run.require_budget(5))
    seed = _seed(run)
    pipeline = make_pipeline(family, spec.task, run.roles, seed=seed)
    space = search_space(family, spec.task)
    params: dict[str, Any] = {}
    tuning = None
    if space:
        splits = run.tune_cv()
        started = time.time()
        clone(pipeline).fit(X.iloc[splits[0][0]], y[splits[0][0]])
        fit_seconds = max(time.time() - started, 0.01)
        # Fits run in parallel across cores; leave headroom for slower configurations.
        workers = os.cpu_count() or 1
        n_iter = int(
            np.clip(budget_seconds * workers / (2 * fit_seconds * len(splits)), 2, max_iter)
        )
        search = RandomizedSearchCV(
            pipeline,
            space,
            n_iter=n_iter,
            scoring=SCORERS[spec.primary_metric],
            cv=splits,
            n_jobs=-1,
            refit=False,
            random_state=seed,
            error_score=np.nan,
        )
        search.fit(X, y)
        run.charge(time.time() - started)
        if not np.isfinite(search.best_score_):
            raise RuntimeError(f"Every {family} configuration failed during tuning")
        params = search.best_params_
        tuning = {
            "configurations_tried": n_iter,
            "tuning_cv_score": round_sig(natural(spec.primary_metric, search.best_score_), 6),
            "seconds": round(time.time() - started, 1),
        }
    estimator = clone(pipeline).set_params(**params)
    return score_candidate(
        run,
        estimator,
        family=family,
        source="sklearn",
        params=params,
        complexity=FAMILIES[family].complexity,
        tuning=tuning,
        notes=notes,
    )


# ------------------------------------------------------------------------- comparing


def leaderboard(run: RunStore) -> dict[str, Any]:
    """Current candidates ranked by mean selection-CV score of the primary metric."""
    metric = run.spec.primary_metric
    current = sorted(run.candidates(current_only=True), key=_sort_key(metric))
    rows = [
        {
            "rank": i + 1,
            "id": r["id"],
            "family": r["family"],
            "source": r["source"],
            f"cv_{metric}": r["cv"][metric]["mean"],
            "cv_std": r["cv"][metric]["std"],
            f"train_{metric}": r["train"][metric],
            "complexity": r["complexity"],
            "threshold": r.get("threshold"),
        }
        for i, r in enumerate(current)
    ]
    stale = len(run.candidates()) - len(current)
    return {
        "metric": metric,
        "lower_is_better": metric in LOWER_IS_BETTER,
        "candidates": rows,
        "stale_candidates": stale,
        "note": f"{stale} candidate(s) were trained on an older feature set and are excluded."
        if stale
        else None,
    }


def corrected_ttest(diffs: np.ndarray, test_train_ratio: float) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test on paired per-fold score differences.

    Plain t-tests over CV folds are overconfident because training sets overlap; the correction
    inflates the variance by ``1/J + n_test/n_train``. Returns (t statistic, two-sided p-value).
    """
    diffs = np.asarray(diffs, dtype=float)
    J = len(diffs)
    variance = np.var(diffs, ddof=1)
    if variance == 0:
        return (0.0, 1.0) if np.mean(diffs) == 0 else (math.copysign(math.inf, np.mean(diffs)), 0.0)
    t = np.mean(diffs) / math.sqrt((1 / J + test_train_ratio) * variance)
    return float(t), float(2 * stats.t.sf(abs(t), df=J - 1))


def compare(run: RunStore, ids: list[str] | None = None) -> dict[str, Any]:
    """Pairwise comparison against the best candidate, plus the 1-SE simplicity recommendation."""
    metric = run.spec.primary_metric
    records = run.candidates(current_only=True)
    if ids:
        missing = sorted(set(ids) - {r["id"] for r in records})
        if missing:
            raise KeyError(f"Not current candidates: {missing}")
        records = [r for r in records if r["id"] in ids]
    if not records:
        raise ValueError("No candidates to compare yet")
    records.sort(key=_sort_key(metric))
    best = records[0]
    folds_best = np.array(best["cv"][metric]["folds"])
    n_folds = len(run.first_repeat_cv())
    ratio = 1 / (n_folds - 1)  # validation rows / training rows within a fold
    se_best = math.sqrt((1 / len(folds_best) + ratio) * np.var(folds_best, ddof=1))
    lower = metric in LOWER_IS_BETTER
    bound = best["cv"][metric]["mean"] + (se_best if lower else -se_best)

    rows = []
    for r in records:
        folds = np.array(r["cv"][metric]["folds"])
        diffs = folds_best - folds if not lower else folds - folds_best
        t, p = corrected_ttest(diffs, ratio) if r is not best else (0.0, 1.0)
        within = r["cv"][metric]["mean"] <= bound if lower else r["cv"][metric]["mean"] >= bound
        rows.append(
            {
                "id": r["id"],
                "family": r["family"],
                "complexity": r["complexity"],
                "cv_mean": r["cv"][metric]["mean"],
                "best_minus_this": round_sig(float(np.mean(diffs)), 4),
                "folds_where_best_wins": f"{int((diffs > 0).sum())}/{len(diffs)}",
                "p_value_vs_best": round_sig(p, 3),
                "within_1se_of_best": bool(within),
            }
        )
    eligible = [row for row in rows if row["within_1se_of_best"]]
    # Simplest first; among equally simple models, the better-ranked one.
    simplest = min(eligible, key=lambda row: (row["complexity"], rows.index(row)))
    return {
        "metric": metric,
        "best": best["id"],
        "one_standard_error": round_sig(se_best, 4),
        "one_se_recommendation": simplest["id"],
        "comparisons": rows,
        "how_to_read": "p_value_vs_best uses the Nadeau-Bengio corrected t-test over identical "
        "selection folds; large p-values mean the difference may be noise. The 1-SE "
        "recommendation is the simplest model whose mean is within one corrected standard error "
        "of the best.",
    }


# ----------------------------------------------------------------------- competitors


def flaml_competitor(
    run: RunStore, *, time_budget: float, notes: str | None = None
) -> dict[str, Any]:
    """Let FLAML search the training rows, then score its winner on the shared selection folds.

    FLAML's best estimator is cloned with its tuned hyperparameters and wrapped in the same
    preprocessing recipe as its family, so it competes on exactly the same footing.
    """
    from flaml import AutoML

    spec = run.spec
    X, y = run.xy_train()
    time_budget = int(min(time_budget, run.require_budget(10)))
    if time_budget < 5:
        raise ValueError("FLAML needs a time budget of at least 5 seconds")
    automl = AutoML()
    started = time.time()
    automl.fit(
        X,
        y,
        task=spec.task,
        metric=spec.primary_metric,
        time_budget=time_budget,
        seed=_seed(run),
        n_jobs=-1,
        verbose=0,
        log_file_name=str(run.root / "flaml.log"),
    )
    run.charge(time.time() - started)
    if automl.model is None:
        raise RuntimeError("FLAML did not finish a trial within its budget; give it more time")
    name = automl.best_estimator
    estimator = Pipeline(
        [
            ("prep", build_preprocessor(FLAML_RECIPES.get(name, "tree"), run.roles)),
            ("model", clone(automl.model.estimator)),
        ]
    )
    return score_candidate(
        run,
        estimator,
        family=f"flaml_{name}",
        source="flaml",
        params={"flaml_estimator": name, **automl.best_config},
        complexity=4,
        tuning={
            "searched_seconds": time_budget,
            "trials": len(automl.config_history),
            "flaml_internal_validation_loss": round_sig(float(automl.best_loss), 6),
        },
        notes=notes,
    )


def build_ensemble(
    run: RunStore, member_ids: list[str], kind: str = "voting", notes: str | None = None
) -> dict[str, Any]:
    """Combine existing candidates (soft voting / averaging, or stacking) as a new candidate."""
    spec = run.spec
    if kind not in ("voting", "stacking"):
        raise ValueError("kind must be 'voting' or 'stacking'")
    if len(set(member_ids)) < 2:
        raise ValueError("An ensemble needs at least two distinct candidates")
    records = {r["id"]: r for r in run.candidates(current_only=True)}
    missing = [m for m in member_ids if m not in records]
    if missing:
        raise KeyError(f"Not current candidates: {missing}")
    if any(records[m]["source"] == "ensemble" for m in member_ids):
        raise ValueError("Ensembles of ensembles are not supported")
    members = [(m, run.candidate_estimator(m)) for m in member_ids]
    classification = spec.task == "classification"
    if kind == "voting":
        if classification and not all(records[m]["has_proba"] for m in member_ids):
            raise ValueError("Soft voting needs probabilities; drop members without predict_proba")
        estimator = (
            VotingClassifier(members, voting="soft") if classification else VotingRegressor(members)
        )
    else:
        estimator = (
            StackingClassifier(members, final_estimator=LogisticRegression(max_iter=2000), cv=3)
            if classification
            else StackingRegressor(members, final_estimator=RidgeCV(), cv=3)
        )
    per_fold = sum(records[m]["fit_seconds"] or 0 for m in member_ids) * (
        4 if kind == "stacking" else 1
    )
    estimate = per_fold * len(run.selection_cv()) / (os.cpu_count() or 1)
    remaining = run.require_budget()
    if estimate > remaining:
        raise ValueError(
            f"Estimated {estimate:.0f}s exceeds the remaining budget ({remaining:.0f}s); use "
            "fewer or cheaper members"
        )
    return score_candidate(
        run,
        estimator,
        family=f"ensemble_{kind}",
        source="ensemble",
        params={"members": member_ids, "kind": kind},
        complexity=5,
        notes=notes,
    )


def tune_threshold(
    run: RunStore, candidate_id: str, objective: str | None = None
) -> dict[str, Any]:
    """Pick a binary decision threshold on the tuning folds; report its effect on selection folds."""
    spec = run.spec
    if spec.task != "classification" or len(run.classes) != 2:
        raise ValueError("Threshold tuning applies to binary classification only")
    objective = objective or spec.threshold_objective or "f1"
    if objective not in THRESHOLD_SCORERS:
        raise ValueError(f"objective must be one of {sorted(THRESHOLD_SCORERS)}")
    X, y = run.xy_train()
    estimator = run.candidate_estimator(candidate_id)
    run.require_budget()
    started = time.time()
    tuner = TunedThresholdClassifierCV(
        estimator, scoring=THRESHOLD_SCORERS[objective], cv=run.tune_cv(), n_jobs=-1
    ).fit(X, y)
    threshold = float(tuner.best_threshold_)
    comparison = {}
    for label, model in (
        ("default", estimator),
        ("tuned", FixedThresholdClassifier(estimator, threshold=threshold)),
    ):
        cv = cross_validate(model, X, y, cv=run.selection_cv(), scoring=DECISION_METRICS, n_jobs=-1)
        comparison[label] = {
            m: round_sig(float(np.mean(cv[f"test_{m}"])), 4) for m in DECISION_METRICS
        }
    run.charge(time.time() - started)
    run.update_candidate(
        candidate_id,
        threshold=round_sig(threshold, 6),
        threshold_objective=objective,
        threshold_cv=comparison,
    )
    return {
        "candidate": candidate_id,
        "objective": objective,
        "threshold": round_sig(threshold, 4),
        "positive_class": run.classes[1],
        "selection_cv_at_default_threshold": comparison["default"],
        "selection_cv_at_tuned_threshold": comparison["tuned"],
        "note": "Chosen on the tuning folds; the selection-fold numbers show how it generalizes. "
        "The final test evaluation will use this threshold.",
    }


# ---------------------------------------------------------------- final evaluation


def _probabilities(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """Class probabilities, or decision scores squashed into [0, 1] (flagged as uncalibrated)."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)), True
    decision = np.asarray(model.decision_function(X))
    if decision.ndim == 1:
        p = 1 / (1 + np.exp(-decision))
        return np.column_stack([1 - p, p]), False
    exp = np.exp(decision - decision.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True), False


def final_estimator(run: RunStore, champion_id: str) -> Any:
    """The champion as it will be deployed: tuned threshold applied if there is one."""
    record = run.candidate(champion_id)
    estimator = run.candidate_estimator(champion_id)
    if record.get("threshold") is not None:
        estimator = FixedThresholdClassifier(estimator, threshold=record["threshold"])
    return estimator


def final_evaluation(
    run: RunStore, champion_id: str, *, importance_repeats: int = 5
) -> dict[str, Any]:
    """Refit the champion on all training rows and score the locked test set (once per run)."""
    cached = run.read_json("champion/final.json")
    if cached and cached["champion"] == champion_id:
        return cached
    spec = run.spec
    metric = spec.primary_metric
    record = run.candidate(champion_id)
    if record["features_hash"] != run.features_hash:
        raise ValueError(f"{champion_id} was trained on an older feature set")

    X, y = run.xy_train()
    model = clone(final_estimator(run, champion_id)).fit(X, y)
    X_test, y_test = run.load_test(champion_id)

    y_true = run.decode_target(y_test)
    y_pred = run.decode_target(model.predict(X_test))
    proba, calibrated = (None, True)
    if spec.task == "classification":
        proba, calibrated = _probabilities(model, X_test)
    classes = run.classes
    test_metrics = evaluate(spec.task, pd.Series(y_true), y_pred, proba, classes)
    if not calibrated and "log_loss" in test_metrics:
        test_metrics["log_loss"] = None
    base_pred, base_proba = _baseline(
        spec.task, pd.Series(run.decode_target(y)), len(y_test), classes
    )
    baseline_metrics = evaluate(spec.task, pd.Series(y_true), base_pred, base_proba, classes)
    for key in ("confusion_matrix", "per_class"):
        baseline_metrics.pop(key, None)

    primary = round_sig(score(metric, pd.Series(y_true), y_pred, proba, classes))
    baseline = round_sig(score(metric, pd.Series(y_true), base_pred, base_proba, classes))
    cv_mean, cv_std = record["cv"][metric]["mean"], record["cv"][metric]["std"]
    threshold_shift = record.get("threshold") is not None and metric in THRESHOLD_DEPENDENT
    consistency = None
    if primary is not None and not threshold_shift:
        low, high = cv_mean - 2 * cv_std, cv_mean + 2 * cv_std
        consistency = {
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "within_2_std_of_cv": bool(low <= primary <= high),
        }

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=SCORERS[metric],
        n_repeats=importance_repeats,
        random_state=_seed(run),
        n_jobs=-1,
    )
    order = np.argsort(importance.importances_mean)[::-1]
    importances = [
        {
            "feature": X_test.columns[i],
            "importance": round_sig(float(importance.importances_mean[i]), 4),
            "std": round_sig(float(importance.importances_std[i]), 4),
        }
        for i in order
    ]

    champion_dir = run.root / "champion"
    joblib.dump(model, champion_dir / "model.joblib")
    predictions = pd.DataFrame({"y_true": y_test, "y_pred": model.predict(X_test)})
    if proba is not None:
        for k in range(proba.shape[1]):
            predictions[f"p_{k}"] = proba[:, k]
    predictions.to_parquet(champion_dir / "test_predictions.parquet", index=False)

    result = {
        "champion": champion_id,
        "family": record["family"],
        "metric": metric,
        "lower_is_better": metric in LOWER_IS_BETTER,
        "rows": {"train": len(y), "test": len(y_test)},
        "test": {metric: primary, "baseline": baseline},
        "beats_baseline": None
        if primary is None or baseline is None
        else bool(better(metric, primary, baseline)),
        "meets_min_score": None
        if spec.min_primary_score is None or primary is None
        else bool(
            primary == spec.min_primary_score or better(metric, primary, spec.min_primary_score)
        ),
        "cv_consistency": consistency,
        "threshold": record.get("threshold"),
        "probabilities_calibrated": calibrated,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "permutation_importance": importances,
        "evaluated_at": now_iso(),
    }
    run.write_json("champion/final.json", result)
    return result
