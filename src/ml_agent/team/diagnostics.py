"""Checks used by the Analysis and Review agents. All run on training rows only."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from ml_agent.automl import LOWER_IS_BETTER
from ml_agent.profiling import round_sig
from ml_agent.team.catalog import make_pipeline
from ml_agent.team.evaluation import SCORERS, natural
from ml_agent.team.preprocess import build_preprocessor, native_categorical_mask
from ml_agent.team.runs import RunStore
from ml_agent.team.schemas import ColumnRoles


def feature_associations(run: RunStore, top_k: int = 20) -> dict[str, Any]:
    """Mutual information of every feature with the target, plus Spearman correlation for numerics."""
    spec, roles = run.spec, run.roles
    X, y = run.xy_train()
    prep = build_preprocessor("tree", roles)
    matrix = prep.fit_transform(X, y)
    names = list(prep.get_feature_names_out())
    discrete = [name in roles.categorical for name in names]
    mi_fn = mutual_info_classif if spec.task == "classification" else mutual_info_regression
    mi = mi_fn(matrix, y, discrete_features=discrete, random_state=0)
    rows = []
    for name, value in zip(names, mi, strict=True):
        row: dict[str, Any] = {"feature": name, "mutual_information": round_sig(float(value), 3)}
        if name in roles.numeric and (spec.task == "regression" or len(run.classes) == 2):
            target = pd.Series(y, index=X.index, dtype=float)
            row["spearman"] = round_sig(float(X[name].corr(target, method="spearman")), 3)
        rows.append(row)
    rows.sort(key=lambda r: r["mutual_information"] or 0, reverse=True)
    return {
        "top_features": rows[:top_k],
        "no_detectable_signal": [
            r["feature"] for r in rows if (r["mutual_information"] or 0) < 1e-3
        ],
        "note": "Mutual information captures non-linear dependence; 0 means no detectable signal "
        "on its own (it may still matter in interactions).",
    }


def _cv_primary(run: RunStore, estimator, splits) -> tuple[float, float, list[float]]:
    metric = run.spec.primary_metric
    X, y = run.xy_train()
    result = cross_validate(estimator, X, y, cv=splits, scoring=SCORERS[metric], n_jobs=-1)
    folds = [natural(metric, v) for v in result["test_score"]]
    return float(np.mean(folds)), float(np.std(folds, ddof=1)), folds


def quick_baselines(run: RunStore) -> dict[str, Any]:
    """Untuned dummy, linear and depth-3 tree models: how much signal is there at all?"""
    spec = run.spec
    splits = run.first_repeat_cv()
    started = time.time()
    results = []
    for family, params, label in (
        ("dummy", {}, "dummy (class prior / mean)"),
        ("linear", {}, "linear, default settings"),
        ("decision_tree", {"model__max_depth": 3}, "decision tree, depth 3"),
    ):
        estimator = make_pipeline(family, spec.task, run.roles, params=params)
        mean, std, _ = _cv_primary(run, estimator, splits)
        results.append(
            {"model": label, spec.primary_metric: round_sig(mean, 4), "std": round_sig(std, 3)}
        )
    run.charge(time.time() - started)
    return {
        "metric": spec.primary_metric,
        "cv": f"{len(splits)}-fold (first selection repeat)",
        "results": results,
        "note": "Quick diagnostics only; they are not leaderboard candidates.",
    }


def check_overfitting(run: RunStore) -> dict[str, Any]:
    """Gap between training and validation scores for each current candidate."""
    metric = run.spec.primary_metric
    lower = metric in LOWER_IS_BETTER
    rows = []
    for r in run.candidates(current_only=True):
        cv, train = r["cv"][metric]["mean"], r["train"][metric]
        gap = (cv - train) if lower else (train - cv)
        relative = gap / abs(cv) if cv else None
        rows.append(
            {
                "id": r["id"],
                f"train_{metric}": train,
                f"cv_{metric}": cv,
                "gap": round_sig(gap, 3),
                "relative_gap": round_sig(relative, 3) if relative is not None else None,
                "flag": "large" if relative is not None and relative > 0.1 else None,
            }
        )
    return {
        "metric": metric,
        "candidates": rows,
        "note": "Tree ensembles often fit training data almost perfectly; judge by whether the CV "
        "score is good and stable, not by the gap alone.",
    }


def _single_feature_roles(roles: ColumnRoles, feature: str) -> ColumnRoles:
    return ColumnRoles(
        numeric=[feature] if feature in roles.numeric else [],
        categorical=[feature] if feature in roles.categorical else [],
        high_cardinality=[feature] if feature in roles.high_cardinality else [],
        dropped=[],
    )


def check_leakage(run: RunStore, max_features: int = 30) -> dict[str, Any]:
    """How close does each feature get to the best model on its own?

    A single feature that recovers most of the best model's lift over the dummy baseline is a
    classic sign of leakage (or of a trivially easy problem worth questioning).
    """
    spec, roles = run.spec, run.roles
    metric = spec.primary_metric
    current = run.candidates(current_only=True)
    if not current:
        raise ValueError("Train some candidates first; leakage is judged against the best one")
    lower = metric in LOWER_IS_BETTER
    best = min(current, key=lambda r: r["cv"][metric]["mean"] * (1 if lower else -1))
    splits = run.first_repeat_cv()
    started = time.time()
    dummy, _, _ = _cv_primary(run, make_pipeline("dummy", spec.task, roles), splits)
    lift_best = (
        (dummy - best["cv"][metric]["mean"]) if lower else (best["cv"][metric]["mean"] - dummy)
    )

    features = feature_associations(run, top_k=max_features)["top_features"]
    candidates = [f["feature"] for f in features if f["feature"] in roles.features]
    rows = []
    for feature in candidates[:max_features]:
        single = _single_feature_roles(roles, feature)
        model = (
            HistGradientBoostingClassifier
            if spec.task == "classification"
            else HistGradientBoostingRegressor
        )(max_iter=50, categorical_features=native_categorical_mask(single), random_state=0)
        estimator = Pipeline([("prep", build_preprocessor("native", single)), ("model", model)])
        mean, _, _ = _cv_primary(run, clone(estimator), splits)
        lift = (dummy - mean) if lower else (mean - dummy)
        share = lift / lift_best if lift_best > 0 else None
        rows.append(
            {
                "feature": feature,
                f"alone_{metric}": round_sig(mean, 4),
                "share_of_best_lift": round_sig(share, 3) if share is not None else None,
            }
        )
    run.charge(time.time() - started)
    rows.sort(key=lambda r: r["share_of_best_lift"] or 0, reverse=True)
    suspicious = [r["feature"] for r in rows if (r["share_of_best_lift"] or 0) >= 0.9]
    return {
        "best_candidate": best["id"],
        f"best_{metric}": best["cv"][metric]["mean"],
        f"dummy_{metric}": round_sig(dummy, 4),
        "features": rows,
        "suspicious": suspicious,
        "note": "share_of_best_lift >= 0.9 means one feature alone nearly matches the best model. "
        "Confirm such features are known at prediction time.",
    }


def check_success(run: RunStore, candidate_id: str) -> dict[str, Any]:
    """Compare a candidate's CV score with the dummy baseline and the spec's minimum score."""
    spec = run.spec
    metric = spec.primary_metric
    lower = metric in LOWER_IS_BETTER
    record = run.candidate(candidate_id)
    dummies = [r for r in run.candidates(current_only=True) if r["family"] == "dummy"]
    cv = record["cv"][metric]["mean"]
    result: dict[str, Any] = {"candidate": candidate_id, f"cv_{metric}": cv}
    if dummies:
        dummy = dummies[0]["cv"][metric]["mean"]
        result[f"dummy_{metric}"] = dummy
        result["beats_dummy"] = bool(cv < dummy if lower else cv > dummy)
    if spec.min_primary_score is not None:
        result["min_required"] = spec.min_primary_score
        result["meets_minimum"] = bool(
            cv <= spec.min_primary_score if lower else cv >= spec.min_primary_score
        )
    result["success_criteria"] = spec.success_criteria
    return result


def rescore(run: RunStore, candidate_id: str) -> dict[str, Any]:
    """Re-run a candidate's selection CV to confirm its scores are reproducible."""
    metric = run.spec.primary_metric
    record = run.candidate(candidate_id)
    started = time.time()
    _, _, folds = _cv_primary(run, run.candidate_estimator(candidate_id), run.selection_cv())
    run.charge(time.time() - started)
    max_diff = float(np.max(np.abs(np.array(folds) - np.array(record["cv"][metric]["folds"]))))
    return {
        "candidate": candidate_id,
        "max_fold_difference": round_sig(max_diff, 3),
        "reproducible": max_diff < 1e-6,
    }
