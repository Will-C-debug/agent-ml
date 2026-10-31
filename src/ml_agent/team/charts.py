"""Charts for the Visualization agent, rendered as PNGs into the run's figures/ folder.

Each chart returns ``facts``: the numbers it shows, so the agent (which cannot see images) can
write accurate captions. Styling follows one system: a fixed categorical order (blue, orange, aqua
- validated for colour-vision deficiency across all pairs), a single blue ramp for magnitude,
thin marks, hairline solid gridlines on the value axis only, and text in ink colours, never in
series colours. Multi-series charts always carry a legend.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap, to_rgb  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import partial_dependence  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import learning_curve  # noqa: E402

from ml_agent.automl import LOWER_IS_BETTER  # noqa: E402
from ml_agent.profiling import round_sig  # noqa: E402
from ml_agent.team.evaluation import SCORERS, natural  # noqa: E402
from ml_agent.team.runs import RunStore  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, in this fixed order
DEEMPHASIS = "#c3c2b7"
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip
MAX_SERIES = len(SERIES)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "savefig.dpi": 150,
        "savefig.facecolor": SURFACE,
    }
)


def _axes(ax, *, grid: str = "y") -> None:
    """Recessive chrome: no top/right spines, hairline solid grid on the value axis only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis=grid, color=GRIDLINE, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)


def _dots(ax, x, y, color, size=64, **kwargs):
    """Markers >= 8px with a 2px surface ring so they stay legible where they overlap."""
    return ax.scatter(
        x, y, s=size, color=color, edgecolors=SURFACE, linewidths=2, zorder=3, **kwargs
    )


def _subtitle(ax, text: str) -> None:
    """A secondary line under the title: lift the title to make room for it."""
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=24)
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9, va="bottom")


def _ranked(run: RunStore, ids: list[str] | None, limit: int) -> list[dict[str, Any]]:
    metric = run.spec.primary_metric
    sign = 1 if metric in LOWER_IS_BETTER else -1
    records = sorted(
        run.candidates(current_only=True), key=lambda r: sign * r["cv"][metric]["mean"]
    )
    if ids:
        by_id = {r["id"]: r for r in records}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise KeyError(f"Not current candidates: {missing}")
        records = [by_id[i] for i in ids]
    if not records:
        raise ValueError("No candidates to chart yet")
    return records[:limit]


def _final(run: RunStore) -> dict[str, Any]:
    final = run.read_json("champion/final.json")
    if final is None:
        raise ValueError("This chart needs the final test evaluation, which hasn't run yet")
    return final


def _require_binary(run: RunStore) -> None:
    if run.spec.task != "classification" or len(run.classes) != 2:
        raise ValueError("This chart is for binary classification")


def _positive_scores(oof: pd.DataFrame) -> np.ndarray:
    return oof["p_1"].to_numpy() if "p_1" in oof else oof["decision"].to_numpy()


# --------------------------------------------------------------------------- charts


def leaderboard_chart(run: RunStore, ids: list[str] | None = None):
    """Dot plot of mean selection-CV score ± 1 std per candidate, best first."""
    metric = run.spec.primary_metric
    records = _ranked(run, ids, limit=15)
    approved = (run.read_json("champion/final.json") or {}).get("champion") or records[0]["id"]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(records) + 1.6))
    _axes(ax, grid="x")
    ys = np.arange(len(records))[::-1]
    for y, r in zip(ys, records, strict=True):
        mean, std = r["cv"][metric]["mean"], r["cv"][metric]["std"] or 0
        color = SERIES[0] if r["id"] == approved else DEEMPHASIS
        ax.plot([mean - std, mean + std], [y, y], color=color, linewidth=2, zorder=2)
        _dots(ax, [mean], [y], color)
    ax.set_yticks(ys, [f"{r['id']}" for r in records])
    dummy = next((r for r in records if r["family"] == "dummy"), None)
    if dummy is not None:
        ax.axvline(dummy["cv"][metric]["mean"], color=MUTED, linewidth=1)
    top = next(r for r in records if r["id"] == approved)
    top_y = ys[records.index(top)]
    ax.annotate(
        f"{top['cv'][metric]['mean']:.4g}",
        (top["cv"][metric]["mean"] + (top["cv"][metric]["std"] or 0), top_y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=INK,
    )
    ax.set_xlabel(
        f"{metric} (selection CV mean ± 1 std, {'lower' if metric in LOWER_IS_BETTER else 'higher'} is better)"
    )
    ax.set_title("Candidate leaderboard")
    _subtitle(ax, f"Highlighted: {approved}" + ("; vertical line: dummy baseline" if dummy else ""))
    facts = {
        "metric": metric,
        "highlighted": approved,
        "candidates": [
            {"id": r["id"], "mean": r["cv"][metric]["mean"], "std": r["cv"][metric]["std"]}
            for r in records
        ],
    }
    return fig, facts


def paired_differences_chart(run: RunStore, ids: list[str] | None = None):
    """Per-fold score difference between the best candidate and each competitor."""
    metric = run.spec.primary_metric
    records = _ranked(run, ids, limit=8)
    if len(records) < 2:
        raise ValueError("Need at least two candidates")
    best, others = records[0], records[1:]
    lower = metric in LOWER_IS_BETTER
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(others) + 1.6))
    _axes(ax, grid="x")
    facts = {"best": best["id"], "metric": metric, "competitors": []}
    ys = np.arange(len(others))[::-1]
    best_folds = np.array(best["cv"][metric]["folds"])
    for y, r in zip(ys, others, strict=True):
        folds = np.array(r["cv"][metric]["folds"])
        diffs = (folds - best_folds) if lower else (best_folds - folds)
        _dots(ax, diffs, np.full(len(diffs), y), SERIES[0], size=48, alpha=0.9)
        facts["competitors"].append(
            {
                "id": r["id"],
                "mean_advantage_of_best": round_sig(float(diffs.mean()), 3),
                "folds_best_wins": f"{int((diffs > 0).sum())}/{len(diffs)}",
            }
        )
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(ys, [r["id"] for r in others])
    ax.set_xlabel(f"Advantage of {best['id']} per fold ({metric}); right of 0 = best wins")
    ax.set_title("How consistently does the best candidate win?")
    _subtitle(ax, "One dot per selection fold; identical folds for every candidate")
    return fig, facts


def roc_chart(run: RunStore, ids: list[str] | None = None):
    """Out-of-fold ROC curves for up to three candidates."""
    _require_binary(run)
    records = _ranked(run, ids, limit=MAX_SERIES)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    _axes(ax, grid="both")
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    facts = {"curves": []}
    for color, r in zip(SERIES, records, strict=False):
        oof = run.candidate_oof(r["id"])
        fpr, tpr, _ = roc_curve(oof["y"], _positive_scores(oof))
        area = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f"{r['id']} (AUC {area:.3f})")
        facts["curves"].append({"id": r["id"], "auc": round_sig(area, 4)})
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title("ROC curves (out-of-fold, training data)")
    _subtitle(ax, f"Positive class: {run.classes[1]}; diagonal = random guessing")
    ax.legend(loc="lower right")
    return fig, facts


def precision_recall_chart(run: RunStore, ids: list[str] | None = None):
    """Out-of-fold precision-recall curves for up to three candidates."""
    _require_binary(run)
    records = _ranked(run, ids, limit=MAX_SERIES)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    _axes(ax, grid="both")
    first = run.candidate_oof(records[0]["id"])
    prevalence = float(np.mean(first["y"] == 1))
    ax.axhline(prevalence, color=AXIS, linewidth=1)
    facts = {"positive_rate": round_sig(prevalence, 3), "curves": []}
    for color, r in zip(SERIES, records, strict=False):
        oof = run.candidate_oof(r["id"])
        scores = _positive_scores(oof)
        precision, recall, _ = precision_recall_curve(oof["y"], scores)
        ap = average_precision_score(oof["y"], scores)
        ax.plot(recall, precision, color=color, label=f"{r['id']} (AP {ap:.3f})")
        facts["curves"].append({"id": r["id"], "average_precision": round_sig(ap, 4)})
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall", ylabel="Precision")
    ax.set_title("Precision-recall curves (out-of-fold, training data)")
    _subtitle(ax, f"Positive class: {run.classes[1]}; horizontal line = positive rate")
    ax.legend(loc="upper right")
    return fig, facts


def confusion_matrix_chart(run: RunStore, ids: list[str] | None = None):
    """Champion's confusion matrix on the locked test set, counts and row percentages."""
    if run.spec.task != "classification":
        raise ValueError("This chart is for classification")
    final = _final(run)
    preds = pd.read_parquet(run.root / "champion" / "test_predictions.parquet")
    labels = run.classes
    matrix = confusion_matrix(preds["y_true"], preds["y_pred"], labels=range(len(labels)))
    share = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    size = 1.1 * len(labels) + 2
    fig, ax = plt.subplots(figsize=(size + 0.8, size))
    ax.imshow(share, cmap=ListedColormap(BLUE_RAMP), vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            r, g, b = to_rgb(BLUE_RAMP[int(share[i, j] * (len(BLUE_RAMP) - 1))])
            ink = "#ffffff" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.45 else INK
            ax.text(j, i, f"{matrix[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", color=ink)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted", ylabel="Actual")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Confusion matrix (locked test set)")
    threshold = final.get("threshold")
    _subtitle(
        ax,
        f"{final['champion']}; row % = share of each actual class"
        + (f"; threshold {threshold:.3g}" if threshold else ""),
    )
    facts = {
        "champion": final["champion"],
        "labels": labels,
        "counts": matrix.tolist(),
        "recall_per_class": {lab: round_sig(float(share[i, i]), 3) for i, lab in enumerate(labels)},
    }
    return fig, facts


def calibration_chart(run: RunStore, ids: list[str] | None = None):
    """Reliability curve of the champion's probabilities on the test set."""
    _require_binary(run)
    final = _final(run)
    if not final.get("probabilities_calibrated", True):
        raise ValueError("The champion has no predicted probabilities to calibrate")
    preds = pd.read_parquet(run.root / "champion" / "test_predictions.parquet")
    true_rate, predicted = calibration_curve(
        preds["y_true"], preds["p_1"], n_bins=10, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(6, 5.4))
    _axes(ax, grid="both")
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    ax.plot(predicted, true_rate, color=SERIES[0])
    _dots(ax, predicted, true_rate, SERIES[0])
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed rate")
    ax.set_title("Calibration (locked test set)")
    _subtitle(ax, f"{final['champion']}; points on the diagonal = trustworthy probabilities")
    gap = float(np.mean(np.abs(true_rate - predicted)))
    return fig, {"champion": final["champion"], "mean_abs_calibration_gap": round_sig(gap, 3)}


def importance_chart(run: RunStore, ids: list[str] | None = None):
    """Permutation importance of the champion's features on the test set."""
    final = _final(run)
    rows = final["permutation_importance"][:15]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(rows) + 1.6))
    _axes(ax, grid="x")
    ys = np.arange(len(rows))[::-1]
    values = [r["importance"] or 0 for r in rows]
    ax.barh(ys, values, height=0.55, color=SERIES[0], xerr=[r["std"] or 0 for r in rows],
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 0})  # fmt: skip
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(ys, [r["feature"] for r in rows])
    ax.set_xlabel(f"Drop in {final['metric']} when the feature is shuffled (± std)")
    ax.set_title("What drives the champion's predictions")
    _subtitle(ax, f"{final['champion']}; permutation importance on the locked test set")
    return fig, {"champion": final["champion"], "top_features": rows[:8]}


def partial_dependence_chart(run: RunStore, ids: list[str] | None = None):
    """How predictions change with each of the top numeric features (small multiples)."""
    final = _final(run)
    model = _fitted_champion(run)
    roles = run.roles
    top = [r["feature"] for r in final["permutation_importance"] if r["feature"] in roles.numeric][
        :3
    ]
    if not top:
        raise ValueError("No numeric features among the champion's important features")
    X, _ = run.xy_train()
    sample = X.sample(min(len(X), 1000), random_state=0)
    fig, axes = plt.subplots(1, len(top), figsize=(3.6 * len(top), 3.6), sharey=True, squeeze=False)
    facts = {"champion": final["champion"], "features": []}
    for ax, feature in zip(axes[0], top, strict=True):
        _axes(ax, grid="y")
        result = partial_dependence(model, sample, [feature], kind="average", grid_resolution=30)
        grid, values = result["grid_values"][0], result["average"][-1]
        ax.plot(grid, values, color=SERIES[0])
        ax.set_xlabel(feature)
        facts["features"].append(
            {
                "feature": feature,
                "from": round_sig(float(values[0]), 3),
                "to": round_sig(float(values[-1]), 3),
                "min": round_sig(float(values.min()), 3),
                "max": round_sig(float(values.max()), 3),
            }
        )
    target_label = (
        f"P({run.spec.target} = {run.classes[1]})"
        if run.spec.task == "classification"
        else f"Predicted {run.spec.target}"
    )
    axes[0][0].set_ylabel(target_label)
    axes[0][0].set_title("Partial dependence of the top numeric features")
    return fig, facts


def learning_curve_chart(run: RunStore, ids: list[str] | None = None):
    """Training vs validation score as the training set grows: would more data help?"""
    import time

    from ml_agent.team.evaluation import final_estimator

    metric = run.spec.primary_metric
    champion = (run.read_json("champion/final.json") or {}).get("champion") or _ranked(run, ids, 1)[
        0
    ]["id"]
    X, y = run.xy_train()
    started = time.time()
    sizes, train_scores, val_scores = learning_curve(
        final_estimator(run, champion), X, y, cv=run.first_repeat_cv(),
        scoring=SCORERS[metric], train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1,
    )  # fmt: skip
    run.charge(time.time() - started)
    train_scores, val_scores = natural(metric, train_scores), natural(metric, val_scores)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    _axes(ax, grid="y")
    facts = {"champion": champion, "metric": metric, "points": []}
    for color, label, scores in (
        (SERIES[0], "Validation", val_scores),
        (SERIES[1], "Training", train_scores),
    ):
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.1, linewidth=0)
        ax.plot(sizes, mean, color=color, label=label)
        _dots(ax, sizes, mean, color)
        ax.annotate(
            label,
            (sizes[-1], mean[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
        )
    for size, tr, va in zip(sizes, train_scores.mean(axis=1), val_scores.mean(axis=1), strict=True):
        facts["points"].append(
            {
                "train_rows": int(size),
                "train": round_sig(float(tr), 4),
                "validation": round_sig(float(va), 4),
            }
        )
    ax.set(xlabel="Training rows", ylabel=metric)
    ax.set_title("Learning curve")
    _subtitle(ax, f"{champion}; a rising validation curve means more data would help")
    ax.legend(loc="lower right")
    return fig, facts


def prediction_error_chart(run: RunStore, ids: list[str] | None = None):
    """Regression: predicted vs actual on the locked test set."""
    if run.spec.task != "regression":
        raise ValueError("This chart is for regression")
    final = _final(run)
    preds = pd.read_parquet(run.root / "champion" / "test_predictions.parquet")
    fig, ax = plt.subplots(figsize=(6, 5.4))
    _axes(ax, grid="both")
    lo = float(min(preds["y_true"].min(), preds["y_pred"].min()))
    hi = float(max(preds["y_true"].max(), preds["y_pred"].max()))
    ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1)
    ax.scatter(preds["y_pred"], preds["y_true"], s=14, color=SERIES[0], alpha=0.5, linewidths=0)
    ax.set(xlabel=f"Predicted {run.spec.target}", ylabel=f"Actual {run.spec.target}")
    ax.set_title("Predicted vs actual (locked test set)")
    _subtitle(ax, f"{final['champion']}; points on the diagonal are perfect predictions")
    residuals = preds["y_true"] - preds["y_pred"]
    return fig, {
        "champion": final["champion"],
        "mean_residual": round_sig(float(residuals.mean()), 3),
        "residual_std": round_sig(float(residuals.std()), 3),
    }


def target_distribution_chart(run: RunStore, ids: list[str] | None = None):
    """Distribution of the target in the training rows."""
    spec = run.spec
    y = run.raw_train()[spec.target].dropna()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _axes(ax, grid="y")
    if spec.task == "classification":
        counts = y.astype(str).value_counts().reindex(run.classes).fillna(0)
        ax.bar(range(len(counts)), counts.to_numpy(), width=0.5, color=SERIES[0])
        ax.set_xticks(range(len(counts)), counts.index)
        ax.set_ylabel("Training rows")
        facts = {"counts": {k: int(v) for k, v in counts.items()}}
    else:
        ax.hist(y, bins=30, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
        ax.set(xlabel=spec.target, ylabel="Training rows")
        facts = {
            "mean": round_sig(float(y.mean())),
            "median": round_sig(float(y.median())),
            "skew": round_sig(float(y.skew()), 3),
        }
    ax.set_title(f"Distribution of {spec.target}")
    return fig, facts


def _fitted_champion(run: RunStore):
    import joblib

    path = run.root / "champion" / "model.joblib"
    if not path.exists():
        raise ValueError("This chart needs the final evaluation, which hasn't run yet")
    return joblib.load(path)


CHARTS: dict[str, tuple[Callable, str]] = {
    "leaderboard": (leaderboard_chart, "Mean CV score ± std per candidate, champion highlighted"),
    "paired_differences": (
        paired_differences_chart,
        "Per-fold advantage of the best candidate over each competitor",
    ),
    "roc": (roc_chart, "Out-of-fold ROC curves, top 3 candidates (binary)"),
    "precision_recall": (
        precision_recall_chart,
        "Out-of-fold precision-recall curves, top 3 (binary)",
    ),
    "confusion_matrix": (
        confusion_matrix_chart,
        "Champion's test-set confusion matrix (classification)",
    ),
    "calibration": (
        calibration_chart,
        "Champion's probability calibration on the test set (binary)",
    ),
    "feature_importance": (importance_chart, "Permutation importance on the test set"),
    "partial_dependence": (
        partial_dependence_chart,
        "Effect of the top numeric features on predictions",
    ),
    "learning_curve": (learning_curve_chart, "Would more data help? (uses compute)"),
    "prediction_error": (
        prediction_error_chart,
        "Predicted vs actual on the test set (regression)",
    ),
    "target_distribution": (target_distribution_chart, "Distribution of the target"),
}


def make_chart(run: RunStore, kind: str, candidate_ids: list[str] | None = None) -> dict[str, Any]:
    """Render one chart to figures/<kind>.png and return its path and the facts it shows."""
    if kind not in CHARTS:
        raise ValueError(f"Unknown chart {kind!r}. Choose from {sorted(CHARTS)}")
    fig, facts = CHARTS[kind][0](run, candidate_ids)
    path = run.root / "figures" / f"{kind}.png"
    try:
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
    finally:
        plt.close(fig)
    return {"kind": kind, "path": str(path), "relative_path": f"figures/{kind}.png", "facts": facts}
