"""LangChain tools for each specialist, bound to one workspace and (after planning) one run.

Every agent gets only what its stage needs. None of these tools can read the locked test set:
only the deterministic final-evaluation step does that.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from ml_agent.config import Settings
from ml_agent.executor import run_code
from ml_agent.profiling import profile_dataframe
from ml_agent.team import charts, diagnostics, evaluation
from ml_agent.team.catalog import FAMILIES, unavailable_reason
from ml_agent.team.runs import RunStore
from ml_agent.team.schemas import ColumnRoles, DroppedColumn
from ml_agent.tools import build_tools
from ml_agent.workspace import Workspace

# Tools whose calls execute agent-written code; they wait for the user's approval.
CODE_TOOLS = ("run_python", "define_features")


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, default=str, ensure_ascii=False)


def planner_tools(ws: Workspace, settings: Settings) -> list[BaseTool]:
    wanted = {"list_datasets", "load_dataset", "preview_dataset", "profile_dataset"}
    return [t for t in build_tools(ws, settings) if t.name in wanted]


def _shared_run_tools(run: RunStore, settings: Settings) -> dict[str, BaseTool]:
    @tool(parse_docstring=True)
    def profile_train(columns: list[str] | None = None) -> str:
        """Profile the run's training rows (after the registered feature function, if any).

        Args:
            columns: Profile only these columns (by default, the first 60).
        """
        return _dump(profile_dataframe(run.train_frame(), target=run.spec.target, columns=columns))

    @tool(parse_docstring=True)
    def preview_train(n_rows: int = 10, columns: list[str] | None = None) -> str:
        """Show the first training rows as CSV text.

        Args:
            n_rows: Number of rows (max 50).
            columns: Optional subset of columns.
        """
        df = run.train_frame()
        if columns:
            df = df[columns]
        return df.head(min(max(n_rows, 1), 50)).to_csv(index=False)

    @tool(parse_docstring=True)
    def run_python(code: str) -> str:
        """Run a Python script on the user's machine for exploration the other tools don't cover.

        Predefined: `pd`, `np`, `load_train(prepared=True)` returning this run's training rows as a
        DataFrame (after prepare(df) when prepared=True), `RUN_DIR` (the run folder) and
        `load_dataset(name)`. The test rows are deliberately unavailable. print() what you want
        to see; nothing persists between calls except files. Save figures under RUN_DIR/figures.

        Args:
            code: Complete Python script to execute.
        """
        return _dump(run_code(run.ws, code, timeout=settings.python_timeout, run_id=run.run_id))

    @tool
    def get_column_roles() -> str:
        """Show how each column is currently treated: numeric, categorical, high_cardinality or dropped."""
        return _dump({**run.roles.model_dump(), "features_hash": run.features_hash})

    return {t.name: t for t in (profile_train, preview_train, run_python, get_column_roles)}


def data_tools(run: RunStore, settings: Settings) -> list[BaseTool]:
    shared = _shared_run_tools(run, settings)

    @tool(parse_docstring=True)
    def define_features(code: str) -> str:
        """Set the run's feature-engineering function and re-infer column roles.

        `code` must define `prepare(df) -> DataFrame`: a row-wise transformation (same rows, same
        order and index) that must not use the target column. It runs on training rows now, and
        on test rows and new data later, so it may only use information available at prediction
        time. Typical uses: parse dates into parts or elapsed time, ratios, flags, text lengths.
        Do NOT impute, scale or encode here - the model pipelines do that inside each CV fold.
        Calling this again replaces the previous function.

        Args:
            code: Python source defining prepare(df).
        """
        run.features_path.write_text(code)
        return _dump(run.register_features())

    @tool(parse_docstring=True)
    def set_column_roles(
        numeric: list[str],
        categorical: list[str],
        high_cardinality: list[str],
        dropped: list[DroppedColumn],
    ) -> str:
        """Set how every column is treated by the model pipelines. Columns not listed are ignored.

        Args:
            numeric: Numeric (or boolean) feature columns.
            categorical: Categorical features with at most ~30 levels (one-hot / ordinal encoded).
            high_cardinality: Categorical features with many levels (target encoded within folds).
            dropped: Columns excluded from the features, each with the reason.
        """
        roles = ColumnRoles(
            numeric=numeric,
            categorical=categorical,
            high_cardinality=high_cardinality,
            dropped=dropped,
        )
        run.set_roles(roles)
        return _dump(
            {"saved": True, "features": roles.features, "features_hash": run.features_hash}
        )

    return [
        shared["profile_train"],
        shared["preview_train"],
        shared["run_python"],
        shared["get_column_roles"],
        define_features,
        set_column_roles,
    ]


def analysis_tools(run: RunStore, settings: Settings) -> list[BaseTool]:
    shared = _shared_run_tools(run, settings)

    @tool(parse_docstring=True)
    def feature_associations(top_k: int = 20) -> str:
        """Rank features by mutual information with the target (non-linear dependence), with Spearman correlation for numeric features.

        Args:
            top_k: Number of features to return.
        """
        return _dump(diagnostics.feature_associations(run, top_k=top_k))

    @tool
    def quick_baselines() -> str:
        """Cross-validate untuned dummy, linear and depth-3 tree models to gauge how much signal there is and whether it looks linear."""
        return _dump(diagnostics.quick_baselines(run))

    return [
        shared["profile_train"],
        shared["preview_train"],
        shared["get_column_roles"],
        shared["run_python"],
        feature_associations,
        quick_baselines,
    ]


def _leaderboard_tool(run: RunStore) -> BaseTool:
    @tool
    def leaderboard() -> str:
        """Current candidates ranked by mean selection-CV score, plus the remaining compute budget."""
        board = evaluation.leaderboard(run)
        board["budget_remaining_seconds"] = round(run.remaining_seconds())
        return _dump(board)

    return leaderboard


def _compare_tool(run: RunStore) -> BaseTool:
    @tool(parse_docstring=True)
    def compare_candidates(candidate_ids: list[str] | None = None) -> str:
        """Paired comparison of candidates against the best one on identical folds: corrected t-test p-values and the 1-SE simplest-model recommendation.

        Args:
            candidate_ids: Limit the comparison to these candidates (default: all current ones).
        """
        return _dump(evaluation.compare(run, candidate_ids))

    return compare_candidates


def modeling_tools(run: RunStore, settings: Settings) -> list[BaseTool]:
    @tool
    def list_model_families() -> str:
        """The scikit-learn model families you can search, whether each is usable for this run, and the remaining compute budget."""
        spec = run.spec
        X, _ = run.xy_train()
        families = [
            {
                "family": name,
                "description": family.description,
                "complexity": family.complexity,
                "unavailable": unavailable_reason(name, spec.task, len(X), spec.primary_metric),
            }
            for name, family in FAMILIES.items()
        ]
        return _dump(
            {
                "families": families,
                "also_available": "run_flaml (FLAML AutoML as a competitor), build_ensemble, "
                "tune_threshold",
                "training_rows": len(X),
                "budget_remaining_seconds": round(run.remaining_seconds()),
            }
        )

    @tool(parse_docstring=True)
    def tune_families(
        families: list[str],
        seconds_per_family: float | None = None,
        max_configs: int = 40,
        notes: str | None = None,
    ) -> str:
        """Random-search each family's hyperparameters on the tuning folds, then score the winner on the shared selection folds as a new candidate.

        Args:
            families: Family names from list_model_families, e.g. ['dummy', 'linear', 'hist_gb'].
            seconds_per_family: Search time per family. Default: a fair share of the remaining budget.
            max_configs: Upper bound on configurations tried per family.
            notes: Why these families / what changed since the previous round.
        """
        share = seconds_per_family or max(run.remaining_seconds() / (len(families) + 2), 10)
        results, errors = [], {}
        for family in families:
            try:
                record = evaluation.tune_family(
                    run, family, budget_seconds=share, max_iter=max_configs, notes=notes
                )
            except (ValueError, KeyError, RuntimeError) as exc:
                errors[family] = str(exc)
                continue
            metric = run.spec.primary_metric
            results.append(
                {
                    "id": record["id"],
                    f"cv_{metric}": record["cv"][metric]["mean"],
                    "cv_std": record["cv"][metric]["std"],
                    f"train_{metric}": record["train"][metric],
                    "configurations_tried": (record["tuning"] or {}).get("configurations_tried", 0),
                    "params": record["params"],
                }
            )
        return _dump(
            {
                "candidates": results,
                "errors": errors or None,
                "budget_remaining_seconds": round(run.remaining_seconds()),
            }
        )

    @tool(parse_docstring=True)
    def run_flaml(time_budget_seconds: float, notes: str | None = None) -> str:
        """Let FLAML AutoML search the training rows, then score its winning model on the same selection folds as every other candidate.

        Args:
            time_budget_seconds: How long FLAML may search.
            notes: Why FLAML is being run now.
        """
        record = evaluation.flaml_competitor(run, time_budget=time_budget_seconds, notes=notes)
        metric = run.spec.primary_metric
        return _dump(
            {
                "id": record["id"],
                "flaml_picked": record["params"].get("flaml_estimator"),
                f"cv_{metric}": record["cv"][metric]["mean"],
                "cv_std": record["cv"][metric]["std"],
                "budget_remaining_seconds": round(run.remaining_seconds()),
            }
        )

    @tool(parse_docstring=True)
    def build_ensemble(
        candidate_ids: list[str], kind: str = "voting", notes: str | None = None
    ) -> str:
        """Combine existing candidates into an ensemble candidate scored on the same folds. Only worth it when members are strong and differ in kind.

        Args:
            candidate_ids: Two to four strong, diverse candidate ids.
            kind: 'voting' (average probabilities / predictions) or 'stacking' (a meta-model learns the blend).
            notes: Why this ensemble.
        """
        record = evaluation.build_ensemble(run, candidate_ids, kind=kind, notes=notes)
        metric = run.spec.primary_metric
        return _dump(
            {
                "id": record["id"],
                f"cv_{metric}": record["cv"][metric]["mean"],
                "cv_std": record["cv"][metric]["std"],
                "budget_remaining_seconds": round(run.remaining_seconds()),
            }
        )

    @tool(parse_docstring=True)
    def tune_threshold(candidate_id: str, objective: str | None = None) -> str:
        """Binary classification: choose the decision threshold on the tuning folds and show precision/recall at the default vs tuned threshold on the selection folds.

        Args:
            candidate_id: Candidate to tune (usually the proposed champion).
            objective: 'f1', 'f2' (favours recall), 'f0.5' (favours precision) or 'balanced_accuracy'. Defaults to the spec's threshold_objective.
        """
        return _dump(evaluation.tune_threshold(run, candidate_id, objective))

    return [
        list_model_families,
        tune_families,
        run_flaml,
        build_ensemble,
        tune_threshold,
        _leaderboard_tool(run),
        _compare_tool(run),
    ]


def review_tools(run: RunStore, settings: Settings) -> list[BaseTool]:
    @tool(parse_docstring=True)
    def get_candidate(candidate_id: str) -> str:
        """Full record of one candidate: hyperparameters, per-fold scores, train score, threshold.

        Args:
            candidate_id: Candidate id such as 'c03-hist_gb'.
        """
        return _dump(run.candidate(candidate_id))

    @tool
    def check_overfitting() -> str:
        """Gap between training and validation scores for every current candidate."""
        return _dump(diagnostics.check_overfitting(run))

    @tool
    def check_leakage() -> str:
        """Score each top feature alone; a feature that nearly matches the best model on its own suggests leakage."""
        return _dump(diagnostics.check_leakage(run))

    @tool(parse_docstring=True)
    def check_success(candidate_id: str) -> str:
        """Compare a candidate with the dummy baseline and the spec's minimum score and success criteria.

        Args:
            candidate_id: Candidate id to check.
        """
        return _dump(diagnostics.check_success(run, candidate_id))

    @tool(parse_docstring=True)
    def rescore_candidate(candidate_id: str) -> str:
        """Re-run a candidate's cross-validation to confirm its scores are reproducible.

        Args:
            candidate_id: Candidate id to re-score.
        """
        return _dump(diagnostics.rescore(run, candidate_id))

    return [
        _leaderboard_tool(run),
        _compare_tool(run),
        get_candidate,
        check_overfitting,
        check_leakage,
        check_success,
        rescore_candidate,
    ]


def visualization_tools(run: RunStore, settings: Settings) -> list[BaseTool]:
    menu = "; ".join(f"{name}: {desc}" for name, (_, desc) in charts.CHARTS.items())

    @tool(parse_docstring=True)
    def make_chart(kind: str, candidate_ids: list[str] | None = None) -> str:
        """Render a chart to the run's figures/ folder and return its path and the numbers it shows (use them for captions - you cannot see the image).

        Args:
            kind: Chart name (see the tool description for the menu).
            candidate_ids: For candidate charts, which candidates to include (default: the top ones).
        """
        return _dump(charts.make_chart(run, kind, candidate_ids))

    make_chart.description += f"\n\nCharts: {menu}"

    @tool(parse_docstring=True)
    def write_report(markdown: str) -> str:
        """Save the final report as report.md in the run folder. Embed figures with relative links like ![Leaderboard](figures/leaderboard.png). Leaderboard and test-metric tables are appended automatically.

        Args:
            markdown: The report body in Markdown, starting with a level-1 title.
        """
        path = run.root / "report.md"
        path.write_text(markdown.strip() + "\n\n" + appendix(run))
        return _dump({"report_path": str(path)})

    return [make_chart, write_report, _leaderboard_tool(run)]


def appendix(run: RunStore) -> str:
    """Deterministic tables appended to every report, so all numbers are available as text."""
    board = evaluation.leaderboard(run)
    metric = board["metric"]
    lines = [
        "## Appendix: leaderboard (selection CV)",
        "",
        f"| rank | candidate | {metric} mean | std | train {metric} | complexity |",
        "|---|---|---|---|---|---|",
    ]
    for row in board["candidates"]:
        lines.append(
            f"| {row['rank']} | {row['id']} | {row[f'cv_{metric}']} | {row['cv_std']} "
            f"| {row[f'train_{metric}']} | {row['complexity']} |"
        )
    final = run.read_json("champion/final.json")
    if final:
        lines += [
            "",
            f"## Appendix: locked test set ({final['rows']['test']} rows), champion {final['champion']}",
            "",
        ]
        if final.get("threshold") is not None:
            objective = run.candidate(final["champion"]).get("threshold_objective")
            lines += [
                f"Decision threshold {final['threshold']:.3g} for `{run.classes[1]}` (tuned for "
                f"{objective}); threshold-dependent metrics such as accuracy reflect that choice.",
                "",
            ]
        lines += [
            "| metric | champion | naive baseline |",
            "|---|---|---|",
        ]
        for key, value in final["test_metrics"].items():
            if key in ("confusion_matrix", "per_class"):
                continue
            lines.append(f"| {key} | {value} | {final['baseline_metrics'].get(key)} |")
    return "\n".join(lines) + "\n"
