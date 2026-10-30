"""LangChain tools that give the agent its ML-engineering capabilities.

Tools are built per workspace with :func:`build_tools`; each returns JSON text for the model.
Exceptions raised here are turned into error tool results by ``ToolErrorMiddleware`` (see
``agent.py``) so the agent can read the message and correct itself.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from ml_agent import automl
from ml_agent.config import Settings
from ml_agent.executor import run_code
from ml_agent.profiling import profile_dataframe
from ml_agent.workspace import Workspace, now_iso, slugify


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, default=str, ensure_ascii=False)


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    """An experiment record without fields that are long and rarely useful to the model."""
    compact = {k: v for k, v in record.items() if k not in ("feature_dtypes", "feature_importance")}
    if "feature_importance" in record:
        compact["top_features"] = record["feature_importance"][:10]
    return compact


def build_tools(ws: Workspace, settings: Settings) -> list[BaseTool]:
    """Create the agent's tools bound to one workspace."""

    @tool(parse_docstring=True)
    def load_dataset(path: str, name: str | None = None) -> str:
        """Import a local data file into the workspace as a named dataset.

        Reads CSV, TSV, Parquet, Feather, JSON, JSON Lines and Excel files. The data is copied
        into the workspace so later steps refer to it by name.

        Args:
            path: File path - absolute, relative to the current directory, or relative to the workspace.
            name: Name to register the dataset under. Defaults to the file name without extension.
        """
        info = ws.import_file(path, name)
        return _dump({**info, "n_columns": len(info["columns"])})

    @tool
    def list_datasets() -> str:
        """List the datasets stored in the workspace with their size and provenance."""
        rows = []
        for name in ws.dataset_names():
            info = ws.dataset_info(name)
            columns = info.pop("columns")
            info["n_columns"] = len(columns)
            info["columns"] = columns[:40] + (["..."] if len(columns) > 40 else [])
            rows.append(info)
        return _dump(rows or "No datasets yet. Import one with load_dataset.")

    @tool(parse_docstring=True)
    def preview_dataset(name: str, n_rows: int = 10, columns: list[str] | None = None) -> str:
        """Show the first rows of a dataset as CSV text.

        Args:
            name: Dataset name.
            n_rows: Number of rows to show (max 50).
            columns: Optional subset of columns to show.
        """
        df = ws.load_dataset(name)
        if columns:
            df = df[columns]
        return df.head(min(max(n_rows, 1), 50)).to_csv(index=False)

    @tool(parse_docstring=True)
    def profile_dataset(
        name: str, target: str | None = None, columns: list[str] | None = None
    ) -> str:
        """Profile a dataset: column types and statistics, missing values, data-quality warnings, and - if a target is given - task type, class balance and possible leakage.

        Always profile with the target before training.

        Args:
            name: Dataset name.
            target: Column to predict. Enables target diagnostics and leakage checks.
            columns: Profile only these columns (by default, the first 60).
        """
        return _dump(profile_dataframe(ws.load_dataset(name), target=target, columns=columns))

    @tool(parse_docstring=True)
    def run_python(code: str, timeout_seconds: int | None = None) -> str:
        """Run a Python script on the user's machine, with the workspace as working directory.

        Use it for data cleaning, feature engineering, joins, custom plots or checks the other
        tools don't cover. Predefined: `pd`, `np`, `ws` (the workspace), `load_dataset(name)` returning
        a DataFrame, and `save_dataset(df, name, note=None)` which stores a DataFrame as a new or
        updated dataset that train_automl can use. Nothing persists between calls except files, so
        load what you need each time and print() the results you want to see. Save figures under
        `reports/`. Prefer writing a new dataset (e.g. `churn_v2`) over overwriting the original.

        Args:
            code: Complete Python script to execute.
            timeout_seconds: Kill the script after this many seconds (default from settings).
        """
        timeout = min(timeout_seconds or settings.python_timeout, settings.python_timeout)
        return _dump(run_code(ws, code, timeout=timeout))

    @tool(parse_docstring=True)
    def train_automl(
        dataset: str,
        target: str,
        task: str = "auto",
        metric: str = "auto",
        time_budget: int = 60,
        estimators: list[str] | None = None,
        drop_columns: list[str] | None = None,
        test_size: float = 0.2,
        seed: int = 42,
        notes: str | None = None,
    ) -> str:
        """Train models with FLAML AutoML and evaluate the winner on a held-out test split.

        A stratified test split is held out first; FLAML then searches estimators and
        hyperparameters on the rest for `time_budget` seconds and refits the best configuration.
        Returns the experiment ID, the leaderboard per estimator, held-out test metrics, the
        naive-baseline metrics and, for binary targets, precision/recall at several decision
        thresholds (operating_points). The model is saved for explain_model, predict and
        export_model.

        Args:
            dataset: Dataset name.
            target: Column to predict.
            task: 'classification', 'regression', or 'auto' to infer from the target.
            metric: Metric to optimize. Classification - accuracy, roc_auc, roc_auc_ovr, roc_auc_ovo, log_loss, f1, micro_f1, macro_f1, ap. Regression - r2, rmse, mae, mse, mape. 'auto' uses roc_auc (binary), log_loss (multiclass) or r2.
            time_budget: Wall-clock seconds for the search. Start with 30-120.
            estimators: Restrict the search, e.g. ['lgbm', 'xgboost', 'rf', 'extra_tree', 'histgb', 'lrl1']. Omit to let FLAML choose.
            drop_columns: Columns to exclude from the features (identifiers, leakage, free text).
            test_size: Fraction of rows held out for the final test evaluation.
            seed: Random seed for the split and the search.
            notes: Why this experiment was run (what changed versus the previous one).
        """
        record = automl.train(
            ws,
            dataset,
            target,
            task=task,
            metric=metric,
            time_budget=time_budget,
            estimators=estimators,
            drop_columns=drop_columns,
            test_size=test_size,
            seed=seed,
            max_time_budget=settings.max_time_budget,
            notes=notes,
        )
        result = _compact_record(record)
        if record["time_budget"]["used"] < record["time_budget"]["requested"]:
            result["warning"] = (
                f"time_budget was capped at {settings.max_time_budget}s by the session settings."
            )
        return _dump(result)

    @tool(parse_docstring=True)
    def list_experiments(dataset: str | None = None) -> str:
        """List past AutoML experiments with their validation and held-out test scores.

        Args:
            dataset: Only show experiments trained on this dataset.
        """
        records = [ws.load_experiment(i) for i in ws.experiment_ids()]
        if dataset:
            records = [r for r in records if r["dataset"] == slugify(dataset)]
        return _dump([automl.summarize_experiment(r) for r in records] or "No experiments yet.")

    @tool(parse_docstring=True)
    def get_experiment(experiment_id: str) -> str:
        """Show the full record of one experiment: data, search settings, best configuration and all metrics.

        Args:
            experiment_id: Experiment ID such as 'exp-001'.
        """
        return _dump(_compact_record(ws.load_experiment(experiment_id)))

    @tool(parse_docstring=True)
    def explain_model(experiment_id: str, top_k: int = 15) -> str:
        """Rank the features of a trained model by permutation importance on its held-out test split.

        Importance is the drop in the experiment's metric when a feature is shuffled; values at or
        below zero mean the feature doesn't help. Also adds the ranking to the model card.

        Args:
            experiment_id: Experiment ID such as 'exp-001'.
            top_k: Number of top features to return.
        """
        return _dump(automl.explain(ws, experiment_id, top_k=top_k))

    @tool(parse_docstring=True)
    def predict(
        experiment_id: str,
        data: str,
        output_path: str | None = None,
        threshold: float | None = None,
    ) -> str:
        """Score new data with a trained model and save the predictions as CSV in the workspace.

        Args:
            experiment_id: Experiment ID such as 'exp-001'.
            data: A workspace dataset name or a path to a data file containing the feature columns.
            output_path: CSV path relative to the workspace's predictions/ folder. Defaults to an automatic name.
            threshold: Binary classification only - predict the positive class when its probability is at least this value instead of 0.5. Pick it from the experiment's operating_points.
        """
        return _dump(automl.predict(ws, experiment_id, data, output_path, threshold))

    @tool(parse_docstring=True)
    def export_model(
        experiment_id: str, name: str | None = None, threshold: float | None = None
    ) -> str:
        """Package a trained model as a self-contained folder with predict.py, pinned requirements and a model card.

        Args:
            experiment_id: Experiment ID such as 'exp-001'.
            name: Folder name under exports/. Defaults to the experiment ID.
            threshold: Binary classification only - decision threshold baked into predict.py.
        """
        return _dump(automl.export(ws, experiment_id, name, threshold))

    @tool(parse_docstring=True)
    def write_report(title: str, markdown: str) -> str:
        """Save a Markdown report to the workspace's reports/ folder.

        Args:
            title: Report title; also used for the file name.
            markdown: Report body in Markdown.
        """
        path = ws.reports_dir / f"{slugify(title)}.md"
        path.write_text(f"# {title}\n\n_Generated {now_iso()}_\n\n{markdown.strip()}\n")
        return _dump({"report_path": str(path)})

    return [
        load_dataset,
        list_datasets,
        preview_dataset,
        profile_dataset,
        run_python,
        train_automl,
        list_experiments,
        get_experiment,
        explain_model,
        predict,
        export_model,
        write_report,
    ]
