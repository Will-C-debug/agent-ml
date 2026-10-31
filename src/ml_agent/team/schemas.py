"""Typed contracts between the agents.

Each specialist returns one of these models as its structured output. They are kept to plain
fields and lists of small objects (no free-form dicts) so they work with Claude's native structured
outputs, which require closed object schemas. Every field is required; optional ones are nullable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ml_agent.automl import CLASSIFICATION_METRICS, REGRESSION_METRICS

Task = Literal["classification", "regression"]
Stage = Literal["data", "analysis", "modeling"]
# F2 favours recall, F0.5 favours precision.
ThresholdObjective = Literal["f1", "f2", "f0.5", "balanced_accuracy"]


class ProblemSpec(BaseModel):
    """What the Planner agent decides: the problem, how success is measured, and the CV scheme."""

    dataset: str = Field(description="Workspace dataset name holding the raw data.")
    target: str = Field(description="Column to predict.")
    task: Task
    primary_metric: str = Field(
        description="Metric that ranks models. Classification: accuracy, roc_auc, roc_auc_ovr, "
        "roc_auc_ovo, log_loss, f1, micro_f1, macro_f1, ap. Regression: r2, rmse, mae, mse, mape."
    )
    secondary_metrics: list[str] = Field(description="Other metrics to report, same names.")
    positive_class: str | None = Field(
        description="Binary classification: the label of the class of interest (e.g. 'yes'). "
        "Null otherwise."
    )
    success_criteria: str = Field(description="What counts as success, in plain words.")
    min_primary_score: float | None = Field(
        description="Numeric bar for the primary metric on held-out data, if the user gave one."
    )
    cv_folds: int = Field(description="Folds for model selection (5 is a good default).")
    test_size: float = Field(description="Fraction of rows locked away as the final test set.")
    group_column: str | None = Field(
        description="Column whose values must not be split across folds (e.g. customer id when "
        "a customer has several rows). Null if rows are independent."
    )
    threshold_objective: ThresholdObjective | None = Field(
        description="Binary classification: what the decision threshold should optimize. "
        "Null to keep 0.5."
    )
    excluded_columns: list[str] = Field(
        description="Columns that must never be features (identifiers, post-outcome fields)."
    )
    constraints: list[str] = Field(description="E.g. 'must be explainable', 'fast predictions'.")
    assumptions: list[str] = Field(description="Assumptions made where the request was vague.")

    @field_validator("primary_metric")
    @classmethod
    def _known_metric(cls, value: str) -> str:
        if value not in (*CLASSIFICATION_METRICS, *REGRESSION_METRICS):
            raise ValueError(f"Unknown metric {value!r}")
        return value

    def check(self) -> None:
        """Validate cross-field rules (raises ValueError with a readable message)."""
        allowed = CLASSIFICATION_METRICS if self.task == "classification" else REGRESSION_METRICS
        for metric in (self.primary_metric, *self.secondary_metrics):
            if metric not in allowed:
                raise ValueError(f"Metric {metric!r} is not valid for {self.task}: {allowed}")
        if not 2 <= self.cv_folds <= 10:
            raise ValueError("cv_folds must be between 2 and 10")
        if not 0.05 <= self.test_size <= 0.4:
            raise ValueError("test_size must be between 0.05 and 0.4")
        if self.task == "regression" and (self.positive_class or self.threshold_objective):
            raise ValueError("positive_class and threshold_objective only apply to classification")


class DroppedColumn(BaseModel):
    column: str
    reason: str


class ColumnRoles(BaseModel):
    """How each feature column is treated by the preprocessing pipelines."""

    numeric: list[str]
    categorical: list[str] = Field(description="Low-cardinality categories (one-hot / ordinal).")
    high_cardinality: list[str] = Field(description="Many-level categories (target encoded).")
    dropped: list[DroppedColumn]

    @property
    def features(self) -> list[str]:
        return [*self.numeric, *self.categorical, *self.high_cardinality]


class Finding(BaseModel):
    title: str
    detail: str


class DataReport(BaseModel):
    """What the Data agent hands over: a clean, documented feature set."""

    summary: str
    feature_engineering: list[str] = Field(description="Features created by prepare(df), and why.")
    column_roles_rationale: list[str]
    leakage_findings: list[Finding]
    data_quality_notes: list[str]


class AnalysisReport(BaseModel):
    """What the Analysis agent found in the training data."""

    summary: str
    key_drivers: list[Finding]
    patterns: list[Finding]
    hypotheses: list[str]
    recommended_families: list[str] = Field(description="Model families worth searching first.")
    preprocessing_advice: list[str]


class ModelingReport(BaseModel):
    """What the Modeling agent tried and which candidate it proposes."""

    summary: str
    candidates_tried: list[str] = Field(description="Candidate ids evaluated in this round.")
    proposed_champion: str = Field(description="Candidate id proposed as the best model.")
    rationale: str
    open_questions: list[str]


class Issue(BaseModel):
    severity: Literal["blocker", "major", "minor"]
    title: str
    evidence: str


class ReviewVerdict(BaseModel):
    """The Review agent's gate decision before the test set is used."""

    decision: Literal["approve", "revise"]
    revisit_stage: Stage | None = Field(description="Where to send the work back to, if revising.")
    champion: str | None = Field(
        description="Approved candidate id (may differ from the Modeling agent's proposal, e.g. a "
        "simpler model within one standard error). Null when revising."
    )
    issues: list[Issue]
    requested_changes: list[str] = Field(description="Concrete instructions for the next round.")
    summary: str


class SignOff(BaseModel):
    """The Review agent's verdict after the one-time test evaluation."""

    trustworthy: bool
    summary: str
    caveats: list[str]


class Figure(BaseModel):
    path: str
    caption: str


class VisualReport(BaseModel):
    """The Visualization agent's deliverable: figures, insights and recommended actions."""

    headline: str
    insights: list[str]
    recommended_actions: list[str]
    figures: list[Figure]
    report_path: str
