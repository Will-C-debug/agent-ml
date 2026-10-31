"""The scikit-learn model families the Modeling agent can search, with their search spaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml_agent.team.preprocess import build_preprocessor, native_categorical_mask
from ml_agent.team.schemas import ColumnRoles

# Metrics that need predict_proba (families without it are skipped for these).
PROBA_METRICS = {"log_loss"}

# Preprocessing recipe for each estimator FLAML may pick.
FLAML_RECIPES = {
    "lgbm": "tree",
    "xgboost": "tree",
    "xgb_limitdepth": "tree",
    "rf": "tree",
    "extra_tree": "tree",
    "histgb": "tree",
    "catboost": "tree",
    "lrl1": "linear",
    "lrl2": "linear",
    "kneighbor": "linear",
    "sgd": "linear",
    "svc": "linear",
    "enet": "linear",
    "lassolars": "linear",
}


@dataclass(frozen=True)
class Family:
    name: str
    description: str
    recipe: str
    complexity: int  # lower is simpler; used to prefer simpler models under the 1-SE rule
    make: Callable[[str, int, ColumnRoles], BaseEstimator]
    space: Callable[[str], dict[str, Any]]
    max_rows: int | None = None
    has_proba: bool = True


def _cls(task: str) -> bool:
    return task == "classification"


def _forest_space(task: str) -> dict[str, Any]:
    space = {
        "n_estimators": randint(100, 500),
        "max_depth": [None, 6, 12, 24],
        "min_samples_leaf": randint(1, 20),
        "max_features": ["sqrt", 0.3, 0.6, 1.0],
    }
    if _cls(task):
        space["class_weight"] = [None, "balanced_subsample"]
    return space


FAMILIES: dict[str, Family] = {
    f.name: f
    for f in [
        Family(
            name="dummy",
            description="Predicts the class prior / training mean. The floor every model must beat.",
            recipe="linear",
            complexity=0,
            make=lambda task, seed, roles: DummyClassifier(strategy="prior")
            if _cls(task)
            else DummyRegressor(),
            space=lambda task: {},
        ),
        Family(
            name="linear",
            description="Logistic regression / ridge on scaled, one-hot encoded features. Fast, "
            "interpretable, strong when effects are roughly additive.",
            recipe="linear",
            complexity=1,
            make=lambda task, seed, roles: LogisticRegression(max_iter=3000)
            if _cls(task)
            else Ridge(),
            space=lambda task: {"C": loguniform(1e-3, 1e2), "class_weight": [None, "balanced"]}
            if _cls(task)
            else {"alpha": loguniform(1e-3, 1e3)},
        ),
        Family(
            name="decision_tree",
            description="A single tree. Very interpretable; usually weaker than ensembles.",
            recipe="tree",
            complexity=2,
            make=lambda task, seed, roles: DecisionTreeClassifier(random_state=seed)
            if _cls(task)
            else DecisionTreeRegressor(random_state=seed),
            space=lambda task: {
                "max_depth": [2, 3, 4, 6, 8, 12, None],
                "min_samples_leaf": randint(1, 50),
                **({"class_weight": [None, "balanced"]} if _cls(task) else {}),
            },
        ),
        Family(
            name="knn",
            description="k-nearest neighbours on scaled features. Captures local structure; "
            "slow to predict on large data.",
            recipe="linear",
            complexity=2,
            max_rows=20_000,
            make=lambda task, seed, roles: KNeighborsClassifier()
            if _cls(task)
            else KNeighborsRegressor(),
            space=lambda task: {
                "n_neighbors": randint(3, 51),
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
        ),
        Family(
            name="svm",
            description="RBF support vector machine. Strong on small/medium data with smooth "
            "boundaries; no probabilities, quadratic training cost.",
            recipe="linear",
            complexity=4,
            max_rows=20_000,
            has_proba=False,
            make=lambda task, seed, roles: SVC(random_state=seed) if _cls(task) else SVR(),
            space=lambda task: {
                "C": loguniform(1e-2, 1e2),
                "gamma": loguniform(1e-4, 1e0),
                **(
                    {"class_weight": [None, "balanced"]}
                    if _cls(task)
                    else {"epsilon": uniform(0.01, 0.5)}
                ),
            },
        ),
        Family(
            name="random_forest",
            description="Random forest. Robust default for tabular data; little tuning needed.",
            recipe="tree",
            complexity=3,
            make=lambda task, seed, roles: RandomForestClassifier(random_state=seed)
            if _cls(task)
            else RandomForestRegressor(random_state=seed),
            space=_forest_space,
        ),
        Family(
            name="extra_trees",
            description="Extremely randomized trees. Like random forest, often smoother and faster.",
            recipe="tree",
            complexity=3,
            make=lambda task, seed, roles: ExtraTreesClassifier(random_state=seed)
            if _cls(task)
            else ExtraTreesRegressor(random_state=seed),
            space=_forest_space,
        ),
        Family(
            name="hist_gb",
            description="Histogram gradient boosting with native categorical and missing-value "
            "support. Usually the strongest single family on tabular data.",
            recipe="native",
            complexity=3,
            make=lambda task, seed, roles: (
                HistGradientBoostingClassifier if _cls(task) else HistGradientBoostingRegressor
            )(random_state=seed, categorical_features=native_categorical_mask(roles)),
            space=lambda task: {
                "learning_rate": loguniform(0.02, 0.3),
                "max_iter": randint(100, 500),
                "max_leaf_nodes": randint(15, 128),
                "min_samples_leaf": randint(5, 100),
                "l2_regularization": loguniform(1e-4, 10),
                **({"class_weight": [None, "balanced"]} if _cls(task) else {}),
            },
        ),
    ]
}


def make_pipeline(
    family: str,
    task: str,
    roles: ColumnRoles,
    *,
    seed: int = 42,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    """A preprocessing + model pipeline for a family; ``params`` use ``model__`` prefixes."""
    spec = FAMILIES[family]
    pipeline = Pipeline(
        [("prep", build_preprocessor(spec.recipe, roles)), ("model", spec.make(task, seed, roles))]
    )
    return pipeline.set_params(**(params or {}))


def search_space(family: str, task: str) -> dict[str, Any]:
    """The family's hyperparameter distributions, prefixed for the pipeline's model step."""
    return {f"model__{k}": v for k, v in FAMILIES[family].space(task).items()}


def unavailable_reason(family: str, task: str, n_rows: int, metric: str) -> str | None:
    """Why a family can't be used for this problem, or None if it can."""
    if family not in FAMILIES:
        return f"unknown family; choose from {sorted(FAMILIES)}"
    spec = FAMILIES[family]
    if spec.max_rows and n_rows > spec.max_rows:
        return f"skipped above {spec.max_rows} training rows (too slow)"
    if task == "classification" and metric in PROBA_METRICS and not spec.has_proba:
        return f"has no predict_proba, which {metric} needs"
    return None
