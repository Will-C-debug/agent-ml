"""Column roles and the preprocessing recipes that turn them into scikit-learn transformers.

All stateful preprocessing (imputation, scaling, encoding) lives inside each model's ``Pipeline``,
so it is fitted on the training part of every CV fold and cannot leak validation data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from ml_agent.profiling import profile_dataframe
from ml_agent.team.schemas import ColumnRoles, DroppedColumn

# Recipes: "linear" for scale-sensitive models, "tree" for tree ensembles, "native" for
# HistGradientBoosting's built-in categorical support.
RECIPES = ("linear", "tree", "native")
MAX_LOW_CARDINALITY = 30
MAX_NATIVE_CATEGORIES = 250  # HistGradientBoosting handles at most 255 bins per feature

_DROP_REASONS = {
    "constant": "constant - carries no signal",
    "id_like": "looks like an identifier",
    "text": "free text - engineer features from it in prepare(df)",
    "date_string": "date stored as text - extract features from it in prepare(df)",
    "datetime": "datetime - extract features from it in prepare(df)",
}


def infer_roles(df: pd.DataFrame, target: str, excluded: list[str] | None = None) -> ColumnRoles:
    """Default roles from the profiler; the Data agent reviews and adjusts them."""
    excluded = set(excluded or [])
    numeric, categorical, high_card, dropped = [], [], [], []
    columns = [c for c in df.columns if c != target]
    for info in profile_dataframe(df[columns], columns=columns)["column_profiles"]:
        name, kind = info["name"], info["kind"]
        if name in excluded:
            dropped.append(DroppedColumn(column=name, reason="excluded by the problem spec"))
        elif kind in _DROP_REASONS:
            dropped.append(DroppedColumn(column=name, reason=_DROP_REASONS[kind]))
        elif kind in ("numeric", "boolean"):
            numeric.append(name)
        elif info["n_unique"] <= MAX_LOW_CARDINALITY:
            categorical.append(name)
        else:
            high_card.append(name)
    return ColumnRoles(
        numeric=numeric, categorical=categorical, high_cardinality=high_card, dropped=dropped
    )


def validate_roles(df: pd.DataFrame, roles: ColumnRoles, target: str) -> None:
    """Raise ValueError if roles reference unknown columns, overlap, or include the target."""
    assigned = [*roles.features, *(d.column for d in roles.dropped)]
    unknown = sorted(set(assigned) - set(df.columns))
    if unknown:
        raise ValueError(f"Unknown columns in roles: {unknown}")
    duplicated = sorted({c for c in assigned if assigned.count(c) > 1})
    if duplicated:
        raise ValueError(f"Columns assigned more than one role: {duplicated}")
    if target in assigned:
        raise ValueError("The target cannot be a feature or appear in the roles")
    if not roles.features:
        raise ValueError("At least one feature column is required")
    bad_numeric = [
        c
        for c in roles.numeric
        if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
    ]
    if bad_numeric:
        raise ValueError(
            f"Columns marked numeric are not numeric: {bad_numeric}. Convert them in prepare(df) "
            "or mark them categorical."
        )
    too_many = [c for c in roles.categorical if df[c].nunique() > MAX_NATIVE_CATEGORIES]
    if too_many:
        raise ValueError(
            f"Columns with more than {MAX_NATIVE_CATEGORIES} levels must be high_cardinality: "
            f"{too_many}"
        )


def coerce_features(df: pd.DataFrame, roles: ColumnRoles) -> pd.DataFrame:
    """Select feature columns with consistent dtypes: floats for numeric, strings for categories.

    The same conversion is repeated in exported ``predict.py`` bundles, so keep them in sync.
    """
    X = df[roles.features].copy()
    for col in roles.numeric:
        X[col] = X[col].astype(float)
    for col in (*roles.categorical, *roles.high_cardinality):
        values = X[col]
        as_text = values.astype(object).where(values.isna(), values.astype(str))
        X[col] = as_text.where(as_text.notna(), np.nan)  # pd.NA -> np.nan for the encoders
    return X


def build_preprocessor(recipe: str, roles: ColumnRoles) -> ColumnTransformer:
    """A ColumnTransformer for one recipe. Output column order: numeric, categorical, high-card."""
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe {recipe!r}")
    numeric, categorical, high_card = roles.numeric, roles.categorical, roles.high_cardinality
    parts: list[tuple] = []
    if recipe == "linear":
        if numeric:
            parts.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                )
            )
        if categorical:
            parts.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="constant", fill_value="<missing>")),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="infrequent_if_exist",
                                    min_frequency=0.01,
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    categorical,
                )
            )
        if high_card:
            parts.append(
                (
                    "high_card",
                    Pipeline(
                        [("encode", TargetEncoder(random_state=0)), ("scale", StandardScaler())]
                    ),
                    high_card,
                )
            )
    else:
        if numeric:
            numeric_step = (
                "passthrough"
                if recipe == "native"
                else SimpleImputer(strategy="median", add_indicator=True)
            )
            parts.append(("num", numeric_step, numeric))
        if categorical:
            # -1 marks missing/unseen categories; HistGradientBoosting treats negatives as missing.
            parts.append(
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                    categorical,
                )
            )
        if high_card:
            parts.append(("high_card", TargetEncoder(random_state=0), high_card))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def native_categorical_mask(roles: ColumnRoles) -> list[bool]:
    """Which output columns of the "native" recipe HistGradientBoosting should treat as categories."""
    return (
        [False] * len(roles.numeric)
        + [True] * len(roles.categorical)
        + [False] * len(roles.high_cardinality)
    )
