"""Dataset profiling: column statistics, data-quality warnings and target diagnostics."""

from __future__ import annotations

import math
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|guid)", re.IGNORECASE)


def round_sig(value: Any, digits: int = 4) -> Any:
    """Round floats to a few significant digits so tool output stays compact."""
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value) or value == 0:
            return value if not math.isnan(value) else None
        return round(value, max(digits - 1 - int(math.floor(math.log10(abs(value)))), 0))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def infer_task(y: pd.Series) -> str:
    """Guess whether a target column calls for classification or regression."""
    y = y.dropna()
    if y.empty:
        raise ValueError("Target column has no non-missing values")
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return "classification"
    n_unique = y.nunique()
    integral = pd.api.types.is_integer_dtype(y) or bool((y % 1 == 0).all())
    if integral and (n_unique <= 2 or (n_unique <= 20 and n_unique / len(y) < 0.05)):
        return "classification"
    return "regression"


def _looks_like_dates(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(50)
    if sample.empty or sample.str.fullmatch(r"[-+]?\d+(\.\d+)?").all():
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.9


def _column_profile(s: pd.Series, n_rows: int) -> tuple[dict[str, Any], list[str]]:
    name = str(s.name)
    non_null = s.dropna()
    n_unique = int(non_null.nunique())
    info: dict[str, Any] = {
        "name": name,
        "dtype": str(s.dtype),
        "missing_pct": round_sig(100 * s.isna().mean(), 3),
        "n_unique": n_unique,
    }
    issues: list[str] = []
    unique_ratio = n_unique / max(len(non_null), 1)

    if n_unique <= 1:
        info["kind"] = "constant"
        issues.append(f"'{name}' is constant; it carries no signal - drop it.")
    elif pd.api.types.is_bool_dtype(s):
        info["kind"] = "boolean"
        info["true_pct"] = round_sig(100 * non_null.astype(bool).mean(), 3)
    elif pd.api.types.is_numeric_dtype(s):
        integral = pd.api.types.is_integer_dtype(s) or bool((non_null % 1 == 0).all())
        if (
            integral
            and unique_ratio > 0.98
            and len(non_null) > 20
            and (ID_NAME.search(name) or s.is_monotonic_increasing)
        ):
            info["kind"] = "id_like"
            issues.append(f"'{name}' looks like an identifier; exclude it from features.")
        else:
            info["kind"] = "numeric"
        desc = non_null.describe()
        info.update(
            {
                "mean": round_sig(desc["mean"]),
                "std": round_sig(desc["std"]),
                "min": round_sig(desc["min"]),
                "p50": round_sig(desc["50%"]),
                "max": round_sig(desc["max"]),
                "skew": round_sig(non_null.skew()) if len(non_null) > 2 else None,
            }
        )
    elif pd.api.types.is_datetime64_any_dtype(s):
        info["kind"] = "datetime"
        info["min"] = str(non_null.min())
        info["max"] = str(non_null.max())
    else:
        as_str = non_null.astype(str)
        mean_len = float(as_str.str.len().mean()) if len(as_str) else 0.0
        if unique_ratio > 0.98 and len(non_null) > 20:
            info["kind"] = "id_like" if mean_len < 40 else "text"
        elif mean_len > 30 and unique_ratio > 0.5:
            info["kind"] = "text"
        elif _looks_like_dates(s):
            info["kind"] = "date_string"
        else:
            info["kind"] = "categorical"
        top = as_str.value_counts().head(5)
        info["top_values"] = {str(k): int(v) for k, v in top.items()}
        if info["kind"] == "id_like":
            issues.append(f"'{name}' has a unique value per row (identifier?); exclude it.")
        elif info["kind"] == "text":
            issues.append(
                f"'{name}' is free text; AutoML will treat it as a high-cardinality category. "
                "Engineer features from it (length, keywords) or drop it."
            )
        elif info["kind"] == "date_string":
            issues.append(
                f"'{name}' holds dates stored as text; parse it and extract features "
                "(year, month, day of week, age/elapsed time) with run_python."
            )
        elif n_unique > 100:
            issues.append(f"'{name}' is a high-cardinality categorical ({n_unique} levels).")

    if info["missing_pct"] and info["missing_pct"] > 40:
        issues.append(f"'{name}' is {info['missing_pct']}% missing.")
    return info, issues


def _target_profile(df: pd.DataFrame, target: str) -> tuple[dict[str, Any], list[str]]:
    y = df[target]
    task = infer_task(y)
    info: dict[str, Any] = {
        "name": target,
        "inferred_task": task,
        "missing_rows": int(y.isna().sum()),
    }
    issues: list[str] = []
    if info["missing_rows"]:
        issues.append(f"{info['missing_rows']} rows have a missing target; training drops them.")
    y = y.dropna()

    if task == "classification":
        counts = y.value_counts()
        info["n_classes"] = int(len(counts))
        info["class_counts"] = {str(k): int(v) for k, v in counts.head(20).items()}
        info["majority_class_pct"] = round_sig(100 * counts.iloc[0] / len(y), 3)
        imbalance = counts.iloc[0] / counts.iloc[-1]
        info["imbalance_ratio"] = round_sig(imbalance, 3)
        if imbalance >= 5:
            issues.append(
                f"Classes are imbalanced ({imbalance:.1f}:1). Prefer roc_auc/ap/macro_f1 over "
                "accuracy, and compare against the majority-class baseline."
            )
        if (counts < 2).any():
            rare = [str(k) for k in counts[counts < 2].index[:10]]
            issues.append(f"Classes with a single example cannot be stratified: {rare}.")
        y_num = y.map({k: i for i, k in enumerate(counts.index)}) if len(counts) == 2 else None
    else:
        info.update(
            {
                "mean": round_sig(y.mean()),
                "std": round_sig(y.std()),
                "min": round_sig(y.min()),
                "max": round_sig(y.max()),
                "skew": round_sig(y.skew()),
            }
        )
        if abs(y.skew()) > 2:
            issues.append(
                "Target is heavily skewed; consider a log transform or a robust metric (mae)."
            )
        y_num = y.astype(float)

    leaks, correlations = [], {}
    for col in df.columns:
        if col == target:
            continue
        x = df.loc[y.index, col]
        if (
            y_num is not None
            and pd.api.types.is_numeric_dtype(x)
            and not pd.api.types.is_bool_dtype(x)
        ):
            mask = x.notna()
            if mask.sum() > 10 and x[mask].nunique() > 1:
                corr = float(np.corrcoef(x[mask].astype(float), y_num[mask])[0, 1])
                if not math.isnan(corr):
                    correlations[col] = corr
                    if abs(corr) > 0.95:
                        leaks.append(col)
        elif not pd.api.types.is_numeric_dtype(x):
            # A categorical column that almost perfectly determines the target is suspicious.
            # Missingness is a level of its own: "only filled in after the outcome" is a classic leak.
            levels = x.astype(object).where(x.notna(), "<missing>").astype(str)
            if not 1 < levels.nunique() < 0.5 * len(levels):
                continue
            grouped = pd.DataFrame({"x": levels, "y": y})
            if task == "classification":
                purity = grouped.groupby("x")["y"].agg(lambda s: s.value_counts().iloc[0]).sum()
                if purity / len(grouped) > 0.99 and info["majority_class_pct"] < 95:
                    leaks.append(col)
            elif y.var() > 0:
                within = grouped.groupby("x")["y"].transform("mean")
                r2 = 1 - ((y - within) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                if r2 > 0.98:
                    leaks.append(col)
    if correlations:
        top = sorted(correlations.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        info["top_numeric_correlations"] = {k: round_sig(v, 3) for k, v in top}
    if leaks:
        info["possible_leakage"] = leaks
        issues.append(
            f"Possible target leakage: {leaks} predict the target almost perfectly. Confirm these "
            "are known at prediction time before using them."
        )
    return info, issues


def profile_dataframe(
    df: pd.DataFrame,
    target: str | None = None,
    columns: list[str] | None = None,
    max_columns: int = 60,
) -> dict[str, Any]:
    """Summarise a DataFrame for an ML engineer: shape, per-column stats and warnings."""
    if target is not None and target not in df.columns:
        raise KeyError(f"Target column {target!r} not found. Columns: {list(df.columns)}")
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        selected = list(columns)
    else:
        selected = list(df.columns[:max_columns])

    profile: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round_sig(df.memory_usage(deep=True).sum() / 1e6, 3),
    }
    if len(selected) < df.shape[1] and not columns:
        profile["note"] = (
            f"Profiled the first {len(selected)} of {df.shape[1]} columns; "
            "pass `columns` to inspect others."
        )

    column_profiles, issues = [], []
    if profile["duplicate_rows"]:
        issues.append(f"{profile['duplicate_rows']} duplicate rows.")
    for col in selected:
        info, col_issues = _column_profile(df[col], len(df))
        column_profiles.append(info)
        issues.extend(col_issues)
    profile["column_profiles"] = column_profiles

    if target is not None:
        target_info, target_issues = _target_profile(df, target)
        profile["target"] = target_info
        issues.extend(target_issues)
    profile["warnings"] = issues
    return profile
