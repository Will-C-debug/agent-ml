"""Locked test split plus fixed cross-validation folds shared by every candidate.

``make_splits`` returns one row per labelled data row (indexed by its position in the raw dataset)
with these columns:

- ``subset``: ``"train"`` or ``"test"``. Test rows are only read by the final evaluation.
- ``tune_fold``: fold id used for hyperparameter search (``-1`` for test rows).
- ``sel_fold_<r>``: fold id in repeat ``r`` of the selection CV that ranks candidates (``-1`` for
  test rows). Selection folds use different seeds from tuning folds, so a model is not ranked on
  exactly the splits it was tuned on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

N_TUNE_FOLDS = 3
N_REPEATS = 2


def _fold_ids(
    y: pd.Series, task: str, n_folds: int, seed: int, groups: pd.Series | None
) -> np.ndarray:
    """Assign each row to one of ``n_folds`` folds."""
    if groups is not None:
        if task == "classification":
            splitter = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
        else:
            splitter = GroupKFold(n_folds, shuffle=True, random_state=seed)
        splits = splitter.split(np.zeros(len(y)), y, groups)
    elif task == "classification":
        splits = StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y
        )
    else:
        splits = KFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)))
    folds = np.full(len(y), -1)
    for k, (_, val) in enumerate(splits):
        folds[val] = k
    return folds


def make_splits(
    y: pd.Series,
    task: str,
    *,
    test_size: float,
    n_folds: int,
    seed: int = 42,
    groups: pd.Series | None = None,
    n_repeats: int = N_REPEATS,
    n_tune_folds: int = N_TUNE_FOLDS,
) -> pd.DataFrame:
    """Split labelled rows into a locked test set and fixed CV folds on the rest."""
    y = y.dropna()
    if groups is not None:
        groups = groups.loc[y.index]
        if groups.isna().any():
            raise ValueError("The group column has missing values")
    if task == "classification":
        counts = y.value_counts()
        if (counts < 2).any():
            raise ValueError(f"Classes with fewer than 2 rows: {list(counts[counts < 2].index)}")
        if counts.min() < n_folds:
            raise ValueError(
                f"The smallest class has {counts.min()} rows, fewer than cv_folds={n_folds}"
            )

    positions = np.arange(len(y))
    if groups is not None:
        if task == "classification":
            n_splits = max(2, round(1 / test_size))
            splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
            train_pos, test_pos = next(splitter.split(positions, y, groups))
        else:
            splitter = GroupShuffleSplit(1, test_size=test_size, random_state=seed)
            train_pos, test_pos = next(splitter.split(positions, y, groups))
    else:
        stratify = y if task == "classification" else None
        train_pos, test_pos = train_test_split(
            positions, test_size=test_size, random_state=seed, stratify=stratify
        )
    train_pos = np.sort(train_pos)

    frame = pd.DataFrame(index=y.index)
    frame["subset"] = "test"
    frame.iloc[train_pos, frame.columns.get_loc("subset")] = "train"
    y_train = y.iloc[train_pos]
    g_train = groups.iloc[train_pos] if groups is not None else None

    frame["tune_fold"] = -1
    frame.iloc[train_pos, frame.columns.get_loc("tune_fold")] = _fold_ids(
        y_train, task, n_tune_folds, seed + 1, g_train
    )
    for r in range(n_repeats):
        column = f"sel_fold_{r}"
        frame[column] = -1
        frame.iloc[train_pos, frame.columns.get_loc(column)] = _fold_ids(
            y_train, task, n_folds, seed + 100 + r, g_train
        )
    return frame


def cv_splits(fold_ids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Turn a fold-id array (over training rows) into scikit-learn (train, validation) indices."""
    fold_ids = np.asarray(fold_ids)
    return [
        (np.flatnonzero(fold_ids != k), np.flatnonzero(fold_ids == k))
        for k in sorted(set(fold_ids.tolist()))
    ]
