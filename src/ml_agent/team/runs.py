"""Everything one team run persists, and the only code path that reads the locked test set.

Layout of ``<workspace>/runs/run-NNN/``::

    run.json            run metadata: spec, label classes, budget and compute used
    splits.parquet      subset (train/test) and fold ids per labelled row
    features.py         the Data agent's prepare(df) feature function (optional)
    data.json           column roles and the feature-set hash candidates are tied to
    candidates/         <id>.json (scores), <id>.joblib (configured estimator), <id>_oof.parquet
    reviews/            one ReviewVerdict per iteration
    champion/           fitted champion, final.json (test results), test predictions
    figures/            charts from the Visualization agent
    report.md           final report
    test_access.json    written when the test set is read; it is read once per run
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from collections.abc import Callable
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml_agent.team.preprocess import coerce_features, infer_roles, validate_roles
from ml_agent.team.schemas import ColumnRoles, ProblemSpec
from ml_agent.team.splits import cv_splits, make_splits
from ml_agent.workspace import Workspace, now_iso, slugify


class BudgetExhausted(RuntimeError):
    """Raised when a run has used its compute budget."""


class RunStore:
    def __init__(self, ws: Workspace, run_id: str):
        self.ws = ws
        self.run_id = run_id
        self.root = ws.runs_dir / run_id
        self._cache: dict[str, Any] = {}

    # ----------------------------------------------------------- lifecycle

    @classmethod
    def create(cls, ws: Workspace) -> RunStore:
        numbers = [
            int(m.group(1))
            for p in ws.runs_dir.iterdir()
            if (m := re.fullmatch(r"run-(\d+)", p.name))
        ]
        run = cls(ws, f"run-{max(numbers, default=0) + 1:03d}")
        for sub in ("candidates", "reviews", "champion", "figures"):
            (run.root / sub).mkdir(parents=True)
        run.write_json("run.json", {"run_id": run.run_id, "created_at": now_iso()})
        return run

    @classmethod
    def open(cls, ws: Workspace, run_id: str) -> RunStore:
        run = cls(ws, run_id)
        if not (run.root / "run.json").exists():
            raise KeyError(f"Unknown run {run_id!r}")
        return run

    # ---------------------------------------------------------------- json

    def read_json(self, name: str, default: Any = None) -> Any:
        path = self.root / name
        return json.loads(path.read_text()) if path.exists() else default

    def write_json(self, name: str, obj: Any) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, default=str))

    @property
    def meta(self) -> dict[str, Any]:
        return self.read_json("run.json", {})

    def update_meta(self, **fields: Any) -> None:
        self.write_json("run.json", {**self.meta, **fields})

    # ------------------------------------------------------------ the spec

    def check_spec(self, spec: ProblemSpec) -> None:
        """Validate a spec against the actual data (raises ValueError/KeyError with the reason)."""
        spec.check()
        raw = self.ws.load_dataset(spec.dataset)
        for col in (spec.target, spec.group_column, *spec.excluded_columns):
            if col is not None and col not in raw.columns:
                raise KeyError(f"Column {col!r} is not in dataset {spec.dataset!r}")
        if spec.target in spec.excluded_columns or spec.target == spec.group_column:
            raise ValueError("The target cannot be excluded or used as the group column")
        if spec.task == "classification":
            labels = sorted(raw[spec.target].dropna().astype(str).unique())
            if len(labels) < 2:
                raise ValueError("The target has fewer than two classes")
            if spec.positive_class is not None:
                if len(labels) != 2:
                    raise ValueError("positive_class only applies to binary targets")
                if spec.positive_class not in labels:
                    raise ValueError(
                        f"positive_class {spec.positive_class!r} is not one of {labels}"
                    )
            binary_only = {"f1", "ap", "roc_auc"} & {spec.primary_metric, *spec.secondary_metrics}
            if len(labels) > 2 and binary_only:
                raise ValueError(
                    f"{sorted(binary_only)} need a binary target; this one has {len(labels)} "
                    "classes. Use roc_auc_ovr, macro_f1 or log_loss."
                )
        elif not pd.api.types.is_numeric_dtype(raw[spec.target]):
            raise ValueError(f"Regression needs a numeric target; {spec.target!r} is not numeric")

    def save_spec(self, spec: ProblemSpec, *, budget_seconds: float, seed: int = 42) -> None:
        self.check_spec(spec)
        self.update_meta(
            spec=spec.model_dump(), budget_seconds=budget_seconds, compute_seconds=0.0, seed=seed
        )

    @property
    def spec(self) -> ProblemSpec:
        if "spec" not in self.meta:
            raise RuntimeError("This run has no problem spec yet")
        return ProblemSpec.model_validate(self.meta["spec"])

    # -------------------------------------------------------------- budget

    def remaining_seconds(self) -> float:
        meta = self.meta
        return max(meta.get("budget_seconds", 0.0) - meta.get("compute_seconds", 0.0), 0.0)

    def charge(self, seconds: float) -> None:
        self.update_meta(compute_seconds=self.meta.get("compute_seconds", 0.0) + seconds)

    def require_budget(self, seconds: float = 1.0) -> float:
        remaining = self.remaining_seconds()
        if remaining < seconds:
            raise BudgetExhausted(
                f"The run's compute budget is used up ({remaining:.0f}s left). Work with the "
                "candidates you have."
            )
        return remaining

    # -------------------------------------------------------------- splits

    def create_splits(self) -> dict[str, Any]:
        """Lock the test set, fix the CV folds and infer default column roles."""
        spec, seed = self.spec, self.meta.get("seed", 42)
        raw = self.ws.load_dataset(spec.dataset)
        y = raw[spec.target]
        groups = raw[spec.group_column] if spec.group_column else None
        frame = make_splits(
            y, spec.task, test_size=spec.test_size, n_folds=spec.cv_folds, seed=seed, groups=groups
        )
        frame.index.name = "row"
        frame.reset_index().to_parquet(self.root / "splits.parquet", index=False)

        classes = None
        if spec.task == "classification":
            labels = sorted(y.dropna().astype(str).unique())
            if spec.positive_class is not None:
                labels = [lab for lab in labels if lab != spec.positive_class] + [
                    spec.positive_class
                ]
            classes = labels
        self.update_meta(classes=classes, n_dropped_missing_target=int(y.isna().sum()))
        self._cache.clear()

        train = self.raw_train()
        roles = infer_roles(
            train, spec.target, [*spec.excluded_columns, *filter(None, [spec.group_column])]
        )
        self._save_data_state(roles=roles, features_file=None)
        summary: dict[str, Any] = {
            "rows_train": int((frame["subset"] == "train").sum()),
            "rows_test": int((frame["subset"] == "test").sum()),
            "rows_dropped_missing_target": self.meta["n_dropped_missing_target"],
            "selection_cv": f"{spec.cv_folds}-fold x {frame.filter(like='sel_fold').shape[1]} repeats",
            "tuning_cv": f"{frame['tune_fold'].max() + 1}-fold",
        }
        if classes:
            counts = train[spec.target].astype(str).value_counts()
            summary["classes"] = classes
            summary["train_class_counts"] = {c: int(counts.get(c, 0)) for c in classes}
        return summary

    @property
    def splits(self) -> pd.DataFrame:
        if "splits" not in self._cache:
            path = self.root / "splits.parquet"
            if not path.exists():
                raise RuntimeError("Splits have not been created for this run yet")
            self._cache["splits"] = pd.read_parquet(path).set_index("row")
        return self._cache["splits"]

    def _train_splits(self) -> pd.DataFrame:
        return self.splits[self.splits["subset"] == "train"]

    def tune_cv(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return cv_splits(self._train_splits()["tune_fold"].to_numpy())

    def selection_cv(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """All selection folds (every repeat), identical for every candidate."""
        train = self._train_splits()
        columns = sorted(c for c in train.columns if c.startswith("sel_fold_"))
        return [split for col in columns for split in cv_splits(train[col].to_numpy())]

    def first_repeat_cv(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """One repeat of the selection folds: a partition used for out-of-fold predictions."""
        return cv_splits(self._train_splits()["sel_fold_0"].to_numpy())

    def raw_train(self) -> pd.DataFrame:
        raw = self.ws.load_dataset(self.spec.dataset)
        return raw.iloc[self._train_splits().index.to_numpy()].reset_index(drop=True)

    # ------------------------------------------------------------ features

    @property
    def features_path(self):
        return self.root / "features.py"

    def _save_data_state(self, *, roles: ColumnRoles, features_file: str | None) -> None:
        source = self.features_path.read_text() if features_file else ""
        digest = hashlib.sha256((source + roles.model_dump_json()).encode()).hexdigest()[:12]
        self.write_json(
            "data.json",
            {"roles": roles.model_dump(), "features_file": features_file, "features_hash": digest},
        )
        self._cache.pop("xy_train", None)

    @property
    def data_state(self) -> dict[str, Any]:
        state = self.read_json("data.json")
        if state is None:
            raise RuntimeError("Splits have not been created for this run yet")
        return state

    @property
    def roles(self) -> ColumnRoles:
        return ColumnRoles.model_validate(self.data_state["roles"])

    @property
    def features_hash(self) -> str:
        return self.data_state["features_hash"]

    def prepare_fn(self) -> Callable[[pd.DataFrame], pd.DataFrame] | None:
        if not self.data_state.get("features_file"):
            return None
        return load_prepare(self.features_path)

    def train_frame(self) -> pd.DataFrame:
        """Training rows after the registered prepare(df), if any."""
        df = self.raw_train()
        prepare = self.prepare_fn()
        return prepare(df) if prepare else df

    def register_features(self) -> dict[str, Any]:
        """Adopt ``features.py`` after checking prepare(df) is row-wise and ignores the target."""
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Write the feature function to {self.features_path} first (def prepare(df): ...)"
            )
        prepare = load_prepare(self.features_path)
        target = self.spec.target
        raw = self.raw_train()
        out = prepare(raw.copy())
        if not isinstance(out, pd.DataFrame) or len(out) != len(raw):
            raise ValueError("prepare(df) must return a DataFrame with one row per input row")
        if not out.index.equals(raw.index):
            raise ValueError("prepare(df) must keep the row order and index (no filtering/sorting)")
        try:
            blind = prepare(raw.drop(columns=[target]))
        except Exception as exc:
            raise ValueError(
                f"prepare(df) fails without the target column ({type(exc).__name__}: {exc}). It "
                "must not use the target - at prediction time there is none."
            ) from exc
        shared = [c for c in blind.columns if c in out.columns]
        changed = [c for c in shared if not blind[c].equals(out[c])]
        if changed:
            raise ValueError(f"prepare(df) output depends on the target: {changed}")

        previous = self.roles
        inferred = infer_roles(
            out, target, [*self.spec.excluded_columns, *filter(None, [self.spec.group_column])]
        )
        self._save_data_state(roles=inferred, features_file="features.py")
        return {
            "columns_added": [c for c in out.columns if c not in raw.columns],
            "columns_removed": [c for c in raw.columns if c not in out.columns],
            "roles": inferred.model_dump(),
            "note": "Column roles were re-inferred from the new feature set; review them."
            if inferred != previous
            else None,
        }

    def set_roles(self, roles: ColumnRoles) -> None:
        validate_roles(self.train_frame(), roles, self.spec.target)
        self._save_data_state(roles=roles, features_file=self.data_state.get("features_file"))

    # ----------------------------------------------------------- matrices

    @property
    def classes(self) -> list[str] | None:
        return self.meta.get("classes")

    def encode_target(self, y: pd.Series) -> np.ndarray:
        if self.spec.task == "regression":
            return y.astype(float).to_numpy()
        index = {label: i for i, label in enumerate(self.classes)}
        return y.astype(str).map(index).to_numpy(dtype=int)

    def decode_target(self, y: np.ndarray) -> np.ndarray:
        if self.spec.task == "regression":
            return np.asarray(y, dtype=float)
        return np.asarray(self.classes, dtype=object)[np.asarray(y, dtype=int)]

    def xy_train(self) -> tuple[pd.DataFrame, np.ndarray]:
        key = self.features_hash
        cached = self._cache.get("xy_train")
        if cached and cached[0] == key:
            return cached[1], cached[2]
        df = self.train_frame()
        X, y = coerce_features(df, self.roles), self.encode_target(df[self.spec.target])
        self._cache["xy_train"] = (key, X, y)
        return X, y

    def load_test(self, champion_id: str) -> tuple[pd.DataFrame, np.ndarray]:
        """The locked test set. Only the final evaluation calls this, once per run."""
        access = self.read_json("test_access.json")
        if access and access["champion"] != champion_id:
            raise RuntimeError(
                f"The test set was already used to evaluate {access['champion']} at "
                f"{access['at']}. Reusing it for another model would turn it into a "
                "validation set; start a new run to evaluate a different champion."
            )
        self.write_json("test_access.json", {"champion": champion_id, "at": now_iso()})
        raw = self.ws.load_dataset(self.spec.dataset)
        test_rows = self.splits.index[self.splits["subset"] == "test"].to_numpy()
        df = raw.iloc[test_rows].reset_index(drop=True)
        prepare = self.prepare_fn()
        df = prepare(df) if prepare else df
        return coerce_features(df, self.roles), self.encode_target(df[self.spec.target])

    # ---------------------------------------------------------- candidates

    def new_candidate_id(self, family: str) -> str:
        existing = list((self.root / "candidates").glob("c*.json"))
        return f"c{len(existing) + 1:02d}-{slugify(family)}"

    def save_candidate(
        self, record: dict[str, Any], estimator: Any, oof: pd.DataFrame | None = None
    ) -> None:
        cid = record["id"]
        joblib.dump(estimator, self.root / "candidates" / f"{cid}.joblib")
        if oof is not None:
            oof.to_parquet(self.root / "candidates" / f"{cid}_oof.parquet", index=False)
        self.write_json(f"candidates/{cid}.json", record)

    def update_candidate(self, cid: str, **fields: Any) -> dict[str, Any]:
        record = {**self.candidate(cid), **fields}
        self.write_json(f"candidates/{cid}.json", record)
        return record

    def candidate(self, cid: str) -> dict[str, Any]:
        record = self.read_json(f"candidates/{cid}.json")
        if record is None:
            known = ", ".join(r["id"] for r in self.candidates()) or "none"
            raise KeyError(f"Unknown candidate {cid!r}. Known: {known}")
        return record

    def candidate_estimator(self, cid: str) -> Any:
        self.candidate(cid)
        return joblib.load(self.root / "candidates" / f"{cid}.joblib")

    def candidate_oof(self, cid: str) -> pd.DataFrame | None:
        path = self.root / "candidates" / f"{cid}_oof.parquet"
        return pd.read_parquet(path) if path.exists() else None

    def candidates(self, *, current_only: bool = False) -> list[dict[str, Any]]:
        records = [
            json.loads(p.read_text()) for p in sorted((self.root / "candidates").glob("c*.json"))
        ]
        if current_only:
            records = [r for r in records if r["features_hash"] == self.features_hash]
        return records


def load_prepare(path) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Import ``prepare`` from a features.py file (fresh import every time)."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(f"_ml_agent_features_{digest}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepare = getattr(module, "prepare", None)
    if not callable(prepare):
        raise ValueError(f"{path} must define prepare(df) -> DataFrame")
    return prepare
