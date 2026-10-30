"""On-disk workspace shared by the agent's tools.

Layout::

    <root>/
      datasets/       <name>.parquet + <name>.json (provenance)
      experiments/    exp-001/ model.joblib, metadata.json, test.parquet, MODEL_CARD.md, flaml.log
      predictions/    batch prediction outputs
      exports/        self-contained model bundles
      reports/        markdown reports written by the agent
      scratch/        code snippets executed by run_python
      .agent/         conversation checkpoints (SQLite)

Datasets and experiments live on disk rather than in agent state so they survive restarts and are
visible to code the agent runs with ``run_python``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_READERS = {
    ".csv": pd.read_csv,
    ".tsv": partial(pd.read_csv, sep="\t"),
    ".txt": partial(pd.read_csv, sep=None, engine="python"),
    ".parquet": pd.read_parquet,
    ".pq": pd.read_parquet,
    ".feather": pd.read_feather,
    ".json": pd.read_json,
    ".jsonl": partial(pd.read_json, lines=True),
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
}

SUPPORTED_EXTENSIONS = tuple(sorted(_READERS))


def slugify(name: str) -> str:
    """Turn a user-supplied name into a safe file stem."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_-").lower()
    if not slug:
        raise ValueError(f"Invalid name: {name!r}")
    return slug


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_table(path: Path) -> pd.DataFrame:
    """Read a tabular file based on its extension."""
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"Unsupported file type {path.suffix!r}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    df = reader(path)
    df.columns = [str(c) for c in df.columns]
    return df


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet, coercing mixed-type object columns to strings if needed."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    try:
        df.to_parquet(path, index=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype("string")
        df.to_parquet(path, index=False)


class Workspace:
    """Directory-backed store for datasets, experiments and generated files."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        for directory in (
            self.datasets_dir,
            self.experiments_dir,
            self.predictions_dir,
            self.exports_dir,
            self.reports_dir,
            self.scratch_dir,
            self.state_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    @property
    def predictions_dir(self) -> Path:
        return self.root / "predictions"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def scratch_dir(self) -> Path:
        return self.root / "scratch"

    @property
    def state_dir(self) -> Path:
        return self.root / ".agent"

    @property
    def checkpoint_db(self) -> Path:
        return self.state_dir / "checkpoints.sqlite"

    # ----------------------------------------------------------------- paths

    def resolve_input(self, path: str) -> Path:
        """Resolve a user-supplied input path: absolute, relative to the CWD, or to the workspace."""
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            resolved = candidate
        elif candidate.exists():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return resolved

    def resolve_output(self, path: str, default_dir: Path) -> Path:
        """Resolve an output path, which must stay inside the workspace."""
        candidate = Path(path).expanduser()
        resolved = (candidate if candidate.is_absolute() else default_dir / candidate).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(
                f"Output paths must be inside the workspace ({self.root}); got {resolved}"
            )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    # -------------------------------------------------------------- datasets

    def dataset_path(self, name: str) -> Path:
        return self.datasets_dir / f"{slugify(name)}.parquet"

    def has_dataset(self, name: str) -> bool:
        return self.dataset_path(name).exists()

    def dataset_names(self) -> list[str]:
        return sorted(p.stem for p in self.datasets_dir.glob("*.parquet"))

    def load_dataset(self, name: str) -> pd.DataFrame:
        path = self.dataset_path(name)
        if not path.exists():
            available = ", ".join(self.dataset_names()) or "none"
            raise KeyError(f"Unknown dataset {name!r}. Available datasets: {available}")
        return pd.read_parquet(path)

    def save_dataset(
        self, df: pd.DataFrame, name: str, *, source: str | None = None, note: str | None = None
    ) -> dict[str, Any]:
        """Store a DataFrame as a named dataset and return its summary."""
        slug = slugify(name)
        write_parquet(df, self.dataset_path(slug))
        meta = {"name": slug, "source": source, "note": note, "created_at": now_iso()}
        (self.datasets_dir / f"{slug}.json").write_text(json.dumps(meta, indent=2))
        return self.dataset_info(slug)

    def import_file(self, path: str, name: str | None = None) -> dict[str, Any]:
        """Read a local file and register it as a dataset."""
        source = self.resolve_input(path)
        df = read_table(source)
        return self.save_dataset(df, name or source.stem, source=str(source))

    def dataset_info(self, name: str) -> dict[str, Any]:
        path = self.dataset_path(name)
        if not path.exists():
            raise KeyError(f"Unknown dataset {name!r}")
        parquet = pq.ParquetFile(path)
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return {
            "name": path.stem,
            "rows": parquet.metadata.num_rows,
            "columns": parquet.schema_arrow.names,
            "source": meta.get("source"),
            "note": meta.get("note"),
            "created_at": meta.get("created_at"),
        }

    # ----------------------------------------------------------- experiments

    def experiment_dir(self, experiment_id: str) -> Path:
        path = self.experiments_dir / slugify(experiment_id)
        if not (path / "metadata.json").exists():
            known = ", ".join(self.experiment_ids()) or "none"
            raise KeyError(f"Unknown experiment {experiment_id!r}. Known experiments: {known}")
        return path

    def experiment_ids(self) -> list[str]:
        return sorted(p.parent.name for p in self.experiments_dir.glob("*/metadata.json"))

    def new_experiment_dir(self) -> tuple[str, Path]:
        numbers = [
            int(m.group(1))
            for p in self.experiments_dir.iterdir()
            if (m := re.fullmatch(r"exp-(\d+)", p.name))
        ]
        experiment_id = f"exp-{max(numbers, default=0) + 1:03d}"
        path = self.experiments_dir / experiment_id
        path.mkdir()
        return experiment_id, path

    def load_experiment(self, experiment_id: str) -> dict[str, Any]:
        return json.loads((self.experiment_dir(experiment_id) / "metadata.json").read_text())

    def save_experiment(self, experiment_id: str, metadata: dict[str, Any]) -> None:
        path = self.experiments_dir / experiment_id / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2, default=str))
