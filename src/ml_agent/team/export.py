"""Package a run's champion as a self-contained folder that runs without ml_agent installed."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from ml_agent.automl import _pinned_requirements
from ml_agent.team.runs import RunStore
from ml_agent.workspace import slugify

_PREDICT_SCRIPT = '''\
"""Batch inference for {run_id}: {family} predicting '{target}' ({task}).

Usage:
    pip install -r requirements.txt
    python predict.py input.csv [output.csv]

The input needs the raw columns that features.py (if present) turns into these features:
    {features}
"""

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "metadata.json").read_text())
MODEL = joblib.load(HERE / "model.joblib")
READERS = {{".csv": pd.read_csv, ".parquet": pd.read_parquet, ".json": pd.read_json}}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    path = HERE / "features.py"
    if not path.exists():
        return df
    spec = importlib.util.spec_from_file_location("bundle_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare(df)


def predict(df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare(df.copy())
    missing = [c for c in META["features"] if c not in prepared.columns]
    if missing:
        raise SystemExit(f"Input is missing required columns: {{missing}}")
    X = prepared[META["features"]].copy()
    for col in META["numeric"]:
        X[col] = X[col].astype(float)
    for col in META["categorical"]:
        values = X[col]
        as_text = values.astype(object).where(values.isna(), values.astype(str))
        X[col] = as_text.where(as_text.notna(), np.nan)
    out = df.copy()
    predictions = MODEL.predict(X)
    if META["task"] == "classification":
        out["prediction"] = np.asarray(META["classes"], dtype=object)[predictions]
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)
            for i, label in enumerate(META["classes"]):
                out[f"proba_{{label}}"] = proba[:, i]
    else:
        out["prediction"] = predictions
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    source = Path(sys.argv[1])
    result = predict(READERS.get(source.suffix.lower(), pd.read_csv)(source))
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_name(source.stem + "_predictions.csv")
    result.to_csv(target, index=False)
    print(f"Wrote {{len(result)}} predictions to {{target}}")
'''


def _model_card(run: RunStore, final: dict[str, Any]) -> str:
    spec = run.spec
    record = run.candidate(final["champion"])
    metric = final["metric"]
    roles = run.roles
    lines = [
        f"# Model card: {run.run_id} champion `{final['champion']}`",
        "",
        f"- **Problem:** predict `{spec.target}` ({spec.task}) from dataset `{spec.dataset}`",
        f"- **Success criteria:** {spec.success_criteria}",
        f"- **Model:** {record['family']} ({record['source']})",
        f"- **Features ({len(roles.features)}):** {', '.join(roles.features)}",
        f"- **Excluded:** {', '.join(f'{d.column} ({d.reason})' for d in roles.dropped) or 'none'}",
    ]
    if spec.task == "classification":
        lines.append(f"- **Classes:** {', '.join(run.classes)}")
    if final.get("threshold") is not None:
        lines.append(
            f"- **Decision threshold:** {final['threshold']} for `{run.classes[1]}` "
            f"(tuned for {record.get('threshold_objective')})"
        )
    lines += [
        "",
        "## Performance",
        "",
        f"| | {metric} |",
        "|---|---|",
        f"| Selection CV (mean ± std) | {record['cv'][metric]['mean']} ± {record['cv'][metric]['std']} |",
        f"| Locked test set ({final['rows']['test']} rows) | {final['test'][metric]} |",
        f"| Naive baseline on test | {final['test']['baseline']} |",
        "",
        "## Hyperparameters",
        "",
        "```json",
        json.dumps(record["params"], indent=2),
        "```",
        "",
        "## Most important features (permutation importance on the test set)",
        "",
    ]
    lines += [
        f"- {row['feature']}: {row['importance']} ± {row['std']}"
        for row in final["permutation_importance"][:10]
    ]
    return "\n".join(lines) + "\n"


def export_champion(run: RunStore, name: str | None = None) -> dict[str, Any]:
    """Write exports/<run>-<name>/ with the fitted champion, features.py and predict.py."""
    final = run.read_json("champion/final.json")
    if final is None:
        raise RuntimeError("Run the final evaluation before exporting")
    spec, roles = run.spec, run.roles
    bundle = run.ws.resolve_output(
        slugify(f"{run.run_id}-{name or final['champion']}"), run.ws.exports_dir
    )
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    shutil.copy2(run.root / "champion" / "model.joblib", bundle / "model.joblib")
    if run.data_state.get("features_file"):
        shutil.copy2(run.features_path, bundle / "features.py")
    metadata = {
        "run_id": run.run_id,
        "champion": final["champion"],
        "family": final["family"],
        "task": spec.task,
        "target": spec.target,
        "classes": run.classes,
        "features": roles.features,
        "numeric": roles.numeric,
        "categorical": [*roles.categorical, *roles.high_cardinality],
        "threshold": final.get("threshold"),
        "metric": final["metric"],
        "test": final["test"],
    }
    (bundle / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (bundle / "predict.py").write_text(
        _PREDICT_SCRIPT.format(
            run_id=run.run_id,
            family=final["family"],
            target=spec.target,
            task=spec.task,
            features=", ".join(roles.features),
        )
    )
    (bundle / "requirements.txt").write_text(_pinned_requirements())
    (bundle / "MODEL_CARD.md").write_text(_model_card(run, final))
    return {
        "bundle_path": str(bundle),
        "files": sorted(p.name for p in bundle.iterdir()),
        "usage": f"cd {bundle} && {Path(sys.executable).name} predict.py <input.csv> [output.csv]",
    }
