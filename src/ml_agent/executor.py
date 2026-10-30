"""Run agent-written Python in a subprocess rooted at the workspace.

This is *not* a security sandbox: the code runs as the current user with full access to the
machine. That is why the agent asks for human approval before every ``run_python`` call unless
auto-approve is switched on.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

from ml_agent.workspace import Workspace

MAX_OUTPUT_CHARS = 8000

# Executed with `python -c PRELUDE <script>`: exposes workspace helpers, then runs the script in
# the same namespace so tracebacks point at the script's own line numbers.
PRELUDE = """\
import os, sys
import numpy as np
import pandas as pd
from ml_agent.workspace import Workspace

ws = Workspace(os.environ["ML_AGENT_WORKSPACE"])
load_dataset = ws.load_dataset

def save_dataset(df, name, note=None):
    info = ws.save_dataset(df, name, source="run_python", note=note)
    print(f"[saved dataset '{info['name']}': {info['rows']} rows x {len(info['columns'])} columns]")
    return info

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
__file__ = sys.argv[1]
with open(__file__) as _f:
    _code = compile(_f.read(), __file__, "exec")
del _f
exec(_code)
"""


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} characters truncated] ...\n{text[-half:]}"


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _dataset_mtimes(ws: Workspace) -> dict[str, float]:
    return {p.stem: p.stat().st_mtime for p in ws.datasets_dir.glob("*.parquet")}


def run_code(ws: Workspace, code: str, timeout: int = 300) -> dict[str, Any]:
    """Execute ``code`` with the workspace as working directory and report what happened."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    script = ws.scratch_dir / f"snippet_{stamp}.py"
    script.write_text(code)
    env = {
        **os.environ,
        "ML_AGENT_WORKSPACE": str(ws.root),
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
    }
    before = _dataset_mtimes(ws)
    started = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PRELUDE, str(script)],
            cwd=ws.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code, stdout, stderr = None, _as_text(exc.stdout), _as_text(exc.stderr)
    after = _dataset_mtimes(ws)

    result: dict[str, Any] = {
        "exit_code": exit_code,
        "seconds": round(time.time() - started, 2),
        "stdout": _truncate(stdout) or "(no output - use print() to show results)",
    }
    if stderr.strip():
        result["stderr"] = _truncate(stderr)
    if timed_out:
        result["error"] = f"Timed out after {timeout}s and was terminated."
    changed = sorted(name for name, mtime in after.items() if before.get(name) != mtime)
    if changed:
        result["datasets_written"] = changed
    result["script"] = str(script)
    return result
