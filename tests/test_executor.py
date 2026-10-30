from ml_agent.executor import run_code


def test_run_code_reads_and_writes_datasets(churn_ws):
    code = (
        "df = load_dataset('churn')\n"
        "df['tenure_years'] = df['tenure'] / 12\n"
        "save_dataset(df, 'churn_v2')\n"
        "print(df.shape)\n"
    )
    result = run_code(churn_ws, code, timeout=60)
    assert result["exit_code"] == 0, result
    assert "(600, 6)" in result["stdout"]
    assert result["datasets_written"] == ["churn_v2"]
    assert "tenure_years" in churn_ws.load_dataset("churn_v2").columns


def test_run_code_reports_errors_with_script_line_numbers(ws):
    result = run_code(ws, "x = 1\nraise ValueError('boom')\n", timeout=60)
    assert result["exit_code"] == 1
    assert "ValueError: boom" in result["stderr"]
    assert "line 2" in result["stderr"]


def test_run_code_timeout(ws):
    result = run_code(ws, "import time\nprint('started', flush=True)\ntime.sleep(10)\n", timeout=3)
    assert result["exit_code"] is None
    assert "Timed out" in result["error"]
