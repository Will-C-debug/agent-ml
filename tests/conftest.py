from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_agent.config import Settings
from ml_agent.workspace import Workspace


def make_churn(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 60, n)
    tickets = rng.poisson(1.2, n)
    plan = rng.choice(["basic", "pro"], n)
    logit = -0.5 - 0.1 * tenure + 1.5 * tickets + 1.0 * (plan == "basic")
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "signup_date": (
                pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 600, n), unit="D")
            ).strftime("%Y-%m-%d"),
            "plan": plan,
            "tenure": tenure,
            "tickets": tickets,
            "constant": 1,
            "cancel_reason": np.where(churned, rng.choice(["price", "moved"], n), None),
            "churned": np.where(churned, "yes", "no"),
        }
    )
    df.loc[rng.random(n) < 0.1, "tenure"] = np.nan
    return df


def make_housing(n: int = 500, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    size = rng.uniform(40, 200, n)
    rooms = rng.integers(1, 6, n)
    city = rng.choice(["a", "b", "c"], n)
    price = 2000 * size + 15000 * rooms + np.where(city == "a", 60000, 0) + rng.normal(0, 20000, n)
    return pd.DataFrame({"size_m2": size, "rooms": rooms, "city": city, "price": price})


@pytest.fixture
def ws(tmp_path) -> Workspace:
    return Workspace(tmp_path / "workspace")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(workspace=tmp_path / "workspace", python_timeout=30, max_time_budget=30)


@pytest.fixture
def churn_csv(tmp_path):
    path = tmp_path / "churn.csv"
    make_churn().to_csv(path, index=False)
    return path


@pytest.fixture
def churn_ws(ws, churn_csv) -> Workspace:
    df = make_churn()
    ws.save_dataset(df.drop(columns=["signup_date", "cancel_reason", "constant"]), "churn")
    return ws
