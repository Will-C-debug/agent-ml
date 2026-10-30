"""Generate a synthetic customer-churn dataset to try the agent on.

The data contains the kinds of problems a real dataset has, so you can watch the agent handle
them: an identifier column, dates stored as text, missing values, class imbalance, and a column
that leaks the target (`cancellation_reason` is only filled in after a customer has churned).

    python examples/make_churn_data.py            # writes examples/data/customers.csv
    python examples/make_churn_data.py --rows 20000 --out-dir somewhere/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_customers(n: int, rng: np.random.Generator, start_id: int = 1) -> pd.DataFrame:
    plan = rng.choice(["basic", "standard", "premium"], n, p=[0.5, 0.35, 0.15])
    base_price = pd.Series(plan).map({"basic": 19.0, "standard": 39.0, "premium": 79.0}).to_numpy()
    tenure = np.clip(rng.exponential(20, n).round().astype(int) + 1, 1, 72)
    tickets = rng.poisson(np.where(plan == "basic", 1.6, 0.9))
    payment = rng.choice(
        ["card", "bank_transfer", "paypal", "invoice"], n, p=[0.45, 0.25, 0.2, 0.1]
    )
    autopay = rng.random(n) < np.where(payment == "invoice", 0.1, 0.65)
    discount = rng.choice([0, 5, 10, 20], n, p=[0.6, 0.2, 0.15, 0.05])
    sessions = rng.gamma(2.0, 12.0, n).round(1)
    nps = np.clip(np.round(rng.normal(7 - 0.6 * tickets, 2)), 0, 10)
    signup = pd.Timestamp("2025-06-30") - pd.to_timedelta(
        tenure * 30 + rng.integers(0, 30, n), unit="D"
    )

    logit = (
        -1.2
        - 0.05 * tenure
        + 0.45 * tickets
        - 0.6 * autopay
        - 0.25 * (nps - 7)
        + 0.8 * (payment == "invoice")
        + 0.5 * (plan == "basic")
        - 0.03 * discount
        - 0.015 * (sessions - 24)
    )
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))
    reasons = np.array(["price", "competitor", "support", "moved", "no_longer_needed"])
    cancellation_reason = np.where(churned, rng.choice(reasons, n), None)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(start_id, start_id + n)],
            "signup_date": signup.strftime("%Y-%m-%d"),
            "plan": plan,
            "region": rng.choice(["north", "south", "east", "west"], n),
            "monthly_charges": (base_price * (1 - discount / 100) + rng.normal(0, 3, n)).round(2),
            "tenure_months": tenure,
            "support_tickets_90d": tickets,
            "avg_session_minutes": sessions,
            "payment_method": payment,
            "autopay": autopay,
            "discount_pct": discount,
            "nps_score": nps,
            "cancellation_reason": cancellation_reason,
            "churned": np.where(churned, "yes", "no"),
        }
    )
    df.loc[rng.random(n) < 0.08, "avg_session_minutes"] = np.nan
    df.loc[rng.random(n) < 0.15, "nps_score"] = np.nan
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--out-dir", default=Path(__file__).parent / "data", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    customers = make_customers(args.rows, rng)
    customers.to_csv(args.out_dir / "customers.csv", index=False)

    # Fresh customers to score once a model exists: no label, no post-churn information.
    new = make_customers(200, rng, start_id=args.rows + 1).drop(
        columns=["churned", "cancellation_reason"]
    )
    new.to_csv(args.out_dir / "new_customers.csv", index=False)

    rate = (customers["churned"] == "yes").mean()
    print(
        f"Wrote {len(customers)} rows to {args.out_dir / 'customers.csv'} (churn rate {rate:.1%})"
    )
    print(
        f"Wrote {len(new)} rows to {args.out_dir / 'new_customers.csv'} (unlabelled, for predictions)"
    )


if __name__ == "__main__":
    main()
