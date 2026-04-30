"""Schema-preserving synthetic PMA dataset generator.

Marginal distributions and column types match the preprocessed PMA dataset
described in the manuscript (12,952 records, 2023-2025 supervisory
window). No real account information is reproduced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GENDERS = ["male", "female"]
AGE_GROUPS = ["young", "adult", "senior"]
ACCOUNT_TYPES = ["current", "savings", "loan", "credit_card"]


def generate(n: int = 12_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(GENDERS, size=n, p=[0.62, 0.38])
    age_numeric = np.clip(rng.normal(loc=44, scale=14, size=n), 18, 90).astype(int)
    age_group = pd.cut(
        age_numeric, bins=[17, 30, 55, 100], labels=AGE_GROUPS
    ).astype(str)

    account_type = rng.choice(ACCOUNT_TYPES, size=n, p=[0.45, 0.25, 0.20, 0.10])
    n_facilities = np.clip(rng.poisson(lam=2.1, size=n), 0, 12)
    exposure_band = pd.cut(
        rng.lognormal(mean=8.5, sigma=0.9, size=n),
        bins=[0, 1e3, 1e4, 5e4, 2e5, 1e7],
        labels=["xs", "s", "m", "l", "xl"],
    ).astype(str)

    utilization = np.clip(rng.beta(2, 5, size=n), 0, 1)
    repayment_status = rng.choice(["on_time", "delayed", "default"], size=n, p=[0.78, 0.17, 0.05])
    delinquency_history = rng.poisson(lam=0.4, size=n)
    year = rng.choice([2023, 2024, 2025], size=n, p=[0.32, 0.34, 0.34])

    base_logit = (
        -1.5
        + 0.6 * (repayment_status != "on_time")
        + 0.4 * (delinquency_history > 0)
        + 0.5 * (utilization > 0.7)
        - 0.3 * (account_type == "savings")
        - 0.05 * np.where(gender == "female", 1.0, 0.0)  # weak protected effect
        - 0.15 * (age_group == "young")
        + rng.normal(0, 0.2, size=n)
    )
    p_risk = 1 / (1 + np.exp(-base_logit))
    risk_flag = (rng.uniform(size=n) < p_risk).astype(int)

    return pd.DataFrame(
        {
            "gender": gender,
            "age_numeric": age_numeric,
            "agegroup": age_group,
            "account_type": account_type,
            "number_of_facilites": n_facilities,
            "exposure_band": exposure_band,
            "utilization_ratio": utilization,
            "repayment_status": repayment_status,
            "delinquency_history": delinquency_history,
            "year": year,
            "risk_flag": risk_flag,
        }
    )


if __name__ == "__main__":
    df = generate(n=12_000, seed=0)
    df.to_csv("pma_synthetic.csv", index=False)
    print(f"Wrote {len(df)} synthetic rows.")
    print(f"Risk-flag prevalence: {df['risk_flag'].mean():.3f}")
    print(f"Female prevalence:    {(df['gender']=='female').mean():.3f}")
