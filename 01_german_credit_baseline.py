"""10-seed XGBoost baseline on the German Credit (UCI / OpenML credit-g)
public benchmark — the externally reproducible counterpart to the
confidential PMA evaluation reported in the CFAM manuscript.

Run::

    python experiments/01_german_credit_baseline.py

Outputs (in ./results/):
    baseline_german_credit.csv     — per-seed metrics
    baseline_german_credit.json    — summary with bootstrap CIs
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfam_core.fairness import (  # noqa: E402
    bootstrap_ci,
    disparate_impact,
    equal_opportunity_difference,
    fmt_ci,
)

warnings.filterwarnings("ignore")

SEEDS = list(range(10))
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _sex_from_status(s: str) -> int:
    """Map OpenML credit-g personal_status to a binary sex indicator.

    NOTE: order matters — must check 'female' before 'male' because
    'male' is a substring of 'female'.
    """
    s = str(s).lower()
    if "female" in s:
        return 0
    if "male" in s:
        return 1
    return 1


def load_german_credit() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, sex_male) for the German Credit benchmark.

    Uses fetch_openml to retrieve the canonical OpenML credit-g dataset.
    """
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml("credit-g", version=1, as_frame=True)
    df = bunch.frame.copy()

    y = (df["class"] == "bad").astype(int)
    df = df.drop(columns=["class"])

    sex_male = df["personal_status"].map(_sex_from_status).rename("sex_male")
    df = df.drop(columns=["personal_status"])
    df = pd.get_dummies(df, drop_first=True)
    return df, y, sex_male


def train_xgb(X_train, y_train, seed: int):
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=seed,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X_test, y_test, sens_test) -> dict:
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "di": float(disparate_impact(pred, sens_test.to_numpy())),
        "eod": float(equal_opportunity_difference(y_test.to_numpy(), pred, sens_test.to_numpy())),
    }


def main() -> None:
    print(f"=" * 60)
    print("German Credit — 10-seed XGBoost baseline")
    print(f"=" * 60)

    X, y, sex = load_german_credit()
    print(f"Loaded credit-g: X.shape={X.shape}  positive prevalence={y.mean():.3f}")

    rows = []
    for s in SEEDS:
        X_tr, X_te, y_tr, y_te, sens_tr, sens_te = train_test_split(
            X, y, sex, test_size=0.30, random_state=s, stratify=y
        )
        clf = train_xgb(X_tr, y_tr, seed=s)
        m = evaluate(clf, X_te, y_te, sens_te)
        m["seed"] = s
        rows.append(m)
        print(
            f"seed={s:2d}  AUC={m['auc']:.4f}  F1={m['f1']:.4f}  "
            f"DI={m['di']:.4f}  EOD={m['eod']:+.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "baseline_german_credit.csv", index=False)

    print("\n=== 10-seed summary (mean [95% bootstrap CI], 2,000 resamples) ===")
    summary = {}
    for col in ["auc", "f1", "precision", "recall", "di", "eod"]:
        m, lo, hi = bootstrap_ci(df[col].tolist(), n_boot=2000, seed=0)
        summary[col] = {"mean": m, "ci_low": lo, "ci_high": hi}
        print(f"  {col:>9}: {fmt_ci(df[col].tolist(), digits=4)}")

    with open(RESULTS_DIR / "baseline_german_credit.json", "w") as f:
        json.dump(
            {"seeds": SEEDS, "per_seed": rows, "summary": summary},
            f,
            indent=2,
            default=str,
        )

    print(f"\nSaved: {RESULTS_DIR / 'baseline_german_credit.csv'}")
    print(f"Saved: {RESULTS_DIR / 'baseline_german_credit.json'}")


if __name__ == "__main__":
    main()
