"""Reweighing comparator on German Credit.

Compares the unweighted XGBoost baseline against the Kamiran-Calders
Reweighing pre-processor in two implementations (hand-rolled + AIF360).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfam_core.fairness import bootstrap_ci, disparate_impact, equal_opportunity_difference, fmt_ci  # noqa: E402
from cfam_core.reweighing import kamiran_calders_weights  # noqa: E402

warnings.filterwarnings("ignore")

SEEDS = list(range(10))
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_BASELINE_PATH = Path(__file__).resolve().parent / "01_german_credit_baseline.py"
_spec = importlib.util.spec_from_file_location("baseline01", _BASELINE_PATH)
baseline01 = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(baseline01)  # type: ignore[union-attr]


def aif360_reweighing_weights(sens_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Canonical AIF360 Reweighing — falls back to hand-rolled if unavailable."""
    try:
        from aif360.algorithms.preprocessing import Reweighing
        from aif360.datasets import BinaryLabelDataset
    except Exception:
        return kamiran_calders_weights(sens_train, y_train)

    df = pd.DataFrame({"sex_male": sens_train.astype(int), "label": y_train.astype(int)})
    bld = BinaryLabelDataset(
        df=df,
        label_names=["label"],
        protected_attribute_names=["sex_male"],
        favorable_label=0,
        unfavorable_label=1,
    )
    rw = Reweighing(
        unprivileged_groups=[{"sex_male": 0}],
        privileged_groups=[{"sex_male": 1}],
    )
    return np.asarray(rw.fit_transform(bld).instance_weights, dtype=float)


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


def _train(X_train, y_train, w, seed):
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=seed,
        eval_metric="logloss", n_jobs=-1, verbosity=0,
    )
    clf.fit(X_train, y_train, sample_weight=w)
    return clf


def run_variant(variant: str, X, y, sens) -> pd.DataFrame:
    rows = []
    for s in SEEDS:
        X_tr, X_te, y_tr, y_te, sens_tr, sens_te = train_test_split(
            X, y, sens, test_size=0.30, random_state=s, stratify=y
        )
        if variant == "baseline":
            w = None
        elif variant == "rw_handrolled":
            w = kamiran_calders_weights(sens_tr.to_numpy(), y_tr.to_numpy())
        elif variant == "rw_aif360":
            w = aif360_reweighing_weights(sens_tr.to_numpy(), y_tr.to_numpy())
        else:
            raise ValueError(variant)
        clf = _train(X_tr, y_tr, w, s)
        m = evaluate(clf, X_te, y_te, sens_te)
        m["seed"] = s
        m["variant"] = variant
        rows.append(m)
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 60)
    print("German Credit — Reweighing comparator (10 seeds)")
    print("=" * 60)

    X, y, sens = baseline01.load_german_credit()

    base = run_variant("baseline", X, y, sens)
    rwh = run_variant("rw_handrolled", X, y, sens)
    rw3 = run_variant("rw_aif360", X, y, sens)

    pd.concat([base, rwh, rw3], ignore_index=True).to_csv(
        RESULTS_DIR / "reweigh_german_credit.csv", index=False
    )

    summary = {}
    for variant, df in [("baseline", base), ("rw_handrolled", rwh), ("rw_aif360", rw3)]:
        s = {col: dict(zip(("mean", "ci_low", "ci_high"),
                           bootstrap_ci(df[col].tolist(), n_boot=2000, seed=0)))
             for col in ["auc", "f1", "precision", "recall", "di", "eod"]}
        summary[variant] = s
        print(f"\n--- {variant} ---")
        for col in ["auc", "f1", "di", "eod"]:
            print(f"   {col:>4}: {fmt_ci(df[col].tolist(), digits=4)}")

    with open(RESULTS_DIR / "reweigh_german_credit.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
