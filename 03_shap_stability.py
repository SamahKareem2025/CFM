"""SHAP rank-stability analysis on German Credit.

Trains XGBoost across SEEDS, computes mean |SHAP| per feature for each seed,
ranks the top-15 features, and reports pairwise Spearman rank correlations
between seeds. High mean off-diagonal rho => explanations are stable.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")

SEEDS = list(range(10))
TOPK = 15
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_BASELINE_PATH = Path(__file__).resolve().parent / "01_german_credit_baseline.py"
_spec = importlib.util.spec_from_file_location("baseline01", _BASELINE_PATH)
baseline01 = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(baseline01)  # type: ignore[union-attr]


def main() -> None:
    import shap
    from xgboost import XGBClassifier

    print("=" * 60)
    print(f"SHAP rank stability — top-{TOPK} features across {len(SEEDS)} seeds")
    print("=" * 60)

    X, y, sens = baseline01.load_german_credit()
    feat_imp = pd.DataFrame(index=X.columns)

    for s in SEEDS:
        X_tr, X_te, y_tr, y_te, _, _ = train_test_split(
            X, y, sens, test_size=0.30, random_state=s, stratify=y
        )
        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=s,
            eval_metric="logloss", n_jobs=-1, verbosity=0,
        )
        clf.fit(X_tr, y_tr)
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_te)
        if isinstance(sv, list):
            sv = sv[1]
        feat_imp[f"seed_{s}"] = np.abs(sv).mean(axis=0)

    feat_imp.to_csv(RESULTS_DIR / "shap_importance_per_seed.csv")

    ranks = feat_imp.apply(lambda c: c.rank(ascending=False))
    top_union = set()
    for s in SEEDS:
        top_union |= set(feat_imp.nlargest(TOPK, f"seed_{s}").index.tolist())
    sub_ranks = ranks.loc[list(top_union)]

    rho_matrix = np.zeros((len(SEEDS), len(SEEDS)))
    for i, si in enumerate(SEEDS):
        for j, sj in enumerate(SEEDS):
            rho_matrix[i, j] = spearmanr(sub_ranks[f"seed_{si}"], sub_ranks[f"seed_{sj}"]).statistic

    off = rho_matrix[~np.eye(len(SEEDS), dtype=bool)]
    print(f"  mean off-diagonal rho = {off.mean():.3f}")
    print(f"  min  off-diagonal rho = {off.min():.3f}")
    print(f"  interpretation: {'HIGH (>0.85)' if off.mean() > 0.85 else 'LOW or moderate'}")

    out = {
        "seeds": SEEDS,
        "top_k": TOPK,
        "rho_mean_off_diag": float(off.mean()),
        "rho_min_off_diag": float(off.min()),
        "rho_matrix": rho_matrix.tolist(),
    }
    with open(RESULTS_DIR / "shap_stability_german_credit.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
