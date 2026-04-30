"""End-to-end CFAM ledger run on German Credit.

For every test-set decision (seed=0):
    1. Compute SHAP top-5 attributions.
    2. Compute global counterfactual fairness rate by flipping `sex_male`.
    3. Build an R-EXP packet with realistic synthetic audit-workflow timestamps.
    4. Anchor the packet on a hash-chained ledger (mined every 50 records).
    5. Record TTA metrics; report ledger summary + audit completion rate.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import uuid
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfam_core.fairness import counterfactual_fairness_rate, disparate_impact, equal_opportunity_difference  # noqa: E402
from cfam_core.ledger import HashChainedLedger  # noqa: E402
from cfam_core.r_exp import (  # noqa: E402
    AuditTimestamps,
    ExplanationArtifacts,
    FairnessMetrics,
    PredictionData,
    RegulatorExplanationPacket,
)
from cfam_core.tta import TimeToAuditMetrics  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 0
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_BASELINE_PATH = Path(__file__).resolve().parent / "01_german_credit_baseline.py"
_spec = importlib.util.spec_from_file_location("baseline01", _BASELINE_PATH)
baseline01 = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(baseline01)  # type: ignore[union-attr]


def _rand_offsets(rng: random.Random) -> tuple[float, float, float]:
    """Sensitivity-tested log-normal-shaped offsets for D2A/R2C simulation.

    These distributions are documented in the manuscript (Section V-D) and
    are *not* claimed to reflect a real PMA workflow — they reflect a
    plausibility envelope that can be replaced by measured data once
    deployed.
    """
    return (
        max(0.5, rng.lognormvariate(0.5, 0.4)),
        max(2.0, rng.lognormvariate(2.5, 0.3)),
        max(1.0, rng.lognormvariate(1.5, 0.4)),
    )


def main() -> None:
    import shap
    from xgboost import XGBClassifier

    print("=" * 60)
    print("CFAM end-to-end ledger run on German Credit (seed=0)")
    print("=" * 60)

    X, y, sens = baseline01.load_german_credit()
    X_tr, X_te, y_tr, y_te, sens_tr, sens_te = train_test_split(
        X, y, sens, test_size=0.30, random_state=SEED, stratify=y
    )

    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED,
        eval_metric="logloss", n_jobs=-1, verbosity=0,
    )
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    X_te_cf = X_te.copy()
    X_te_cf["sex_male"] = 1 - sens_te.to_numpy().astype(int)
    proba_cf = clf.predict_proba(X_te_cf)[:, 1]
    cfr_global = counterfactual_fairness_rate(proba, proba_cf, threshold=0.5)
    di = disparate_impact(pred, sens_te.to_numpy())
    eod = equal_opportunity_difference(y_te.to_numpy(), pred, sens_te.to_numpy())
    print(f"  Global CFR = {cfr_global:.4f}  DI = {di:.4f}  EOD = {eod:+.4f}")

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_te)
    if isinstance(sv, list):
        sv = sv[1]

    ledger = HashChainedLedger("German_Credit_CFAM_Ledger")
    tta = TimeToAuditMetrics()
    rng = random.Random(SEED)

    fairness_template = FairnessMetrics(
        fairness_status="PASS" if (0.8 <= di <= 1.25 and abs(eod) <= 0.05) else "FAIL",
        disparate_impact=float(di),
        equal_opportunity_difference=float(eod),
        counterfactual_fairness_rate=float(cfr_global),
        counterfactual_violation=bool(cfr_global > 0.20),
        protected_attributes_used=["sex_male"],
    )

    base_time = datetime.now()
    for i, idx in enumerate(X_te.index):
        topk = np.argsort(np.abs(sv[i]))[::-1][:5]
        shap_top = {str(X_te.columns[k]): float(sv[i, k]) for k in topk}

        d2a_off, r2c_off, close_off = _rand_offsets(rng)
        decision_time = base_time + timedelta(seconds=i * 5)
        ts = AuditTimestamps(
            decision_logged=decision_time,
            audit_requested=decision_time + timedelta(hours=d2a_off),
            audit_completed=decision_time + timedelta(hours=d2a_off + r2c_off),
            audit_closed=decision_time + timedelta(hours=d2a_off + r2c_off + close_off),
        )

        packet = RegulatorExplanationPacket(
            packet_id=f"R_EXP_{uuid.uuid4().hex[:16]}",
            model_id="CFAM_XGBoost_v1.0",
            model_version="1.0.0",
            decision_id=f"GC_{i:04d}_{uuid.uuid4().hex[:6]}",
            prediction_data=PredictionData(
                binary_decision=bool(pred[i]),
                decision_threshold=0.5,
                credit_score=float(proba[i]),
                risk_flag=bool(pred[i]),
            ),
            fairness_assessment=fairness_template,
            explanations=ExplanationArtifacts(
                shap_values=shap_top,
                feature_importance={},
                counterfactual_explanations=[],
                local_feature_attributions=[],
            ),
            timestamps=ts,
            protected_attributes=["sex_male"],
        )
        ledger.add_r_exp_packet(packet)
        tta.record(packet)
        if (i + 1) % 50 == 0:
            ledger.mine_block()

    ledger.mine_block()

    summary = {
        "seed": SEED,
        "ledger": ledger.summary(),
        "tta": tta.summary(),
        "global_cfr": float(cfr_global),
        "disparate_impact": float(di),
        "eod": float(eod),
    }
    with open(RESULTS_DIR / "ledger_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Ledger: blocks={summary['ledger']['blocks']}  "
          f"tx={summary['ledger']['transactions']}  "
          f"integrity_OK={summary['ledger']['integrity_ok']}")
    print(f"  D2A mean={summary['tta']['d2a']['mean']:.2f}h  "
          f"R2C mean={summary['tta']['r2c']['mean']:.2f}h  "
          f"audit_completion_rate={summary['tta']['audit_completion_rate']:.1%}")


if __name__ == "__main__":
    main()
