"""Tests for the R-EXP packet schema and hash determinism."""
from __future__ import annotations

import json
from datetime import datetime

from cfam_core.r_exp import (
    AuditTimestamps,
    ExplanationArtifacts,
    FairnessMetrics,
    PredictionData,
    RegulatorExplanationPacket,
)


def _make_packet(decision_id: str = "d1") -> RegulatorExplanationPacket:
    return RegulatorExplanationPacket(
        packet_id="pkt_1",
        model_id="m",
        model_version="0.1",
        decision_id=decision_id,
        prediction_data=PredictionData(True, 0.5, 0.7, True),
        fairness_assessment=FairnessMetrics("PASS", 1.0, 0.0, 0.0, False, ["sex"]),
        explanations=ExplanationArtifacts(
            shap_values={"f1": 0.1, "f2": -0.2},
            feature_importance={"f1": 0.6},
            counterfactual_explanations=[{"feature": "f1", "delta": 0.1}],
            local_feature_attributions=[],
        ),
        timestamps=AuditTimestamps(
            decision_logged=datetime(2026, 1, 1, 10, 0),
            audit_requested=datetime(2026, 1, 1, 12, 0),
            audit_completed=datetime(2026, 1, 1, 18, 0),
            audit_closed=datetime(2026, 1, 2, 0, 0),
        ),
        protected_attributes=["sex"],
    )


def test_to_json_is_valid_json():
    p = _make_packet()
    parsed = json.loads(p.to_json())
    assert parsed["decision_id"] == "d1"
    assert parsed["fairness_assessment"]["fairness_status"] == "PASS"


def test_hash_is_deterministic():
    p1 = _make_packet("d1")
    p2 = _make_packet("d1")
    assert p1.compute_hash() == p2.compute_hash()


def test_hash_changes_when_decision_id_changes():
    assert _make_packet("d1").compute_hash() != _make_packet("d2").compute_hash()
