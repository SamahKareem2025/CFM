"""Tests for TimeToAuditMetrics."""
from __future__ import annotations

from datetime import datetime

import pytest

from cfam_core.r_exp import (
    AuditTimestamps,
    ExplanationArtifacts,
    FairnessMetrics,
    PredictionData,
    RegulatorExplanationPacket,
)
from cfam_core.tta import TimeToAuditMetrics


def _packet(decision_id: str, d2a_h: float, r2c_h: float):
    base = datetime(2026, 1, 1, 0, 0)
    return RegulatorExplanationPacket(
        packet_id="p",
        model_id="m",
        model_version="0",
        decision_id=decision_id,
        prediction_data=PredictionData(True, 0.5, 0.5, False),
        fairness_assessment=FairnessMetrics("PASS", 1.0, 0.0, 0.0, False, []),
        explanations=ExplanationArtifacts({}, {}, [], []),
        timestamps=AuditTimestamps(
            decision_logged=base,
            audit_requested=base,
            audit_completed=base.replace(microsecond=0).fromtimestamp(base.timestamp() + d2a_h * 3600),
            audit_closed=base.replace(microsecond=0).fromtimestamp(base.timestamp() + r2c_h * 3600),
        ),
    )


def test_d2a_r2c_simple():
    tta = TimeToAuditMetrics()
    tta.record(_packet("d1", d2a_h=10.0, r2c_h=20.0))
    s = tta.summary()
    assert pytest.approx(s["d2a"]["mean"], abs=1e-3) == 10.0
    assert pytest.approx(s["r2c"]["mean"], abs=1e-3) == 20.0
    assert s["audit_completion_rate"] == 1.0


def test_below_24h_threshold():
    tta = TimeToAuditMetrics()
    tta.record(_packet("a", 5, 30))
    tta.record(_packet("b", 50, 60))
    s = tta.summary()
    assert s["d2a"]["below_24h"] == 0.5
    assert s["r2c"]["below_48h"] == 0.5
