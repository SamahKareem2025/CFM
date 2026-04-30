"""Tests for HashChainedLedger integrity & audit trail lookup."""
from __future__ import annotations

from datetime import datetime

import pytest

from cfam_core.ledger import HashChainedLedger
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
        model_id="test_model",
        model_version="0.1",
        decision_id=decision_id,
        prediction_data=PredictionData(True, 0.5, 0.7, True),
        fairness_assessment=FairnessMetrics("PASS", 1.0, 0.0, 0.0, False, ["sex"]),
        explanations=ExplanationArtifacts({}, {}, [], []),
        timestamps=AuditTimestamps(
            decision_logged=datetime(2026, 1, 1, 10, 0),
            audit_requested=datetime(2026, 1, 1, 12, 0),
            audit_completed=datetime(2026, 1, 1, 18, 0),
            audit_closed=datetime(2026, 1, 2, 0, 0),
        ),
    )


def test_genesis_block():
    ledger = HashChainedLedger("test")
    assert len(ledger.chain) == 1
    assert ledger.chain[0]["block_type"] == "genesis"
    assert ledger.verify_integrity()


def test_mine_and_verify():
    ledger = HashChainedLedger("test")
    ledger.add_r_exp_packet(_make_packet("d1"))
    ledger.add_r_exp_packet(_make_packet("d2"))
    assert ledger.mine_block() is not None
    assert ledger.verify_integrity()
    assert ledger.summary()["transactions"] == 2


def test_audit_trail_includes_pending():
    ledger = HashChainedLedger("test")
    ledger.add_r_exp_packet(_make_packet("d1"))
    trail = ledger.get_audit_trail("d1")
    assert len(trail) == 1
    assert trail[0]["status"] == "pending"
    ledger.mine_block()
    trail = ledger.get_audit_trail("d1")
    assert trail[0]["status"] == "mined"


def test_tamper_detection():
    ledger = HashChainedLedger("test")
    ledger.add_r_exp_packet(_make_packet("d1"))
    ledger.mine_block()
    ledger.chain[1]["transactions"][0]["decision_id"] = "evil"
    assert ledger.verify_integrity() is False
