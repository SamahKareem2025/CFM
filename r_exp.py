"""Regulator Explanation Packet (R-EXP) schema.

A R-EXP packet is the standardised, regulator-ready record produced by the CFAM
pipeline for every supervisory decision. It bundles four components into a
single JSON-serialisable, hashable artefact:

    1. PredictionData       — the model output and decision-level metadata
    2. FairnessMetrics      — DI / EOD / CFR + protected-attribute trace
    3. ExplanationArtifacts — local SHAP attributions + counterfactuals
    4. AuditTimestamps      — decision_logged / requested / completed / closed

The optional LedgerMetadata block records the on-ledger anchor (block hash,
transaction id, ledger location) once the packet has been mined.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class PredictionData:
    binary_decision: bool
    decision_threshold: float
    credit_score: float
    risk_flag: bool


@dataclass
class FairnessMetrics:
    fairness_status: str  # "PASS" or "FAIL"
    disparate_impact: float
    equal_opportunity_difference: float
    counterfactual_fairness_rate: float
    counterfactual_violation: bool
    protected_attributes_used: List[str]


@dataclass
class ExplanationArtifacts:
    shap_values: Dict[str, float]
    feature_importance: Dict[str, float]
    counterfactual_explanations: List[Dict[str, Any]]
    local_feature_attributions: List[Dict[str, Any]]


@dataclass
class AuditTimestamps:
    decision_logged: datetime
    audit_requested: Optional[datetime] = None
    audit_completed: Optional[datetime] = None
    audit_closed: Optional[datetime] = None


@dataclass
class LedgerMetadata:
    block_hash: str
    previous_block_hash: str
    transaction_id: str
    block_timestamp: datetime
    ledger_location: str


@dataclass
class RegulatorExplanationPacket:
    packet_id: str
    model_id: str
    model_version: str
    decision_id: str

    prediction_data: PredictionData
    fairness_assessment: FairnessMetrics
    explanations: ExplanationArtifacts
    timestamps: AuditTimestamps

    packet_version: str = "1.0"
    system_config: Dict[str, Any] = field(default_factory=dict)
    ledger_metadata: Optional[LedgerMetadata] = None

    protected_attributes: List[str] = field(default_factory=list)
    fairness_rules_applied: List[str] = field(default_factory=list)
    audit_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        def _conv(o: Any) -> Any:
            if isinstance(o, datetime):
                return o.isoformat()
            return str(o)

        return json.dumps(self.to_dict(), default=_conv, indent=2, sort_keys=True)

    def compute_hash(self) -> str:
        """SHA-256 over a canonical JSON serialisation (sorted keys)."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def build_r_exp(
    *,
    decision_id: str,
    model_id: str,
    model_version: str,
    prediction: PredictionData,
    fairness: FairnessMetrics,
    explanations: ExplanationArtifacts,
    decision_time: datetime,
    audit_request_offset_h: float = 2.0,
    audit_completion_offset_h: float = 18.0,
    audit_closure_offset_h: float = 6.0,
    protected_attributes: Optional[List[str]] = None,
) -> RegulatorExplanationPacket:
    """Assemble a R-EXP packet with realistic audit-workflow timestamps.

    Offsets are *deterministic* by default — randomise externally if a
    stochastic workflow simulation is required (see experiments/04_ledger_run).
    """
    audit_requested = decision_time + timedelta(hours=audit_request_offset_h)
    audit_completed = audit_requested + timedelta(hours=audit_completion_offset_h)
    audit_closed = audit_completed + timedelta(hours=audit_closure_offset_h)

    return RegulatorExplanationPacket(
        packet_id=f"R_EXP_{uuid.uuid4().hex[:16]}",
        model_id=model_id,
        model_version=model_version,
        decision_id=decision_id,
        prediction_data=prediction,
        fairness_assessment=fairness,
        explanations=explanations,
        timestamps=AuditTimestamps(
            decision_logged=decision_time,
            audit_requested=audit_requested,
            audit_completed=audit_completed,
            audit_closed=audit_closed,
        ),
        protected_attributes=list(protected_attributes or []),
        fairness_rules_applied=["DI_threshold", "EOD_threshold", "CF_test"],
        audit_flags=["tamper_evident_log", "tta_tracking"],
    )
