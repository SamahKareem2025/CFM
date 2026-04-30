"""CFAM — Causal Fairness Auditing & Monitoring core library.

Public API:
    RegulatorExplanationPacket, build_r_exp
    HashChainedLedger
    TimeToAuditMetrics
    disparate_impact, equal_opportunity_difference, counterfactual_fairness_rate
    bootstrap_ci, fmt_ci
    kamiran_calders_weights
"""
from cfam_core.r_exp import (
    RegulatorExplanationPacket,
    PredictionData,
    FairnessMetrics,
    ExplanationArtifacts,
    AuditTimestamps,
    LedgerMetadata,
    build_r_exp,
)
from cfam_core.ledger import HashChainedLedger
from cfam_core.tta import TimeToAuditMetrics
from cfam_core.fairness import (
    disparate_impact,
    equal_opportunity_difference,
    counterfactual_fairness_rate,
    bootstrap_ci,
    fmt_ci,
)
from cfam_core.reweighing import kamiran_calders_weights

__version__ = "1.0.0"

__all__ = [
    "RegulatorExplanationPacket",
    "PredictionData",
    "FairnessMetrics",
    "ExplanationArtifacts",
    "AuditTimestamps",
    "LedgerMetadata",
    "build_r_exp",
    "HashChainedLedger",
    "TimeToAuditMetrics",
    "disparate_impact",
    "equal_opportunity_difference",
    "counterfactual_fairness_rate",
    "bootstrap_ci",
    "fmt_ci",
    "kamiran_calders_weights",
    "__version__",
]
