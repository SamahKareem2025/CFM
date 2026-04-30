"""Time-to-Audit (TTA) metrics for CFAM.

Computes:
    * Decision-to-Audit (D2A) — hours between decision_logged and audit_completed.
    * Request-to-Closure (R2C) — hours between audit_requested and audit_closed.
    * Mean Time-to-Audit (MTTA) — mean of (D2A + R2C) / 2 across decisions.

Metrics are computed **directly from R-EXP packet timestamps** rather than via
ledger lookup. This is intentional: it makes TTA reporting independent of the
mining schedule (a packet that is still in `current_transactions` reports the
same TTA as one already mined into a block).
"""
from __future__ import annotations

from datetime import datetime
from statistics import mean, median
from typing import Dict, List, Optional

import pandas as pd

from cfam_core.r_exp import RegulatorExplanationPacket


def _to_datetime(x) -> Optional[datetime]:
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    return datetime.fromisoformat(str(x))


def _hours_between(start, end) -> Optional[float]:
    s, e = _to_datetime(start), _to_datetime(end)
    if s is None or e is None:
        return None
    return (e - s).total_seconds() / 3600.0


class TimeToAuditMetrics:
    """Track and summarise TTA metrics across a stream of R-EXP packets."""

    def __init__(self) -> None:
        self._records: List[Dict[str, object]] = []

    def record(self, packet: RegulatorExplanationPacket) -> Dict[str, Optional[float]]:
        ts = packet.timestamps
        d2a = _hours_between(ts.decision_logged, ts.audit_completed)
        r2c = _hours_between(ts.audit_requested, ts.audit_closed)
        row = {
            "decision_id": packet.decision_id,
            "d2a_hours": d2a,
            "r2c_hours": r2c,
            "computed_at": datetime.now(),
        }
        self._records.append(row)
        return {"d2a_hours": d2a, "r2c_hours": r2c}

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def summary(self) -> Dict[str, object]:
        df = self.to_dataframe()
        if df.empty:
            return {"error": "no records"}

        d2a = [v for v in df["d2a_hours"] if v is not None]
        r2c = [v for v in df["r2c_hours"] if v is not None]

        out: Dict[str, object] = {
            "total_decisions": int(len(df)),
            "audit_completion_rate": float(df["d2a_hours"].notna().mean()),
            "audit_closure_rate": float(df["r2c_hours"].notna().mean()),
        }
        if d2a:
            out["d2a"] = {
                "mean": float(mean(d2a)),
                "median": float(median(d2a)),
                "p95": float(pd.Series(d2a).quantile(0.95)),
                "below_24h": float(sum(1 for v in d2a if v <= 24) / len(d2a)),
            }
        if r2c:
            out["r2c"] = {
                "mean": float(mean(r2c)),
                "median": float(median(r2c)),
                "p95": float(pd.Series(r2c).quantile(0.95)),
                "below_48h": float(sum(1 for v in r2c if v <= 48) / len(r2c)),
            }
        if d2a and r2c:
            out["mtta"] = {"mean": float((mean(d2a) + mean(r2c)) / 2.0)}
        return out
