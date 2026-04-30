"""Hash-chained, append-only ledger simulation for CFAM audit logging.

This module implements a permissioned-ledger *simulation* — not a production
blockchain. All security claims are limited to evidentiary integrity and
tamper evidence:

    * append-only semantics,
    * SHA-256-linked blocks (genesis included),
    * canonical block-hash recomputation for integrity verification,
    * audit-trail lookup that searches both mined blocks AND pending tx.

Out of scope: distributed consensus, Byzantine fault tolerance, adversarial
threat modelling, key-compromise resistance.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from cfam_core.r_exp import LedgerMetadata, RegulatorExplanationPacket


def _canonical_block_string(
    index: int,
    timestamp: datetime,
    transactions: List[Dict[str, Any]],
    previous_hash: str,
) -> str:
    """Deterministic, cross-platform string representation of a block."""
    return (
        f"{index}|{timestamp.isoformat()}|"
        f"{json.dumps(transactions, default=str, sort_keys=True)}|{previous_hash}"
    )


def _compute_block_hash(
    index: int,
    timestamp: datetime,
    transactions: List[Dict[str, Any]],
    previous_hash: str,
) -> str:
    return hashlib.sha256(
        _canonical_block_string(index, timestamp, transactions, previous_hash).encode("utf-8")
    ).hexdigest()


class HashChainedLedger:
    """Append-only, hash-chained ledger with selective on-ledger anchoring.

    Transactions are buffered until :meth:`mine_block` is called, at which
    point they are sealed into a new block and the buffer is cleared. The
    genesis block is created at construction time.
    """

    def __init__(self, ledger_name: str = "CFAM_Ledger") -> None:
        self.ledger_name = ledger_name
        self.chain: List[Dict[str, Any]] = []
        self.current_transactions: List[Dict[str, Any]] = []
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        ts = datetime.now()
        genesis = {
            "index": 0,
            "timestamp": ts,
            "transactions": [],
            "previous_hash": "0" * 64,
            "block_type": "genesis",
            "ledger_name": self.ledger_name,
        }
        genesis["block_hash"] = _compute_block_hash(0, ts, [], "0" * 64)
        self.chain.append(genesis)

    def add_transaction(self, transaction: Dict[str, Any]) -> str:
        tx_id = f"tx_{uuid.uuid4().hex[:16]}"
        transaction = dict(transaction)
        transaction["transaction_id"] = tx_id
        transaction.setdefault("timestamp", datetime.now())
        self.current_transactions.append(transaction)
        return tx_id

    def add_r_exp_packet(self, packet: RegulatorExplanationPacket) -> Dict[str, Any]:
        """Anchor an R-EXP packet on the ledger.

        The full R-EXP is *not* placed on-chain (off-chain storage is the
        caller's responsibility); only its SHA-256 hash plus a small summary.
        """
        last_block = self.chain[-1]
        packet.ledger_metadata = LedgerMetadata(
            block_hash="",
            previous_block_hash=last_block["block_hash"],
            transaction_id=f"tx_{uuid.uuid4().hex[:16]}",
            block_timestamp=datetime.now(),
            ledger_location=f"{self.ledger_name}_Channel_1",
        )

        r_exp_hash = packet.compute_hash()
        tx = {
            "type": "R_EXP_DECISION",
            "decision_id": packet.decision_id,
            "r_exp_hash": r_exp_hash,
            "model_id": packet.model_id,
            "fairness_status": packet.fairness_assessment.fairness_status,
            "timestamps": {
                "decision_logged": packet.timestamps.decision_logged,
                "audit_requested": packet.timestamps.audit_requested,
                "audit_completed": packet.timestamps.audit_completed,
                "audit_closed": packet.timestamps.audit_closed,
            },
        }
        tx_id = self.add_transaction(tx)
        return {"transaction_id": tx_id, "r_exp_hash": r_exp_hash, "block_pending": True}

    def mine_block(self) -> Optional[Dict[str, Any]]:
        if not self.current_transactions:
            return None
        ts = datetime.now()
        index = len(self.chain)
        prev = self.chain[-1]["block_hash"]
        txs = list(self.current_transactions)
        block = {
            "index": index,
            "timestamp": ts,
            "transactions": txs,
            "previous_hash": prev,
            "block_type": "regular",
            "block_hash": _compute_block_hash(index, ts, txs, prev),
        }
        self.chain.append(block)
        self.current_transactions.clear()
        return block

    def verify_integrity(self) -> bool:
        for i in range(1, len(self.chain)):
            cur, prev = self.chain[i], self.chain[i - 1]
            if cur["previous_hash"] != prev["block_hash"]:
                return False
            recomputed = _compute_block_hash(
                cur["index"], cur["timestamp"], cur["transactions"], cur["previous_hash"]
            )
            if recomputed != cur["block_hash"]:
                return False
        return True

    def get_audit_trail(self, decision_id: str) -> List[Dict[str, Any]]:
        """Return all transactions for ``decision_id`` from mined and pending."""
        trail: List[Dict[str, Any]] = []
        for block in self.chain:
            for tx in block.get("transactions", []):
                if tx.get("decision_id") == decision_id:
                    trail.append(
                        {
                            "block_index": block["index"],
                            "block_timestamp": block["timestamp"],
                            "transaction": tx,
                            "status": "mined",
                        }
                    )
        for tx in self.current_transactions:
            if tx.get("decision_id") == decision_id:
                trail.append(
                    {
                        "block_index": None,
                        "block_timestamp": None,
                        "transaction": tx,
                        "status": "pending",
                    }
                )
        return trail

    def summary(self) -> Dict[str, Any]:
        return {
            "ledger_name": self.ledger_name,
            "blocks": len(self.chain),
            "transactions": sum(len(b.get("transactions", [])) for b in self.chain),
            "pending_transactions": len(self.current_transactions),
            "integrity_ok": self.verify_integrity(),
        }
