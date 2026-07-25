"""
Transaction Sync Protocol — federation via transaction & intention sync.

Extends the existing DeltaTransport with:

    1. TransactionSyncProtocol: Full transaction (not just delta) sync
       between federated HCIR workspaces, with conflict detection.

    2. IntentionSet: Lightweight declaration of *intended* state changes
       that peers can review before committing, enabling pre-commit
       coordination in distributed cognitive architectures.

Usage::

    sync = TransactionSyncProtocol(device_id="node_a")

    # Export a committed transaction for federation
    envelope = sync.export_transaction(committed_tx)

    # Import and apply a remote transaction
    success = sync.import_transaction(envelope, local_workspace)

    # Declare intentions before commit
    intentions = IntentionSet(device_id="node_a")
    intentions.declare("plan_goal_g1", "Will add goal G1")
    peer_intents = intentions.to_dict()  # Send to peers
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.identity import HCIRObjectID
from hbllm.hcir.transactions import (
    HCIRTransaction,
    TransactionAnnotation,
    TransactionOperation,
)
from hbllm.hcir.types import Provenance
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Transaction Envelope (for federation)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TransactionEnvelope:
    """A signed envelope carrying a committed transaction between peers.

    Unlike DeltaPacket (which carries raw deltas), this carries
    the full transaction including provenance, annotations, and status.
    """

    envelope_id: str = field(default_factory=lambda: f"env_{HCIRObjectID().uuid}")
    source_device_id: str = "local"
    target_device_id: str = "broadcast"
    tenant_id: str = "default"
    transaction_id: str = ""
    transaction_author: str = ""
    operations: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    signature: str = ""

    def compute_signature(self, secret_key: str = "hcir_federation_key") -> str:
        payload = f"{self.envelope_id}:{self.source_device_id}:{self.transaction_id}:{len(self.operations)}:{secret_key}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# IntentionSet — pre-commit coordination
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Intention:
    """A single declared intention."""

    intention_id: str
    description: str
    target_node_ids: list[str] = field(default_factory=list)
    declared_at: float = field(default_factory=time.time)


class IntentionSet:
    """Lightweight declaration of intended state changes.

    Peers broadcast their intentions before committing, enabling
    pre-commit conflict detection across federated workspaces.

    Usage::

        intentions = IntentionSet(device_id="node_a")
        intentions.declare("plan_goal", "Will create goal G1", ["g1"])
        # Send to peers for review
        conflict = other_intentions.check_conflict(intentions)
    """

    def __init__(self, device_id: str = "local") -> None:
        self._device_id = device_id
        self._intentions: list[Intention] = []

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def intentions(self) -> list[Intention]:
        return list(self._intentions)

    def declare(
        self,
        intention_id: str,
        description: str,
        target_node_ids: list[str] | None = None,
    ) -> Intention:
        """Declare an intention to modify specific nodes."""
        intent = Intention(
            intention_id=intention_id,
            description=description,
            target_node_ids=target_node_ids or [],
        )
        self._intentions.append(intent)
        return intent

    def check_conflict(self, other: IntentionSet) -> list[str]:
        """Check for conflicting intentions with another set.

        Returns list of node IDs that both sets intend to modify.
        """
        my_targets = set()
        for i in self._intentions:
            my_targets.update(i.target_node_ids)

        their_targets = set()
        for i in other.intentions:
            their_targets.update(i.target_node_ids)

        return sorted(my_targets & their_targets)

    def clear(self) -> None:
        self._intentions.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self._device_id,
            "intentions": [
                {
                    "intention_id": i.intention_id,
                    "description": i.description,
                    "target_node_ids": i.target_node_ids,
                    "declared_at": i.declared_at,
                }
                for i in self._intentions
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# TransactionSyncProtocol
# ═══════════════════════════════════════════════════════════════════════════


class TransactionSyncProtocol:
    """Full transaction sync between federated HCIR workspaces.

    Unlike the DeltaTransportProtocol (which syncs raw graph deltas),
    this syncs *committed transactions* with full provenance, annotations,
    and governance metadata.

    Usage::

        sync = TransactionSyncProtocol(device_id="node_a")
        envelope = sync.export_transaction(committed_tx)
        success = sync.import_transaction(envelope, local_workspace)
    """

    def __init__(
        self,
        device_id: str = "local",
        secret_key: str = "hcir_federation_key",
    ) -> None:
        self._device_id = device_id
        self._secret_key = secret_key
        self._synced_tx_ids: set[str] = set()
        self._conflict_count: int = 0

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def synced_count(self) -> int:
        return len(self._synced_tx_ids)

    @property
    def conflict_count(self) -> int:
        return self._conflict_count

    def export_transaction(
        self,
        transaction: HCIRTransaction,
        target_device_id: str = "broadcast",
        tenant_id: str = "default",
    ) -> TransactionEnvelope:
        """Export a committed transaction as a signed envelope.

        Only committed transactions should be exported.
        """
        envelope = TransactionEnvelope(
            source_device_id=self._device_id,
            target_device_id=target_device_id,
            tenant_id=tenant_id,
            transaction_id=transaction.id,
            transaction_author=transaction.author,
            operations=[op.model_dump() for op in transaction.operations],
            annotations=[a.model_dump() for a in transaction.annotations],
            provenance=transaction.provenance.model_dump(),
        )
        envelope.signature = envelope.compute_signature(self._secret_key)
        return envelope

    def import_transaction(
        self,
        envelope: TransactionEnvelope,
        target_workspace: HCIRWorkspaceState,
    ) -> bool:
        """Import and apply a remote transaction envelope.

        Verifies signature, deduplicates, and commits.
        """
        # Verify signature
        expected_sig = envelope.compute_signature(self._secret_key)
        if envelope.signature != expected_sig:
            logger.warning("Envelope %s signature mismatch", envelope.envelope_id)
            return False

        # Deduplicate
        if envelope.transaction_id in self._synced_tx_ids:
            logger.debug("Transaction %s already synced", envelope.transaction_id)
            return True

        # Reconstruct operations
        ops = []
        for op_data in envelope.operations:
            ops.append(TransactionOperation.model_validate(op_data))

        # Commit via TransactionManager
        from hbllm.hcir.kernel.transaction_manager import TransactionManager

        tx_mgr = TransactionManager(target_workspace)
        tx = HCIRTransaction(
            author=f"sync:{envelope.source_device_id}:{envelope.transaction_author}",
            operations=ops,
            provenance=Provenance(
                created_by=f"federation:{envelope.source_device_id}",
                reason=f"Synced from {envelope.source_device_id}",
            ),
            annotations=[
                TransactionAnnotation(
                    author="TransactionSyncProtocol",
                    assertion=f"Imported from {envelope.source_device_id}",
                    severity="info",
                )
            ],
        )
        result = tx_mgr.commit(tx)

        if result.is_committed:
            self._synced_tx_ids.add(envelope.transaction_id)
            logger.info(
                "Synced transaction %s from %s",
                envelope.transaction_id,
                envelope.source_device_id,
            )
            return True
        else:
            self._conflict_count += 1
            logger.warning(
                "Failed to sync transaction %s from %s",
                envelope.transaction_id,
                envelope.source_device_id,
            )
            return False
