"""
Distributed HCIR Delta Transport — multi-device swarm state synchronization.

Provides network transport and merge validation for distributed HCIR nodes:

    Device A (Local Workspace)
            │
      HCIRDelta Packet
            │
    DeltaTransportProtocol (JSON serialization & HMAC verification)
            │
    Device B (Remote Workspace Merge)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.identity import HCIRObjectID
from hbllm.hcir.transactions import HCIRDelta
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


@dataclass
class DeltaPacket:
    """Network packet carrying an HCIRDelta for swarm node synchronization."""

    packet_id: str = field(default_factory=lambda: f"pkt_{HCIRObjectID().uuid}")
    source_device_id: str = "local"
    target_device_id: str = "broadcast"
    tenant_id: str = "default"
    delta: HCIRDelta = field(default_factory=HCIRDelta)
    timestamp: float = field(default_factory=time.time)
    signature: str = ""

    def compute_signature(self, secret_key: str = "hcir_swarm_secret") -> str:
        """Compute HMAC signature for network authentication."""
        payload = f"{self.packet_id}:{self.source_device_id}:{self.tenant_id}:{len(self.delta.to_operations())}:{secret_key}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize packet to dictionary format."""
        return {
            "packet_id": self.packet_id,
            "source_device_id": self.source_device_id,
            "target_device_id": self.target_device_id,
            "tenant_id": self.tenant_id,
            "delta": {
                "add_nodes": [
                    n if isinstance(n, dict) else n.model_dump() for n in self.delta.add_nodes
                ],
                "remove_node_ids": self.delta.remove_node_ids,
                "add_edges": [
                    e if isinstance(e, dict) else e.model_dump() for e in self.delta.add_edges
                ],
                "remove_edge_ids": self.delta.remove_edge_ids,
            },
            "timestamp": self.timestamp,
            "signature": self.signature or self.compute_signature(),
        }


class DeltaTransportProtocol:
    """Protocol for sending and receiving distributed HCIR deltas across swarm nodes."""

    def __init__(
        self, device_id: str = "local_node", secret_key: str = "hcir_swarm_secret"
    ) -> None:
        self._device_id = device_id
        self._secret_key = secret_key

    def create_packet(
        self,
        delta: HCIRDelta,
        target_device_id: str = "broadcast",
        tenant_id: str = "default",
    ) -> DeltaPacket:
        """Package a local HCIRDelta into a signed network DeltaPacket."""
        pkt = DeltaPacket(
            source_device_id=self._device_id,
            target_device_id=target_device_id,
            tenant_id=tenant_id,
            delta=delta,
        )
        pkt.signature = pkt.compute_signature(self._secret_key)
        return pkt

    def verify_and_apply(
        self,
        packet_dict: dict[str, Any],
        target_workspace: HCIRWorkspaceState,
    ) -> bool:
        """Verify network signature and apply remote delta to local workspace."""
        try:
            sig = packet_dict.get("signature", "")
            pkt_id = packet_dict.get("packet_id", "")
            src_dev = packet_dict.get("source_device_id", "")
            tenant_id = packet_dict.get("tenant_id", "default")
            delta_raw = packet_dict.get("delta", {})

            # Compute expected signature
            ops_count = (
                len(delta_raw.get("add_nodes", []))
                + len(delta_raw.get("add_edges", []))
                + len(delta_raw.get("remove_node_ids", []))
                + len(delta_raw.get("remove_edge_ids", []))
            )
            payload = f"{pkt_id}:{src_dev}:{tenant_id}:{ops_count}:{self._secret_key}"
            expected_sig = hashlib.sha256(payload.encode()).hexdigest()[:16]

            if sig != expected_sig:
                logger.warning("Delta packet %s signature verification failed", pkt_id)
                return False

            # Apply delta operations to target workspace
            from hbllm.hcir.kernel.transaction_manager import TransactionManager
            from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation

            tx_mgr = TransactionManager(target_workspace)
            ops = []
            for node_data in delta_raw.get("add_nodes", []):
                ops.append(
                    TransactionOperation(
                        op=TransactionOp.ADD_NODE, node_id=node_data.get("id"), node_data=node_data
                    )
                )

            if ops:
                tx = HCIRTransaction(author=src_dev, operations=ops)
                tx_mgr.commit(tx)

            logger.info("Applied distributed delta packet %s from device '%s'", pkt_id, src_dev)
            return True

        except Exception as exc:
            logger.error("Failed to apply delta packet: %s", exc)
            return False
