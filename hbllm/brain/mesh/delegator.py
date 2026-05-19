"""Contract Negotiation and Delegation.

Manages the Accept, Decline, or RetryLater lifecycle for
capsules exchanged between sovereign nodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hbllm.brain.mesh.capsule import TaskCapsule

logger = logging.getLogger(__name__)


class DelegationResponse(StrEnum):
    ACCEPT = "accept"
    DECLINE = "decline"
    RETRY_LATER = "retry_later"


@dataclass
class ContractOffer:
    capsule: TaskCapsule
    offered_by: str


class ContractDelegator:
    """Handles the negotiation of task execution contracts."""

    def __init__(self, local_node_id: str) -> None:
        self.local_node_id = local_node_id
        # In a real system, these would be network RPC clients
        self.network_peers: dict[str, Any] = {}

    def propose_contract(self, target_node_id: str, capsule: TaskCapsule) -> DelegationResponse:
        """Send a TaskCapsule to another node for execution."""
        if not capsule.is_valid:
            logger.error("Cannot propose invalid capsule %s", capsule.capsule_id)
            return DelegationResponse.DECLINE

        capsule.add_hop(self.local_node_id)

        logger.info("Proposing capsule %s to node %s", capsule.capsule_id, target_node_id)

        peer = self.network_peers.get(target_node_id)
        if not peer:
            # Simulate a network failure or disconnected peer
            logger.warning("Peer %s unreachable.", target_node_id)
            return DelegationResponse.RETRY_LATER

        # Synchronous mock of an async RPC call to peer.evaluate_contract
        return peer.evaluate_contract(ContractOffer(capsule=capsule, offered_by=self.local_node_id))

    def evaluate_contract(
        self, offer: ContractOffer, current_memory_pressure: float
    ) -> DelegationResponse:
        """Evaluate an incoming contract based on local sovereign constraints."""
        if not offer.capsule.is_valid:
            return DelegationResponse.DECLINE

        # Example Sovereign Limits
        if current_memory_pressure > 0.8:
            logger.warning(
                "Declining contract %s due to high memory pressure.", offer.capsule.capsule_id
            )
            return DelegationResponse.DECLINE

        if offer.capsule.priority.value > 2 and current_memory_pressure > 0.5:
            # Defer background/archival tasks if under moderate pressure
            return DelegationResponse.RETRY_LATER

        logger.info("Accepted contract %s from %s", offer.capsule.capsule_id, offer.offered_by)
        return DelegationResponse.ACCEPT
