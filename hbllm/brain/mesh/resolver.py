"""Distributed Conflict Resolver.

Uses Hierarchical Hybrid Coordination and Domain Authority
to resolve contradictory beliefs or plans between nodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hbllm.brain.mesh.registry import NodeType

logger = logging.getLogger(__name__)


@dataclass
class ConflictProposal:
    node_id: str
    node_type: NodeType
    domain: str
    proposed_value: Any
    is_safety_critical: bool = False


class DistributedConflictResolver:
    """Resolves conflicts using Domain Authority and Safety hierarchies."""

    # Domain Authority mappings
    AUTHORITY = {
        "vehicle": NodeType.CAR,
        "biometrics": NodeType.PHONE,
        "filesystem": NodeType.DESKTOP,
        "long_term_memory": NodeType.CLOUD_SERVER,
    }

    def resolve(self, proposals: list[ConflictProposal]) -> ConflictProposal | None:
        """Resolve a conflict between multiple node proposals."""
        if not proposals:
            return None
        if len(proposals) == 1:
            return proposals[0]

        # Rule 1: Safety Reflexes always override utility
        safety_critical = [p for p in proposals if p.is_safety_critical]
        if safety_critical:
            # If multiple safety critical, we just pick the first for now
            # In a real system, human intervention might be required if safety reflexes conflict
            logger.warning(
                "Resolving conflict via SAFETY OVERRIDE: selected %s", safety_critical[0].node_id
            )
            return safety_critical[0]

        # Rule 2: Domain Authority (Locality beats consensus)
        domain = proposals[0].domain
        authoritative_type = self.AUTHORITY.get(domain)

        if authoritative_type:
            for p in proposals:
                if p.node_type == authoritative_type:
                    logger.info(
                        "Resolving conflict via DOMAIN AUTHORITY: selected %s for domain %s",
                        p.node_id,
                        domain,
                    )
                    return p

        # Rule 3: Tie-breaker - trust the primary node (e.g., PHONE)
        for p in proposals:
            if p.node_type == NodeType.PHONE:
                logger.info("Resolving conflict via PRIMARY COORDINATOR: selected %s", p.node_id)
                return p

        # Rule 4: Arbitrary tie breaker if all else fails
        logger.info("Resolving conflict via ARBITRARY FALLBACK: selected %s", proposals[0].node_id)
        return proposals[0]
