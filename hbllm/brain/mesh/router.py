"""Epistemic and Compute Routing.

Routes tasks to nodes based on knowledge quality (Domain Authority)
as well as compute, power, and latency constraints.
"""

from __future__ import annotations

import logging

from hbllm.brain.mesh.registry import NodeProfile, NodeRegistry, NodeType, TaskPriorityClass

logger = logging.getLogger(__name__)


class EpistemicRouter:
    """Matches a task domain to the node with the highest Domain Authority."""

    DOMAIN_MAP = {
        "vehicle": NodeType.CAR,
        "biometrics": NodeType.PHONE,
        "filesystem": NodeType.DESKTOP,
        "long_term_memory": NodeType.CLOUD_SERVER,
        "heavy_simulation": NodeType.CLOUD_SERVER,
        "local_environment": NodeType.PHONE,
    }

    def __init__(self, registry: NodeRegistry) -> None:
        self.registry = registry

    def route_task(
        self, domain: str, priority: TaskPriorityClass, required_compute: str = "low"
    ) -> str | None:
        """Find the optimal node for a task based on epistemic authority and compute limits."""
        active_nodes = self.registry.get_active_nodes()
        if not active_nodes:
            return None

        # 1. Epistemic Routing (Knowledge Quality Priority)
        target_type = self.DOMAIN_MAP.get(domain)
        if target_type:
            for node in active_nodes:
                if node.node_type == target_type:
                    # Check if the authoritative node can handle it
                    if self._check_qos_constraints(node, priority, required_compute):
                        logger.info(
                            "Epistemic routing matched domain '%s' to node %s", domain, node.node_id
                        )
                        return node.node_id

        # 2. Compute/Power Fallback Routing
        # If no authoritative node, or authoritative node is overloaded, find any capable node
        candidates = []
        for node in active_nodes:
            if self._check_qos_constraints(node, priority, required_compute):
                candidates.append(node)

        if not candidates:
            logger.warning("No nodes meet QoS constraints for domain '%s'", domain)
            return None

        # Sort by battery level (highest first) as a simple heuristic
        candidates.sort(key=lambda n: n.capabilities.battery_level, reverse=True)
        return candidates[0].node_id

    def _check_qos_constraints(
        self, node: NodeProfile, priority: TaskPriorityClass, required_compute: str
    ) -> bool:
        """Ensure the node has enough resources to execute the priority class."""
        # Hard limits
        if required_compute == "high" and not (
            node.capabilities.has_gpu or node.capabilities.has_npu
        ):
            return False

        if priority == TaskPriorityClass.REFLEX and node.capabilities.network_latency_ms > 50.0:
            return False  # Too slow for reflex

        if (
            priority == TaskPriorityClass.BACKGROUND
            and node.capabilities.battery_level < 0.2
            and not node.capabilities.is_charging
        ):
            return False  # Conserve battery, drop background tasks

        return True
