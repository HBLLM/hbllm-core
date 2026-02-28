"""
Load Balancer — distributes requests across multiple instances of the same capability.

Strategies:
  - round_robin:       Cycle through healthy nodes
  - least_loaded:      Pick the node with lowest latency (from health data)
  - capability_match:  Prefer exact capability match, then fallback
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hbllm.network.node import HealthStatus, NodeInfo

if TYPE_CHECKING:
    from hbllm.network.circuit_breaker import CircuitBreakerRegistry
    from hbllm.network.registry import ServiceRegistry

logger = logging.getLogger(__name__)


class LoadBalancer:
    """
    Distributes requests across healthy nodes providing the same capability.

    Integrates with ServiceRegistry for discovery and CircuitBreaker to
    skip nodes with open circuits.
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        circuit_breakers: CircuitBreakerRegistry,
        strategy: str = "round_robin",
        health_weight: float = 0.7,
        latency_weight: float = 0.3,
    ):
        self._registry = registry
        self._circuit_breakers = circuit_breakers
        self._strategy = strategy
        self._health_weight = health_weight
        self._latency_weight = latency_weight

        # Round-robin counters per capability
        self._rr_counters: dict[str, int] = {}

    async def select(
        self,
        capability: str,
        strategy: str | None = None,
    ) -> NodeInfo | None:
        """
        Select the best node for a capability using the configured strategy.

        Args:
            capability: The capability to find a node for.
            strategy: Override the default strategy for this call.

        Returns:
            NodeInfo of the selected node, or None if none available.
        """
        # Discover healthy nodes with this capability
        candidates = await self._registry.discover(
            capability=capability,
            healthy_only=True,
        )

        # Filter out nodes with open circuit breakers
        available = []
        for node in candidates:
            breaker = self._circuit_breakers.get(node.node_id)
            if breaker.can_execute():
                available.append(node)

        if not available:
            logger.warning("No available nodes for capability '%s'", capability)
            return None

        # Single node — short-circuit
        if len(available) == 1:
            return available[0]

        active_strategy = strategy or self._strategy

        if active_strategy == "round_robin":
            return self._select_round_robin(capability, available)
        elif active_strategy == "least_loaded":
            return await self._select_least_loaded(capability, available)
        elif active_strategy == "capability_match":
            return self._select_capability_match(capability, available)
        else:
            logger.warning("Unknown strategy '%s', defaulting to round_robin", active_strategy)
            return self._select_round_robin(capability, available)

    def _select_round_robin(
        self,
        capability: str,
        candidates: list[NodeInfo],
    ) -> NodeInfo:
        """Simple round-robin selection."""
        counter = self._rr_counters.get(capability, 0)
        selected = candidates[counter % len(candidates)]
        self._rr_counters[capability] = counter + 1
        logger.debug(
            "RoundRobin selected '%s' for '%s' (%d candidates)",
            selected.node_id, capability, len(candidates),
        )
        return selected

    async def _select_least_loaded(
        self,
        capability: str,
        candidates: list[NodeInfo],
    ) -> NodeInfo:
        """Select the node with the lowest weighted score (latency + health)."""
        best: NodeInfo | None = None
        best_score = float("inf")

        for node in candidates:
            health = await self._registry.get_health(node.node_id)
            if not health:
                continue

            # Lower latency = better, healthy = 0, degraded = 0.5
            health_penalty = 0.0
            if health.status == HealthStatus.DEGRADED:
                health_penalty = 0.5

            score = (
                self._latency_weight * health.latency_ms
                + self._health_weight * health_penalty * 1000  # scale to ms range
            )

            if score < best_score:
                best_score = score
                best = node

        if best is None:
            best = candidates[0]

        logger.debug(
            "LeastLoaded selected '%s' for '%s' (score=%.1f, %d candidates)",
            best.node_id, capability, best_score, len(candidates),
        )
        return best

    def _select_capability_match(
        self,
        capability: str,
        candidates: list[NodeInfo],
    ) -> NodeInfo:
        """
        Prefer nodes whose capabilities exactly match the request.
        Falls back to round-robin among partial matches.
        """
        # Exact matches: capability appears first in the node's capability list
        exact = [n for n in candidates if n.capabilities and n.capabilities[0] == capability]
        if exact:
            return self._select_round_robin(capability, exact)

        # Partial matches
        return self._select_round_robin(capability, candidates)

    def reset_counters(self) -> None:
        """Reset all round-robin counters."""
        self._rr_counters.clear()
