"""
Fallback Chain — routes to alternative nodes when primary is unavailable.

Each capability has an ordered list of nodes to try. If the primary is down
(circuit open), the next node in the chain handles the request with a
degraded-mode disclaimer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hbllm.network.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError
from hbllm.network.registry import ServiceRegistry

logger = logging.getLogger(__name__)


@dataclass
class FallbackResult:
    """Result of a fallback chain resolution."""

    target_node_id: str
    is_fallback: bool  # True if not the primary node
    primary_node_id: str | None  # The preferred node (if different from target)
    degraded_message: str | None  # Warning message for the user
    chain_position: int  # 0 = primary, 1+ = fallback


@dataclass
class FallbackChainConfig:
    """Configuration for a single fallback chain."""

    capability: str
    chain: list[str]  # Ordered list of node capabilities to try
    degraded_messages: dict[str, str] = field(default_factory=dict)


class FallbackManager:
    """
    Manages fallback chains for graceful degradation.

    When a node is unavailable (circuit open or not registered),
    resolves to the next available node in the fallback chain.
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        circuit_breakers: CircuitBreakerRegistry,
    ):
        self._registry = registry
        self._circuit_breakers = circuit_breakers
        self._chains: dict[str, FallbackChainConfig] = {}

        # Default degraded messages
        self._default_messages: dict[str, str] = {
            "coding": "⚠️ Code module offline — General may have reduced accuracy for code",
            "math": "⚠️ Math specialist offline — using general reasoning",
            "science": "⚠️ Science module offline — using general knowledge",
            "creative": "⚠️ Creative module offline — using general capabilities",
            "general": "⚠️ Running in minimal mode — limited capabilities",
        }

    def register_chain(
        self,
        capability: str,
        chain: list[str],
        degraded_messages: dict[str, str] | None = None,
    ) -> None:
        """Register a fallback chain for a capability."""
        self._chains[capability] = FallbackChainConfig(
            capability=capability,
            chain=chain,
            degraded_messages=degraded_messages or {},
        )
        logger.info("Registered fallback chain for '%s': %s", capability, chain)

    async def resolve(self, capability: str) -> FallbackResult | None:
        """
        Resolve a capability to an available node, trying the fallback chain.

        Returns None if no node in the chain is available.
        """
        chain_config = self._chains.get(capability)
        if not chain_config:
            # No chain configured — try direct discovery
            nodes = await self._registry.discover(capability=capability, healthy_only=True)
            if nodes:
                return FallbackResult(
                    target_node_id=nodes[0].node_id,
                    is_fallback=False,
                    primary_node_id=None,
                    degraded_message=None,
                    chain_position=0,
                )
            return None

        primary_node_id: str | None = None

        for position, cap in enumerate(chain_config.chain):
            # Find healthy nodes with this capability
            nodes = await self._registry.discover(capability=cap, healthy_only=True)

            for node_info in nodes:
                # Check circuit breaker
                breaker = self._circuit_breakers.get(node_info.node_id)
                if not breaker.can_execute():
                    continue

                # Track the primary (first in chain)
                if position == 0:
                    primary_node_id = node_info.node_id

                is_fallback = position > 0
                degraded_msg = None

                if is_fallback:
                    degraded_msg = (
                        chain_config.degraded_messages.get(capability)
                        or self._default_messages.get(capability)
                        or f"⚠️ {capability} module offline — using {cap} as fallback"
                    )

                return FallbackResult(
                    target_node_id=node_info.node_id,
                    is_fallback=is_fallback,
                    primary_node_id=primary_node_id,
                    degraded_message=degraded_msg,
                    chain_position=position,
                )

        logger.warning("No available node found in fallback chain for '%s'", capability)
        return None

    async def get_system_status(self) -> dict[str, dict[str, str]]:
        """
        Get the status of all capabilities and their fallback state.

        Returns a dict of capability → {status, active_node, message}.
        """
        status: dict[str, dict[str, str]] = {}

        for capability, chain_config in self._chains.items():
            result = await self.resolve(capability)
            if result is None:
                status[capability] = {
                    "status": "offline",
                    "active_node": "none",
                    "message": f"❌ {capability} — no available nodes",
                }
            elif result.is_fallback:
                status[capability] = {
                    "status": "degraded",
                    "active_node": result.target_node_id,
                    "message": result.degraded_message or "Running on fallback",
                }
            else:
                status[capability] = {
                    "status": "healthy",
                    "active_node": result.target_node_id,
                    "message": f"✅ {capability} — operating normally",
                }

        return status
