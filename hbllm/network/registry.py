"""
Service Registry — tracks all active nodes in the HBLLM network.

Handles node registration, discovery, and health monitoring.
Starts as an in-process dict; can be backed by etcd/Consul for distribution.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from hbllm.network.node import HealthStatus, NodeHealth, NodeInfo, NodeType

if TYPE_CHECKING:
    from hbllm.network.bus import MessageBus

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """
    Tracks all active nodes and their health status.

    Provides node discovery by type/capability and periodic health checking.
    """

    def __init__(
        self,
        health_check_interval: float = 10.0,
        node_timeout: float = 30.0,
    ):
        self._nodes: dict[str, NodeInfo] = {}
        self._health: dict[str, NodeHealth] = {}
        self._health_check_interval = health_check_interval
        self._node_timeout = node_timeout
        self._health_task: asyncio.Task[None] | None = None
        self._running = False
        self._bus: MessageBus | None = None

    async def start(self, bus: MessageBus | None = None) -> None:
        """Start the registry and health check loop."""
        self._bus = bus
        self._running = True
        self._health_task = asyncio.create_task(self._health_check_loop())
        logger.info("ServiceRegistry started (interval=%.1fs)", self._health_check_interval)

    async def stop(self) -> None:
        """Stop the registry."""
        self._running = False
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        logger.info("ServiceRegistry stopped")

    async def register(self, node_info: NodeInfo) -> None:
        """Register a node with the registry."""
        self._nodes[node_info.node_id] = node_info
        self._health[node_info.node_id] = NodeHealth(
            node_id=node_info.node_id,
            status=HealthStatus.STARTING,
            last_heartbeat=time.monotonic(),
            capabilities_available=node_info.capabilities,
        )
        logger.info(
            "Registered node: %s (type=%s, capabilities=%s)",
            node_info.node_id,
            node_info.node_type.value,
            node_info.capabilities,
        )

    async def deregister(self, node_id: str) -> None:
        """Remove a node from the registry."""
        self._nodes.pop(node_id, None)
        self._health.pop(node_id, None)
        logger.info("Deregistered node: %s", node_id)

    async def update_health(self, health: NodeHealth) -> None:
        """Update health status for a node."""
        health.last_heartbeat = time.monotonic()
        self._health[health.node_id] = health

    async def discover(
        self,
        node_type: NodeType | None = None,
        capability: str | None = None,
        healthy_only: bool = True,
    ) -> list[NodeInfo]:
        """
        Discover nodes matching criteria.

        Args:
            node_type: Filter by node type
            capability: Filter by capability string
            healthy_only: Only return healthy/degraded nodes
        """
        results = []
        for node_id, info in self._nodes.items():
            # Filter by type
            if node_type and info.node_type != node_type:
                continue

            # Filter by capability
            if capability and capability not in info.capabilities:
                continue

            # Filter by health
            if healthy_only:
                health = self._health.get(node_id)
                if health and health.status in (HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN):
                    continue

            results.append(info)

        # Sort by priority (higher first)
        results.sort(key=lambda n: n.priority, reverse=True)
        return results

    async def get_health(self, node_id: str) -> NodeHealth | None:
        """Get current health for a specific node."""
        return self._health.get(node_id)

    async def get_all_health(self) -> dict[str, NodeHealth]:
        """Get health status of all registered nodes."""
        return dict(self._health)

    async def get_available_capabilities(self) -> set[str]:
        """Get all capabilities currently available from healthy nodes."""
        capabilities: set[str] = set()
        for node_id, info in self._nodes.items():
            health = self._health.get(node_id)
            if health and health.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED):
                capabilities.update(info.capabilities)
        return capabilities

    async def is_node_healthy(self, node_id: str) -> bool:
        """Check if a specific node is healthy."""
        health = self._health.get(node_id)
        if not health:
            return False
        return health.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    async def _health_check_loop(self) -> None:
        """Periodically check for timed-out nodes."""
        while self._running:
            try:
                await asyncio.sleep(self._health_check_interval)
                now = time.monotonic()

                for node_id, health in list(self._health.items()):
                    elapsed = now - health.last_heartbeat
                    if elapsed > self._node_timeout:
                        if health.status != HealthStatus.UNHEALTHY:
                            logger.warning(
                                "Node %s timed out (%.1fs since last heartbeat), marking unhealthy",
                                node_id,
                                elapsed,
                            )
                            health.status = HealthStatus.UNHEALTHY
                            health.message = f"No heartbeat for {elapsed:.0f}s"

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in health check loop")

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        healthy = sum(
            1
            for h in self._health.values()
            if h.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        )
        return f"<ServiceRegistry nodes={len(self._nodes)} healthy={healthy}>"
