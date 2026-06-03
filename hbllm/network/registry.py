"""
Service Registry — tracks all active nodes in the HBLLM network.

Handles node registration, discovery, and health monitoring.
Starts as an in-process dict; can be backed by etcd/Consul for distribution.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import DeviceTier, HealthStatus, NodeHealth, NodeInfo, NodeType
from hbllm.security.identity import NodeIdentity

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
        self._revoked_nodes: set[str] = set()
        self._vector_clocks: dict[str, Any] = {}  # node_id -> VectorClock
        self._owner_public_key: str | None = None  # Root of Trust

    async def start(self, bus: MessageBus | None = None) -> None:
        """Start the registry and health check loop."""
        self._bus = bus
        self._running = True
        self._health_task = asyncio.create_task(self._health_check_loop())

        if self._bus:
            # Lifecycle listener
            await self._bus.subscribe("node.lifecycle", self._handle_lifecycle_message)

        logger.info("ServiceRegistry started (interval=%.1fs)", self._health_check_interval)

    async def _handle_lifecycle_message(self, message: Message) -> Message | None:
        """Handle incoming node lifecycle messages."""
        if message.type == MessageType.NODE_DEREGISTERED:
            node_id = message.source_node_id
            logger.info("[Registry] Received dying gasp from node: %s", node_id)
            await self.deregister(node_id)
        return None

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

    def set_owner(self, public_key_b64: str) -> None:
        """Set the Root of Trust (Owner's public key)."""
        self._owner_public_key = public_key_b64
        logger.info("ServiceRegistry Root of Trust set: %s...", public_key_b64[:10])

    async def register(self, node_info: NodeInfo) -> None:
        """Register a node with the registry."""
        # Trust Chain Validation
        if self._owner_public_key:
            from hbllm.security.trust_chain import TrustChain

            tc = TrustChain()
            if not node_info.owner_signature or not tc.verify_node_registration(
                node_id=node_info.node_id,
                public_key_b64=node_info.public_key or "",
                owner_signature_b64=node_info.owner_signature,
                owner_public_key_b64=self._owner_public_key,
            ):
                logger.warning(
                    "[Registry] Node '%s' registration REJECTED: Invalid owner signature",
                    node_info.node_id,
                )
                raise PermissionError(f"Node '{node_info.node_id}' is not signed by the Owner.")
        else:
            logger.debug(
                "[Registry] Node '%s' registered in permissive (standalone) mode.",
                node_info.node_id,
            )
        self._nodes[node_info.node_id] = node_info
        self._health[node_info.node_id] = NodeHealth(
            node_id=node_info.node_id,
            status=HealthStatus.STARTING,
            last_heartbeat=time.monotonic(),
            capabilities_available=node_info.capabilities,
        )
        logger.info(
            "Registered node: %s (type=%s, capabilities=%s, metadata=%s)",
            node_info.node_id,
            node_info.node_type.value,
            node_info.capabilities,
            node_info.capability_metadata,
        )

    async def deregister(self, node_id: str) -> None:
        """Remove a node from the registry."""
        self._nodes.pop(node_id, None)
        self._health.pop(node_id, None)
        self._vector_clocks.pop(node_id, None)
        logger.info("Deregistered node: %s", node_id)

    async def revoke_node(self, node_id: str, reason: str = "Compromised") -> None:
        """Revoke a node's identity globally, preventing it from participating in the network."""
        self._revoked_nodes.add(node_id)
        await self.deregister(node_id)
        logger.critical("Node '%s' has been REVOKED. Reason: %s", node_id, reason)
        if self._bus:
            from hbllm.network.messages import Message, MessageType

            await self._bus.publish(
                "system.security.revocation",
                Message(
                    type=MessageType.EVENT,
                    topic="system.security.revocation",
                    source_node_id="system",
                    payload={"revoked_node_id": node_id, "reason": reason},
                ),
            )

    async def is_revoked(self, node_id: str) -> bool:
        """Check if a node has been revoked."""
        return node_id in self._revoked_nodes

    async def update_health(self, health: NodeHealth) -> None:
        """Update health status for a node."""
        health.last_heartbeat = time.monotonic()
        self._health[health.node_id] = health

    async def discover(
        self,
        node_type: NodeType | None = None,
        capability: str | None = None,
        healthy_only: bool = True,
        device_tier: DeviceTier | str | None = None,
    ) -> list[NodeInfo]:
        """
        Discover nodes matching criteria.

        Args:
            node_type: Filter by node type
            capability: Filter by capability string
            healthy_only: Only return healthy/degraded nodes
            device_tier: Filter by hardware tier
        """

        tier_val = None
        if device_tier:
            tier_val = device_tier.value if isinstance(device_tier, DeviceTier) else device_tier

        results = []
        for node_id, info in self._nodes.items():
            # Filter by type
            if node_type and info.node_type != node_type:
                continue

            # Filter by capability
            if capability and capability not in info.capabilities:
                continue

            # Filter by Device Tier
            if tier_val and info.device_tier != tier_val:
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

    async def has_permission(self, node_id: str, scope: str) -> bool:
        """Check if a node has permission to access a specific scope."""
        parent_id = node_id.split(".")[0]
        if parent_id in ("api_server", "system"):
            return True
        info = self._nodes.get(parent_id)
        if not info:
            return False
        # 'admin' scope grants all permissions
        return "admin" in info.scopes or scope in info.scopes or scope == "public"

    async def verify_message(self, message: Message) -> bool:
        """Verify the cryptographic signature and causal ordering of a message."""
        source_id = message.source_node_id
        parent_id = source_id.split(".")[0]

        if source_id in self._revoked_nodes or parent_id in self._revoked_nodes:
            logger.warning("[Registry] verify_message failed: Node '%s' is REVOKED", source_id)
            return False

        info = self._nodes.get(parent_id)
        if not info:
            logger.debug(
                "[Registry] verify_message failed: Node '%s' (parent '%s') not found",
                source_id,
                parent_id,
            )
            return False
        if not info.public_key:
            logger.debug(
                "[Registry] verify_message failed: Node '%s' has no public key",
                message.source_node_id,
            )
            return False

        if not message.signature:
            logger.debug(
                "[Registry] verify_message failed: Message from '%s' has no signature",
                message.source_node_id,
            )
            return False

        is_valid = NodeIdentity.verify(
            public_key_b64=info.public_key,
            data=message.signable_data,
            signature_b64=message.signature,
        )
        if not is_valid:
            logger.debug(
                "[Registry] verify_message failed: Invalid signature for node '%s'",
                message.source_node_id,
            )
            return False

        # Verify Vector Clock for Replay Protection
        if message.vector_clock:
            from hbllm.network.clocks import VectorClock

            msg_clock = VectorClock.from_dict(message.source_node_id, message.vector_clock)
            current_clock = self._vector_clocks.get(message.source_node_id)

            if current_clock:
                cmp = msg_clock.compare(current_clock)
                if cmp in ("before", "equal"):
                    logger.warning(
                        "[Registry] verify_message failed: Replay or causality violation from node '%s'",
                        message.source_node_id,
                    )
                    return False
                current_clock.update(msg_clock)
            else:
                self._vector_clocks[message.source_node_id] = msg_clock

        return True

    async def get_authority_score(self, node_id: str) -> int:
        """Get the authority score of a node (0-100)."""
        info = self._nodes.get(node_id)
        if not info:
            return 0
        return info.authority_score

    async def is_node_healthy(self, node_id: str) -> bool:
        """Check if a specific node is healthy."""
        if node_id in self._revoked_nodes:
            return False
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
