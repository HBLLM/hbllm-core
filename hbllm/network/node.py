"""
Node abstraction — every HBLLM component is a Node.

Nodes register with the MessageBus, handle messages, and report health.
This is the fundamental building block of the distributed architecture.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from hbllm.network.clocks import VectorClock
from hbllm.network.messages import Message, MessageType
from hbllm.security.identity import NodeIdentity

if TYPE_CHECKING:
    from hbllm.network.bus import MessageBus


class NodeType(StrEnum):
    """Types of nodes in the HBLLM network."""

    ROUTER = "router"
    CORE = "core"
    DOMAIN_MODULE = "domain_module"
    MEMORY = "memory"
    PLANNER = "planner"
    LEARNER = "learner"
    DETECTOR = "detector"
    SPAWNER = "spawner"
    META = "meta"
    PERCEPTION = "perception"
    ACTION = "action"
    SYSTEM = "system"
    GATEWAY = "gateway"


class DeviceTier(StrEnum):
    """Tiering for hardware capabilities in the personal AI network."""

    MOBILE = "mobile"  # Phone, tablet (high latency, low power)
    EDGE = "edge"  # Raspberry Pi, localized IoT (low latency, minimal power)
    SERVER = "server"  # Desktop, home server (low latency, high power)
    CLOUD = "cloud"  # Remote GPU, shared API (high latency, infinite power)


logger = logging.getLogger(__name__)


class HealthStatus(StrEnum):
    """Health states for a node."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"


class NodeHealth(BaseModel):
    """Health report from a node."""

    node_id: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_heartbeat: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0
    capabilities_available: list[str] = []
    message: str = ""


class NodeInfo(BaseModel):
    """Registration information for a node."""

    node_id: str
    node_type: NodeType
    capabilities: list[str] = []
    capability_metadata: dict[str, Any] = {}  # Metadata for capabilities (e.g., capacity: 5)
    scopes: list[str] = ["public"]  # Permissions: which topic groups this node can access
    public_key: str | None = None  # Ed25519 public key (Base64)
    authority_score: int = 50  # 0-100, used for conflict resolution (higher wins)
    description: str = ""
    fallback_for: list[str] = []  # List of node_ids this node can substitute for
    priority: int = 0  # Higher = preferred when multiple nodes serve same capability
    device_tier: DeviceTier = DeviceTier.SERVER  # Default to server tier
    owner_signature: str | None = None  # Signature from the Owner Node


class Node(ABC):
    """
    Abstract base class for all HBLLM nodes.

    Every component in the system (router, domain modules, memory, planner, etc.)
    extends this class. Nodes communicate exclusively via the MessageBus.
    """

    def __init__(
        self,
        node_id: str,
        node_type: NodeType,
        capabilities: list[str] | None = None,
        capability_metadata: dict[str, Any] | None = None,
        scopes: list[str] | None = None,
        device_tier: DeviceTier = DeviceTier.SERVER,
    ) -> None:
        self.node_id = node_id
        self.node_type = node_type
        self.capabilities = capabilities or []
        self.capability_metadata = capability_metadata or {}
        self.scopes = scopes or ["public"]
        self.device_tier = device_tier
        self.authority_score = 50  # Default authority
        self.description = ""
        self._bus: MessageBus | None = None
        self._running = False
        self._start_time = 0.0
        # Trust Model: Identity
        self.node_identity = NodeIdentity.generate()  # Temporary key if not loaded
        # Authority Hierarchy: Vector Clock
        self.clock = VectorClock(node_id=self.node_id)

    @property
    def bus(self) -> MessageBus:
        """Get the message bus. Raises if node hasn't started."""
        if self._bus is None:
            raise RuntimeError(f"Node {self.node_id} has not been started yet.")
        return self._bus

    @property
    def uptime(self) -> float:
        """Seconds since the node started."""
        if self._start_time == 0.0:
            return 0.0
        return time.monotonic() - self._start_time

    def get_info(self) -> NodeInfo:
        """Get registration info for this node."""
        return NodeInfo(
            node_id=self.node_id,
            node_type=self.node_type,
            capabilities=self.capabilities,
            capability_metadata=self.capability_metadata,
            scopes=self.scopes,
            public_key=self.node_identity.public_key_b64,
            authority_score=self.authority_score,
            description=self.description,
            device_tier=self.device_tier,
        )

    async def start(self, bus: MessageBus) -> None:
        """Start the node and register with the bus."""
        self._bus = bus
        self._start_time = time.monotonic()
        self._running = True
        await self.on_start()

    async def stop(self) -> None:
        """Stop the node and clean up resources."""
        if self._bus and self._running:
            try:
                # Dying Gasp: Notify the bus we are leaving
                await self.publish(
                    "node.lifecycle",
                    Message(
                        type=MessageType.NODE_DEREGISTERED,
                        source_node_id=self.node_id,
                        target_node_id="system",
                        topic="node.lifecycle",
                        payload={"reason": "graceful_shutdown"},
                    ),
                )
            except Exception:
                logger.warning("Failed to send dying gasp for node %s", self.node_id)

        self._running = False
        await self.on_stop()
        self._bus = None

    @abstractmethod
    async def on_start(self) -> None:
        """Called when the node is started. Set up subscriptions here."""
        ...

    @abstractmethod
    async def on_stop(self) -> None:
        """Called when the node is stopping. Clean up resources here."""
        ...

    async def publish(self, topic: str, message: Message) -> None:
        """Sign and publish a message to the bus."""
        self.clock.increment()
        message.vector_clock = self.clock.to_dict()
        message.signature = self.node_identity.sign(message.signable_data)
        await self.bus.publish(topic, message)

    async def request(self, topic: str, message: Message, timeout: float = 90.0) -> Message:
        """Sign and send a request message to the bus."""
        self.clock.increment()
        message.vector_clock = self.clock.to_dict()
        message.signature = self.node_identity.sign(message.signable_data)
        return await self.bus.request(topic, message, timeout=timeout)

    @abstractmethod
    async def handle_message(self, message: Message) -> Message | None:
        """
        Handle an incoming message.

        Subclasses implement this. Instead of subscribing to handle_message
        directly, nodes should ideally subscribe to `process_message` which
        wraps this method with `pre_handle` and `post_handle` hooks.
        """
        ...

    async def process_message(self, message: Message) -> Message | None:
        """
        Wrapper around handle_message that invokes lifecycle hooks.
        Nodes should subscribe to this method to enable telemetry/security hooks.
        """
        await self.pre_handle(message)
        response = await self.handle_message(message)
        await self.post_handle(message, response)
        return response

    async def pre_handle(self, message: Message) -> None:
        """Hook called before handle_message. Override for custom tracing or security."""
        pass

    async def post_handle(self, message: Message, response: Message | None) -> None:
        """Hook called after handle_message. Override for custom telemetry."""
        pass

    async def health_check(self) -> NodeHealth:
        """
        Report the health of this node.

        Override in subclasses for custom health checks.
        """
        return NodeHealth(
            node_id=self.node_id,
            status=HealthStatus.HEALTHY if self._running else HealthStatus.UNHEALTHY,
            uptime_seconds=self.uptime,
            capabilities_available=self.capabilities,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.node_id} type={self.node_type.value}>"
