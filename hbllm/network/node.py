"""
Node abstraction — every HBLLM component is a Node.

Nodes register with the MessageBus, handle messages, and report health.
This is the fundamental building block of the distributed architecture.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from hbllm.network.bus import MessageBus
    from hbllm.network.messages import Message


class NodeType(str, Enum):
    """Types of nodes in the HBLLM network."""

    ROUTER = "router"
    DOMAIN_MODULE = "domain_module"
    MEMORY = "memory"
    PLANNER = "planner"
    LEARNER = "learner"
    DETECTOR = "detector"
    SPAWNER = "spawner"
    META = "meta"


class HealthStatus(str, Enum):
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
    description: str = ""
    fallback_for: list[str] = []  # List of node_ids this node can substitute for
    priority: int = 0  # Higher = preferred when multiple nodes serve same capability


class Node(ABC):
    """
    Abstract base class for all HBLLM nodes.

    Every component in the system (router, domain modules, memory, planner, etc.)
    extends this class. Nodes communicate exclusively via the MessageBus.
    """

    def __init__(self, node_id: str, node_type: NodeType, capabilities: list[str] | None = None):
        self.node_id = node_id
        self.node_type = node_type
        self.capabilities = capabilities or []
        self.description = ""
        self._bus: MessageBus | None = None
        self._running = False
        self._start_time = 0.0

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
            description=self.description,
        )

    async def start(self, bus: MessageBus) -> None:
        """Start the node and register with the bus."""
        self._bus = bus
        self._start_time = time.monotonic()
        self._running = True
        await self.on_start()

    async def stop(self) -> None:
        """Stop the node and clean up resources."""
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

    @abstractmethod
    async def handle_message(self, message: Message) -> Message | None:
        """
        Handle an incoming message.

        Returns a response Message, or None if no response is needed.
        """
        ...

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

    async def publish(self, topic: str, message: Message) -> None:
        """Convenience: publish a message via the bus."""
        await self.bus.publish(topic, message)

    async def request(self, topic: str, message: Message, timeout: float = 30.0) -> Message:
        """Convenience: send a request and wait for response via the bus."""
        return await self.bus.request(topic, message, timeout=timeout)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.node_id} type={self.node_type.value}>"
