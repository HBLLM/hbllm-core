"""
Node State Engine — The anchor system for adaptive topology stability.

Tracks the dynamic state of the local node: its current role, health, load,
active transports, and peer graph view. This is the "truth" about what this
node currently IS, as opposed to what it CAN do (capabilities).

Without this engine, adaptive topology becomes unstable:
  - Role shifting breaks consistency.
  - Mesh behavior drifts into chaos.
  - The RIL cannot make informed routing decisions.

The NodeState Engine is queried by the RIL scoring model to factor in
real-time health and load when selecting transports.
"""

from __future__ import annotations

import logging
import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NodeRole(StrEnum):
    """Dynamic role a node can assume in the network."""

    STANDALONE = "standalone"  # No network participation
    EDGE = "edge"  # Connects upstream to a Hub
    RELAY = "relay"  # Forwards traffic between peers
    COORDINATOR = "coordinator"  # Central Hub / authority node


class NodeStateStatus(StrEnum):
    """Overall health status of this node."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"


class PeerInfo(BaseModel):
    """Information about a known peer node."""

    node_id: str
    role: NodeRole = NodeRole.STANDALONE
    transport_id: str = ""  # Which transport connects us to this peer
    transport_type: str = ""  # "websocket", "webrtc", "redis", etc.
    capabilities: list[str] = Field(default_factory=list)
    device_tier: str = "server"
    latency_ms: float = 0.0
    last_seen: float = Field(default_factory=time.monotonic)
    is_reachable: bool = True
    authority_score: int = 50


class TransportInfo(BaseModel):
    """Summary of an active transport."""

    transport_id: str
    transport_type: str
    state: str = "disconnected"
    latency_ms: float = 0.0
    error_rate: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0


class NodeStateSnapshot(BaseModel):
    """Serializable snapshot of the full node state."""

    node_id: str
    role: NodeRole
    status: NodeStateStatus
    device_tier: str
    authority_score: int

    # Load metrics
    cpu_load: float = 0.0  # 0.0 - 1.0
    memory_load: float = 0.0  # 0.0 - 1.0
    task_queue_depth: int = 0
    active_tasks: int = 0

    # Network
    active_transports: list[TransportInfo] = Field(default_factory=list)
    known_peers: list[PeerInfo] = Field(default_factory=list)
    reachable_peer_count: int = 0

    # Timing
    uptime_seconds: float = 0.0
    last_updated: float = 0.0


class NodeStateEngine:
    """
    Tracks the dynamic state of the local node.

    This is the anchor system that stabilizes:
      - Role shifting (edge ↔ relay ↔ coordinator)
      - Mesh behavior
      - Hybrid topology decisions

    The RIL queries this engine to factor in real-time health and load
    when scoring transports.
    """

    def __init__(
        self,
        node_id: str,
        role: NodeRole = NodeRole.STANDALONE,
        device_tier: str = "server",
        authority_score: int = 50,
    ) -> None:
        self.node_id = node_id
        self._role = role
        self._device_tier = device_tier
        self._authority_score = authority_score
        self._status = NodeStateStatus.STARTING
        self._start_time = time.monotonic()

        # Load metrics
        self._cpu_load: float = 0.0
        self._memory_load: float = 0.0
        self._task_queue_depth: int = 0
        self._active_tasks: int = 0

        # Peer graph
        self._peers: dict[str, PeerInfo] = {}

        # Active transport summaries
        self._transports: dict[str, TransportInfo] = {}

        # Role transition history (for debugging/auditing)
        self._role_history: list[dict[str, Any]] = []

        self._last_updated = time.monotonic()

    # ── Properties ────────────────────────────────────────────────────

    @property
    def role(self) -> NodeRole:
        return self._role

    @property
    def status(self) -> NodeStateStatus:
        return self._status

    @property
    def device_tier(self) -> str:
        return self._device_tier

    @property
    def authority_score(self) -> int:
        return self._authority_score

    @property
    def uptime(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def load_score(self) -> float:
        """
        Combined load score (0.0 = idle, 1.0 = fully loaded).

        Used by the RIL as a penalty in transport scoring.
        """
        return (
            self._cpu_load * 0.4
            + self._memory_load * 0.3
            + min(self._task_queue_depth / 100.0, 1.0) * 0.3
        )

    @property
    def is_overloaded(self) -> bool:
        return self.load_score > 0.85

    @property
    def reachable_peers(self) -> list[PeerInfo]:
        return [p for p in self._peers.values() if p.is_reachable]

    # ── Role Management ───────────────────────────────────────────────

    def set_role(self, new_role: NodeRole, reason: str = "") -> None:
        """
        Change the node's role. Records transition history.

        Role changes are the most sensitive operation in the adaptive
        topology. They should only happen when:
          - Network partition detected (promote to relay/coordinator)
          - Hub reconnected (demote back to edge)
          - Manual configuration change
        """
        old_role = self._role
        if old_role == new_role:
            return

        self._role = new_role
        self._role_history.append(
            {
                "from": old_role.value,
                "to": new_role.value,
                "reason": reason,
                "timestamp": time.monotonic(),
            }
        )
        self._last_updated = time.monotonic()

        logger.info(
            "NodeState[%s] role changed: %s → %s (reason: %s)",
            self.node_id,
            old_role.value,
            new_role.value,
            reason,
        )

    # ── Status Management ─────────────────────────────────────────────

    def set_status(self, status: NodeStateStatus) -> None:
        """Update the overall health status."""
        if self._status != status:
            logger.info(
                "NodeState[%s] status changed: %s → %s",
                self.node_id,
                self._status.value,
                status.value,
            )
            self._status = status
            self._last_updated = time.monotonic()

    def update_load(
        self,
        cpu_load: float | None = None,
        memory_load: float | None = None,
        task_queue_depth: int | None = None,
        active_tasks: int | None = None,
    ) -> None:
        """Update load metrics. Auto-derives status from load."""
        if cpu_load is not None:
            self._cpu_load = min(max(cpu_load, 0.0), 1.0)
        if memory_load is not None:
            self._memory_load = min(max(memory_load, 0.0), 1.0)
        if task_queue_depth is not None:
            self._task_queue_depth = task_queue_depth
        if active_tasks is not None:
            self._active_tasks = active_tasks

        # Auto-derive status from load
        if self.is_overloaded:
            self.set_status(NodeStateStatus.OVERLOADED)
        elif self.load_score > 0.6:
            self.set_status(NodeStateStatus.DEGRADED)
        elif self._status in (NodeStateStatus.OVERLOADED, NodeStateStatus.DEGRADED):
            self.set_status(NodeStateStatus.HEALTHY)

        self._last_updated = time.monotonic()

    # ── Peer Management ───────────────────────────────────────────────

    def register_peer(self, peer: PeerInfo) -> None:
        """Register or update a known peer."""
        peer.last_seen = time.monotonic()
        self._peers[peer.node_id] = peer
        self._last_updated = time.monotonic()
        logger.debug(
            "NodeState[%s] peer registered: %s (role=%s, transport=%s)",
            self.node_id,
            peer.node_id,
            peer.role.value,
            peer.transport_type,
        )

    def remove_peer(self, node_id: str) -> None:
        """Remove a peer from the graph."""
        self._peers.pop(node_id, None)
        self._last_updated = time.monotonic()

    def mark_peer_unreachable(self, node_id: str) -> None:
        """Mark a peer as unreachable (e.g., after timeout)."""
        peer = self._peers.get(node_id)
        if peer:
            peer.is_reachable = False
            self._last_updated = time.monotonic()
            logger.warning("NodeState[%s] peer unreachable: %s", self.node_id, node_id)

    def mark_peer_reachable(self, node_id: str, latency_ms: float = 0.0) -> None:
        """Mark a peer as reachable (e.g., after heartbeat)."""
        peer = self._peers.get(node_id)
        if peer:
            peer.is_reachable = True
            peer.latency_ms = latency_ms
            peer.last_seen = time.monotonic()
            self._last_updated = time.monotonic()

    def find_peer_by_capability(self, capability: str) -> list[PeerInfo]:
        """Find reachable peers that advertise a specific capability."""
        return [p for p in self._peers.values() if p.is_reachable and capability in p.capabilities]

    # ── Transport Tracking ────────────────────────────────────────────

    def update_transport(self, info: TransportInfo) -> None:
        """Update transport summary from transport metrics."""
        self._transports[info.transport_id] = info
        self._last_updated = time.monotonic()

    def remove_transport(self, transport_id: str) -> None:
        """Remove a transport from tracking."""
        self._transports.pop(transport_id, None)
        self._last_updated = time.monotonic()

    # ── Snapshot ──────────────────────────────────────────────────────

    def snapshot(self) -> NodeStateSnapshot:
        """Create a serializable snapshot of the current node state."""
        return NodeStateSnapshot(
            node_id=self.node_id,
            role=self._role,
            status=self._status,
            device_tier=self._device_tier,
            authority_score=self._authority_score,
            cpu_load=self._cpu_load,
            memory_load=self._memory_load,
            task_queue_depth=self._task_queue_depth,
            active_tasks=self._active_tasks,
            active_transports=list(self._transports.values()),
            known_peers=list(self._peers.values()),
            reachable_peer_count=len(self.reachable_peers),
            uptime_seconds=self.uptime,
            last_updated=self._last_updated,
        )

    def __repr__(self) -> str:
        return (
            f"<NodeStateEngine id={self.node_id} role={self._role.value} "
            f"status={self._status.value} load={self.load_score:.2f} "
            f"peers={len(self._peers)}>"
        )
