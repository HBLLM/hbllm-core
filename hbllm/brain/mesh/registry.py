"""Node Registry and Cognitive QoS Definitions.

Defines node types, capabilities, and the TaskPriorityClass
to establish the distributed mesh nervous system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


class NodeType(StrEnum):
    """Classification of sovereign nodes in the mesh."""

    PHONE = "phone"
    DESKTOP = "desktop"
    CAR = "car"
    CLOUD_SERVER = "cloud_server"
    WATCH = "watch"


class TaskPriorityClass(IntEnum):
    """Cognitive Quality of Service (QoS). Lower number = higher priority."""

    REFLEX = 0  # Safety critical (e.g., car braking)
    REALTIME = 1  # Latency critical (e.g., voice response)
    INTERACTIVE = 2  # Human waiting (e.g., UI updates)
    BACKGROUND = 3  # Deep simulation, heavy planning
    ARCHIVAL = 4  # Memory compaction, semantic folding


@dataclass
class NodeCapabilities:
    """Hardware and cognitive limits of a node."""

    max_context_tokens: int = 4096
    has_gpu: bool = False
    has_npu: bool = False
    battery_level: float = 1.0  # 0.0 to 1.0
    is_charging: bool = True
    network_latency_ms: float = 0.0
    specializations: list[str] = field(
        default_factory=list
    )  # e.g. ["vehicle_control", "biometrics"]


@dataclass
class NodeProfile:
    """A registered node in the mesh."""

    node_id: str
    node_type: NodeType
    capabilities: NodeCapabilities = field(default_factory=NodeCapabilities)
    last_heartbeat: float = field(default_factory=time.time)
    is_active: bool = True

    def update_heartbeat(self, capabilities: NodeCapabilities | None = None) -> None:
        self.last_heartbeat = time.time()
        if capabilities:
            self.capabilities = capabilities


class NodeRegistry:
    """Tracks all known sovereign nodes in the personal cluster."""

    def __init__(self, local_node_id: str, local_node_type: NodeType) -> None:
        self.local_node_id = local_node_id
        self.local_node_type = local_node_type
        self.nodes: dict[str, NodeProfile] = {}

    def register_node(self, profile: NodeProfile) -> None:
        self.nodes[profile.node_id] = profile

    def get_active_nodes(self, max_latency_ms: float = 1000.0) -> list[NodeProfile]:
        """Return nodes that are active and reachable."""
        now = time.time()
        active = []
        for node in self.nodes.values():
            if node.is_active and (now - node.last_heartbeat) < 60.0:
                if node.capabilities.network_latency_ms <= max_latency_ms:
                    active.append(node)
        return active
