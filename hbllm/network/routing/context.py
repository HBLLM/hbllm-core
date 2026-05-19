"""
Execution Context — Traceability model for distributed cognitive routing.

Every message routed through the RIL carries an ExecutionContext.
This ensures deterministic debugging and prevents "lost reasoning chains"
across multiple physical devices.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class RouteHop(BaseModel):
    """A single hop in the routing trace history."""

    transport_id: str
    transport_type: str  # "inprocess", "websocket", "redis", "webrtc"
    node_id: str
    timestamp: float = Field(default_factory=time.monotonic)
    latency_ms: float = 0.0


class ExecutionContext(BaseModel):
    """
    Traceability context attached to every routed message.

    Tracks the full lifecycle of a message across the distributed
    cognitive network, enabling debugging, auditing, and loop detection.
    """

    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Origin
    origin_node: str = ""
    origin_transport: str = ""

    # Target
    target_node: str | None = None
    capability_required: str | None = None

    # Routing decision
    selected_transport: str | None = None
    routing_score: float = 0.0
    fallback_path: list[str] = Field(default_factory=list)

    # Trace history (ordered list of hops)
    trace_history: list[RouteHop] = Field(default_factory=list)

    # Constraints
    max_hops: int = 5  # Prevent cyclic routing chains
    created_at: float = Field(default_factory=time.monotonic)
    deadline: float | None = None  # Absolute monotonic deadline

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def hop_count(self) -> int:
        return len(self.trace_history)

    @property
    def is_expired(self) -> bool:
        if self.deadline is None:
            return False
        return time.monotonic() > self.deadline

    @property
    def has_exceeded_max_hops(self) -> bool:
        return self.hop_count >= self.max_hops

    @property
    def total_latency_ms(self) -> float:
        return sum(hop.latency_ms for hop in self.trace_history)

    def add_hop(
        self,
        transport_id: str,
        transport_type: str,
        node_id: str,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a routing hop."""
        self.trace_history.append(
            RouteHop(
                transport_id=transport_id,
                transport_type=transport_type,
                node_id=node_id,
                latency_ms=latency_ms,
            )
        )

    def visited_node(self, node_id: str) -> bool:
        """Check if a node has already been visited (loop detection)."""
        return any(hop.node_id == node_id for hop in self.trace_history)
