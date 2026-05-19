"""
Capability Registry — The source of truth for the merged network view.

This registry maintains a unified view of all known nodes and their
capabilities across the entire distributed network. It answers the question:

    "Who can do X, and how do I reach them?"

The registry is populated from:
  1. Local ServiceRegistry (nodes registered in this process).
  2. Peer advertisements received via transports (gossip, heartbeats).
  3. Manual configuration (static topology files).

The RIL queries this registry when it needs to route a message to a
specific capability that no local node can handle.

Design principle: The registry is a READ-OPTIMIZED truth store. Writes
happen infrequently (node join/leave/heartbeat), reads happen on every
message routing decision.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CapabilityEntry(BaseModel):
    """A single capability advertised by a node."""

    node_id: str
    capability: str
    transport_id: str = ""  # Which transport reaches this node
    transport_type: str = ""  # "inprocess", "websocket", "redis", "webrtc"
    device_tier: str = "server"
    authority_score: int = 50
    latency_ms: float = 0.0  # Last measured latency to this node
    load: float = 0.0  # 0.0-1.0, last known load
    is_local: bool = False  # True if this node is in our process
    is_reachable: bool = True
    last_updated: float = Field(default_factory=time.monotonic)
    ttl_seconds: float = 60.0  # How long this entry is valid without refresh
    metadata: dict[str, Any] = Field(default_factory=dict)


class CapabilityRegistry:
    """
    Merged view of all capabilities across the distributed network.

    Provides fast lookups for the RIL:
      - "Which nodes can handle 'tool.search'?"
      - "What's the lowest-latency node with 'gpu_inference'?"
      - "Are there any local nodes with 'mcp.filesystem'?"

    Entries have TTLs and are automatically pruned when stale.
    """

    def __init__(self, default_ttl: float = 60.0) -> None:
        self._default_ttl = default_ttl
        # Primary index: capability -> list of entries
        self._by_capability: dict[str, list[CapabilityEntry]] = {}
        # Secondary index: node_id -> list of entries
        self._by_node: dict[str, list[CapabilityEntry]] = {}
        # Stats
        self._total_registrations: int = 0
        self._total_evictions: int = 0

    # ── Registration ──────────────────────────────────────────────────

    def register(
        self,
        node_id: str,
        capabilities: list[str],
        transport_id: str = "",
        transport_type: str = "",
        device_tier: str = "server",
        authority_score: int = 50,
        latency_ms: float = 0.0,
        load: float = 0.0,
        is_local: bool = False,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        """
        Register a node's capabilities.

        If the node was previously registered, its entries are replaced
        (not merged) to ensure freshness.
        """
        # Remove old entries for this node
        self.deregister(node_id)

        ttl = ttl_seconds or self._default_ttl
        entries: list[CapabilityEntry] = []

        for cap in capabilities:
            entry = CapabilityEntry(
                node_id=node_id,
                capability=cap,
                transport_id=transport_id,
                transport_type=transport_type,
                device_tier=device_tier,
                authority_score=authority_score,
                latency_ms=latency_ms,
                load=load,
                is_local=is_local,
                ttl_seconds=ttl,
                metadata=metadata or {},
            )
            entries.append(entry)

            # Index by capability
            if cap not in self._by_capability:
                self._by_capability[cap] = []
            self._by_capability[cap].append(entry)

        # Index by node
        self._by_node[node_id] = entries
        self._total_registrations += 1

        logger.debug(
            "CapabilityRegistry: registered %s with %d capabilities (transport=%s)",
            node_id,
            len(capabilities),
            transport_type,
        )

    def deregister(self, node_id: str) -> None:
        """Remove all entries for a node."""
        old_entries = self._by_node.pop(node_id, [])
        for entry in old_entries:
            cap_list = self._by_capability.get(entry.capability, [])
            self._by_capability[entry.capability] = [e for e in cap_list if e.node_id != node_id]
            # Clean up empty lists
            if not self._by_capability[entry.capability]:
                del self._by_capability[entry.capability]

        if old_entries:
            logger.debug("CapabilityRegistry: deregistered %s", node_id)

    # ── Queries ───────────────────────────────────────────────────────

    def find_by_capability(
        self,
        capability: str,
        local_only: bool = False,
        reachable_only: bool = True,
        sort_by_latency: bool = True,
    ) -> list[CapabilityEntry]:
        """
        Find nodes that advertise a specific capability.

        Args:
            capability: The capability to search for (e.g., "tool.search").
            local_only: Only return entries from the local process.
            reachable_only: Only return entries that are currently reachable.
            sort_by_latency: Sort results by latency (fastest first).

        Returns:
            List of CapabilityEntry, filtered and sorted.
        """
        self._prune_stale()

        entries = self._by_capability.get(capability, [])
        results = []

        for entry in entries:
            if local_only and not entry.is_local:
                continue
            if reachable_only and not entry.is_reachable:
                continue
            results.append(entry)

        if sort_by_latency:
            results.sort(key=lambda e: e.latency_ms)

        return results

    def find_best_for_capability(
        self, capability: str, local_only: bool = False
    ) -> CapabilityEntry | None:
        """
        Find the single best node for a capability.

        Scoring:
          - Local preference: +100
          - Low latency: +50 * (1 - normalized_latency)
          - Low load: +30 * (1 - load)
          - Authority: +20 * (authority / 100)
        """
        candidates = self.find_by_capability(
            capability, local_only=local_only, reachable_only=True, sort_by_latency=False
        )
        if not candidates:
            return None

        def score(entry: CapabilityEntry) -> float:
            s = 0.0
            if entry.is_local:
                s += 100.0
            # Latency (lower is better, cap at 200ms for scoring)
            s += 50.0 * max(0, 1.0 - entry.latency_ms / 200.0)
            # Load (lower is better)
            s += 30.0 * (1.0 - entry.load)
            # Authority
            s += 20.0 * (entry.authority_score / 100.0)
            return s

        return max(candidates, key=score)

    def get_node_capabilities(self, node_id: str) -> list[str]:
        """Get all capabilities registered by a specific node."""
        entries = self._by_node.get(node_id, [])
        return [e.capability for e in entries if e.is_reachable]

    def get_all_capabilities(self) -> list[str]:
        """Get a deduplicated list of all known capabilities."""
        self._prune_stale()
        return list(self._by_capability.keys())

    def get_all_nodes(self) -> list[str]:
        """Get all registered node IDs."""
        return list(self._by_node.keys())

    def get_network_summary(self) -> dict[str, Any]:
        """Get a summary of the network state."""
        self._prune_stale()
        total_entries = sum(len(v) for v in self._by_node.values())
        reachable = sum(
            1 for entries in self._by_node.values() if any(e.is_reachable for e in entries)
        )
        return {
            "total_nodes": len(self._by_node),
            "reachable_nodes": reachable,
            "total_capabilities": len(self._by_capability),
            "total_entries": total_entries,
            "total_registrations": self._total_registrations,
            "total_evictions": self._total_evictions,
        }

    # ── Health Updates ────────────────────────────────────────────────

    def update_node_health(
        self,
        node_id: str,
        latency_ms: float | None = None,
        load: float | None = None,
        is_reachable: bool | None = None,
    ) -> None:
        """Update health metrics for a node's entries."""
        entries = self._by_node.get(node_id, [])
        for entry in entries:
            if latency_ms is not None:
                entry.latency_ms = latency_ms
            if load is not None:
                entry.load = load
            if is_reachable is not None:
                entry.is_reachable = is_reachable
            entry.last_updated = time.monotonic()

    # ── TTL Management ────────────────────────────────────────────────

    def _prune_stale(self) -> None:
        """Remove entries whose TTL has expired."""
        now = time.monotonic()
        stale_nodes: list[str] = []

        for node_id, entries in self._by_node.items():
            fresh = [e for e in entries if (now - e.last_updated) < e.ttl_seconds]
            if not fresh:
                stale_nodes.append(node_id)
            else:
                self._by_node[node_id] = fresh

        for node_id in stale_nodes:
            self.deregister(node_id)
            self._total_evictions += 1
            logger.debug("CapabilityRegistry: evicted stale node %s", node_id)

    def refresh_node(self, node_id: str) -> None:
        """Refresh the TTL for all entries of a node (e.g., on heartbeat)."""
        now = time.monotonic()
        for entry in self._by_node.get(node_id, []):
            entry.last_updated = now

    # ── Stats ─────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self._by_node)

    @property
    def capability_count(self) -> int:
        return len(self._by_capability)

    def __repr__(self) -> str:
        return f"<CapabilityRegistry nodes={self.node_count} capabilities={self.capability_count}>"
