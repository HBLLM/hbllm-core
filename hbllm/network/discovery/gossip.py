"""
Gossip Protocol — State synchronization across distributed HBLLM nodes.

Gossip is STATE SYNC ONLY. It propagates routing hints, capability updates,
and node health across the network using epidemic-style message spreading.

Gossip does NOT:
  - Discover peers (that's mDNS's job).
  - Route messages (that's the RIL's job).
  - Make cognitive decisions (that's the Cognition layer's job).

Safety mechanisms:
  - TTL (Time-to-Live): Each gossip message has a max-hop count to prevent
    infinite propagation.
  - Seen-set: Each node tracks message IDs it has already processed to
    prevent reprocessing.
  - Trust boundaries: Gossip messages carry the originator's node_id, and
    stale entries are auto-pruned via TTL.
  - Convergence: Uses "pull-push" gossip — on each interval, pick a random
    peer and exchange digests.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from hbllm.network.discovery.registry import CapabilityRegistry
    from hbllm.network.node_state import NodeStateEngine


# ── Gossip Data Models ────────────────────────────────────────────────


class GossipEntry(BaseModel):
    """A single entry in the gossip state table."""

    node_id: str
    key: str  # e.g., "capabilities", "health", "role"
    value: Any  # The actual data
    version: int = 0  # Monotonically increasing version per entry
    timestamp: float = Field(default_factory=time.monotonic)
    originator: str = ""  # Who first published this entry


class GossipMessage(BaseModel):
    """A gossip message exchanged between peers."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    source_node: str
    entries: list[GossipEntry] = Field(default_factory=list)
    ttl: int = 3  # Max remaining hops
    created_at: float = Field(default_factory=time.monotonic)


class GossipDigest(BaseModel):
    """A compact summary of a node's gossip state for pull-push exchange."""

    node_id: str
    key: str
    version: int


# ── Gossip Engine ─────────────────────────────────────────────────────


class GossipSync:
    """
    Epidemic-style state synchronization engine.

    Periodically picks a random peer and exchanges state digests. If a peer
    has newer data, we pull it. If we have newer data, we push it.

    The gossip engine updates the CapabilityRegistry and NodeState
    engine when it receives new state from peers.

    Usage:
        gossip = GossipSync(node_id="homeserver", max_hops=3)
        gossip.set_node_state(node_state_engine)
        gossip.set_capability_registry(registry)
        gossip.set_send_fn(my_send_function)
        await gossip.start()
    """

    def __init__(
        self,
        node_id: str,
        max_hops: int = 3,
        gossip_interval: float = 10.0,
        seen_cache_size: int = 1000,
        entry_ttl: float = 120.0,
    ) -> None:
        self.node_id = node_id
        self.max_hops = max_hops
        self.gossip_interval = gossip_interval
        self.entry_ttl = entry_ttl

        # Local state table: (node_id, key) -> GossipEntry
        self._state: dict[tuple[str, str], GossipEntry] = {}

        # Seen message IDs to prevent reprocessing
        self._seen: set[str] = set()
        self._seen_cache_size = seen_cache_size

        # External connections
        self._node_state: NodeStateEngine | None = None
        self._capability_registry: CapabilityRegistry | None = None
        self._send_fn: Any | None = None  # async fn(peer_node_id, GossipMessage)
        self._get_peers_fn: Any | None = None  # fn() -> list[str]

        # Background task
        self._running = False
        self._gossip_task: asyncio.Task[None] | None = None

        # Stats
        self.messages_sent: int = 0
        self.messages_received: int = 0
        self.entries_merged: int = 0
        self.entries_rejected: int = 0

    # ── Configuration ─────────────────────────────────────────────────

    def set_node_state(self, engine: NodeStateEngine) -> None:
        """Attach the NodeState engine for state updates."""
        self._node_state = engine

    def set_capability_registry(self, registry: CapabilityRegistry) -> None:
        """Attach the CapabilityRegistry for capability updates."""
        self._capability_registry = registry

    def set_send_fn(self, fn: Any) -> None:
        """Set the function used to send gossip to a peer.
        Signature: async fn(peer_node_id: str, message: GossipMessage) -> None
        """
        self._send_fn = fn

    def set_get_peers_fn(self, fn: Any) -> None:
        """Set the function to get current peer list.
        Signature: fn() -> list[str]
        """
        self._get_peers_fn = fn

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the periodic gossip loop."""
        self._running = True
        # Publish own state first
        self._publish_local_state()
        self._gossip_task = asyncio.create_task(self._gossip_loop())
        logger.info(
            "GossipSync started (node=%s, interval=%.1fs, max_hops=%d)",
            self.node_id,
            self.gossip_interval,
            self.max_hops,
        )

    async def stop(self) -> None:
        """Stop the gossip loop."""
        self._running = False
        if self._gossip_task and not self._gossip_task.done():
            self._gossip_task.cancel()
            try:
                await self._gossip_task
            except asyncio.CancelledError:
                pass
        logger.info("GossipSync stopped (node=%s)", self.node_id)

    # ── Publishing Local State ────────────────────────────────────────

    def _publish_local_state(self) -> None:
        """Publish the local node's state into the gossip table."""
        if self._node_state:
            snap = self._node_state.snapshot()
            self._update_entry(self.node_id, "role", snap.role.value)
            self._update_entry(self.node_id, "status", snap.status.value)
            self._update_entry(self.node_id, "device_tier", snap.device_tier)
            self._update_entry(self.node_id, "load", snap.cpu_load)
            self._update_entry(self.node_id, "authority_score", snap.authority_score)

        if self._capability_registry:
            caps = self._capability_registry.get_node_capabilities(self.node_id)
            if caps:
                self._update_entry(self.node_id, "capabilities", caps)

    def _update_entry(self, node_id: str, key: str, value: Any) -> None:
        """Update a local gossip entry, incrementing version."""
        table_key = (node_id, key)
        existing = self._state.get(table_key)
        version = (existing.version + 1) if existing else 1

        self._state[table_key] = GossipEntry(
            node_id=node_id,
            key=key,
            value=value,
            version=version,
            originator=self.node_id,
        )

    # ── Receiving Gossip ──────────────────────────────────────────────

    async def receive(self, message: GossipMessage) -> None:
        """
        Process an incoming gossip message from a peer.

        Merge entries that are newer than our local state.
        Re-broadcast (with decremented TTL) if TTL > 0.
        """
        # Duplicate check
        if message.message_id in self._seen:
            return
        self._seen.add(message.message_id)
        self._trim_seen_cache()

        self.messages_received += 1

        merged_count = 0
        for entry in message.entries:
            table_key = (entry.node_id, entry.key)
            existing = self._state.get(table_key)

            # Merge only if newer
            if existing is None or entry.version > existing.version:
                self._state[table_key] = entry
                merged_count += 1
                self.entries_merged += 1

                # Apply to external registries
                self._apply_entry(entry)
            else:
                self.entries_rejected += 1

        if merged_count > 0:
            logger.debug(
                "Gossip: merged %d entries from %s (ttl=%d)",
                merged_count,
                message.source_node,
                message.ttl,
            )

        # Re-broadcast with decremented TTL
        if message.ttl > 1 and merged_count > 0:
            fwd = message.model_copy(deep=True)
            fwd.ttl -= 1
            fwd.source_node = self.node_id
            await self._broadcast(fwd, exclude=message.source_node)

    def _apply_entry(self, entry: GossipEntry) -> None:
        """Apply a merged gossip entry to the local registries."""
        if entry.node_id == self.node_id:
            return  # Don't apply our own state back

        # Update NodeState peer info
        if self._node_state and entry.key == "capabilities":
            from hbllm.network.node_state import NodeRole, PeerInfo

            capabilities = entry.value if isinstance(entry.value, list) else []
            role_entry = self._state.get((entry.node_id, "role"))
            role_str = role_entry.value if role_entry else "edge"
            try:
                role = NodeRole(role_str)
            except ValueError:
                role = NodeRole.EDGE

            tier_entry = self._state.get((entry.node_id, "device_tier"))
            device_tier = tier_entry.value if tier_entry else "server"

            self._node_state.register_peer(
                PeerInfo(
                    node_id=entry.node_id,
                    role=role,
                    capabilities=capabilities,
                    device_tier=device_tier,
                )
            )

        # Update CapabilityRegistry
        if self._capability_registry and entry.key == "capabilities":
            capabilities = entry.value if isinstance(entry.value, list) else []
            if capabilities:
                tier_entry = self._state.get((entry.node_id, "device_tier"))
                device_tier = tier_entry.value if tier_entry else "server"
                auth_entry = self._state.get((entry.node_id, "authority_score"))
                authority = auth_entry.value if auth_entry else 50

                self._capability_registry.register(
                    node_id=entry.node_id,
                    capabilities=capabilities,
                    device_tier=device_tier,
                    authority_score=authority,
                )

    # ── Gossip Loop ───────────────────────────────────────────────────

    async def _gossip_loop(self) -> None:
        """Periodic loop: refresh local state, pick random peer, exchange."""
        import random

        while self._running:
            try:
                await asyncio.sleep(self.gossip_interval)

                # Refresh local state
                self._publish_local_state()

                # Prune stale entries
                self._prune_stale()

                # Get peer list
                peers = self._get_peer_list()
                if not peers:
                    continue

                # Pick a random peer for pull-push gossip
                target = random.choice(peers)

                # Build gossip message with all entries
                entries = list(self._state.values())
                msg = GossipMessage(
                    source_node=self.node_id,
                    entries=entries,
                    ttl=self.max_hops,
                )

                if self._send_fn:
                    try:
                        await self._send_fn(target, msg)
                        self.messages_sent += 1
                    except Exception as e:
                        logger.debug("Gossip: failed to send to %s: %s", target, e)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in gossip loop")

    async def _broadcast(self, message: GossipMessage, exclude: str = "") -> None:
        """Broadcast a gossip message to all known peers except the sender."""
        if not self._send_fn:
            return

        peers = self._get_peer_list()
        for peer_id in peers:
            if peer_id == exclude or peer_id == self.node_id:
                continue
            try:
                await self._send_fn(peer_id, message)
                self.messages_sent += 1
            except Exception as e:
                logger.debug("Gossip: broadcast to %s failed: %s", peer_id, e)

    def _get_peer_list(self) -> list[str]:
        """Get the list of known peer node IDs."""
        if self._get_peers_fn:
            return self._get_peers_fn()
        if self._node_state:
            return [p.node_id for p in self._node_state.reachable_peers]
        return []

    # ── Pruning ───────────────────────────────────────────────────────

    def _prune_stale(self) -> None:
        """Remove entries older than entry_ttl."""
        now = time.monotonic()
        stale_keys = [
            k for k, entry in self._state.items() if (now - entry.timestamp) > self.entry_ttl
        ]
        for k in stale_keys:
            self._state.pop(k, None)
        if stale_keys:
            logger.debug("Gossip: pruned %d stale entries", len(stale_keys))

    def _trim_seen_cache(self) -> None:
        """Trim the seen-set if it grows too large."""
        if len(self._seen) > self._seen_cache_size:
            # Keep the most recent half
            excess = len(self._seen) - (self._seen_cache_size // 2)
            for _ in range(excess):
                self._seen.pop()

    # ── Queries ───────────────────────────────────────────────────────

    def get_digest(self) -> list[GossipDigest]:
        """Get a compact digest of the local gossip state."""
        return [
            GossipDigest(
                node_id=entry.node_id,
                key=entry.key,
                version=entry.version,
            )
            for entry in self._state.values()
        ]

    def get_state_for_node(self, node_id: str) -> dict[str, Any]:
        """Get all gossip state for a specific node."""
        return {key: entry.value for (nid, key), entry in self._state.items() if nid == node_id}

    def get_stats(self) -> dict[str, Any]:
        """Get gossip engine statistics."""
        return {
            "node_id": self.node_id,
            "state_entries": len(self._state),
            "seen_cache": len(self._seen),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "entries_merged": self.entries_merged,
            "entries_rejected": self.entries_rejected,
        }
