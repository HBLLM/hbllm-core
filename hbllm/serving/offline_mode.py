"""Offline Mode Manager — graceful degradation when network is unavailable.

Monitors network connectivity and switches behavior when offline:
    - Routes LLM calls to local model (via DualLLMRouter)
    - Caches recent cloud responses for replay
    - Queues outbound requests (push notifications, webhooks) for retry
    - Disables web search tools
    - Publishes system.offline / system.online events

Architecture:
    1. Periodic connectivity check (configurable interval)
    2. State machine: ONLINE → DEGRADED → OFFLINE → RECOVERING → ONLINE
    3. Request queue with TTL (outbound requests expire after max_age)
    4. Cloud response cache (LRU, configurable size)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ConnectivityState(str, Enum):
    """Network connectivity states."""

    ONLINE = "online"
    DEGRADED = "degraded"  # Intermittent connectivity
    OFFLINE = "offline"
    RECOVERING = "recovering"  # Just came back online, draining queue


@dataclass
class QueuedRequest:
    """An outbound request queued while offline."""

    request_id: str
    request_type: str  # "push", "webhook", "api_call", "memory_sync"
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    max_age_s: float = 3600.0  # 1 hour default TTL
    retry_count: int = 0
    max_retries: int = 3

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.max_age_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_type": self.request_type,
            "age_s": time.time() - self.created_at,
            "retry_count": self.retry_count,
            "expired": self.is_expired,
        }


@dataclass
class CachedResponse:
    """A cached cloud response for offline replay."""

    cache_key: str
    response: Any
    cached_at: float = field(default_factory=time.time)
    ttl_s: float = 1800.0  # 30 minutes

    @property
    def is_stale(self) -> bool:
        return time.time() - self.cached_at > self.ttl_s


class OfflineManager:
    """Manages graceful degradation when network is unavailable.

    Usage::

        offline = OfflineManager()
        await offline.start()

        # Check before making cloud calls
        if offline.is_online:
            response = await cloud_api.call(...)
            offline.cache_response("query_key", response)
        else:
            cached = offline.get_cached("query_key")
            if cached:
                use(cached)
            else:
                use_local_model(...)

        # Queue outbound when offline
        if not offline.is_online:
            offline.enqueue("push", {"device": "...", "msg": "..."})
    """

    def __init__(
        self,
        check_interval_s: float = 30.0,
        check_endpoints: list[str] | None = None,
        cache_max_size: int = 200,
        queue_max_size: int = 500,
        bus: Any = None,
    ) -> None:
        self.check_interval_s = check_interval_s
        self.check_endpoints = check_endpoints or [
            "8.8.8.8",  # Google DNS
            "1.1.1.1",  # Cloudflare DNS
        ]
        self._cache_max_size = cache_max_size
        self._queue_max_size = queue_max_size
        self.bus = bus

        # State
        self._state = ConnectivityState.ONLINE
        self._state_since = time.time()
        self._check_task: asyncio.Task[None] | None = None
        self._running = False

        # Queue and cache
        self._queue: list[QueuedRequest] = []
        self._cache: OrderedDict[str, CachedResponse] = OrderedDict()
        self._request_counter = 0

        # Telemetry
        self._total_checks = 0
        self._total_failures = 0
        self._total_queued = 0
        self._total_drained = 0
        self._total_expired = 0
        self._state_transitions = 0
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def state(self) -> ConnectivityState:
        return self._state

    @property
    def is_online(self) -> bool:
        return self._state in (ConnectivityState.ONLINE, ConnectivityState.RECOVERING)

    @property
    def is_offline(self) -> bool:
        return self._state == ConnectivityState.OFFLINE

    async def start(self) -> None:
        """Start the connectivity monitoring loop."""
        self._running = True
        self._check_task = asyncio.create_task(self._monitor_loop())
        logger.info("OfflineManager started (interval=%.0fs)", self.check_interval_s)

    async def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("OfflineManager stopped")

    async def _monitor_loop(self) -> None:
        """Periodic connectivity check loop."""
        while self._running:
            try:
                is_connected = await self._check_connectivity()
                self._total_checks += 1

                if is_connected and self._state == ConnectivityState.OFFLINE:
                    await self._transition(ConnectivityState.RECOVERING)
                    await self._drain_queue()
                    await self._transition(ConnectivityState.ONLINE)

                elif is_connected and self._state == ConnectivityState.DEGRADED:
                    await self._transition(ConnectivityState.ONLINE)

                elif not is_connected and self._state == ConnectivityState.ONLINE:
                    self._total_failures += 1
                    # First failure → degraded (might be transient)
                    await self._transition(ConnectivityState.DEGRADED)

                elif not is_connected and self._state == ConnectivityState.DEGRADED:
                    self._total_failures += 1
                    # Second failure → offline
                    await self._transition(ConnectivityState.OFFLINE)

                elif not is_connected:
                    self._total_failures += 1

                await asyncio.sleep(self.check_interval_s)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Connectivity check error: %s", e)
                await asyncio.sleep(self.check_interval_s)

    async def _check_connectivity(self) -> bool:
        """Check if we can reach any of the configured endpoints."""
        for endpoint in self.check_endpoints:
            try:
                # Use asyncio to attempt a TCP connection (port 53 for DNS)
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(endpoint, 53),
                    timeout=5.0,
                )
                writer.close()
                await writer.wait_closed()
                return True
            except (OSError, asyncio.TimeoutError):
                continue
        return False

    async def _transition(self, new_state: ConnectivityState) -> None:
        """Transition to a new connectivity state."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._state_since = time.time()
        self._state_transitions += 1

        logger.info("Network state: %s → %s", old_state.value, new_state.value)

        # Publish state change event
        if self.bus:
            try:
                from hbllm.network.messages import Message, MessageType

                topic = (
                    "system.offline"
                    if new_state == ConnectivityState.OFFLINE
                    else "system.online"
                    if new_state == ConnectivityState.ONLINE
                    else f"system.network.{new_state.value}"
                )
                await self.bus.publish(
                    topic,
                    Message(
                        type=MessageType.EVENT,
                        source_node_id="offline_manager",
                        topic=topic,
                        payload={
                            "state": new_state.value,
                            "previous_state": old_state.value,
                            "queued_requests": len(self._queue),
                        },
                    ),
                )
            except Exception as e:
                logger.debug("Failed to publish state change: %s", e)

    # ── Queue Management ────────────────────────────────────────────────

    def enqueue(
        self,
        request_type: str,
        payload: dict[str, Any],
        max_age_s: float = 3600.0,
    ) -> str:
        """Queue an outbound request for delivery when online.

        Returns the request_id.
        """
        if len(self._queue) >= self._queue_max_size:
            # Prune expired first
            self._prune_expired()
            if len(self._queue) >= self._queue_max_size:
                # Drop oldest non-critical
                self._queue.pop(0)

        self._request_counter += 1
        request_id = f"req_{self._request_counter}_{int(time.time())}"

        self._queue.append(
            QueuedRequest(
                request_id=request_id,
                request_type=request_type,
                payload=payload,
                max_age_s=max_age_s,
            )
        )
        self._total_queued += 1

        logger.debug("Queued %s request: %s", request_type, request_id)
        return request_id

    async def _drain_queue(self) -> None:
        """Drain queued requests when connectivity is restored."""
        self._prune_expired()

        if not self._queue:
            return

        logger.info("Draining %d queued requests", len(self._queue))
        drained = []

        for req in list(self._queue):
            if req.is_expired:
                self._total_expired += 1
                drained.append(req)
                continue

            try:
                # Execute the queued request
                await self._execute_queued(req)
                self._total_drained += 1
                drained.append(req)
            except Exception as e:
                req.retry_count += 1
                if req.retry_count >= req.max_retries:
                    logger.warning(
                        "Queued request %s exhausted retries: %s",
                        req.request_id,
                        e,
                    )
                    drained.append(req)
                else:
                    logger.debug(
                        "Queued request %s retry %d failed: %s",
                        req.request_id,
                        req.retry_count,
                        e,
                    )

        for req in drained:
            if req in self._queue:
                self._queue.remove(req)

    async def _execute_queued(self, req: QueuedRequest) -> None:
        """Execute a single queued request. Override for real implementations."""
        logger.debug(
            "Executing queued %s request: %s",
            req.request_type,
            req.request_id,
        )
        # Default: just log it. Real implementation would route to appropriate handler.

    def _prune_expired(self) -> None:
        """Remove expired requests from the queue."""
        before = len(self._queue)
        self._queue = [r for r in self._queue if not r.is_expired]
        pruned = before - len(self._queue)
        if pruned:
            self._total_expired += pruned

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    def get_queue_summary(self) -> dict[str, int]:
        """Get a summary of queued requests by type."""
        summary: dict[str, int] = {}
        for req in self._queue:
            summary[req.request_type] = summary.get(req.request_type, 0) + 1
        return summary

    # ── Response Cache ──────────────────────────────────────────────────

    def cache_response(
        self,
        key: str,
        response: Any,
        ttl_s: float = 1800.0,
    ) -> None:
        """Cache a cloud response for offline replay."""
        if len(self._cache) >= self._cache_max_size:
            # Evict oldest
            self._cache.popitem(last=False)

        self._cache[key] = CachedResponse(
            cache_key=key,
            response=response,
            ttl_s=ttl_s,
        )

    def get_cached(self, key: str) -> Any | None:
        """Retrieve a cached response if available and fresh."""
        entry = self._cache.get(key)
        if entry is None:
            self._cache_misses += 1
            return None

        if entry.is_stale:
            del self._cache[key]
            self._cache_misses += 1
            return None

        self._cache_hits += 1
        # Move to end (LRU)
        self._cache.move_to_end(key)
        return entry.response

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()

    # ── Telemetry ───────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Offline manager statistics."""
        return {
            "state": self._state.value,
            "state_duration_s": time.time() - self._state_since,
            "total_checks": self._total_checks,
            "total_failures": self._total_failures,
            "state_transitions": self._state_transitions,
            "queue_size": len(self._queue),
            "queue_by_type": self.get_queue_summary(),
            "total_queued": self._total_queued,
            "total_drained": self._total_drained,
            "total_expired": self._total_expired,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }
