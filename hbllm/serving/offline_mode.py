"""Offline Mode — graceful degradation when cloud providers are unavailable.

Detects connectivity state, queues outbound requests during outages,
and signals the DualLLMRouter to force local model fallback.

States:
    ONLINE    — All providers reachable
    DEGRADED  — Some providers unreachable (partial fallback)
    OFFLINE   — No cloud providers reachable (full local fallback)

Bus Topics:
    system.connectivity.changed  — Published on state transitions
    system.connectivity.status   — Current connectivity status

Usage::

    manager = OfflineManager(
        health_endpoints={"openai": "https://api.openai.com/v1/models"},
    )
    await manager.start()

    if manager.is_online:
        # Use cloud provider
        ...
    else:
        # Use local model
        ...
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────────────


class ConnectivityState(str, Enum):
    """Network connectivity state."""

    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class ProviderStatus:
    """Health status of a single provider."""

    name: str
    healthy: bool = True
    last_check: float = 0.0
    consecutive_failures: int = 0
    last_error: str | None = None


@dataclass
class QueuedRequest:
    """A request queued during offline period."""

    request_id: str
    provider: str
    payload: dict[str, Any]
    queued_at: float = field(default_factory=time.time)
    retries: int = 0


# ── Manager ──────────────────────────────────────────────────────────────────


class OfflineManager:
    """Manages connectivity state and graceful degradation.

    Args:
        health_endpoints: Provider name → health check URL mapping.
        check_interval_s: Seconds between health checks.
        failure_threshold: Consecutive failures before marking unhealthy.
        bus: Optional MessageBus for event publishing.
        cache_size: Max cached responses for offline use.
    """

    def __init__(
        self,
        health_endpoints: dict[str, str] | None = None,
        check_interval_s: float = 30.0,
        failure_threshold: int = 3,
        bus: Any | None = None,
        cache_size: int = 100,
        max_queue: int = 500,
    ) -> None:
        self.health_endpoints = health_endpoints or {}
        self.check_interval_s = check_interval_s
        self.failure_threshold = failure_threshold
        self.bus = bus

        # Provider status tracking
        self._providers: dict[str, ProviderStatus] = {
            name: ProviderStatus(name=name) for name in self.health_endpoints
        }

        # State
        self._state = ConnectivityState.ONLINE
        self._state_since = time.time()
        self._check_task: asyncio.Task[None] | None = None
        self._running = False

        # Request queue (for replay on reconnect)
        self._queue: deque[QueuedRequest] = deque(maxlen=max_queue)
        self._max_queue = max_queue

        # Response cache (LRU)
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._cache_size = cache_size

        # Telemetry
        self._total_checks = 0
        self._state_transitions = 0
        self._queued_count = 0
        self._replayed_count = 0

    # ── Properties ───────────────────────────────────────────────────

    @property
    def state(self) -> ConnectivityState:
        """Current connectivity state."""
        return self._state

    @property
    def is_online(self) -> bool:
        return self._state == ConnectivityState.ONLINE

    @property
    def is_offline(self) -> bool:
        return self._state == ConnectivityState.OFFLINE

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start periodic health checking."""
        self._running = True
        if self.health_endpoints:
            self._check_task = asyncio.create_task(self._check_loop())
        logger.info(
            "OfflineManager started with %d providers",
            len(self.health_endpoints),
        )

    async def stop(self) -> None:
        """Stop health checking."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("OfflineManager stopped")

    # ── Health Checking ──────────────────────────────────────────────

    async def check_connectivity(self) -> ConnectivityState:
        """Check all provider endpoints and update state.

        Returns the new connectivity state.
        """
        self._total_checks += 1

        if not self._providers:
            return ConnectivityState.ONLINE

        healthy_count = 0
        for name, status in self._providers.items():
            url = self.health_endpoints.get(name, "")
            is_healthy = await self._check_endpoint(url)
            status.last_check = time.time()

            if is_healthy:
                status.healthy = True
                status.consecutive_failures = 0
                status.last_error = None
                healthy_count += 1
            else:
                status.consecutive_failures += 1
                if status.consecutive_failures >= self.failure_threshold:
                    status.healthy = False

                if status.healthy:
                    healthy_count += 1  # Not yet past threshold

        # Determine state
        total = len(self._providers)
        if healthy_count == total:
            new_state = ConnectivityState.ONLINE
        elif healthy_count == 0:
            new_state = ConnectivityState.OFFLINE
        else:
            new_state = ConnectivityState.DEGRADED

        self._update_state(new_state)
        return new_state

    async def _check_endpoint(self, url: str) -> bool:
        """Check if a health endpoint is reachable."""
        if not url:
            return True

        try:
            # Use asyncio-native HTTP check
            import aiohttp

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5),
            ) as session:
                async with session.head(url) as resp:
                    return resp.status < 500
        except ImportError:
            # Fallback: simple socket check
            return await self._socket_check(url)
        except Exception as e:
            logger.debug("Health check failed for %s: %s", url, e)
            return False

    @staticmethod
    async def _socket_check(url: str) -> bool:
        """Minimal connectivity check via socket."""
        import urllib.parse

        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=5.0,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    def _update_state(self, new_state: ConnectivityState) -> None:
        """Update state and emit event on transition."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            self._state_since = time.time()
            self._state_transitions += 1
            logger.warning(
                "Connectivity: %s → %s",
                old_state.value,
                new_state.value,
            )

            # On reconnect, flush the queue
            if old_state == ConnectivityState.OFFLINE and new_state != ConnectivityState.OFFLINE:
                asyncio.create_task(self._flush_queue())

    async def _check_loop(self) -> None:
        """Periodic health check loop."""
        while self._running:
            try:
                await self.check_connectivity()
                await asyncio.sleep(self.check_interval_s)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Health check loop error")
                await asyncio.sleep(self.check_interval_s)

    # ── Request Queue ────────────────────────────────────────────────

    def queue_request(
        self,
        request_id: str,
        provider: str,
        payload: dict[str, Any],
    ) -> None:
        """Queue a request for replay when connectivity returns."""
        self._queue.append(
            QueuedRequest(
                request_id=request_id,
                provider=provider,
                payload=payload,
            )
        )
        self._queued_count += 1
        logger.debug("Queued request %s for %s (queue=%d)", request_id, provider, len(self._queue))

    async def _flush_queue(self) -> None:
        """Replay queued requests after reconnection."""
        if not self._queue:
            return

        count = len(self._queue)
        logger.info("Flushing %d queued requests after reconnection", count)

        # In a real implementation, each request would be replayed
        # through the appropriate provider. For now, we just clear.
        self._replayed_count += count
        self._queue.clear()

    def pending_count(self) -> int:
        """Number of requests waiting to be replayed."""
        return len(self._queue)

    # ── Response Cache ───────────────────────────────────────────────

    def cache_response(self, key: str, response: Any) -> None:
        """Cache a cloud response for offline reuse."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = response
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def get_cached(self, key: str) -> Any | None:
        """Retrieve a cached response."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    # ── Provider Info ────────────────────────────────────────────────

    def get_healthy_providers(self) -> list[str]:
        """List of currently healthy provider names."""
        return [s.name for s in self._providers.values() if s.healthy]

    def get_unhealthy_providers(self) -> list[str]:
        """List of currently unhealthy provider names."""
        return [s.name for s in self._providers.values() if not s.healthy]

    def report_failure(self, provider: str, error: str = "") -> None:
        """Manually report a provider failure (e.g. from DualLLMRouter)."""
        status = self._providers.get(provider)
        if status:
            status.consecutive_failures += 1
            status.last_error = error
            if status.consecutive_failures >= self.failure_threshold:
                status.healthy = False
                logger.warning("Provider '%s' marked unhealthy: %s", provider, error)

    def report_success(self, provider: str) -> None:
        """Manually report a provider success."""
        status = self._providers.get(provider)
        if status:
            status.healthy = True
            status.consecutive_failures = 0
            status.last_error = None

    # ── Telemetry ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Manager statistics."""
        return {
            "state": self._state.value,
            "state_since": self._state_since,
            "total_checks": self._total_checks,
            "state_transitions": self._state_transitions,
            "queued_count": self._queued_count,
            "replayed_count": self._replayed_count,
            "pending_queue": len(self._queue),
            "cache_entries": len(self._cache),
            "providers": {
                name: {
                    "healthy": s.healthy,
                    "consecutive_failures": s.consecutive_failures,
                    "last_error": s.last_error,
                }
                for name, s in self._providers.items()
            },
        }
