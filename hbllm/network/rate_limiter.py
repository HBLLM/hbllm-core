"""
Rate Limiter Interceptor for MessageBus.

Implements a Token Bucket / Leaky Bucket style Quality of Service (QoS)
enforcer to prevent "Noisy Neighbor" starvation across tenants.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict

from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


class RateLimitInterceptor:
    """
    Proactively intercepts messages on the bus and throttles them if a tenant
    exceeds their allowed request quotas.

    Uses per-tenant locks instead of a global lock to avoid cross-tenant
    serialisation bottlenecks.
    """

    def __init__(
        self,
        target_rpm: float = 60.0,
        burst_multiplier: float = 1.5,
        max_tracked_tenants: int = 1000,
    ):
        """
        Args:
            target_rpm: Target maximum Requests Per Minute (steady state).
            burst_multiplier: Allowance for bursts beyond the target RPM.
            max_tracked_tenants: LRU cap on per-tenant tracking data.
        """
        self.target_rpm = target_rpm
        self.burst_limit = target_rpm * burst_multiplier
        self._max_tracked_tenants = max_tracked_tenants

        # Per-tenant state — protected by per-tenant locks
        self.tokens: dict[str, float] = {}
        self.last_refill: dict[str, float] = {}
        self._locks: OrderedDict[str, asyncio.Lock] = OrderedDict()

    def _get_lock(self, tenant: str) -> asyncio.Lock:
        """Get or create a per-tenant lock with LRU eviction."""
        if tenant in self._locks:
            self._locks.move_to_end(tenant)
            return self._locks[tenant]

        # Evict LRU entries if at capacity
        while len(self._locks) >= self._max_tracked_tenants:
            evicted_tenant, _ = self._locks.popitem(last=False)
            self.tokens.pop(evicted_tenant, None)
            self.last_refill.pop(evicted_tenant, None)

        lock = asyncio.Lock()
        self._locks[tenant] = lock
        return lock

    async def intercept(self, message: Message) -> Message | None:
        """
        Evaluate if the message should be allowed based on rate limits.

        Returns:
            The original message if allowed. None if dropped (rate limited).
        """
        tenant = message.tenant_id
        if not tenant or tenant == "system":
            return message

        # Exempt high-frequency streaming topics from rate limiting
        topic = getattr(message, "topic", "") or ""
        if topic.startswith("sensory.audio") or topic.startswith("sensory.transcription"):
            return message

        now = time.monotonic()
        lock = self._get_lock(tenant)

        async with lock:
            # Initialize bucket for new tenant
            if tenant not in self.tokens:
                self.tokens[tenant] = self.burst_limit
                self.last_refill[tenant] = now

            # Refill tokens based on elapsed time (1 RPM = 1/60 tokens per second)
            elapsed = now - self.last_refill[tenant]
            refill_amount = elapsed * (self.target_rpm / 60.0)

            self.tokens[tenant] = min(self.burst_limit, self.tokens[tenant] + refill_amount)
            self.last_refill[tenant] = now

            # Evaluate QoS drop
            if self.tokens[tenant] >= 1.0:
                self.tokens[tenant] -= 1.0
                return message
            else:
                logger.warning(
                    "[RateLimiter] Dropping message from %s (QoS quota exceeded)", tenant
                )
                return None
