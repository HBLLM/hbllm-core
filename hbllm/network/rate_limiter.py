"""
Rate Limiter Interceptor for MessageBus.

Implements a Token Bucket / Leaky Bucket style Quality of Service (QoS)
enforcer to prevent "Noisy Neighbor" starvation across tenants.
"""

from __future__ import annotations

import asyncio
import logging
import time

from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


class RateLimitInterceptor:
    """
    Proactively intercepts messages on the bus and throttles them if a tenant
    exceeds their allowed request quotas.
    """

    def __init__(self, target_rpm: float = 60.0, burst_multiplier: float = 1.5):
        """
        Args:
            target_rpm: Target maximum Requests Per Minute (steady state).
            burst_multiplier: Allowance for bursts beyond the target RPM.
        """
        self.target_rpm = target_rpm
        self.burst_limit = target_rpm * burst_multiplier
        self.tokens: dict[str, float] = {}
        self.last_refill: dict[str, float] = {}
        self.lock = asyncio.Lock()

    async def intercept(self, message: Message) -> Message | None:
        """
        Evaluate if the message should be allowed based on rate limits.

        Returns:
            The original message if allowed. None if dropped (rate limited).
        """
        tenant = message.tenant_id
        if not tenant or tenant == "system":
            return message

        now = time.monotonic()

        async with self.lock:
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
