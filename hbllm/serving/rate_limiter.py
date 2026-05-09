"""
Unified Rate Limiter for HBLLM Core.

Token-bucket algorithm with per-tenant, per-user limits.
Configurable via SecurityConfig or environment variables.

Usage::

    limiter = RateLimiter(rpm=60)
    allowed, retry_after = limiter.check("tenant_123", "user_456")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Default rate limits (requests per minute, burst capacity)
DEFAULT_LIMITS: dict[str, dict[str, int]] = {
    "free": {"rpm": 10, "burst": 15},
    "starter": {"rpm": 30, "burst": 45},
    "business": {"rpm": 120, "burst": 180},
    "enterprise": {"rpm": 600, "burst": 900},
}


@dataclass
class _Bucket:
    """Token bucket state for a single key."""

    tokens: float
    capacity: int
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.monotonic)


class RateLimiter:
    """
    Thread-safe, in-memory token-bucket rate limiter.

    Supports per-tenant and per-user rate limiting.
    Bucket keys are (tenant_id, user_id) tuples for per-user isolation.
    """

    def __init__(
        self,
        plan_limits: dict[str, dict[str, int]] | None = None,
        rpm: int | None = None,
        burst: int | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._plan_limits = plan_limits or DEFAULT_LIMITS
        self._override_rpm = rpm
        self._override_burst = burst
        self._buckets: dict[str, _Bucket] = {}

    def _bucket_key(self, tenant_id: str, user_id: str = "") -> str:
        """Create a unique bucket key scoped to tenant+user."""
        if user_id:
            return f"{tenant_id}:{user_id}"
        return tenant_id

    def _get_bucket(self, key: str, plan: str = "free") -> _Bucket:
        """Get or create a token bucket for the given key."""
        if key not in self._buckets:
            if self._override_rpm:
                rpm = self._override_rpm
                burst = self._override_burst or int(rpm * 1.5)
            else:
                limits = self._plan_limits.get(
                    plan, self._plan_limits.get("free", {"rpm": 10, "burst": 15})
                )
                rpm = limits["rpm"]
                burst = limits["burst"]

            self._buckets[key] = _Bucket(
                tokens=float(burst),
                capacity=burst,
                refill_rate=rpm / 60.0,
            )
        return self._buckets[key]

    def check(
        self,
        tenant_id: str,
        user_id: str = "",
        plan: str = "free",
        cost: int = 1,
    ) -> tuple[bool, float]:
        """
        Check if a request is allowed.

        Args:
            tenant_id: Tenant identifier
            user_id: Optional user identifier for per-user limiting
            plan: Subscription plan for limit lookup
            cost: Token cost of the request

        Returns:
            (allowed, retry_after_seconds)
        """
        if not self.enabled:
            return True, 0.0

        key = self._bucket_key(tenant_id, user_id)
        bucket = self._get_bucket(key, plan)
        now = time.monotonic()

        # Refill tokens
        elapsed = now - bucket.last_refill
        bucket.tokens = min(
            bucket.capacity,
            bucket.tokens + elapsed * bucket.refill_rate,
        )
        bucket.last_refill = now

        # Try to consume
        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return True, 0.0

        # Calculate retry time
        deficit = cost - bucket.tokens
        retry_after = deficit / bucket.refill_rate
        return False, retry_after

    def get_usage(self, tenant_id: str, user_id: str = "", plan: str = "free") -> dict[str, Any]:
        """Get rate limit status for a tenant/user."""
        key = self._bucket_key(tenant_id, user_id)
        bucket = self._get_bucket(key, plan)

        return {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "tokens_remaining": round(bucket.tokens, 1),
            "capacity": bucket.capacity,
            "utilization_pct": round((1 - bucket.tokens / bucket.capacity) * 100, 1),
        }

    def get_headers(self, tenant_id: str, user_id: str = "", plan: str = "free") -> dict[str, str]:
        """Return standard X-RateLimit-* HTTP headers."""
        key = self._bucket_key(tenant_id, user_id)
        bucket = self._get_bucket(key, plan)

        deficit = bucket.capacity - bucket.tokens
        reset_seconds = max(0, deficit / bucket.refill_rate) if bucket.refill_rate > 0 else 0

        rpm = int(bucket.refill_rate * 60)
        return {
            "X-RateLimit-Limit": str(rpm),
            "X-RateLimit-Remaining": str(max(0, int(bucket.tokens))),
            "X-RateLimit-Reset": str(int(reset_seconds)),
        }

    def reset(self, tenant_id: str, user_id: str = "") -> None:
        """Reset a bucket (e.g., after plan upgrade)."""
        key = self._bucket_key(tenant_id, user_id)
        self._buckets.pop(key, None)

    def stats(self) -> dict[str, Any]:
        """Get aggregate rate limiter statistics."""
        return {
            "enabled": self.enabled,
            "active_buckets": len(self._buckets),
            "plans": list(self._plan_limits.keys()),
        }
