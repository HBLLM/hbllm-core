"""
HTTP Rate Limiting Middleware — per-tenant token bucket for API endpoints.

Supports two backends:
  - In-memory (default): fast, single-instance only
  - Redis (multi-replica): cluster-safe sliding window counter

Prevents any single tenant from flooding the API. Uses configurable RPM
and burst limits per tenant.

Skip list:
  - Health check endpoints (/health, /health/live, /health/ready)
  - Metrics endpoints (/metrics, /routing/stats)
  - CORS preflight (OPTIONS)
  - Static file paths (/admin/static)
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Paths exempt from rate limiting
_EXEMPT_PATHS = frozenset(
    {
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/routing/stats",
        "/docs",
        "/openapi.json",
    }
)

_EXEMPT_PREFIXES = (
    "/admin/static",
    "/studio/",
)


# ── Rate Limit Backend Protocol ──────────────────────────────────────────────


class RateLimitBackend(Protocol):
    """Protocol for pluggable rate limiting backends."""

    def try_consume(self, tenant_id: str, rpm: float, burst: float) -> bool:
        """Try to consume one token. Returns True if allowed."""
        ...

    @property
    def retry_after(self) -> float:
        """Seconds until next token is available (after a failed consume)."""
        ...


# ── In-Memory Backend (Single Instance) ──────────────────────────────────────


class _TenantBucket:
    """Token bucket for a single tenant."""

    __slots__ = ("tokens", "last_refill", "rpm", "burst")

    def __init__(self, rpm: float, burst: float) -> None:
        self.rpm = rpm
        self.burst = burst
        self.tokens = burst
        self.last_refill = time.monotonic()

    def try_consume(self) -> bool:
        """Try to consume one token. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        # Refill: rpm/60 tokens per second
        self.tokens = min(self.burst, self.tokens + elapsed * (self.rpm / 60.0))
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    @property
    def retry_after(self) -> float:
        """Seconds until one token is available."""
        if self.tokens >= 1.0:
            return 0.0
        deficit = 1.0 - self.tokens
        return deficit / (self.rpm / 60.0)


class InMemoryRateLimitBackend:
    """In-memory token bucket backend. Fast but single-instance only."""

    def __init__(self) -> None:
        self._buckets: dict[str, _TenantBucket] = {}
        self._last_retry_after: float = 0.0

    def try_consume(self, tenant_id: str, rpm: float, burst: float) -> bool:
        if tenant_id not in self._buckets:
            self._buckets[tenant_id] = _TenantBucket(rpm=rpm, burst=burst)
        bucket = self._buckets[tenant_id]
        result = bucket.try_consume()
        if not result:
            self._last_retry_after = bucket.retry_after
        return result

    @property
    def retry_after(self) -> float:
        return self._last_retry_after


# ── Redis Backend (Multi-Replica) ────────────────────────────────────────────


class RedisRateLimitBackend:
    """Redis-backed sliding window rate limiter for multi-replica deployments.

    Uses a simple INCR + EXPIRE pattern with a 60-second window.
    Each tenant gets a key ``rl:{tenant_id}`` that counts requests
    in the current minute window. Atomic and cluster-safe.
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._redis: Any = None
        self._last_retry_after: float = 1.0

    def _get_redis(self) -> Any:
        """Lazy-initialize Redis connection."""
        if self._redis is None:
            try:
                import redis as redis_lib

                self._redis = redis_lib.from_url(self._redis_url, decode_responses=True)
            except Exception as e:
                logger.error("[RateLimit] Failed to connect to Redis: %s", e)
                raise
        return self._redis

    def try_consume(self, tenant_id: str, rpm: float, burst: float) -> bool:
        """Consume one token using Redis INCR with 60s sliding window."""
        try:
            r = self._get_redis()
            key = f"rl:{tenant_id}"

            # Atomic increment
            count = r.incr(key)

            # Set expiry on first request in window
            if count == 1:
                r.expire(key, 60)

            # Check against limit (use burst as the cap for the window)
            if count > int(burst):
                ttl = r.ttl(key)
                self._last_retry_after = max(1.0, float(ttl)) if ttl > 0 else 1.0
                return False

            return True

        except Exception as e:
            # Redis failure: fail-open (allow request) and log
            logger.warning("[RateLimit] Redis error, failing open: %s", e)
            return True

    @property
    def retry_after(self) -> float:
        return self._last_retry_after


# ── Backend Factory ──────────────────────────────────────────────────────────


def _create_backend(
    redis_url: str | None = None,
) -> InMemoryRateLimitBackend | RedisRateLimitBackend:
    """Create the appropriate rate limit backend.

    Uses Redis if ``redis_url`` is provided or ``HBLLM_REDIS_URL`` env var is set
    and the system is running in multi-replica mode (``HBLLM_BUS_BACKEND=redis``).
    Falls back to in-memory otherwise.
    """
    url = redis_url or os.environ.get("HBLLM_REDIS_URL", "")
    bus_backend = os.environ.get("HBLLM_BUS_BACKEND", "")

    if url and bus_backend == "redis":
        try:
            backend = RedisRateLimitBackend(url)
            backend._get_redis()  # Verify connection
            logger.info("[RateLimit] Using Redis backend at %s", url.split("@")[-1])
            return backend
        except Exception as e:
            logger.warning("[RateLimit] Redis unavailable (%s), falling back to in-memory", e)

    logger.info("[RateLimit] Using in-memory backend (single-instance)")
    return InMemoryRateLimitBackend()


# ── Middleware ───────────────────────────────────────────────────────────────


class HTTPRateLimitMiddleware(BaseHTTPMiddleware):
    """Per-tenant rate limiting middleware for FastAPI.

    Uses a token bucket algorithm with configurable RPM and burst multiplier.
    Tenant ID is read from ``request.state.tenant_id`` (set by JWTAuthMiddleware).

    Returns 429 Too Many Requests with Retry-After header when exceeded.

    Supports both in-memory (single-instance) and Redis (multi-replica) backends.
    The backend is selected automatically based on environment configuration.
    """

    def __init__(
        self,
        app: Any,
        default_rpm: float = 60.0,
        burst_multiplier: float = 1.5,
        tenant_limits: dict[str, float] | None = None,
        redis_url: str | None = None,
    ) -> None:
        """
        Args:
            app: The ASGI application.
            default_rpm: Default requests per minute per tenant.
            burst_multiplier: Burst allowance multiplier over RPM.
            tenant_limits: Optional per-tenant RPM overrides.
            redis_url: Optional Redis URL for multi-replica rate limiting.
        """
        super().__init__(app)
        self.default_rpm = default_rpm
        self.burst_multiplier = burst_multiplier
        self.tenant_limits = tenant_limits or {}
        self._backend = _create_backend(redis_url)

        # Keep legacy _buckets reference for test backward compatibility
        if isinstance(self._backend, InMemoryRateLimitBackend):
            self._buckets = self._backend._buckets

    def _get_rpm(self, tenant_id: str) -> float:
        """Get the RPM limit for a tenant."""
        return self.tenant_limits.get(tenant_id, self.default_rpm)

    def _get_bucket(self, tenant_id: str) -> _TenantBucket:
        """Get or create a bucket for a tenant (backward compatibility).

        Only works with the in-memory backend. Used by existing tests.
        """
        if isinstance(self._backend, InMemoryRateLimitBackend):
            rpm = self._get_rpm(tenant_id)
            burst = rpm * self.burst_multiplier
            # Trigger bucket creation via try_consume if needed
            if tenant_id not in self._backend._buckets:
                self._backend._buckets[tenant_id] = _TenantBucket(rpm=rpm, burst=burst)
            return self._backend._buckets[tenant_id]
        raise RuntimeError("_get_bucket() is only available with the in-memory backend")

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Skip exempt paths
        path = request.url.path
        if (
            request.method == "OPTIONS"
            or path in _EXEMPT_PATHS
            or path.startswith(_EXEMPT_PREFIXES)
        ):
            return await call_next(request)

        # Get tenant ID (set by JWTAuthMiddleware upstream)
        tenant_id = getattr(request.state, "tenant_id", None)
        if not tenant_id:
            # No tenant context yet — let auth middleware handle it
            return await call_next(request)

        # System tenant is exempt
        if tenant_id == "system":
            return await call_next(request)

        rpm = self._get_rpm(tenant_id)
        burst = rpm * self.burst_multiplier

        if not self._backend.try_consume(tenant_id, rpm, burst):
            retry_after = max(1, int(self._backend.retry_after))
            logger.warning(
                "[RateLimit] Tenant %s exceeded %d RPM (retry_after=%ds)",
                tenant_id,
                int(rpm),
                retry_after,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": f"Rate limit exceeded. Retry after {retry_after}s.",
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
