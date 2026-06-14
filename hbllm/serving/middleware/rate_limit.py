"""
HTTP Rate Limiting Middleware — per-tenant token bucket for API endpoints.

Prevents any single tenant from flooding the API. Uses an in-memory
token bucket per tenant with configurable RPM and burst limits.

Skip list:
  - Health check endpoints (/health, /health/live, /health/ready)
  - Metrics endpoints (/metrics, /routing/stats)
  - CORS preflight (OPTIONS)
  - Static file paths (/admin/static)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

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


class HTTPRateLimitMiddleware(BaseHTTPMiddleware):
    """Per-tenant rate limiting middleware for FastAPI.

    Uses a token bucket algorithm with configurable RPM and burst multiplier.
    Tenant ID is read from ``request.state.tenant_id`` (set by JWTAuthMiddleware).

    Returns 429 Too Many Requests with Retry-After header when exceeded.
    """

    def __init__(
        self,
        app: Any,
        default_rpm: float = 60.0,
        burst_multiplier: float = 1.5,
        tenant_limits: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            app: The ASGI application.
            default_rpm: Default requests per minute per tenant.
            burst_multiplier: Burst allowance multiplier over RPM.
            tenant_limits: Optional per-tenant RPM overrides.
        """
        super().__init__(app)
        self.default_rpm = default_rpm
        self.burst_multiplier = burst_multiplier
        self.tenant_limits = tenant_limits or {}
        self._buckets: dict[str, _TenantBucket] = {}

    def _get_bucket(self, tenant_id: str) -> _TenantBucket:
        """Get or create a bucket for a tenant."""
        if tenant_id not in self._buckets:
            rpm = self.tenant_limits.get(tenant_id, self.default_rpm)
            burst = rpm * self.burst_multiplier
            self._buckets[tenant_id] = _TenantBucket(rpm=rpm, burst=burst)
        return self._buckets[tenant_id]

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

        bucket = self._get_bucket(tenant_id)
        if not bucket.try_consume():
            retry_after = max(1, int(bucket.retry_after))
            logger.warning(
                "[RateLimit] Tenant %s exceeded %d RPM (retry_after=%ds)",
                tenant_id,
                int(bucket.rpm),
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
