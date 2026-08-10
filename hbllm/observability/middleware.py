"""
OpenTelemetry ASGI Tracing Middleware for HBLLM.

Creates a span for each HTTP request with tenant context, request ID,
and status code. Wires into the FastAPI middleware stack alongside
the existing Prometheus middleware.

If tracing is not initialized (OTEL_ENABLED != 1), this middleware
passes through all requests without overhead.

Usage::

    from hbllm.observability.middleware import TracingMiddleware

    app.add_middleware(TracingMiddleware)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Paths that don't need tracing (health checks, metrics, static)
_SKIP_PATHS = frozenset(
    {
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/metrics/prometheus",
    }
)

_SKIP_PREFIXES = (
    "/admin/static",
    "/studio/static",
    "/portal/static",
)


class TracingMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that creates OpenTelemetry spans for HTTP requests.

    Span attributes:
      - ``http.method``: Request method
      - ``http.url``: Full request URL
      - ``http.status_code``: Response status code
      - ``http.route``: URL path
      - ``hbllm.tenant_id``: Tenant ID (from auth middleware)
      - ``hbllm.request_id``: Request ID (from request state)
      - ``hbllm.duration_ms``: Request duration in milliseconds
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        from hbllm.observability.tracing import _OTEL_AVAILABLE

        # Fast path: if tracing is disabled, pass through
        if not _OTEL_AVAILABLE:
            return await call_next(request)

        # Skip health checks and static paths
        path = request.url.path
        if path in _SKIP_PATHS or path.startswith(_SKIP_PREFIXES):
            return await call_next(request)

        # Build span attributes
        attributes: dict[str, Any] = {
            "http.method": request.method,
            "http.url": str(request.url),
            "http.route": path,
        }

        # Add tenant context if available
        tenant_id = getattr(request.state, "tenant_id", None)
        if tenant_id:
            attributes["hbllm.tenant_id"] = tenant_id

        request_id = getattr(request.state, "request_id", None)
        if request_id:
            attributes["hbllm.request_id"] = request_id

        from hbllm.observability.tracing import trace_span

        start = time.monotonic()
        with trace_span(f"{request.method} {path}", attributes=attributes) as span:
            try:
                response = await call_next(request)
                duration_ms = (time.monotonic() - start) * 1000

                # Record response attributes on the span
                span.set_attribute("http.status_code", str(response.status_code))
                span.set_attribute("hbllm.duration_ms", f"{duration_ms:.1f}")

                return response

            except Exception as exc:
                duration_ms = (time.monotonic() - start) * 1000
                span.set_attribute("hbllm.duration_ms", f"{duration_ms:.1f}")
                span.record_exception(exc)
                raise
