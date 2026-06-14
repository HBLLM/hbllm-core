"""
Security Headers Middleware — Defense-in-Depth HTTP Header Hardening.

Adds security headers to every response. These are applied at the
application level as defense-in-depth (even if Nginx also sets them).

Headers added:
  - Strict-Transport-Security (HSTS)
  - X-Content-Type-Options
  - X-Frame-Options
  - X-XSS-Protection
  - Referrer-Policy
  - Content-Security-Policy
  - Permissions-Policy
  - Cache-Control (for API responses)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Default security headers applied to every response
DEFAULT_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
}

# API responses should not be cached by default
API_CACHE_CONTROL = "no-store, no-cache, must-revalidate, max-age=0"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that adds security headers to all responses.

    Works alongside Nginx headers as defense-in-depth. Application-level
    headers ensure protection even when accessed directly (dev, testing).

    Parameters:
        hsts: Enable HSTS header. Set False for local development.
        custom_headers: Additional headers to merge with defaults.
    """

    def __init__(
        self,
        app: object,
        *,
        hsts: bool = True,
        custom_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._headers = dict(DEFAULT_SECURITY_HEADERS)
        if not hsts:
            self._headers.pop("Strict-Transport-Security", None)
        if custom_headers:
            self._headers.update(custom_headers)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)

        # Apply security headers
        for header, value in self._headers.items():
            response.headers.setdefault(header, value)

        # API responses should not be cached
        path = request.url.path
        if path.startswith("/v1/") or path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", API_CACHE_CONTROL)

        return response
