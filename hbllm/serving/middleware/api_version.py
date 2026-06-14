"""
API Version Header Middleware.

Injects `X-API-Version` response header on all requests.
Validates `Accept-Version` request header if present.

Current API version: v1.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Supported API versions
CURRENT_VERSION = "v1"
SUPPORTED_VERSIONS = {"v1"}


class APIVersionMiddleware(BaseHTTPMiddleware):
    """Middleware that adds API version headers and validates version requests."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Check Accept-Version header if present
        requested_version = request.headers.get("Accept-Version")
        if requested_version and requested_version not in SUPPORTED_VERSIONS:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "unsupported_api_version",
                    "message": f"API version '{requested_version}' is not supported",
                    "supported_versions": sorted(SUPPORTED_VERSIONS),
                    "current_version": CURRENT_VERSION,
                },
            )

        response = await call_next(request)

        # Inject version headers on all responses
        response.headers["X-API-Version"] = CURRENT_VERSION
        response.headers["X-Supported-Versions"] = ", ".join(sorted(SUPPORTED_VERSIONS))

        return response
