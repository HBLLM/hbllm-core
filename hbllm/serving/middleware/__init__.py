"""
Request ID Middleware — injects a unique X-Request-ID into every request/response.

Enables end-to-end request tracing through logs, error responses, and monitoring.
If the client sends an X-Request-ID header, it is preserved; otherwise a new UUID is generated.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every request and response."""

    HEADER_NAME = "X-Request-ID"

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Prefer client-provided request ID, else generate one
        request_id = request.headers.get(self.HEADER_NAME) or str(uuid.uuid4())

        # Store on request.state for downstream use (error handlers, logging)
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[self.HEADER_NAME] = request_id
        return response
