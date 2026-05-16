"""
Input validation and sanitization middleware for HBLLM API.
"""

import logging
import re
from collections.abc import Awaitable, Callable

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RequestSizeLimiter(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int = 10 * 1024 * 1024):  # Default 10MB
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_upload_size:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "detail": f"Payload too large. Max size is {self.max_upload_size} bytes."
                        },
                    )
            except ValueError:
                pass
        return await call_next(request)


class ContentTypeValidator(BaseHTTPMiddleware):
    def __init__(self, app, allowed_types: list[str] | None = None):
        super().__init__(app)
        self.allowed_types = allowed_types or ["application/json"]

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        if request.method in ("POST", "PUT", "PATCH"):
            content_type = request.headers.get("content-type", "")
            if not any(allowed in content_type for allowed in self.allowed_types):
                return JSONResponse(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    content={
                        "detail": f"Unsupported media type. Allowed types: {', '.join(self.allowed_types)}"
                    },
                )
        return await call_next(request)


class InputSanitizer(BaseHTTPMiddleware):
    """
    Basic prompt injection and sanitization checks on incoming JSON bodies.
    """

    # Very basic patterns for generic injection attempts (system prompt bypass)
    BLOCKED_PATTERNS = [
        re.compile(r"ignore (?:all )?(?:previous )?instructions", re.IGNORECASE),
        re.compile(r"disregard (?:all )?(?:previous )?(?:instructions|directions)", re.IGNORECASE),
        re.compile(r"you are (?:now )?a(?:n)? (?:assistant|expert|AI|developer)", re.IGNORECASE),
        re.compile(r"system directive overwrite", re.IGNORECASE),
        re.compile(r"\[system\]", re.IGNORECASE),
        re.compile(r"(?:Do Anything Now|DAN) mode", re.IGNORECASE),
        re.compile(r"forget (?:everything|all instructions)", re.IGNORECASE),
        re.compile(r"\\n\\n(?:Human|Assistant|System):", re.IGNORECASE),
        re.compile(r"print your (?:initial )?prompt", re.IGNORECASE),
    ]

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        if request.method in ("POST", "PUT", "PATCH"):
            # We must consume the body to check it, then put it back
            try:
                body_bytes = await request.body()
                if body_bytes:
                    body_text = body_bytes.decode("utf-8", errors="ignore")

                    for pattern in self.BLOCKED_PATTERNS:
                        if pattern.search(body_text):
                            logger.warning(
                                "Potential prompt injection detected: %s", pattern.pattern
                            )
                            return JSONResponse(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                content={"detail": "Request rejected by input sanitizer."},
                            )

                    # Put body back for downstream consumers
                    async def receive():
                        return {"type": "http.request", "body": body_bytes}

                    request._receive = receive
            except Exception as e:
                logger.error("Error sanitizing input: %s", e)

        return await call_next(request)
