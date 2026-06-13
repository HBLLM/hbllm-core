"""
Structured Error Handling for HBLLM Serving Layer.

Provides:
  - HBLLMError: Base exception with error codes and safe messages
  - Structured error response format for API consumers
  - Error sanitization to prevent leaking internal details
"""

from __future__ import annotations

import logging
import uuid
from enum import Enum
from typing import Any

from fastapi import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Standardized error codes for API responses."""

    # Client errors (4xx)
    INVALID_REQUEST = "INVALID_REQUEST"
    MISSING_FIELD = "MISSING_FIELD"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    INPUT_REJECTED = "INPUT_REJECTED"
    POLICY_VIOLATION = "POLICY_VIOLATION"

    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    PIPELINE_ERROR = "PIPELINE_ERROR"
    PIPELINE_TIMEOUT = "PIPELINE_TIMEOUT"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    BRAIN_DEGRADED = "BRAIN_DEGRADED"
    GATEWAY_ERROR = "GATEWAY_ERROR"


class HBLLMError(Exception):
    """Base exception for HBLLM API errors.

    Carries a safe, user-facing message and an internal detail
    that is only logged (never sent to the client).
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = 500,
        *,
        internal_detail: str = "",
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.internal_detail = internal_detail
        super().__init__(message)


def error_response(
    code: ErrorCode,
    message: str,
    status_code: int,
    *,
    request_id: str = "",
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build a standardized JSON error response."""
    body: dict[str, Any] = {
        "error": {
            "code": code.value,
            "message": message,
        },
    }
    if request_id:
        body["error"]["request_id"] = request_id
    if extra:
        body["error"].update(extra)
    return JSONResponse(status_code=status_code, content=body)


def sanitize_exception_message(exc: Exception) -> str:
    """Return a safe error message that doesn't leak internals.

    Internal details like file paths, stack frames, SQL errors, and
    provider API keys are stripped out.
    """
    msg = str(exc)

    # Strip common internal leak patterns
    leak_indicators = [
        "Traceback",
        "/Users/",
        "/home/",
        "/app/",
        'File "',
        "line ",
        "sqlite3.",
        "asyncpg.",
        "sqlalchemy.",
        "KeyError:",
        "AttributeError:",
        "sk-",  # API key prefixes
        "Bearer ",
    ]

    for indicator in leak_indicators:
        if indicator in msg:
            return "An internal error occurred. Please try again or contact support."

    # Truncate long messages
    if len(msg) > 200:
        return msg[:200] + "..."

    return msg


# ─── FastAPI Exception Handlers ──────────────────────────────────────────────


async def hbllm_error_handler(request: Request, exc: HBLLMError) -> JSONResponse:
    """Handle HBLLMError exceptions with structured responses."""
    request_id = getattr(request.state, "request_id", "")

    if exc.internal_detail:
        logger.error(
            "HBLLMError [%s] request_id=%s: %s | internal: %s",
            exc.code.value,
            request_id,
            exc.message,
            exc.internal_detail,
        )
    else:
        logger.warning("HBLLMError [%s] request_id=%s: %s", exc.code.value, request_id, exc.message)

    return error_response(exc.code, exc.message, exc.status_code, request_id=request_id)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler that prevents leaking internal details."""
    request_id = getattr(request.state, "request_id", "")

    logger.exception(
        "Unhandled exception request_id=%s path=%s: %s",
        request_id,
        request.url.path,
        exc,
    )

    safe_message = sanitize_exception_message(exc)
    return error_response(
        ErrorCode.INTERNAL_ERROR,
        safe_message,
        500,
        request_id=request_id,
    )
