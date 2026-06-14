"""
Audit Logging Middleware — Automatic Request/Response Audit Trail.

Integrates the existing AuditLog into the ASGI middleware stack,
automatically recording every API request with:
  - Tenant & user identity
  - HTTP method, path, status code
  - Request latency
  - IP address and user agent
  - Action classification (auth, chat, admin, etc.)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from hbllm.security.audit_log import AuditAction, AuditLog, AuditSeverity

logger = logging.getLogger(__name__)

# Paths that should NOT be audit-logged (high-frequency, low-value)
SKIP_PATHS = frozenset(
    {
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/favicon.ico",
    }
)

# Map path prefixes to audit actions for richer classification
PATH_ACTION_MAP: list[tuple[str, str, AuditAction]] = [
    ("POST", "/v1/chat", AuditAction.CHAT_MESSAGE),
    ("GET", "/v1/chat", AuditAction.DATA_ACCESSED),
    ("GET", "/v1/conversations", AuditAction.DATA_ACCESSED),
    ("POST", "/v1/memory", AuditAction.DATA_CREATED),
    ("PUT", "/v1/memory", AuditAction.DATA_UPDATED),
    ("DELETE", "/v1/memory", AuditAction.DATA_DELETED),
    ("POST", "/v1/tools", AuditAction.TOOL_EXECUTED),
    ("POST", "/v1/admin", AuditAction.ADMIN_ACTION),
    ("PUT", "/v1/admin", AuditAction.ADMIN_ACTION),
    ("GET", "/v1/admin/audit", AuditAction.DATA_ACCESSED),
    ("POST", "/v1/data/export", AuditAction.DATA_EXPORTED),
]


class AuditMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that logs every API request to the AuditLog.

    Automatically captures:
      - Identity context (tenant_id, user_id from request.state)
      - Request metadata (method, path, IP, user agent)
      - Response status and latency
      - Classified action type

    Failed requests (4xx/5xx) are logged with WARNING/CRITICAL severity.
    """

    def __init__(self, app: object, audit_log: AuditLog) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.audit_log = audit_log

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path

        # Skip high-frequency health/metrics endpoints
        if path in SKIP_PATHS:
            return await call_next(request)

        start = time.monotonic()
        status_code = 500  # Default in case of unhandled exception

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.monotonic() - start
            self._record(request, status_code, duration)

    def _record(self, request: Request, status_code: int, duration: float) -> None:
        """Record the request in the audit log."""
        method = request.method
        path = request.url.path

        # Extract identity from request.state (set by auth middleware)
        tenant_id = getattr(request.state, "tenant_id", "anonymous")
        user_id = getattr(request.state, "user_id", "")

        # Classify the action
        action = self._classify_action(method, path)

        # Determine severity based on status code
        if status_code >= 500:
            severity = AuditSeverity.CRITICAL
        elif status_code >= 400:
            severity = AuditSeverity.WARNING
        else:
            severity = AuditSeverity.INFO

        # Get client info
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")[:256]

        try:
            self.audit_log.log(
                action=action,
                tenant_id=tenant_id,
                user_id=user_id,
                actor=user_id or "anonymous",
                resource=f"{method} {path}",
                ip_address=ip_address,
                user_agent=user_agent,
                severity=severity,
                success=status_code < 400,
                details={
                    "status_code": status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "method": method,
                    "path": path,
                    "query": str(request.url.query) if request.url.query else None,
                },
            )
        except Exception as e:
            # Audit logging must never crash the request
            logger.error("Failed to write audit log entry: %s", e)

    @staticmethod
    def _classify_action(method: str, path: str) -> AuditAction:
        """Map a request to an audit action type."""
        for rule_method, rule_prefix, action in PATH_ACTION_MAP:
            if method == rule_method and path.startswith(rule_prefix):
                return action
        return AuditAction.DATA_ACCESSED

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract the real client IP, respecting X-Forwarded-For."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client = request.client
        if client:
            return client.host
        return ""
