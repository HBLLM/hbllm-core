"""
RBAC Middleware — Route-Level Permission Enforcement.

Extracts user identity from JWT claims and checks permissions against
the RBACGuard before allowing requests to proceed.

Protected paths are mapped to required permissions. Unprotected paths
(health checks, docs) pass through without checks.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from hbllm.security.rbac import Permission, RBACGuard

logger = logging.getLogger(__name__)

# ── Route → Permission Mapping ────────────────────────────────────────

# Maps (method, path_prefix) to required Permission.
# Order matters: first match wins. More specific paths should come first.
ROUTE_PERMISSIONS: list[tuple[str, str, Permission]] = [
    # Admin endpoints
    ("GET", "/v1/admin/audit", Permission.ADMIN_VIEW_AUDIT),
    ("POST", "/v1/admin/users", Permission.ADMIN_MANAGE_USERS),
    ("PUT", "/v1/admin/users", Permission.ADMIN_MANAGE_USERS),
    ("DELETE", "/v1/admin/users", Permission.ADMIN_MANAGE_USERS),
    ("POST", "/v1/admin/config", Permission.ADMIN_MANAGE_CONFIG),
    ("PUT", "/v1/admin/config", Permission.ADMIN_MANAGE_CONFIG),
    # Data operations
    ("POST", "/v1/data/export", Permission.DATA_EXPORT),
    ("DELETE", "/v1/data", Permission.DATA_DELETE),
    # Tool & shell execution
    ("POST", "/v1/tools/shell", Permission.TOOL_SHELL),
    ("POST", "/v1/tools", Permission.TOOL_EXECUTE),
    # Memory
    ("POST", "/v1/memory", Permission.MEMORY_WRITE),
    ("PUT", "/v1/memory", Permission.MEMORY_WRITE),
    ("DELETE", "/v1/memory", Permission.MEMORY_WRITE),
    ("GET", "/v1/memory", Permission.MEMORY_READ),
    # Chat (send)
    ("POST", "/v1/chat", Permission.CHAT_SEND),
    # Chat (read)
    ("GET", "/v1/chat", Permission.CHAT_READ),
    ("GET", "/v1/conversations", Permission.CHAT_READ),
]

# Paths that skip RBAC entirely
UNPROTECTED_PATHS = frozenset(
    {
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/",
    }
)


class RBACMiddleware(BaseHTTPMiddleware):
    """ASGI middleware for route-level permission enforcement.

    Expects JWT claims to be present in ``request.state`` with:
      - ``tenant_id``: The tenant ID
      - ``user_id``: The user ID
      - ``role``: (Optional) pre-resolved role from JWT

    If claims are missing, the request is treated as unauthenticated
    and receives a 401 response for protected routes.
    """

    def __init__(self, app: object, rbac_guard: RBACGuard) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.guard = rbac_guard

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        method = request.method

        # Skip unprotected paths
        if path in UNPROTECTED_PATHS or path.startswith("/health"):
            return await call_next(request)

        # Find required permission for this route
        required_permission = self._match_permission(method, path)
        if required_permission is None:
            # No explicit permission mapping → allow through
            return await call_next(request)

        # Extract identity from request state (set by auth middleware)
        tenant_id = getattr(request.state, "tenant_id", None)
        user_id = getattr(request.state, "user_id", None)

        if not tenant_id or not user_id:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Authentication required for this endpoint",
                },
            )

        # Check permission
        if not self.guard.check(tenant_id, user_id, required_permission):
            role = self.guard.get_role(tenant_id, user_id)
            logger.warning(
                "RBAC denied: user=%s role=%s tenant=%s needs=%s path=%s",
                user_id,
                role.value,
                tenant_id,
                required_permission.value,
                path,
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": "forbidden",
                    "message": f"Insufficient permissions. Required: {required_permission.value}",
                    "role": role.value,
                    "required_permission": required_permission.value,
                },
            )

        return await call_next(request)

    @staticmethod
    def _match_permission(method: str, path: str) -> Permission | None:
        """Find the required permission for a given method+path."""
        for rule_method, rule_prefix, permission in ROUTE_PERMISSIONS:
            if method == rule_method and path.startswith(rule_prefix):
                return permission
        return None
