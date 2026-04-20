"""
FastAPI Middleware for JWT-based Tenant Authentication.

Ensures that all requests to the API are authenticated with a valid JWT.
The resolved `tenant_id` is securely injected into `request.state.tenant_id`.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Awaitable, Callable
from typing import Any

import jwt
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, secret_key: str | None = None) -> None:
        super().__init__(app)
        self.secret_key = secret_key or os.environ.get("HBLLM_JWT_SECRET", "")

        if not self.secret_key:
            # In production (HBLLM_ENV=production), refuse to start without
            # an explicit secret to prevent accidental insecure deployments.
            if os.environ.get("HBLLM_ENV", "").lower() == "production":
                raise ValueError(
                    "HBLLM_JWT_SECRET must be set in production. "
                    'Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
                )
            # In development, generate a random ephemeral secret per boot.
            self.secret_key = secrets.token_urlsafe(32)
            logger.warning(
                "⚠️  HBLLM_JWT_SECRET is not set — using a random ephemeral secret. "
                "JWTs will NOT survive server restarts. Set HBLLM_JWT_SECRET for persistence."
            )

        self.algorithms: list[str] = ["HS256"]

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Pass-through for health checks, studio dashboard, static files, and CORS preflight
        if (
            request.method == "OPTIONS"
            or request.url.path
            in [
                "/health",
                "/metrics",
                "/docs",
                "/openapi.json",
            ]
            or request.url.path.startswith(("/admin/static", "/studio/"))
        ):
            return await call_next(request)
        # If an upstream cloud middleware (e.g. ApiSecurityMiddleware) has already
        # authenticated the request and injected the tenant context, skip JWT verification.
        if hasattr(request.state, "tenant_id") and request.state.tenant_id:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401, content={"detail": "Missing or invalid Authorization header"}
            )

        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=self.algorithms)
            tenant_id = payload.get("tenant_id")
            if not tenant_id:
                return JSONResponse(
                    status_code=401, content={"detail": "Token payload missing 'tenant_id'"}
                )

            # Securely inject the verified tenant_id into the request state
            request.state.tenant_id = tenant_id

        except jwt.ExpiredSignatureError:
            return JSONResponse(status_code=401, content={"detail": "Token has expired"})
        except jwt.InvalidTokenError:
            return JSONResponse(status_code=401, content={"detail": "Invalid token signature"})

        response = await call_next(request)
        return response
