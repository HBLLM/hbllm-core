"""
FastAPI Middleware for JWT-based Tenant Authentication.

Ensures that all requests to the API are authenticated with a valid JWT.
The resolved `tenant_id` is securely injected into `request.state.tenant_id`.
"""

import os
from collections.abc import Awaitable, Callable

import jwt
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class JWTAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, secret_key: str | None = None):
        super().__init__(app)
        self.secret_key = secret_key or os.environ.get(
            "HBLLM_JWT_SECRET", "default_insecure_secret"
        )
        self.algorithms = ["HS256"]

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
    ) -> JSONResponse:
        # Pass-through for health checks and static files
        if request.url.path in [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
        ] or request.url.path.startswith("/admin/static"):
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
