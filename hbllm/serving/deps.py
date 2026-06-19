"""
FastAPI Dependency Injection for HBLLM route modules.

Provides typed dependency functions for accessing shared state
(brain, bus, provider) without relying on module-level globals.

Usage in route modules::

    from hbllm.serving.deps import get_brain, get_bus

    @router.get("/endpoint")
    async def handler(brain = Depends(get_brain)):
        ...
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request

from hbllm.serving.state import _state


def get_brain() -> Any:
    """Get the Brain instance. Raises 503 if not initialized."""
    brain = _state.get("brain")
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    return brain


def get_brain_optional() -> Any | None:
    """Get the Brain instance or None if not initialized."""
    return _state.get("brain")


def get_bus() -> Any:
    """Get the MessageBus instance. Raises 503 if not initialized."""
    bus = _state.get("bus")
    if bus is None:
        raise HTTPException(status_code=503, detail="Message bus not initialized")
    return bus


def get_bus_optional() -> Any | None:
    """Get the MessageBus instance or None."""
    return _state.get("bus")


def get_provider() -> Any | None:
    """Get the LLM provider (from brain or standalone)."""
    provider = _state.get("provider")
    if provider is None:
        brain = _state.get("brain")
        if brain is not None:
            provider = getattr(brain, "provider", None)
    return provider


def get_mode() -> str:
    """Get the current serving mode ('full' or 'provider')."""
    return _state.get("mode", "unknown")


def get_tenant_id(request: Request) -> str:
    """Extract tenant_id from request state (set by JWTAuthMiddleware)."""
    return getattr(request.state, "tenant_id", "default")
