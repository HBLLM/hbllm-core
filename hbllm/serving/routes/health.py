"""
Health & Monitoring Routes.

Endpoints:
  GET  /health          — Basic health check
  GET  /health/live     — Kubernetes liveness probe
  GET  /health/ready    — Kubernetes readiness probe
  GET  /metrics         — MessageBus performance metrics
  GET  /routing/stats   — DualLLMRouter statistics
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from starlette.responses import JSONResponse

from hbllm.serving.state import _state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class HealthResponse:
    """Inline response model — avoids circular import from api.py schemas."""

    pass


@router.get("/health")
async def health_check() -> Any:
    """Check server health and node count."""
    brain = _state.get("brain")
    node_count = len(brain.nodes) if brain else 0
    mode = _state.get("mode", "unknown")
    return {
        "status": "healthy",
        "nodes_registered": node_count,
        "bus_type": _state.get("bus_type", "unknown"),
        "provider_mode": mode,
    }


@router.get("/health/live")
async def health_live() -> dict[str, str]:
    """Kubernetes liveness probe — returns 200 if the process is running."""
    return {"status": "alive"}


@router.get("/health/ready")
async def health_ready() -> Any:
    """Kubernetes readiness probe — checks brain, bus, and provider availability."""
    checks: dict[str, Any] = {}
    overall_ready = True

    # Brain check
    brain = _state.get("brain")
    if brain:
        checks["brain"] = "ok"
    elif _state.get("mode") == "provider":
        checks["brain"] = "degraded (provider-only mode)"
    else:
        checks["brain"] = "not_initialized"
        overall_ready = False

    # Bus check
    bus = _state.get("bus")
    checks["bus"] = "ok" if bus else "not_initialized"

    # Provider check
    provider = _state.get("provider")
    if not provider and brain:
        provider = getattr(brain, "provider", None)
    checks["provider"] = "ok" if provider else "not_configured"

    status_code = 200 if overall_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if overall_ready else "not_ready", "checks": checks},
    )


@router.get("/metrics")
async def metrics() -> Any:
    """Return real-time MessageBus performance metrics."""
    brain = _state.get("brain")
    if not brain or not hasattr(brain.bus, "metrics"):
        return {"error": "Bus not initialized or metrics unavailable"}
    return brain.bus.metrics.snapshot()


@router.get("/routing/stats")
async def routing_stats() -> Any:
    """Return Dual LLM Router statistics (local vs external usage)."""
    brain = _state.get("brain")
    if not brain:
        return {"error": "Brain not initialized"}
    dual_router = getattr(brain, "dual_router", None)
    if dual_router is None:
        return {"status": "single_model", "message": "No dual router configured"}
    return dual_router.snapshot()
