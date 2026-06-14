"""
Health & Monitoring Routes — extracted from api.py for maintainability.

Provides:
  - /health          — Basic health check
  - /health/live     — Kubernetes liveness probe
  - /health/ready    — Kubernetes readiness probe
  - /metrics         — Internal bus metrics
  - /routing/stats   — DualLLMRouter statistics
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Request

from hbllm.serving.state import _state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request) -> dict[str, Any]:
    """Basic health check endpoint."""
    brain = _state.get("brain")
    bus = _state.get("bus")

    result: dict[str, Any] = {
        "status": "healthy" if brain else "degraded",
        "brain_initialized": brain is not None,
        "draining": brain.is_draining if brain else False,
        "timestamp": time.time(),
    }

    if bus:
        from hbllm.network.tracing import BusMetrics

        metrics = getattr(bus, "metrics", None)
        if metrics and isinstance(metrics, BusMetrics):
            result["bus_metrics"] = metrics.snapshot()

    return result


@router.get("/health/live")
async def liveness_probe() -> dict[str, str]:
    """Kubernetes liveness probe — is the process alive?"""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_probe() -> dict[str, Any]:
    """Kubernetes readiness probe — is the brain ready to serve?"""
    brain = _state.get("brain")
    if brain is None:
        return {"status": "not_ready", "reason": "brain_not_initialized"}
    if brain.is_draining:
        return {"status": "not_ready", "reason": "draining"}
    return {"status": "ready"}


@router.get("/routing/stats")
async def routing_stats() -> dict[str, Any]:
    """DualLLMRouter statistics — local vs external usage."""
    brain = _state.get("brain")
    if brain is None:
        return {"error": "Brain not initialized"}
    dual_router = getattr(brain, "dual_router", None)
    if dual_router is None:
        return {"status": "no_dual_router", "message": "DualLLMRouter not configured"}
    return dual_router.snapshot()
