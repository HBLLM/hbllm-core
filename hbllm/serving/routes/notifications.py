"""
Notification & Proactive Output API routes.

Endpoints:
    GET  /v1/notifications           → List unread notifications
    GET  /v1/notifications/all       → List all notifications (incl. read)
    POST /v1/notifications/read      → Mark notification(s) as read
    GET  /v1/notifications/stream    → SSE stream for real-time push
    GET  /v1/autonomy/status         → AutonomyCore telemetry
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["notifications"])


def _get_brain(request: Request) -> Any:
    """Extract brain from app state."""
    state = getattr(request.app, "_state", None) or {}
    brain = state.get("brain")
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    return brain


# ── Notification Endpoints ───────────────────────────────────────────────────


@router.get("/v1/notifications")
async def get_notifications(
    request: Request,
    category: str | None = None,
    limit: int = Query(default=50, le=200),
) -> Any:
    """Get unread notifications for the current tenant."""
    brain = _get_brain(request)
    gateway = getattr(brain, "notification_gateway", None)
    if not gateway:
        return {"notifications": [], "unread_count": 0}

    tenant_id = getattr(request.state, "tenant_id", "default")

    from hbllm.serving.notifications import NotificationCategory

    cat = NotificationCategory(category) if category else None
    notifications = gateway.get_unread(tenant_id, category=cat, limit=limit)

    return {
        "notifications": [n.to_dict() for n in notifications],
        "unread_count": gateway.unread_count(tenant_id),
    }


@router.get("/v1/notifications/all")
async def get_all_notifications(
    request: Request,
    limit: int = Query(default=100, le=500),
    include_read: bool = False,
) -> Any:
    """Get all notifications (including read) for the current tenant."""
    brain = _get_brain(request)
    gateway = getattr(brain, "notification_gateway", None)
    if not gateway:
        return {"notifications": [], "total": 0}

    tenant_id = getattr(request.state, "tenant_id", "default")
    notifications = gateway.get_all(tenant_id, limit=limit, include_read=include_read)

    return {
        "notifications": [n.to_dict() for n in notifications],
        "total": len(notifications),
    }


@router.post("/v1/notifications/read")
async def mark_notifications_read(request: Request) -> Any:
    """Mark notification(s) as read.

    Body:
        {"notification_id": "abc123"}     → mark one
        {"all": true}                     → mark all
    """
    brain = _get_brain(request)
    gateway = getattr(brain, "notification_gateway", None)
    if not gateway:
        raise HTTPException(status_code=503, detail="Notification gateway not initialized")

    tenant_id = getattr(request.state, "tenant_id", "default")
    body = await request.json()

    if body.get("all"):
        count = gateway.mark_all_read(tenant_id)
        return {"marked_read": count}

    notification_id = body.get("notification_id")
    if not notification_id:
        raise HTTPException(status_code=400, detail="notification_id or all=true required")

    success = gateway.mark_read(tenant_id, notification_id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")

    return {"marked_read": 1}


# ── SSE Stream ───────────────────────────────────────────────────────────────


@router.get("/v1/notifications/stream")
async def notification_stream(request: Request) -> StreamingResponse:
    """Server-Sent Events stream for real-time proactive notifications.

    Connect with EventSource:
        const es = new EventSource('/v1/notifications/stream');
        es.onmessage = (e) => console.log(JSON.parse(e.data));
    """
    brain = _get_brain(request)
    sse_channel = getattr(brain, "sse_channel", None)
    if not sse_channel:
        raise HTTPException(status_code=503, detail="SSE channel not initialized")

    tenant_id = getattr(request.state, "tenant_id", "default")

    async def event_generator():
        """Yield SSE events from the proactive channel."""
        queue = sse_channel.get_queue(tenant_id)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    data = json.dumps(event.to_dict())
                    yield f"event: notification\ndata: {data}\n\n"
                except (TimeoutError, asyncio.TimeoutError):
                    # Send keepalive
                    yield ": keepalive\n\n"
                except asyncio.CancelledError:
                    break
        finally:
            sse_channel.remove_tenant(tenant_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Autonomy Status ─────────────────────────────────────────────────────────


@router.get("/v1/autonomy/status")
async def autonomy_status(request: Request) -> Any:
    """Get AutonomyCore telemetry — cognitive heartbeat status."""
    brain = _get_brain(request)
    autonomy = getattr(brain, "autonomy_core", None)
    if not autonomy:
        return {"status": "inactive", "message": "AutonomyCore not initialized"}

    snapshot = autonomy.snapshot()
    snapshot["status"] = "active" if snapshot.get("running") else "stopped"

    # Add proactive processor stats
    proactive = getattr(brain, "proactive_processor", None)
    if proactive:
        snapshot["proactive"] = proactive.snapshot()

    # Add notification stats
    gateway = getattr(brain, "notification_gateway", None)
    if gateway:
        snapshot["notifications"] = gateway.stats()

    return snapshot
