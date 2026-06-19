"""
Studio Cognitive Endpoints — Notifications, Habits, Digest, Threads.

Exposes the cognitive-layer nodes for observability and manual
interaction from the Studio dashboard.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from hbllm.serving.studio.helpers import (
    get_brain,
    get_node,
    get_tenant_id,
    require_bus,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Notification Gateway ─────────────────────────────────────────────────────


@router.get("/studio/notifications")
async def studio_notifications_status(request: Request):
    """Notification gateway status, channel health, and delivery stats."""

    node = get_node("NotificationGateway")
    if not node:
        return {
            "status": "not_loaded",
            "channels": [],
            "total_sent": 0,
            "total_failed": 0,
        }

    stats = node.stats() if hasattr(node, "stats") else {}
    channels = []
    channel_registry = getattr(node, "_channels", {})
    for name, ch in channel_registry.items():
        channels.append(
            {
                "name": name,
                "enabled": getattr(ch, "enabled", True),
                "type": type(ch).__name__,
            }
        )

    return {
        "status": "active",
        "channels": channels,
        **stats,
    }


@router.post("/studio/notifications/test")
async def studio_notifications_test(request: Request):
    """Send a test notification through the gateway.

    Body::

        {
            "channel": "push",
            "title": "Test",
            "body": "Hello from Studio"
        }
    """
    bus = require_bus()
    body = await request.json()
    tenant_id = get_tenant_id(request)

    from hbllm.network.messages import Message, MessageType

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="studio",
        tenant_id=tenant_id,
        topic="system.notification",
        payload={
            "channel": body.get("channel", "push"),
            "title": body.get("title", "Studio Test"),
            "body": body.get("body", "Test notification from Studio dashboard"),
            "priority": body.get("priority", "normal"),
        },
    )
    await bus.publish("system.notification", msg)

    return {"status": "sent", "channel": body.get("channel", "push")}


# ── Habit Tracker ────────────────────────────────────────────────────────────


@router.get("/studio/habits")
async def studio_habits_status(request: Request):
    """Habit tracker status and active habits."""

    node = get_node("HabitTracker")
    if not node:
        return {
            "status": "not_loaded",
            "habits": [],
            "total_tracked": 0,
        }

    stats = node.stats() if hasattr(node, "stats") else {}
    habits = []
    habit_registry = getattr(node, "_habits", {})
    for tenant_habits in habit_registry.values():
        if isinstance(tenant_habits, dict):
            for habit_id, habit in tenant_habits.items():
                habits.append(
                    {
                        "id": habit_id,
                        "name": getattr(habit, "name", habit_id),
                        "frequency": getattr(habit, "frequency", "unknown"),
                        "streak": getattr(habit, "streak", 0),
                        "last_triggered": getattr(habit, "last_triggered", 0),
                    }
                )

    return {
        "status": "active",
        "habits": habits,
        **stats,
    }


@router.post("/studio/habits")
async def studio_habits_create(request: Request):
    """Create or update a habit via the bus.

    Body::

        {
            "name": "Morning standup",
            "cron": "0 9 * * 1-5",
            "action": "remind"
        }
    """
    bus = require_bus()
    body = await request.json()
    tenant_id = get_tenant_id(request)

    from hbllm.network.messages import Message, MessageType

    msg = Message(
        type=MessageType.COMMAND,
        source_node_id="studio",
        tenant_id=tenant_id,
        topic="habits.create",
        payload={
            "name": body.get("name", "Unnamed habit"),
            "cron": body.get("cron"),
            "frequency": body.get("frequency", "daily"),
            "action": body.get("action", "remind"),
            "metadata": body.get("metadata", {}),
        },
    )
    await bus.publish("habits.create", msg)

    return {"status": "created", "name": body.get("name")}


# ── Activity Digest ──────────────────────────────────────────────────────────


@router.get("/studio/digest")
async def studio_digest_status(request: Request):
    """Latest activity digest summary."""

    node = get_node("ActivityDigest")
    if not node:
        return {
            "status": "not_loaded",
            "latest": None,
            "total_generated": 0,
        }

    stats = node.stats() if hasattr(node, "stats") else {}
    latest = getattr(node, "_latest_digest", None)

    return {
        "status": "active",
        "latest": latest.to_dict() if latest and hasattr(latest, "to_dict") else latest,
        **stats,
    }


@router.post("/studio/digest/generate")
async def studio_digest_generate(request: Request):
    """Force-generate an activity digest."""
    bus = require_bus()
    tenant_id = get_tenant_id(request)

    from hbllm.network.messages import Message, MessageType

    msg = Message(
        type=MessageType.COMMAND,
        source_node_id="studio",
        tenant_id=tenant_id,
        topic="digest.generate",
        payload={"force": True},
    )
    await bus.publish("digest.generate", msg)

    return {"status": "generating"}


# ── Conversation Threads ─────────────────────────────────────────────────────


@router.get("/studio/threads")
async def studio_threads_status(request: Request):
    """Active conversation threads per tenant."""

    node = get_node("ConversationThread")
    if not node:
        return {
            "status": "not_loaded",
            "threads": [],
            "total_messages": 0,
        }

    stats = node.stats() if hasattr(node, "stats") else {}
    threads = []
    thread_registry = getattr(node, "_threads", {})
    for tid, thread in thread_registry.items():
        messages = getattr(thread, "messages", [])
        threads.append(
            {
                "thread_id": tid,
                "message_count": len(messages) if isinstance(messages, list) else 0,
                "last_activity": getattr(thread, "last_activity", 0),
            }
        )

    return {
        "status": "active",
        "threads": threads,
        **stats,
    }
