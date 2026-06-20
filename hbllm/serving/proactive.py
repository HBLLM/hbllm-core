"""
ProactiveProcessor — bridge between autonomy actions and user-facing output.

When the AutonomyCore generates a cognitive action (proactive reminder,
anomaly detection, background insight, goal progress), this processor
routes it through the CognitivePipeline for LLM enrichment and delivers
the result via NotificationGateway + real-time push channels (SSE/WS).

Bus Topics:
    cognitive.process       → Subscribed (from AutonomyCore)
    proactive.output        → Published (enriched result for delivery)
    notification.push       → Published (queued for NotificationGateway)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.bus import MessageBus, Subscription
from hbllm.network.messages import Message, MessageType
from hbllm.serving.notifications import (
    NotificationCategory,
    NotificationGateway,
    NotificationPriority,
)

logger = logging.getLogger(__name__)


# ── Category Mapping ─────────────────────────────────────────────────────────

_CATEGORY_MAP: dict[str, NotificationCategory] = {
    "user_action": NotificationCategory.SYSTEM,
    "sensor": NotificationCategory.SYSTEM,
    "system_alert": NotificationCategory.SYSTEM,
    "device_change": NotificationCategory.SYSTEM,
    "background": NotificationCategory.INSIGHT,
    "internal": NotificationCategory.INSIGHT,
    "reminder": NotificationCategory.REMINDER,
    "goal_progress": NotificationCategory.GOAL,
    "routine": NotificationCategory.HABIT,
    "security": NotificationCategory.SECURITY,
    "digest": NotificationCategory.DIGEST,
}

_PRIORITY_MAP: dict[str, NotificationPriority] = {
    "tier3_heavy_reasoning": NotificationPriority.HIGH,
    "tier2_fast_router": NotificationPriority.INFO,
    "tier1_reflex": NotificationPriority.SUGGESTION,
}


# ── SSE Channel ──────────────────────────────────────────────────────────────


@dataclass
class ProactiveEvent:
    """A proactive output event ready for delivery."""

    tenant_id: str
    title: str
    body: str
    priority: NotificationPriority = NotificationPriority.INFO
    category: NotificationCategory = NotificationCategory.INSIGHT
    source_event_id: str = ""
    correlation_id: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "title": self.title,
            "body": self.body,
            "priority": self.priority.value,
            "category": self.category.value,
            "source_event_id": self.source_event_id,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class SSEChannel:
    """Server-Sent Events channel for real-time proactive push.

    Each tenant gets their own asyncio.Queue for push delivery.
    FastAPI/Starlette SSE endpoints drain from these queues.
    """

    def __init__(self, max_queue_size: int = 100) -> None:
        self._queues: dict[str, asyncio.Queue[ProactiveEvent]] = {}
        self._max_queue_size = max_queue_size

    def get_queue(self, tenant_id: str) -> asyncio.Queue[ProactiveEvent]:
        """Get or create the SSE queue for a tenant."""
        if tenant_id not in self._queues:
            self._queues[tenant_id] = asyncio.Queue(maxsize=self._max_queue_size)
        return self._queues[tenant_id]

    async def push(self, event: ProactiveEvent) -> bool:
        """Push an event to the tenant's SSE queue.

        Returns False if the queue is full (event dropped).
        """
        queue = self.get_queue(event.tenant_id)
        try:
            queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            logger.warning(
                "SSE queue full for tenant '%s', dropping event: %s",
                event.tenant_id,
                event.title,
            )
            return False

    async def stream(self, tenant_id: str) -> AsyncIterator[ProactiveEvent]:
        """Async iterator for SSE streaming. Used by API endpoints."""
        queue = self.get_queue(tenant_id)
        while True:
            event = await queue.get()
            yield event

    def remove_tenant(self, tenant_id: str) -> None:
        """Remove a tenant's queue (on disconnect)."""
        self._queues.pop(tenant_id, None)


# ── ProactiveProcessor ───────────────────────────────────────────────────────


class ProactiveProcessor:
    """Routes AutonomyCore cognitive actions to user-facing output.

    Flow:
        1. AutonomyCore emits ``cognitive.process`` or proactive messages
        2. ProactiveProcessor receives them via bus subscription
        3. For messages needing LLM enrichment: routes through pipeline
        4. Delivers final result via:
           a. NotificationGateway (persistent, pollable)
           b. SSE channel (real-time push)
           c. Bus topic ``proactive.output`` (for other subscribers)

    Usage::

        processor = ProactiveProcessor(
            gateway=notification_gateway,
            pipeline=cognitive_pipeline,  # Optional: for LLM enrichment
        )
        await processor.start(bus)
    """

    def __init__(
        self,
        gateway: NotificationGateway,
        pipeline: Any | None = None,
        *,
        sse_channel: SSEChannel | None = None,
        enrich_via_llm: bool = True,
        enrichment_timeout: float = 15.0,
    ) -> None:
        self.gateway = gateway
        self.pipeline = pipeline
        self.sse = sse_channel or SSEChannel()
        self._enrich_via_llm = enrich_via_llm and pipeline is not None
        self._enrichment_timeout = enrichment_timeout

        self._bus: MessageBus | None = None
        self._subscriptions: list[Subscription] = []

        # Telemetry
        self._events_received = 0
        self._events_enriched = 0
        self._events_delivered = 0
        self._enrichment_failures = 0

    async def start(self, bus: MessageBus) -> None:
        """Subscribe to cognitive action topics on the bus."""
        self._bus = bus

        self._subscriptions = [
            await bus.subscribe("cognitive.process", self._handle_cognitive_action),
            await bus.subscribe("proactive.push", self._handle_direct_push),
        ]
        logger.info(
            "ProactiveProcessor started (enrich=%s, timeout=%.1fs)",
            self._enrich_via_llm,
            self._enrichment_timeout,
        )

    async def stop(self) -> None:
        """Unsubscribe and clean up."""
        if self._bus:
            for sub in self._subscriptions:
                await self._bus.unsubscribe(sub)
        self._subscriptions.clear()
        logger.info(
            "ProactiveProcessor stopped (received=%d, delivered=%d, enrichment_failures=%d)",
            self._events_received,
            self._events_delivered,
            self._enrichment_failures,
        )

    async def _handle_cognitive_action(self, msg: Message) -> None:
        """Handle a cognitive action from AutonomyCore."""
        self._events_received += 1

        payload = msg.payload
        tier = payload.get("tier", "tier1_reflex")
        category = payload.get("category", "background")
        source = payload.get("source", "unknown")
        event_id = payload.get("event_id", "")
        original_payload = payload.get("original_payload", {})

        # Determine if this needs LLM enrichment
        needs_enrichment = (
            self._enrich_via_llm and tier in ("tier2_fast_router", "tier3_heavy_reasoning")
        )

        # Build the proactive event
        title = self._generate_title(source, category, original_payload)
        body = original_payload.get("content", original_payload.get("text", ""))

        if needs_enrichment and self.pipeline:
            body = await self._enrich_with_llm(body, source, category)

        tenant_id = msg.tenant_id or "default"

        event = ProactiveEvent(
            tenant_id=tenant_id,
            title=title,
            body=body,
            priority=_PRIORITY_MAP.get(tier, NotificationPriority.INFO),
            category=_CATEGORY_MAP.get(category, NotificationCategory.INSIGHT),
            source_event_id=event_id,
            correlation_id=msg.correlation_id or "",
            metadata={"source": source, "tier": tier},
        )

        await self._deliver(event)

    async def _handle_direct_push(self, msg: Message) -> None:
        """Handle a direct push notification (bypass enrichment)."""
        self._events_received += 1

        payload = msg.payload
        event = ProactiveEvent(
            tenant_id=msg.tenant_id or "default",
            title=payload.get("title", "Notification"),
            body=payload.get("body", ""),
            priority=NotificationPriority(payload.get("priority", "info")),
            category=NotificationCategory(payload.get("category", "system")),
            metadata=payload.get("metadata", {}),
        )

        await self._deliver(event)

    async def _enrich_with_llm(self, text: str, source: str, category: str) -> str:
        """Enrich a proactive message with LLM context."""
        if not self.pipeline or not text:
            return text

        try:
            prompt = (
                f"The system detected a background event from '{source}' "
                f"(category: {category}). Summarize this concisely for the user "
                f"as a brief notification (1-2 sentences max):\n\n{text}"
            )
            result = await asyncio.wait_for(
                self.pipeline.process(
                    text=prompt,
                    tenant_id="system",
                    session_id="proactive",
                ),
                timeout=self._enrichment_timeout,
            )
            if not result.error and result.text:
                self._events_enriched += 1
                return result.text
        except (TimeoutError, asyncio.TimeoutError):
            logger.debug("LLM enrichment timed out for proactive event")
            self._enrichment_failures += 1
        except Exception:
            logger.warning("LLM enrichment failed", exc_info=True)
            self._enrichment_failures += 1

        return text

    async def _deliver(self, event: ProactiveEvent) -> None:
        """Deliver a proactive event through all channels."""
        self._events_delivered += 1

        # 1. NotificationGateway (persistent)
        self.gateway.push(
            tenant_id=event.tenant_id,
            title=event.title,
            body=event.body,
            priority=event.priority,
            category=event.category,
            metadata=event.metadata,
        )

        # 2. SSE channel (real-time)
        await self.sse.push(event)

        # 3. Bus broadcast (for other subscribers)
        if self._bus:
            broadcast = Message(
                type=MessageType.EVENT,
                source_node_id="proactive_processor",
                topic="proactive.output",
                tenant_id=event.tenant_id,
                payload=event.to_dict(),
            )
            await self._bus.publish("proactive.output", broadcast)

    def _generate_title(
        self, source: str, category: str, payload: dict[str, Any]
    ) -> str:
        """Generate a human-readable title from event metadata."""
        # Use explicit title if provided
        if "title" in payload:
            return str(payload["title"])

        # Generate from source
        title_map = {
            "sensor.anomaly": "Sensor Alert",
            "device.change": "Device Update",
            "system.critical": "System Alert",
            "internal.reminder": "Reminder",
            "internal.reflection": "Background Insight",
            "internal.deferred_goal": "Goal Update",
        }

        for prefix, title in title_map.items():
            if source.startswith(prefix) or source == prefix:
                return title

        if category == "goal_progress":
            return "Goal Progress"
        if category == "routine":
            return "Routine Suggestion"

        return f"Update: {source.split('.')[-1].replace('_', ' ').title()}"

    def snapshot(self) -> dict[str, Any]:
        """Telemetry snapshot."""
        return {
            "events_received": self._events_received,
            "events_enriched": self._events_enriched,
            "events_delivered": self._events_delivered,
            "enrichment_failures": self._enrichment_failures,
            "gateway_stats": self.gateway.stats(),
        }
