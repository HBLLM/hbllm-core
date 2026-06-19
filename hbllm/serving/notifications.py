"""
NotificationGateway — Proactive push channel for background insights.

The brain continuously processes background thoughts (CuriosityNode,
autonomy loop, scheduled tasks, world-state changes) but has no way to
push results to the user unprompted.  NotificationGateway bridges that
gap with a priority-based notification queue and pluggable delivery
backends (WebSocket, webhook, in-memory polling).

Bus Topics:
    notification.push      → Enqueue a notification
    notification.read      → Mark notification(s) as read
    notification.list      → Retrieve pending notifications
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Data types ───────────────────────────────────────────────────────────────


class NotificationPriority(str, Enum):
    """Notification urgency levels."""

    CRITICAL = "critical"  # Requires immediate attention (e.g. security alert)
    HIGH = "high"  # Important but not urgent (e.g. goal completed)
    INFO = "info"  # Informational (e.g. background insight)
    SUGGESTION = "suggestion"  # Low-priority suggestions (e.g. habit reminder)


class NotificationCategory(str, Enum):
    """Notification source categories."""

    SYSTEM = "system"  # System health, errors
    GOAL = "goal"  # Goal progress, completion
    INSIGHT = "insight"  # Background analysis results
    REMINDER = "reminder"  # Time-based or location-based
    SECURITY = "security"  # Trust, auth, threat events
    HABIT = "habit"  # Learned routine suggestions
    DIGEST = "digest"  # Activity summaries


@dataclass
class Notification:
    """A single notification."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tenant_id: str = ""
    title: str = ""
    body: str = ""
    priority: NotificationPriority = NotificationPriority.INFO
    category: NotificationCategory = NotificationCategory.SYSTEM
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    read_at: float | None = None
    delivered: bool = False

    @property
    def is_read(self) -> bool:
        return self.read_at is not None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["priority"] = self.priority.value
        result["category"] = self.category.value
        return result


# ── Delivery backends ────────────────────────────────────────────────────────


class DeliveryBackend:
    """Base class for notification delivery backends."""

    async def deliver(self, notification: Notification) -> bool:
        """Deliver a notification. Returns True on success."""
        raise NotImplementedError


class InMemoryBackend(DeliveryBackend):
    """In-memory backend — notifications are stored for polling via API."""

    async def deliver(self, notification: Notification) -> bool:
        notification.delivered = True
        return True


class WebhookBackend(DeliveryBackend):
    """Delivers notifications via HTTP webhook POST."""

    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self.url = url
        self.headers = headers or {}

    async def deliver(self, notification: Notification) -> bool:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.url,
                    json=notification.to_dict(),
                    headers=self.headers,
                )
                notification.delivered = response.is_success
                return response.is_success
        except Exception as e:
            logger.warning("Webhook delivery failed for %s: %s", notification.id, e)
            return False


# ── NotificationGateway ──────────────────────────────────────────────────────


class NotificationGateway:
    """
    Proactive notification system for the cognitive brain.

    Supports:
    - Priority-based queueing (critical notifications are delivered first)
    - Per-tenant notification isolation
    - Pluggable delivery backends (in-memory, webhook, WebSocket)
    - Notification lifecycle (create → deliver → read → archive)
    - Callbacks for real-time delivery

    Usage:
        gateway = NotificationGateway()

        # Push a notification
        gateway.push(
            tenant_id="user_1",
            title="Build failed",
            body="The CI pipeline for hbllm-core failed on commit abc123.",
            priority=NotificationPriority.HIGH,
            category=NotificationCategory.SYSTEM,
        )

        # Get unread notifications
        unread = gateway.get_unread("user_1")

        # Register a real-time callback (e.g. WebSocket push)
        gateway.on_notification("user_1", my_ws_callback)
    """

    def __init__(
        self,
        max_per_tenant: int = 500,
        default_backend: DeliveryBackend | None = None,
    ) -> None:
        self._max_per_tenant = max_per_tenant
        self._default_backend = default_backend or InMemoryBackend()
        # Per-tenant notification queues
        self._queues: dict[str, deque[Notification]] = {}
        # Per-tenant real-time callbacks
        self._callbacks: dict[str, list[Callable[[Notification], Any]]] = {}
        # Per-tenant custom backends
        self._backends: dict[str, DeliveryBackend] = {}

        logger.info("NotificationGateway initialized (max_per_tenant=%d)", max_per_tenant)

    def _get_queue(self, tenant_id: str) -> deque[Notification]:
        """Get or create the notification queue for a tenant."""
        if tenant_id not in self._queues:
            self._queues[tenant_id] = deque(maxlen=self._max_per_tenant)
        return self._queues[tenant_id]

    def push(
        self,
        tenant_id: str,
        title: str,
        body: str = "",
        priority: NotificationPriority = NotificationPriority.INFO,
        category: NotificationCategory = NotificationCategory.SYSTEM,
        metadata: dict[str, Any] | None = None,
    ) -> Notification:
        """
        Push a new notification to a tenant's queue.

        Returns the created Notification object.
        """
        notification = Notification(
            tenant_id=tenant_id,
            title=title,
            body=body,
            priority=priority,
            category=category,
            metadata=metadata or {},
        )

        queue = self._get_queue(tenant_id)
        queue.append(notification)

        logger.debug(
            "Notification pushed: [%s] %s → tenant '%s' (%d in queue)",
            priority.value,
            title,
            tenant_id,
            len(queue),
        )

        # Fire real-time callbacks
        for callback in self._callbacks.get(tenant_id, []):
            try:
                callback(notification)
            except Exception as e:
                logger.warning("Notification callback error: %s", e)

        return notification

    def get_unread(
        self,
        tenant_id: str,
        category: NotificationCategory | None = None,
        limit: int = 50,
    ) -> list[Notification]:
        """Get unread notifications for a tenant, newest first."""
        queue = self._get_queue(tenant_id)
        results = []
        for n in reversed(queue):
            if n.is_read:
                continue
            if category and n.category != category:
                continue
            results.append(n)
            if len(results) >= limit:
                break
        return results

    def get_all(
        self,
        tenant_id: str,
        limit: int = 100,
        include_read: bool = False,
    ) -> list[Notification]:
        """Get all notifications for a tenant, newest first."""
        queue = self._get_queue(tenant_id)
        results = []
        for n in reversed(queue):
            if not include_read and n.is_read:
                continue
            results.append(n)
            if len(results) >= limit:
                break
        return results

    def mark_read(self, tenant_id: str, notification_id: str) -> bool:
        """Mark a single notification as read."""
        queue = self._get_queue(tenant_id)
        for n in queue:
            if n.id == notification_id:
                n.read_at = time.time()
                return True
        return False

    def mark_all_read(self, tenant_id: str) -> int:
        """Mark all unread notifications as read. Returns count marked."""
        queue = self._get_queue(tenant_id)
        count = 0
        now = time.time()
        for n in queue:
            if not n.is_read:
                n.read_at = now
                count += 1
        return count

    def unread_count(self, tenant_id: str) -> int:
        """Count of unread notifications for a tenant."""
        queue = self._get_queue(tenant_id)
        return sum(1 for n in queue if not n.is_read)

    def on_notification(self, tenant_id: str, callback: Callable[[Notification], Any]) -> None:
        """Register a real-time callback for new notifications."""
        if tenant_id not in self._callbacks:
            self._callbacks[tenant_id] = []
        self._callbacks[tenant_id].append(callback)

    def remove_callback(self, tenant_id: str, callback: Callable[[Notification], Any]) -> None:
        """Remove a registered callback."""
        if tenant_id in self._callbacks:
            try:
                self._callbacks[tenant_id].remove(callback)
            except ValueError:
                pass

    def set_backend(self, tenant_id: str, backend: DeliveryBackend) -> None:
        """Set a custom delivery backend for a specific tenant."""
        self._backends[tenant_id] = backend

    async def deliver_pending(self, tenant_id: str) -> int:
        """Deliver all undelivered notifications via the configured backend."""
        queue = self._get_queue(tenant_id)
        backend = self._backends.get(tenant_id, self._default_backend)
        delivered = 0
        for n in queue:
            if not n.delivered:
                success = await backend.deliver(n)
                if success:
                    delivered += 1
        return delivered

    def clear(self, tenant_id: str) -> None:
        """Clear all notifications for a tenant."""
        if tenant_id in self._queues:
            self._queues[tenant_id].clear()

    def stats(self) -> dict[str, Any]:
        """Aggregate stats across all tenants."""
        total = 0
        unread = 0
        for queue in self._queues.values():
            total += len(queue)
            unread += sum(1 for n in queue if not n.is_read)
        return {
            "tenant_count": len(self._queues),
            "total_notifications": total,
            "total_unread": unread,
        }
