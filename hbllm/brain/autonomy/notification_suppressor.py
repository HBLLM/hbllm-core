"""Notification Suppressor — anti-annoyance system.

Prevents notification fatigue by:
    1. Rate limiting per category (max 3 suggestions/hour)
    2. Detecting user fatigue (> 50% dismissed → reduce frequency)
    3. Batching non-urgent notifications into periodic digests
    4. Respecting Do Not Disturb schedules
    5. Exponential backoff on rejected notification types

"The best assistant knows when to shut up."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DNDSchedule:
    """Do Not Disturb schedule (24-hour format)."""

    enabled: bool = False
    start_hour: int = 22  # 10 PM
    start_minute: int = 0
    end_hour: int = 7  # 7 AM
    end_minute: int = 0

    def is_active(self) -> bool:
        """Check if DND is currently active."""
        if not self.enabled:
            return False

        import datetime

        now = datetime.datetime.now()
        current_minutes = now.hour * 60 + now.minute
        start_minutes = self.start_hour * 60 + self.start_minute
        end_minutes = self.end_hour * 60 + self.end_minute

        if start_minutes <= end_minutes:
            # Same-day range (e.g., 09:00 - 17:00)
            return start_minutes <= current_minutes < end_minutes
        else:
            # Overnight range (e.g., 22:00 - 07:00)
            return current_minutes >= start_minutes or current_minutes < end_minutes


@dataclass
class CategoryLimits:
    """Rate limits per notification category."""

    max_per_hour: int = 10
    max_suggestions_per_hour: int = 3
    max_digests_per_day: int = 1
    max_reminders_per_hour: int = 5


class NotificationSuppressor:
    """Intelligent notification suppression engine.

    Usage::

        suppressor = NotificationSuppressor()

        # Before sending a notification:
        if suppressor.should_send(
            category="suggestion",
            priority="suggestion",
            tenant_id="user1",
        ):
            send_notification(...)
        else:
            suppressor.defer(notification)  # Add to batch digest
    """

    def __init__(
        self,
        limits: CategoryLimits | None = None,
        dnd_schedules: dict[str, DNDSchedule] | None = None,
    ) -> None:
        self.limits = limits or CategoryLimits()
        self._dnd_schedules: dict[str, DNDSchedule] = dnd_schedules or {}

        # Per-tenant tracking
        self._sent_timestamps: dict[
            str, dict[str, list[float]]
        ] = {}  # tenant → category → [timestamps]
        self._dismissal_counts: dict[str, dict[str, int]] = {}  # tenant → category → dismiss count
        self._delivery_counts: dict[str, dict[str, int]] = {}  # tenant → category → delivery count
        self._backoff_until: dict[
            str, dict[str, float]
        ] = {}  # tenant → category → backoff deadline

        # Deferred notifications for batching
        self._deferred: dict[str, list[dict[str, Any]]] = {}  # tenant → [notification dicts]
        self._max_deferred = 50

        # Telemetry
        self._total_checked = 0
        self._total_suppressed = 0
        self._total_deferred = 0

    def set_dnd(self, tenant_id: str, schedule: DNDSchedule) -> None:
        """Set a Do Not Disturb schedule for a tenant."""
        self._dnd_schedules[tenant_id] = schedule

    def should_send(
        self,
        category: str,
        priority: str,
        tenant_id: str = "default",
    ) -> bool:
        """Decide whether a notification should be sent now.

        Args:
            category: Notification category (e.g., "suggestion", "reminder").
            priority: Priority level ("critical", "high", "info", "suggestion").
            tenant_id: Tenant identifier.

        Returns:
            True if the notification should be sent, False if suppressed.
        """
        self._total_checked += 1

        # Critical notifications always pass through
        if priority == "critical":
            self._record_sent(tenant_id, category)
            return True

        # Check DND schedule
        dnd = self._dnd_schedules.get(tenant_id)
        if dnd and dnd.is_active():
            if priority != "critical":
                self._total_suppressed += 1
                return False

        # Check backoff
        backoff = self._backoff_until.get(tenant_id, {}).get(category, 0)
        if time.time() < backoff:
            self._total_suppressed += 1
            return False

        # Check rate limits
        if not self._check_rate_limit(tenant_id, category, priority):
            self._total_suppressed += 1
            return False

        # Check fatigue
        if self._is_fatigued(tenant_id, category):
            self._total_suppressed += 1
            return False

        self._record_sent(tenant_id, category)
        return True

    def record_dismissal(self, tenant_id: str, category: str) -> None:
        """Record that the user dismissed a notification (for fatigue detection)."""
        if tenant_id not in self._dismissal_counts:
            self._dismissal_counts[tenant_id] = {}
        self._dismissal_counts[tenant_id][category] = (
            self._dismissal_counts[tenant_id].get(category, 0) + 1
        )

        # Apply exponential backoff if user keeps dismissing
        dismissals = self._dismissal_counts[tenant_id].get(category, 0)
        if dismissals >= 3:
            # Backoff: 5 min, 15 min, 45 min, 2h, 6h...
            backoff_s = min(21600, 300 * (3 ** (dismissals - 3)))
            if tenant_id not in self._backoff_until:
                self._backoff_until[tenant_id] = {}
            self._backoff_until[tenant_id][category] = time.time() + backoff_s
            logger.info(
                "Notification backoff for tenant=%s category=%s: %ds (dismissals=%d)",
                tenant_id,
                category,
                backoff_s,
                dismissals,
            )

    def defer(self, tenant_id: str, notification: dict[str, Any]) -> None:
        """Defer a notification for later batch delivery."""
        if tenant_id not in self._deferred:
            self._deferred[tenant_id] = []
        if len(self._deferred[tenant_id]) < self._max_deferred:
            self._deferred[tenant_id].append(notification)
            self._total_deferred += 1

    def get_deferred_batch(self, tenant_id: str) -> list[dict[str, Any]]:
        """Get and clear deferred notifications for a tenant."""
        batch = self._deferred.pop(tenant_id, [])
        return batch

    def _check_rate_limit(self, tenant_id: str, category: str, priority: str) -> bool:
        """Check if sending would exceed rate limits."""
        now = time.time()
        one_hour_ago = now - 3600

        if tenant_id not in self._sent_timestamps:
            return True

        timestamps = self._sent_timestamps.get(tenant_id, {}).get(category, [])
        recent = [t for t in timestamps if t > one_hour_ago]

        # Category-specific limits
        if category == "suggestion" and len(recent) >= self.limits.max_suggestions_per_hour:
            return False
        elif category == "reminder" and len(recent) >= self.limits.max_reminders_per_hour:
            return False
        elif category == "digest":
            one_day_ago = now - 86400
            daily = [t for t in timestamps if t > one_day_ago]
            if len(daily) >= self.limits.max_digests_per_day:
                return False
        elif len(recent) >= self.limits.max_per_hour:
            return False

        return True

    def _is_fatigued(self, tenant_id: str, category: str) -> bool:
        """Check if user is fatigued with this category (> 50% dismissal rate)."""
        dismissals = self._dismissal_counts.get(tenant_id, {}).get(category, 0)
        deliveries = self._delivery_counts.get(tenant_id, {}).get(category, 0)

        if deliveries < 5:
            return False  # Not enough data

        dismissal_rate = dismissals / deliveries if deliveries > 0 else 0
        return dismissal_rate > 0.5

    def _record_sent(self, tenant_id: str, category: str) -> None:
        """Record a sent notification timestamp."""
        if tenant_id not in self._sent_timestamps:
            self._sent_timestamps[tenant_id] = {}
        if category not in self._sent_timestamps[tenant_id]:
            self._sent_timestamps[tenant_id][category] = []

        self._sent_timestamps[tenant_id][category].append(time.time())

        # Track delivery count
        if tenant_id not in self._delivery_counts:
            self._delivery_counts[tenant_id] = {}
        self._delivery_counts[tenant_id][category] = (
            self._delivery_counts[tenant_id].get(category, 0) + 1
        )

        # Prune old timestamps (keep last 100)
        if len(self._sent_timestamps[tenant_id][category]) > 100:
            self._sent_timestamps[tenant_id][category] = self._sent_timestamps[tenant_id][category][
                -50:
            ]

    def stats(self) -> dict[str, Any]:
        """Suppressor statistics."""
        return {
            "total_checked": self._total_checked,
            "total_suppressed": self._total_suppressed,
            "total_deferred": self._total_deferred,
            "suppression_rate": self._total_suppressed / max(1, self._total_checked),
            "active_dnd_schedules": sum(1 for s in self._dnd_schedules.values() if s.is_active()),
            "active_backoffs": sum(
                1
                for tenant in self._backoff_until.values()
                for deadline in tenant.values()
                if deadline > time.time()
            ),
        }
