"""
ActivityDigest — Summarizes missed activity during user absence.

When the user returns after being away, the brain should greet them with
a natural-language catch-up briefing:

    "While you were away: your CI build succeeded, 2 goals were completed,
     and I found an interesting pattern in your server logs."

ActivityDigest aggregates events from the notification queue, completed
goals, background insights, and system health changes into a concise
digest grouped by importance.

Integrations:
    ProjectGraph     → Enriches digest with active project goal status
    UserModelEngine  → Adapts digest verbosity to user preferences

Bus Topics:
    digest.generate   → Trigger a digest for a tenant
    digest.result     → The generated digest text
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass
class DigestItem:
    """A single item in the activity digest."""

    category: str  # "goal", "notification", "insight", "system", "habit"
    title: str
    detail: str = ""
    importance: float = 0.5  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "title": self.title,
            "detail": self.detail,
            "importance": round(self.importance, 2),
            "timestamp": self.timestamp,
        }


@dataclass
class Digest:
    """A complete activity digest for a tenant."""

    tenant_id: str
    items: list[DigestItem] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)
    period_start: float = 0.0
    period_end: float = field(default_factory=time.time)

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    @property
    def duration_hours(self) -> float:
        if self.period_start == 0:
            return 0.0
        return (self.period_end - self.period_start) / 3600

    def to_natural_language(self) -> str:
        """Generate a human-readable summary of the digest."""
        if self.is_empty:
            return "Nothing significant happened while you were away."

        # Group by category
        by_category: dict[str, list[DigestItem]] = defaultdict(list)
        for item in sorted(self.items, key=lambda i: i.importance, reverse=True):
            by_category[item.category].append(item)

        parts = []
        hours = self.duration_hours

        if hours > 0:
            if hours < 1:
                parts.append(f"In the last {int(hours * 60)} minutes:")
            elif hours < 24:
                parts.append(f"In the last {hours:.0f} hour{'s' if hours >= 2 else ''}:")
            else:
                days = hours / 24
                parts.append(f"In the last {days:.0f} day{'s' if days >= 2 else ''}:")
        else:
            parts.append("Here's what happened:")

        # Critical items first
        critical = [i for i in self.items if i.importance >= 0.8]
        if critical:
            for item in critical[:3]:
                parts.append(f"  ⚠️ {item.title}")
                if item.detail:
                    parts.append(f"     {item.detail}")

        # Goals
        goals = by_category.get("goal", [])
        if goals:
            completed = [g for g in goals if "completed" in g.title.lower()]
            in_progress = [g for g in goals if "progress" in g.title.lower()]
            if completed:
                parts.append(
                    f"  ✅ {len(completed)} goal{'s' if len(completed) > 1 else ''} completed"
                )
                for g in completed[:3]:
                    parts.append(f"     • {g.detail or g.title}")
            if in_progress:
                parts.append(
                    f"  🔄 {len(in_progress)} goal{'s' if len(in_progress) > 1 else ''} made progress"
                )

        # Insights
        insights = by_category.get("insight", [])
        if insights:
            parts.append(
                f"  💡 {len(insights)} insight{'s' if len(insights) > 1 else ''} discovered"
            )
            for i in insights[:2]:
                parts.append(f"     • {i.title}")

        # System events
        system = by_category.get("system", [])
        if system:
            parts.append(f"  🔧 {len(system)} system event{'s' if len(system) > 1 else ''}")
            for s in system[:2]:
                parts.append(f"     • {s.title}")

        # Notifications
        notifs = by_category.get("notification", [])
        if notifs:
            parts.append(f"  📬 {len(notifs)} notification{'s' if len(notifs) > 1 else ''} pending")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "generated_at": self.generated_at,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "duration_hours": round(self.duration_hours, 1),
            "item_count": len(self.items),
            "items": [item.to_dict() for item in self.items],
            "summary": self.to_natural_language(),
        }


# ── ActivityDigest Engine ────────────────────────────────────────────────────


class ActivityDigestEngine:
    """
    Generates catch-up briefings from accumulated events.

    Collects events from various sources (notifications, goals, insights,
    system health) and produces a prioritized, human-readable digest.

    Usage:
        engine = ActivityDigestEngine()

        # Record events as they happen
        engine.record_event("user_1", DigestItem(
            category="goal",
            title="Goal completed: Deploy v2.0",
            importance=0.8,
        ))

        # When user returns, generate digest
        digest = engine.generate_digest("user_1")
        print(digest.to_natural_language())
    """

    def __init__(
        self,
        max_events_per_tenant: int = 500,
        user_model: Any | None = None,
        project_graph: Any | None = None,
    ) -> None:
        self._max_events = max_events_per_tenant
        self._user_model = user_model  # Optional UserModelEngine
        self._project_graph = project_graph  # Optional ProjectGraph
        # Per-tenant event buffers (accumulated since last digest)
        self._buffers: dict[str, list[DigestItem]] = defaultdict(list)
        # Track when each tenant was last active
        self._last_active: dict[str, float] = {}
        # Track when last digest was generated
        self._last_digest: dict[str, float] = {}

        logger.info(
            "ActivityDigestEngine initialized (max_events=%d, user_model=%s, project_graph=%s)",
            max_events_per_tenant,
            "connected" if user_model else "none",
            "connected" if project_graph else "none",
        )

    def record_event(self, tenant_id: str, item: DigestItem) -> None:
        """Record an event for a tenant's next digest."""
        buffer = self._buffers[tenant_id]
        buffer.append(item)

        # Evict oldest if over capacity
        if len(buffer) > self._max_events:
            self._buffers[tenant_id] = buffer[-self._max_events :]

    def record_activity(self, tenant_id: str) -> None:
        """Mark a tenant as currently active (resets absence timer)."""
        self._last_active[tenant_id] = time.time()

    def get_absence_duration(self, tenant_id: str) -> float:
        """Get how long a tenant has been away, in seconds."""
        last = self._last_active.get(tenant_id, 0.0)
        if last == 0.0:
            return 0.0
        return time.time() - last

    def generate_digest(
        self,
        tenant_id: str,
        since: float | None = None,
        max_items: int = 20,
    ) -> Digest:
        """
        Generate a digest of activity since the given timestamp.

        If `since` is None, uses the last digest time or last active time.
        """
        now = time.time()
        if since is None:
            since = self._last_digest.get(
                tenant_id,
                self._last_active.get(tenant_id, now - 86400),  # Default: last 24h
            )

        buffer = self._buffers.get(tenant_id, [])

        # Filter to events within the time window
        relevant = [item for item in buffer if item.timestamp >= since]

        # Sort by importance (descending), then recency
        relevant.sort(key=lambda i: (i.importance, i.timestamp), reverse=True)

        # Truncate to max items
        items = relevant[:max_items]

        digest = Digest(
            tenant_id=tenant_id,
            items=items,
            period_start=since,
            period_end=now,
        )

        # Enrich with project context if ProjectGraph is connected
        if self._project_graph:
            self._enrich_with_project_context(digest, tenant_id)

        # Respect user verbosity preference
        if self._user_model:
            self._apply_verbosity_preference(digest, tenant_id)

        # Update tracking
        self._last_digest[tenant_id] = now
        # Clear consumed events
        self._buffers[tenant_id] = [item for item in buffer if item.timestamp > now]

        logger.debug(
            "Generated digest for tenant '%s': %d items over %.1f hours",
            tenant_id,
            len(digest.items),
            digest.duration_hours,
        )
        return digest

    def has_pending_events(self, tenant_id: str) -> bool:
        """Check if a tenant has undigested events."""
        return len(self._buffers.get(tenant_id, [])) > 0

    def pending_count(self, tenant_id: str) -> int:
        """Count pending events for a tenant."""
        return len(self._buffers.get(tenant_id, []))

    def stats(self) -> dict[str, Any]:
        """Aggregate stats across all tenants."""
        return {
            "tenant_count": len(self._buffers),
            "total_pending": sum(len(b) for b in self._buffers.values()),
            "tenants_with_events": sum(1 for b in self._buffers.values() if b),
            "user_model_connected": self._user_model is not None,
            "project_graph_connected": self._project_graph is not None,
        }

    # ── Integration Helpers ────────────────────────────────────────────

    def _enrich_with_project_context(self, digest: Digest, tenant_id: str) -> None:
        """Add project goal status to the digest."""
        try:
            # Get all projects and check for active goals
            projects = self._project_graph.list_projects(tenant_id)
            for project in projects[:5]:  # Max 5 projects in digest
                project_id = project.get("id", "") if isinstance(project, dict) else getattr(project, "id", "")
                if not project_id:
                    continue
                goals = self._project_graph.get_active_goals(project_id)
                if goals:
                    project_name = project.get("name", project_id) if isinstance(project, dict) else getattr(project, "name", project_id)
                    digest.items.append(DigestItem(
                        category="goal",
                        title=f"{project_name}: {len(goals)} active goal(s)",
                        detail=", ".join(
                            getattr(g, "name", str(g))[:40] for g in goals[:3]
                        ),
                        importance=0.6,
                    ))
        except Exception as e:
            logger.debug("Failed to enrich digest with project context: %s", e)

    def _apply_verbosity_preference(self, digest: Digest, tenant_id: str) -> None:
        """Trim digest items based on user verbosity preference."""
        try:
            model = self._user_model.get_model(tenant_id)
            verbosity_pref = model.preferences.get("verbosity")
            if verbosity_pref and verbosity_pref.value == "concise":
                # Keep only high-importance items for concise users
                digest.items = [
                    item for item in digest.items if item.importance >= 0.5
                ][:10]
        except Exception as e:
            logger.debug("Failed to apply verbosity preference: %s", e)
