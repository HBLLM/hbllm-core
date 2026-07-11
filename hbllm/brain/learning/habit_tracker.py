"""
HabitTracker — Temporal pattern mining for proactive user assistance.

Monitors episodic memory to discover user routines and behavioral patterns:
- "User always checks email at 9 AM"
- "User prefers concise answers on mobile"
- "User usually asks about project X on Mondays"

Detected habits feed into PersonaEngine (context modulation) and
NotificationGateway (proactive suggestions at the right time).

Integrations:
    UserModelEngine  → Cross-validates detected habits against learned active_hours

Bus Topics:
    habit.detected    → New habit pattern discovered
    habit.suggestion  → Time-triggered habit-based suggestion
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass
class HabitPattern:
    """A detected behavioral pattern."""

    id: str
    tenant_id: str
    description: str
    # When this habit typically occurs
    hour_of_day: int | None = None  # 0-23
    day_of_week: int | None = None  # 0=Monday, 6=Sunday
    # What the habit involves
    action_type: str = ""  # e.g. "query", "tool_use", "goal_create"
    topic: str = ""  # e.g. "email", "code_review", "project_x"
    # Confidence and frequency
    confidence: float = 0.0  # 0.0 to 1.0
    occurrence_count: int = 0
    total_observations: int = 0
    # Metadata
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_triggered: float = 0.0
    suppressed: bool = False  # User can suppress unwanted habits

    @property
    def frequency(self) -> float:
        """How often this pattern occurs relative to observations."""
        if self.total_observations == 0:
            return 0.0
        return self.occurrence_count / self.total_observations

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "description": self.description,
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "action_type": self.action_type,
            "topic": self.topic,
            "confidence": round(self.confidence, 3),
            "occurrence_count": self.occurrence_count,
            "total_observations": self.total_observations,
            "frequency": round(self.frequency, 3),
            "suppressed": self.suppressed,
        }


@dataclass
class InteractionEvent:
    """A single user interaction event for pattern analysis."""

    tenant_id: str
    timestamp: float
    action_type: str  # "query", "tool_use", "goal_create", "feedback"
    topic: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def hour(self) -> int:
        return time.localtime(self.timestamp).tm_hour

    @property
    def weekday(self) -> int:
        return time.localtime(self.timestamp).tm_wday


# ── HabitTracker ─────────────────────────────────────────────────────────────


class HabitTracker:
    """
    Discovers and tracks user behavioral patterns over time.

    Analyzes interaction events to find temporal regularities:
    - Hourly patterns (e.g. morning email check)
    - Weekly patterns (e.g. Monday planning sessions)
    - Topic affinities (e.g. always asks about X after Y)

    Usage:
        tracker = HabitTracker()

        # Record interactions
        tracker.record_event(InteractionEvent(
            tenant_id="user_1",
            timestamp=time.time(),
            action_type="query",
            topic="email",
        ))

        # Check for mature habits
        habits = tracker.get_habits("user_1", min_confidence=0.6)

        # Get suggestions for current time
        suggestions = tracker.get_suggestions("user_1")
    """

    def __init__(
        self,
        min_observations: int = 5,
        confidence_threshold: float = 0.5,
        max_events_per_tenant: int = 2000,
        user_model: Any | None = None,
    ) -> None:
        self._min_observations = min_observations
        self._confidence_threshold = confidence_threshold
        self._max_events = max_events_per_tenant
        self._user_model = user_model  # Optional UserModelEngine for cross-validation

        # Per-tenant event history
        self._events: dict[str, list[InteractionEvent]] = defaultdict(list)
        # Per-tenant discovered habits
        self._habits: dict[str, dict[str, HabitPattern]] = defaultdict(dict)

        logger.info(
            "HabitTracker initialized (min_obs=%d, confidence=%.1f, user_model=%s)",
            min_observations,
            confidence_threshold,
            "connected" if user_model else "none",
        )

    def record_event(self, event: InteractionEvent) -> None:
        """Record a user interaction event for pattern analysis."""
        events = self._events[event.tenant_id]
        events.append(event)

        # Evict oldest events if over capacity
        if len(events) > self._max_events:
            self._events[event.tenant_id] = events[-self._max_events :]

        # Re-analyze patterns periodically (every 10 events)
        if len(events) % 10 == 0:
            self._analyze_patterns(event.tenant_id)

    def _analyze_patterns(self, tenant_id: str) -> None:
        """Analyze event history to discover or update habit patterns."""
        events = self._events.get(tenant_id, [])
        if len(events) < self._min_observations:
            return

        # Analyze hourly patterns per action type
        self._analyze_hourly_patterns(tenant_id, events)
        # Analyze weekly patterns per action type
        self._analyze_weekly_patterns(tenant_id, events)
        # Analyze topic affinities
        self._analyze_topic_patterns(tenant_id, events)

    def _analyze_hourly_patterns(self, tenant_id: str, events: list[InteractionEvent]) -> None:
        """Find actions that cluster around specific hours."""
        # Group events by (action_type, hour)
        hourly: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        action_totals: dict[str, int] = defaultdict(int)

        for event in events:
            hourly[event.action_type][event.hour] += 1
            action_totals[event.action_type] += 1

        for action_type, hour_counts in hourly.items():
            total = action_totals[action_type]
            if total < self._min_observations:
                continue

            for hour, count in hour_counts.items():
                ratio = count / total
                # A habit requires concentration: > 30% of actions at this hour
                if ratio < 0.3:
                    continue

                # Confidence based on consistency and sample size
                confidence = self._compute_confidence(count, total, ratio)

                # Cross-validate with UserModel's learned active_hours
                confidence = self._cross_validate_with_user_model(
                    tenant_id,
                    confidence,
                    hour=hour,
                )

                if confidence < self._confidence_threshold:
                    continue

                habit_id = f"hourly_{action_type}_{hour}"
                habit = self._habits[tenant_id].get(habit_id)

                if habit is None:
                    habit = HabitPattern(
                        id=habit_id,
                        tenant_id=tenant_id,
                        description=(
                            f"You typically {action_type.replace('_', ' ')} around {hour:02d}:00"
                        ),
                        hour_of_day=hour,
                        action_type=action_type,
                        confidence=confidence,
                        occurrence_count=count,
                        total_observations=total,
                    )
                    self._habits[tenant_id][habit_id] = habit
                    logger.info(
                        "New habit detected: %s (%.0f%%)", habit.description, confidence * 100
                    )
                else:
                    habit.confidence = confidence
                    habit.occurrence_count = count
                    habit.total_observations = total
                    habit.last_seen = time.time()

    def _analyze_weekly_patterns(self, tenant_id: str, events: list[InteractionEvent]) -> None:
        """Find actions that cluster around specific days of the week."""
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        weekly: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        action_totals: dict[str, int] = defaultdict(int)

        for event in events:
            weekly[event.action_type][event.weekday] += 1
            action_totals[event.action_type] += 1

        for action_type, day_counts in weekly.items():
            total = action_totals[action_type]
            if total < self._min_observations:
                continue

            for day, count in day_counts.items():
                ratio = count / total
                if ratio < 0.25:  # More lenient for weekly (7 days vs 24 hours)
                    continue

                confidence = self._compute_confidence(count, total, ratio)
                if confidence < self._confidence_threshold:
                    continue

                habit_id = f"weekly_{action_type}_{day}"
                habit = self._habits[tenant_id].get(habit_id)

                if habit is None:
                    habit = HabitPattern(
                        id=habit_id,
                        tenant_id=tenant_id,
                        description=(
                            f"You typically {action_type.replace('_', ' ')} on {day_names[day]}s"
                        ),
                        day_of_week=day,
                        action_type=action_type,
                        confidence=confidence,
                        occurrence_count=count,
                        total_observations=total,
                    )
                    self._habits[tenant_id][habit_id] = habit
                    logger.info(
                        "New habit detected: %s (%.0f%%)", habit.description, confidence * 100
                    )
                else:
                    habit.confidence = confidence
                    habit.occurrence_count = count
                    habit.total_observations = total
                    habit.last_seen = time.time()

    def _analyze_topic_patterns(self, tenant_id: str, events: list[InteractionEvent]) -> None:
        """Find recurring topic interests."""
        topic_counts: dict[str, int] = defaultdict(int)
        total_with_topic = 0

        for event in events:
            if event.topic:
                topic_counts[event.topic] += 1
                total_with_topic += 1

        if total_with_topic < self._min_observations:
            return

        for topic, count in topic_counts.items():
            ratio = count / total_with_topic
            if ratio < 0.15:
                continue

            confidence = self._compute_confidence(count, total_with_topic, ratio)
            if confidence < self._confidence_threshold:
                continue

            habit_id = f"topic_{topic}"
            if habit_id not in self._habits[tenant_id]:
                habit = HabitPattern(
                    id=habit_id,
                    tenant_id=tenant_id,
                    description=f"You frequently engage with '{topic}'",
                    action_type="topic_interest",
                    topic=topic,
                    confidence=confidence,
                    occurrence_count=count,
                    total_observations=total_with_topic,
                )
                self._habits[tenant_id][habit_id] = habit

    def _compute_confidence(self, count: int, total: int, ratio: float) -> float:
        """
        Compute confidence score combining frequency ratio and sample size.

        Uses a Bayesian-inspired formula that rewards both consistency
        (high ratio) and sufficient evidence (high count).
        """
        # Sample size factor: approaches 1.0 as count grows
        sample_factor = 1.0 - math.exp(-count / self._min_observations)
        # Combine ratio and sample size
        return ratio * sample_factor

    def _cross_validate_with_user_model(
        self,
        tenant_id: str,
        confidence: float,
        hour: int | None = None,
        day: int | None = None,
    ) -> float:
        """Cross-validate a habit's confidence against UserModel data.

        If the user's learned active_hours confirms the habit hour,
        boost confidence. If the hour is known-inactive, penalize.
        """
        if not self._user_model:
            return confidence

        try:
            model = self._user_model.get_model(tenant_id)
            active_hours = getattr(model, "active_hours", {})

            if hour is not None and active_hours:
                activity_level = active_hours.get(hour, 0.5)
                if activity_level > 0.6:
                    # User is known-active at this hour — boost confidence
                    confidence = min(1.0, confidence * 1.15)
                elif activity_level < 0.1:
                    # User is known-inactive — habit is probably noise
                    confidence *= 0.7

            active_days = getattr(model, "active_days", {})
            if day is not None and active_days:
                day_activity = active_days.get(day, 0.5)
                if day_activity > 0.6:
                    confidence = min(1.0, confidence * 1.1)
                elif day_activity < 0.1:
                    confidence *= 0.8

        except Exception as e:
            logger.debug("Failed to cross-validate with UserModel: %s", e)

        return confidence

    def get_habits(
        self,
        tenant_id: str,
        min_confidence: float = 0.0,
        include_suppressed: bool = False,
    ) -> list[HabitPattern]:
        """Get all detected habits for a tenant, sorted by confidence."""
        habits = list(self._habits.get(tenant_id, {}).values())
        if not include_suppressed:
            habits = [h for h in habits if not h.suppressed]
        if min_confidence > 0:
            habits = [h for h in habits if h.confidence >= min_confidence]
        return sorted(habits, key=lambda h: h.confidence, reverse=True)

    def get_suggestions(self, tenant_id: str) -> list[HabitPattern]:
        """
        Get habits that are relevant right now (matching current time).

        Returns habits whose hour_of_day or day_of_week matches the
        current time, and haven't been triggered recently.
        """
        now = time.localtime()
        current_hour = now.tm_hour
        current_day = now.tm_wday
        current_ts = time.time()

        suggestions = []
        for habit in self.get_habits(tenant_id, min_confidence=self._confidence_threshold):
            # Skip recently triggered habits (cooldown: 1 hour)
            if current_ts - habit.last_triggered < 3600:
                continue

            match = False
            if habit.hour_of_day is not None and habit.hour_of_day == current_hour:
                match = True
            if habit.day_of_week is not None and habit.day_of_week == current_day:
                match = True

            if match:
                suggestions.append(habit)

        return suggestions

    def suppress_habit(self, tenant_id: str, habit_id: str) -> bool:
        """Suppress a habit so it no longer generates suggestions."""
        habits = self._habits.get(tenant_id, {})
        if habit_id in habits:
            habits[habit_id].suppressed = True
            logger.info("Habit suppressed: %s for tenant '%s'", habit_id, tenant_id)
            return True
        return False

    def stats(self, tenant_id: str) -> dict[str, Any]:
        """Get habit tracking stats for a tenant."""
        events = self._events.get(tenant_id, [])
        habits = self._habits.get(tenant_id, {})
        return {
            "total_events": len(events),
            "total_habits": len(habits),
            "active_habits": sum(1 for h in habits.values() if not h.suppressed),
            "suppressed_habits": sum(1 for h in habits.values() if h.suppressed),
            "high_confidence": sum(
                1 for h in habits.values() if h.confidence >= 0.7 and not h.suppressed
            ),
        }
