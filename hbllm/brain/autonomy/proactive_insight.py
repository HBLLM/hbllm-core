"""Proactive Insight Generator — context-aware suggestions.

Generates proactive insights by correlating:
    1. Temporal patterns (from TemporalPatternDetector)
    2. Spatial context (from SpatialMemory)
    3. World state (from WorldStateEngine)
    4. User habits (from ValueMemory)
    5. Calendar events (from external calendar feeds)
    6. User model (from UserModelEngine) — stress, interests, preferences
    7. Project context (from ProjectGraph) — active goals, blockers

Produces suggestions like:
    "It's Monday 9am — you usually start coding now. Open VS Code?"
    "You're in the kitchen and it's dinner time — want a recipe suggestion?"
    "Battery at 15% — you have a 2-hour meeting in 30 minutes. Plug in."
    "You have a blocker on the auth module — want me to investigate?"

Architecture:
    Runs as a slow-path check during the AutonomyCore's tick loop.
    Returns Message objects that get published via the notification pipeline.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


class ProactiveInsightGenerator:
    """Generates context-aware proactive suggestions.

    This is the "brain" of proactive behavior — it correlates multiple
    data sources to produce useful, timely suggestions.

    Usage::

        generator = ProactiveInsightGenerator(
            temporal_detector=temporal_detector,
            spatial_memory=spatial_memory,
            world_engine=world_engine,
        )
        insights = await generator.generate_insights(tenant_id="user1")
    """

    def __init__(
        self,
        temporal_detector: Any | None = None,
        spatial_memory: Any | None = None,
        world_engine: Any | None = None,
        value_db: Any | None = None,
        user_model: Any | None = None,
        project_graph: Any | None = None,
    ) -> None:
        self.temporal = temporal_detector
        self.spatial = spatial_memory
        self.world = world_engine
        self.value_db = value_db
        self._user_model = user_model  # Optional UserModelEngine
        self._project_graph = project_graph  # Optional ProjectGraph

        # Track what insights we've already generated (prevent repeats)
        self._generated: dict[str, float] = {}  # insight_key → last_generated_time
        self._cooldown_s = 3600  # 1 hour between same insight

        # Telemetry
        self._total_generated = 0
        self._total_suppressed = 0

    async def generate_insights(
        self,
        tenant_id: str = "default",
    ) -> list[Message]:
        """Generate proactive insights based on current context.

        Returns a list of Message objects ready for the notification pipeline.
        """
        insights: list[Message] = []
        now = time.time()
        dt = datetime.now(timezone.utc)

        # 1. Temporal pattern insights
        if self.temporal:
            insights.extend(self._temporal_insights(tenant_id, dt))

        # 2. Spatial + temporal combo insights
        if self.spatial and self.world:
            insights.extend(self._spatial_temporal_insights(tenant_id, dt))

        # 3. Hardware-aware insights
        if self.world:
            insights.extend(self._hardware_context_insights(tenant_id, dt))

        # 4. Routine-based insights
        insights.extend(self._routine_insights(tenant_id, dt))

        # 5. Project-aware insights (active goals, blockers)
        if self._project_graph:
            insights.extend(self._project_insights(tenant_id, dt))

        # Stress-based suppression: when user is stressed, reduce suggestion volume
        max_insights = self._get_max_insights(tenant_id)

        # Filter out recently generated insights
        filtered: list[Message] = []
        for msg in insights:
            key = msg.payload.get("insight_key", "")
            if key:
                last = self._generated.get(key, 0)
                if now - last < self._cooldown_s:
                    self._total_suppressed += 1
                    continue
                self._generated[key] = now

            self._total_generated += 1
            filtered.append(msg)

            if len(filtered) >= max_insights:
                break

        # Prune stale insight keys (older than 24 hours) to prevent memory growth
        stale_cutoff = now - 86400
        stale_keys = [k for k, v in self._generated.items() if v < stale_cutoff]
        for key in stale_keys:
            del self._generated[key]

        return filtered

    def _get_max_insights(self, tenant_id: str) -> int:
        """Compute max insights per cycle based on user stress level."""
        if not self._user_model:
            return 5  # Default

        try:
            model = self._user_model.get_model(tenant_id)
            if model.stress_level > 0.7:
                return 1  # High stress → minimal suggestions
            if model.stress_level > 0.4:
                return 3  # Moderate stress → fewer suggestions
            return 5
        except Exception:
            return 5

    def _temporal_insights(
        self,
        tenant_id: str,
        dt: datetime,
    ) -> list[Message]:
        """Generate insights from temporal patterns."""
        messages: list[Message] = []

        try:
            patterns = self.temporal.get_stored_patterns(tenant_id)
        except Exception:
            return []

        for pattern in patterns[:5]:  # Max 5 pattern insights
            if pattern.confidence < 0.6:
                continue

            # Check if current time matches the pattern
            if pattern.pattern_type == "time_of_day":
                _block = pattern.parameters.get("block", "")
                start_h = pattern.parameters.get("start_hour", 0)
                end_h = pattern.parameters.get("end_hour", 24)
                if start_h <= dt.hour < end_h:
                    messages.append(
                        self._make_insight(
                            title=f"🕐 {pattern.domain.title()} Time",
                            body=pattern.description,
                            priority="suggestion",
                            insight_key=f"temporal_{pattern.pattern_id}",
                            tenant_id=tenant_id,
                        )
                    )

            elif pattern.pattern_type == "day_of_week":
                day_num = pattern.parameters.get("day_of_week", -1)
                if dt.weekday() == day_num:
                    messages.append(
                        self._make_insight(
                            title=f"📅 {pattern.domain.title()} Day",
                            body=pattern.description,
                            priority="suggestion",
                            insight_key=f"temporal_{pattern.pattern_id}",
                            tenant_id=tenant_id,
                        )
                    )

        return messages

    def _spatial_temporal_insights(
        self,
        tenant_id: str,
        dt: datetime,
    ) -> list[Message]:
        """Generate insights from spatial + temporal context."""
        messages: list[Message] = []

        # Get world state for location hints
        world_state = self.world.get_state() if self.world else {}
        _iot_devices = world_state.get("iot_devices", {})  # noqa: F841 — reserved for IoT presence

        # Check if user's location is known
        try:
            location_domains = self.spatial.get_domains_by_location(tenant_id)
        except Exception:
            return []

        for location_id, domains in location_domains.items():
            if not domains:
                continue

            # Check if user might be in this location (via IoT motion)
            # This is a heuristic — real implementation would use presence detection
            top_domain = domains[0]

            # Time-appropriate suggestions
            hour = dt.hour
            if top_domain == "cooking" and 17 <= hour <= 20:
                messages.append(
                    self._make_insight(
                        title="👨‍🍳 Dinner Time",
                        body=f"You usually cook around this time in the {location_id}. "
                        f"Want a recipe suggestion?",
                        priority="suggestion",
                        insight_key=f"spatial_cooking_{location_id}",
                        tenant_id=tenant_id,
                    )
                )
            elif top_domain == "exercise" and 6 <= hour <= 8:
                messages.append(
                    self._make_insight(
                        title="🏋️ Workout Time",
                        body=f"You typically exercise in the morning at the {location_id}. "
                        f"Ready for today's workout?",
                        priority="suggestion",
                        insight_key=f"spatial_exercise_{location_id}",
                        tenant_id=tenant_id,
                    )
                )

        return messages

    def _hardware_context_insights(
        self,
        tenant_id: str,
        dt: datetime,
    ) -> list[Message]:
        """Generate insights from hardware state + calendar context."""
        messages: list[Message] = []

        world_state = self.world.get_state() if self.world else {}
        hardware = world_state.get("hardware", {})
        calendar = world_state.get("calendar", {})

        battery = hardware.get("battery_level") or hardware.get("battery")
        if battery is not None:
            # Low battery + upcoming calendar event
            if battery < 0.2 and calendar:
                next_event = calendar.get("next_event", {})
                if next_event:
                    minutes_until = next_event.get("minutes_until", 999)
                    if minutes_until < 60:
                        messages.append(
                            self._make_insight(
                                title="🔋 Charge Before Meeting",
                                body=f"Battery at {battery:.0%} and you have "
                                f"'{next_event.get('summary', 'an event')}' in "
                                f"{minutes_until} minutes. Consider plugging in.",
                                priority="high",
                                insight_key="battery_meeting",
                                tenant_id=tenant_id,
                            )
                        )

        return messages

    def _routine_insights(
        self,
        tenant_id: str,
        dt: datetime,
    ) -> list[Message]:
        """Generate time-based routine insights."""
        messages: list[Message] = []
        hour = dt.hour

        # Hydration reminder (every 2 hours during work)
        if 9 <= hour <= 17 and hour % 2 == 0 and dt.minute < 15:
            messages.append(
                self._make_insight(
                    title="💧 Stay Hydrated",
                    body="Time for a water break! You've been working for a while.",
                    priority="suggestion",
                    insight_key=f"hydration_{hour}",
                    tenant_id=tenant_id,
                )
            )

        # Break reminder (after extended work)
        if hour in (11, 15) and dt.minute < 15:
            messages.append(
                self._make_insight(
                    title="☕ Take a Break",
                    body="Consider a short break to stretch and rest your eyes.",
                    priority="suggestion",
                    insight_key=f"break_{hour}",
                    tenant_id=tenant_id,
                )
            )

        return messages

    def _make_insight(
        self,
        title: str,
        body: str,
        priority: str = "suggestion",
        insight_key: str = "",
        tenant_id: str = "default",
    ) -> Message:
        """Create a proactive insight message."""
        return Message(
            type=MessageType.EVENT,
            source_node_id="proactive_insight_generator",
            topic="proactive.push",
            tenant_id=tenant_id,
            payload={
                "title": title,
                "body": body,
                "priority": priority,
                "category": "insight",
                "insight_key": insight_key,
            },
        )

    def stats(self) -> dict[str, Any]:
        """Generator statistics."""
        return {
            "total_generated": self._total_generated,
            "total_suppressed": self._total_suppressed,
            "tracked_insights": len(self._generated),
            "cooldown_s": self._cooldown_s,
            "user_model_connected": self._user_model is not None,
            "project_graph_connected": self._project_graph is not None,
        }

    # ── Project-Aware Insights ────────────────────────────────────────

    def _project_insights(
        self,
        tenant_id: str,
        dt: datetime,
    ) -> list[Message]:
        """Generate insights from active projects, goals, and blockers."""
        messages: list[Message] = []

        try:
            active_goals = self._project_graph.get_active_goals(tenant_id)
        except Exception:
            active_goals = []

        if not active_goals:
            return []

        # Check for stale goals (no progress in 3+ days)
        now = time.time()
        for goal in active_goals[:3]:  # Max 3 project insights
            last_activity = goal.get("last_activity", now)
            days_stale = (now - last_activity) / 86400.0

            if days_stale > 3.0:
                project_name = goal.get("project_name", "a project")
                goal_desc = goal.get("description", "a goal")[:60]
                messages.append(
                    self._make_insight(
                        title=f"📌 Stale Goal: {project_name}",
                        body=f"'{goal_desc}' hasn't had activity in "
                        f"{int(days_stale)} days. Want me to investigate or reprioritize?",
                        priority="normal",
                        insight_key=f"stale_goal_{goal.get('goal_id', '')}",
                        tenant_id=tenant_id,
                    )
                )

        return messages
