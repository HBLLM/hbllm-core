"""Proactive Insight Generator — context-aware suggestions.

Generates proactive insights by correlating:
    1. Temporal patterns (from TemporalPatternDetector)
    2. Spatial context (from SpatialMemory)
    3. World state (from WorldStateEngine)
    4. User habits (from ValueMemory)
    5. Calendar events (from external calendar feeds)

Produces suggestions like:
    "It's Monday 9am — you usually start coding now. Open VS Code?"
    "You're in the kitchen and it's dinner time — want a recipe suggestion?"
    "Battery at 15% — you have a 2-hour meeting in 30 minutes. Plug in."

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
    ) -> None:
        self.temporal = temporal_detector
        self.spatial = spatial_memory
        self.world = world_engine
        self.value_db = value_db

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

        return filtered

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
        }
