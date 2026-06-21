"""Social Timing — contextual communication timing.

Determines the optimal time and channel to deliver messages:
    - "Don't tell them about the bill during dinner"
    - "Hold non-urgent updates until morning"
    - "Urgent security alerts: always immediate"

Uses:
    1. Time-of-day preferences
    2. User engagement state (from InterruptDetector)
    3. Message urgency classification
    4. Social context (meeting, dinner, sleep)
    5. Channel preference (push, voice, display)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DeliveryDecision:
    """Decision about when and how to deliver a message."""

    deliver_now: bool = True
    delay_until: float | None = None  # Unix timestamp
    channel: str = "push"  # "push", "voice", "display", "email", "hold"
    reason: str = ""
    priority_override: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "deliver_now": self.deliver_now,
            "delay_until": self.delay_until,
            "channel": self.channel,
            "reason": self.reason,
        }


@dataclass
class SocialContext:
    """Current social context for timing decisions."""

    in_meeting: bool = False
    meeting_end_time: float | None = None
    in_focus_mode: bool = False
    focus_end_time: float | None = None
    is_sleeping: bool = False
    wake_time: float | None = None
    is_driving: bool = False
    social_setting: str = ""  # "dinner", "company", "alone"


class SocialTimingEngine:
    """Determines optimal message delivery timing.

    Usage::

        engine = SocialTimingEngine()
        decision = engine.evaluate(
            priority="suggestion",
            category="weather",
            social_context=social_context,
        )
        if decision.deliver_now:
            send_notification(...)
        else:
            schedule_delivery(decision.delay_until)
    """

    # Priority levels ordered by urgency
    PRIORITY_LEVELS = {
        "critical": 4,  # Always deliver immediately
        "high": 3,  # Deliver unless sleeping
        "normal": 2,  # Respect social context
        "suggestion": 1,  # Hold for optimal timing
        "info": 0,  # Batch and deliver during idle
    }

    # Quiet hours (UTC) — user-configurable
    DEFAULT_QUIET_START = 23  # 11 PM
    DEFAULT_QUIET_END = 7  # 7 AM

    def __init__(
        self,
        quiet_start_hour: int = 23,
        quiet_end_hour: int = 7,
        batch_info_interval_s: float = 1800,  # Batch info messages every 30 min
    ) -> None:
        self.quiet_start = quiet_start_hour
        self.quiet_end = quiet_end_hour
        self.batch_interval = batch_info_interval_s

        self._social_context = SocialContext()
        self._held_messages: list[dict[str, Any]] = []
        self._last_batch_delivery = 0.0

        # Telemetry
        self._delivered_now = 0
        self._delayed = 0
        self._held = 0

    def update_context(self, context: SocialContext) -> None:
        """Update the current social context."""
        self._social_context = context

    def evaluate(
        self,
        priority: str = "normal",
        category: str = "",
        content: str = "",
        social_context: SocialContext | None = None,
    ) -> DeliveryDecision:
        """Evaluate the optimal delivery timing for a message.

        Args:
            priority: Message priority level.
            category: Message category (e.g., "weather", "security", "iot").
            content: Message content (for context-aware decisions).
            social_context: Override the stored social context.

        Returns:
            DeliveryDecision with timing and channel recommendation.
        """
        ctx = social_context or self._social_context
        now = time.time()
        dt = datetime.now(timezone.utc)
        hour = dt.hour
        priority_level = self.PRIORITY_LEVELS.get(priority, 2)

        # Rule 1: Critical messages always deliver immediately
        if priority_level >= 4:
            self._delivered_now += 1
            return DeliveryDecision(
                deliver_now=True,
                channel=self._select_channel(ctx, urgent=True),
                reason="Critical priority — immediate delivery",
            )

        # Rule 2: Security alerts always deliver (high+)
        if category == "security" and priority_level >= 3:
            self._delivered_now += 1
            return DeliveryDecision(
                deliver_now=True,
                channel=self._select_channel(ctx, urgent=True),
                reason="Security alert — immediate delivery",
            )

        # Rule 3: Sleeping — hold everything except critical
        if ctx.is_sleeping:
            self._delayed += 1
            wake = ctx.wake_time or self._next_quiet_end(now)
            return DeliveryDecision(
                deliver_now=False,
                delay_until=wake,
                channel="hold",
                reason="User is sleeping — held until wake time",
            )

        # Rule 4: In meeting — hold normal and below
        if ctx.in_meeting and priority_level < 3:
            self._delayed += 1
            end = ctx.meeting_end_time or now + 3600
            return DeliveryDecision(
                deliver_now=False,
                delay_until=end,
                channel="hold",
                reason="User is in a meeting — held until meeting ends",
            )

        # Rule 5: Focus mode — hold suggestions and info
        if ctx.in_focus_mode and priority_level < 2:
            self._delayed += 1
            end = ctx.focus_end_time or now + 1800
            return DeliveryDecision(
                deliver_now=False,
                delay_until=end,
                channel="hold",
                reason="Focus mode active — held until focus ends",
            )

        # Rule 6: Driving — voice only for high+
        if ctx.is_driving:
            if priority_level >= 3:
                self._delivered_now += 1
                return DeliveryDecision(
                    deliver_now=True,
                    channel="voice",
                    reason="User is driving — voice delivery",
                )
            else:
                self._held += 1
                return DeliveryDecision(
                    deliver_now=False,
                    channel="hold",
                    reason="User is driving — non-urgent held",
                )

        # Rule 7: Quiet hours — hold suggestions and info
        if self._is_quiet_hours(hour) and priority_level < 2:
            self._delayed += 1
            return DeliveryDecision(
                deliver_now=False,
                delay_until=self._next_quiet_end(now),
                channel="hold",
                reason="Quiet hours — held until morning",
            )

        # Rule 8: Info messages — batch delivery
        if priority_level == 0:
            if now - self._last_batch_delivery < self.batch_interval:
                self._held += 1
                return DeliveryDecision(
                    deliver_now=False,
                    delay_until=self._last_batch_delivery + self.batch_interval,
                    channel="hold",
                    reason="Info message — batched for next delivery window",
                )
            else:
                self._last_batch_delivery = now

        # Default: deliver now
        self._delivered_now += 1
        return DeliveryDecision(
            deliver_now=True,
            channel=self._select_channel(ctx),
            reason="No delivery restrictions — sending now",
        )

    def _select_channel(
        self,
        ctx: SocialContext,
        urgent: bool = False,
    ) -> str:
        """Select the best delivery channel based on context."""
        if ctx.is_driving:
            return "voice"
        if urgent:
            return "push"
        if ctx.social_setting in ("dinner", "company"):
            return "display"  # Silent display notification
        return "push"

    def _is_quiet_hours(self, hour: int) -> bool:
        """Check if current hour is within quiet hours."""
        if self.quiet_start > self.quiet_end:
            # Crosses midnight (e.g., 23:00 - 07:00)
            return hour >= self.quiet_start or hour < self.quiet_end
        return self.quiet_start <= hour < self.quiet_end

    def _next_quiet_end(self, now: float) -> float:
        """Calculate the next quiet hours end timestamp."""
        dt = datetime.fromtimestamp(now, tz=timezone.utc)
        if dt.hour < self.quiet_end:
            # Same day
            target = dt.replace(hour=self.quiet_end, minute=0, second=0)
        else:
            # Next day
            target = dt.replace(hour=self.quiet_end, minute=0, second=0)
            target = target.replace(day=target.day + 1)
        return target.timestamp()

    def stats(self) -> dict[str, Any]:
        return {
            "delivered_now": self._delivered_now,
            "delayed": self._delayed,
            "held": self._held,
            "quiet_hours": f"{self.quiet_start:02d}:00-{self.quiet_end:02d}:00",
            "current_context": {
                "in_meeting": self._social_context.in_meeting,
                "in_focus": self._social_context.in_focus_mode,
                "sleeping": self._social_context.is_sleeping,
                "driving": self._social_context.is_driving,
            },
        }
