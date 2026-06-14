"""Idle Detector — tracks user activity and emits idle/return events.

Monitors the time since the last user interaction on the MessageBus.
When idle exceeds a threshold, emits ``system.user_idle`` for sleep/
consolidation triggers. When the user returns, emits ``system.user_returned``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


class IdleDetector:
    """Proactive handler that detects user idle/return transitions.

    Tracks the timestamp of the last bus message with a ``user.*`` topic.
    When idle exceeds the threshold, emits a single idle event. When the
    user returns, emits a return event with the idle duration.

    Usage::

        detector = IdleDetector(idle_threshold_s=300)
        autonomy_core.add_proactive_handler("idle_detector", detector.check)

        # Also subscribe to user messages to track activity:
        bus.subscribe("user.*", detector.on_user_activity)
    """

    def __init__(
        self,
        idle_threshold_s: float = 300.0,  # 5 minutes
        deep_idle_threshold_s: float = 1800.0,  # 30 minutes
    ) -> None:
        self.idle_threshold_s = idle_threshold_s
        self.deep_idle_threshold_s = deep_idle_threshold_s

        self._last_activity: float = time.monotonic()
        self._is_idle = False
        self._idle_since: float = 0.0
        self._idle_event_emitted = False
        self._deep_idle_emitted = False

    async def on_user_activity(self, msg: Message) -> None:
        """Call this from a bus subscription on ``user.*`` topics."""
        was_idle = self._is_idle
        self._last_activity = time.monotonic()
        self._is_idle = False
        self._idle_event_emitted = False
        self._deep_idle_emitted = False

        if was_idle:
            idle_duration = time.monotonic() - self._idle_since
            logger.info(
                "[IdleDetector] User returned after %.0fs idle",
                idle_duration,
            )

    async def check(self) -> list[Message] | None:
        """Proactive handler callback — check idle state.

        Returns:
            List of Messages for idle/return transitions, None if no change.
        """
        now = time.monotonic()
        idle_duration = now - self._last_activity
        messages: list[Message] = []

        if idle_duration >= self.idle_threshold_s and not self._idle_event_emitted:
            # User just went idle
            self._is_idle = True
            self._idle_since = self._last_activity
            self._idle_event_emitted = True

            messages.append(
                Message(
                    type=MessageType.EVENT,
                    source_node_id="autonomy.watcher.idle_detector",
                    topic="system.user_idle",
                    payload={
                        "idle_duration_s": round(idle_duration, 1),
                        "threshold_s": self.idle_threshold_s,
                        "level": "idle",
                        "_urgency": 0.2,
                    },
                )
            )

            logger.info(
                "[IdleDetector] User idle for %.0fs (threshold: %.0fs)",
                idle_duration,
                self.idle_threshold_s,
            )

        if (
            idle_duration >= self.deep_idle_threshold_s
            and not self._deep_idle_emitted
            and self._is_idle
        ):
            # Deep idle — trigger heavier consolidation
            self._deep_idle_emitted = True
            messages.append(
                Message(
                    type=MessageType.EVENT,
                    source_node_id="autonomy.watcher.idle_detector",
                    topic="system.user_idle",
                    payload={
                        "idle_duration_s": round(idle_duration, 1),
                        "threshold_s": self.deep_idle_threshold_s,
                        "level": "deep_idle",
                        "_urgency": 0.1,
                    },
                )
            )

            logger.info(
                "[IdleDetector] Deep idle reached (%.0fs). Suitable for consolidation.",
                idle_duration,
            )

        return messages if messages else None

    @property
    def is_idle(self) -> bool:
        return self._is_idle

    @property
    def idle_duration(self) -> float:
        """Current idle duration in seconds."""
        return time.monotonic() - self._last_activity

    def snapshot(self) -> dict[str, Any]:
        """Introspection snapshot."""
        return {
            "is_idle": self._is_idle,
            "idle_duration_s": round(self.idle_duration, 1),
            "idle_threshold_s": self.idle_threshold_s,
            "deep_idle_threshold_s": self.deep_idle_threshold_s,
            "last_activity_ago_s": round(time.monotonic() - self._last_activity, 1),
        }
