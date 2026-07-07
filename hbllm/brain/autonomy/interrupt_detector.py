"""User Interrupt Detector — engagement state tracking.

Detects when the user is actively engaged and adjusts system behavior:
    - Active conversation → suppress low-priority background notifications
    - Voice activity → hold audio notifications
    - Recent interaction → batch non-urgent messages
    - Idle → safe to deliver proactive insights

States:
    ENGAGED     — user actively interacting (< 30s since last message)
    LISTENING   — voice activity detected
    PRESENT     — recent interaction (< 5 min)
    IDLE        — no interaction for > 5 min
    DEEP_IDLE   — no interaction for > 30 min (safe for digests)

Publishes state changes to `autonomy.user.state` on the MessageBus.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any

from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


class UserState(str, Enum):
    """User engagement states."""

    ENGAGED = "engaged"
    LISTENING = "listening"
    PRESENT = "present"
    IDLE = "idle"
    DEEP_IDLE = "deep_idle"


# State → minimum notification priority allowed
STATE_PRIORITY_THRESHOLDS: dict[UserState, str] = {
    UserState.ENGAGED: "critical",  # Only critical interrupts during conversation
    UserState.LISTENING: "critical",  # Only critical during voice activity
    UserState.PRESENT: "high",  # High and critical when recently active
    UserState.IDLE: "info",  # Everything when idle
    UserState.DEEP_IDLE: "suggestion",  # Even suggestions during deep idle
}


class InterruptDetector:
    """Tracks user engagement state to gate notifications.

    Usage::

        detector = InterruptDetector(bus=message_bus)
        await detector.start()

        # Check before sending a notification
        if detector.should_deliver(priority="info"):
            send_notification(...)
    """

    def __init__(
        self,
        bus: Any | None = None,
        engaged_timeout_s: float = 30.0,
        present_timeout_s: float = 300.0,
        deep_idle_timeout_s: float = 1800.0,
    ) -> None:
        self.bus = bus
        self.engaged_timeout_s = engaged_timeout_s
        self.present_timeout_s = present_timeout_s
        self.deep_idle_timeout_s = deep_idle_timeout_s

        self._last_user_input: float = 0.0
        self._last_voice_activity: float = 0.0
        self._current_state = UserState.IDLE
        self._state_since: float = time.monotonic()

        # Telemetry
        self._state_changes = 0
        self._suppressed_count = 0

    async def start(self) -> None:
        """Subscribe to relevant bus topics for engagement tracking."""
        if self.bus:
            # Track user messages
            await self.bus.subscribe("user.input", self._on_user_input)
            await self.bus.subscribe("user.action", self._on_user_input)
            # Track voice activity
            await self.bus.subscribe("perception.audio.speech_detected", self._on_voice_activity)
            await self.bus.subscribe("perception.audio.vad.active", self._on_voice_activity)

            logger.info("InterruptDetector started")

    async def _on_user_input(self, msg: Message) -> None:
        """User sent a message or performed an action."""
        self._last_user_input = time.monotonic()
        await self._update_state()

    async def _on_voice_activity(self, msg: Message) -> None:
        """Voice activity detected."""
        self._last_voice_activity = time.monotonic()
        await self._update_state()

    async def _update_state(self) -> None:
        """Recompute the user engagement state."""
        now = time.monotonic()
        since_input = now - self._last_user_input if self._last_user_input > 0 else float("inf")
        since_voice = (
            now - self._last_voice_activity if self._last_voice_activity > 0 else float("inf")
        )

        # Determine new state
        if since_voice < 5.0:
            new_state = UserState.LISTENING
        elif since_input < self.engaged_timeout_s:
            new_state = UserState.ENGAGED
        elif since_input < self.present_timeout_s:
            new_state = UserState.PRESENT
        elif since_input < self.deep_idle_timeout_s:
            new_state = UserState.IDLE
        else:
            new_state = UserState.DEEP_IDLE

        if new_state != self._current_state:
            old_state = self._current_state
            self._current_state = new_state
            self._state_since = now
            self._state_changes += 1

            logger.info(
                "User state: %s → %s (input %.0fs ago, voice %.0fs ago)",
                old_state.value,
                new_state.value,
                since_input,
                since_voice,
            )

            # Publish state change
            if self.bus:
                await self.bus.publish(
                    "autonomy.user.state",
                    Message(
                        type=MessageType.EVENT,
                        source_node_id="interrupt_detector",
                        topic="autonomy.user.state",
                        payload={
                            "state": new_state.value,
                            "previous_state": old_state.value,
                            "since_input_s": since_input,
                            "since_voice_s": since_voice,
                        },
                    ),
                )

    @property
    def state(self) -> UserState:
        """Get the current engagement state (lazy refresh)."""
        # Refresh on read since state can decay over time
        now = time.monotonic()
        since_input = now - self._last_user_input if self._last_user_input > 0 else float("inf")

        old_state = self._current_state
        if since_input >= self.deep_idle_timeout_s:
            self._current_state = UserState.DEEP_IDLE
        elif since_input >= self.present_timeout_s:
            self._current_state = UserState.IDLE
        elif since_input >= self.engaged_timeout_s:
            self._current_state = UserState.PRESENT

        # Emit bus event on passive state decay (mirrors _update_state behavior)
        if self._current_state != old_state:
            self._state_since = now
            self._state_changes += 1
            logger.info(
                "User state (passive decay): %s → %s (input %.0fs ago)",
                old_state.value,
                self._current_state.value,
                since_input,
            )
            if self.bus:
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        self.bus.publish(
                            "autonomy.user.state",
                            Message(
                                type=MessageType.EVENT,
                                source_node_id="interrupt_detector",
                                topic="autonomy.user.state",
                                payload={
                                    "state": self._current_state.value,
                                    "previous_state": old_state.value,
                                    "since_input_s": since_input,
                                    "decay": True,
                                },
                            ),
                        )
                    )
                except RuntimeError:
                    pass  # No running event loop (e.g., called from sync context)

        return self._current_state

    def should_deliver(self, priority: str) -> bool:
        """Check if a notification with given priority should be delivered now.

        Args:
            priority: "critical", "high", "info", or "suggestion"

        Returns:
            True if the notification should be delivered given current state.
        """
        priority_rank = {"critical": 4, "high": 3, "normal": 2, "info": 2, "suggestion": 1}
        threshold = STATE_PRIORITY_THRESHOLDS.get(self.state, "info")

        incoming_rank = priority_rank.get(priority, 0)
        threshold_rank = priority_rank.get(threshold, 0)

        allowed = incoming_rank >= threshold_rank

        if not allowed:
            self._suppressed_count += 1
            logger.debug(
                "Notification suppressed: priority=%s, state=%s, threshold=%s",
                priority,
                self.state.value,
                threshold,
            )

        return allowed

    def stats(self) -> dict[str, Any]:
        """Interrupt detector statistics."""
        return {
            "current_state": self.state.value,
            "state_changes": self._state_changes,
            "suppressed_count": self._suppressed_count,
            "state_duration_s": time.monotonic() - self._state_since,
        }
