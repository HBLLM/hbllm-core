"""
Interruption Manager — Decides whether incoming events should interrupt
the current cognitive task or be deferred to the queue.

Humans don't respond to every stimulus immediately. The Interruption
Manager gives HBLLM the same discipline:

    1. Score the urgency of the incoming event.
    2. Compare against the importance of the current focus.
    3. Decide: INTERRUPT (switch now), QUEUE (process after current), or DROP.

Urgency factors:
    - Source priority (user message > timer > background insight)
    - Content signals (explicit mention, question, error vs. info)
    - User relationship (high-trust users get higher priority)
    - Temporal pressure (deadline proximity)
    - Cognitive load (if brain is overloaded, raise threshold)

Integration::

    EventBus / ConversationBus message arrives
        ↓
    InterruptionManager.evaluate(event)
        ↓
    InterruptDecision (INTERRUPT / QUEUE / DROP)
        ↓
    Gateway / Executive acts accordingly

Usage::

    from hbllm.brain.attention.interruption_manager import InterruptionManager

    manager = InterruptionManager()
    decision = manager.evaluate(event, current_focus=focus)

    if decision.action == "interrupt":
        # Switch to the new event immediately
        ...
    elif decision.action == "queue":
        # Add to the deferred queue
        ...
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════


class InterruptAction(StrEnum):
    """What to do with an incoming event."""

    INTERRUPT = "interrupt"  # Switch to this now
    QUEUE = "queue"  # Process after current task
    DROP = "drop"  # Discard (noise, duplicate, etc.)


class EventSource(StrEnum):
    """Source category of an incoming event."""

    USER_DIRECT = "user_direct"  # User typed a message
    USER_INDIRECT = "user_indirect"  # User action (file save, navigation)
    SYSTEM_CRITICAL = "system_critical"  # Errors, security alerts
    SYSTEM_INFO = "system_info"  # Health checks, metrics
    TIMER = "timer"  # Scheduled event
    PLUGIN = "plugin"  # Plugin notification
    BACKGROUND = "background"  # Autonomous reasoning output
    SENSOR = "sensor"  # IoT, camera, microphone


# Base urgency score per source type
_SOURCE_BASE_URGENCY: dict[EventSource, float] = {
    EventSource.USER_DIRECT: 0.9,
    EventSource.SYSTEM_CRITICAL: 0.95,
    EventSource.USER_INDIRECT: 0.5,
    EventSource.TIMER: 0.6,
    EventSource.PLUGIN: 0.4,
    EventSource.BACKGROUND: 0.3,
    EventSource.SYSTEM_INFO: 0.2,
    EventSource.SENSOR: 0.5,
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class IncomingEvent:
    """An event requesting attention."""

    source: EventSource = EventSource.BACKGROUND
    content: str = ""
    priority_hint: float = 0.0  # 0.0–1.0 from the source
    tenant_id: str = "default"
    user_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class CurrentFocus:
    """What the brain is currently working on."""

    task_id: str = ""
    description: str = ""
    importance: float = 0.5  # 0.0–1.0
    elapsed_ms: float = 0.0  # How long we've been on this
    interruptible: bool = True  # Some tasks can't be interrupted
    user_id: str = ""  # Who initiated the current task


@dataclass
class InterruptDecision:
    """The result of an interruption evaluation."""

    action: InterruptAction
    urgency_score: float  # Computed urgency of the event
    focus_importance: float  # Importance of current focus
    reason: str = ""
    deferred_until_ms: float = 0.0  # If QUEUE, estimated wait


# ═══════════════════════════════════════════════════════════════════════════
# Content Signals
# ═══════════════════════════════════════════════════════════════════════════

# Words/patterns that boost urgency
_URGENCY_SIGNALS: dict[str, float] = {
    "urgent": 0.15,
    "emergency": 0.2,
    "error": 0.1,
    "failed": 0.1,
    "critical": 0.15,
    "asap": 0.1,
    "help": 0.05,
    "broken": 0.08,
    "crash": 0.12,
    "down": 0.05,
    "security": 0.1,
    "deadline": 0.08,
}


# ═══════════════════════════════════════════════════════════════════════════
# Interruption Manager
# ═══════════════════════════════════════════════════════════════════════════


class InterruptionManager:
    """Decides whether an incoming event should interrupt the current focus.

    Uses a multi-factor scoring system:
        urgency = base_urgency(source) + content_signals + priority_hint
        threshold = focus_importance × cognitive_load_factor

    If urgency > threshold → INTERRUPT
    If urgency > threshold × 0.5 → QUEUE
    Else → DROP
    """

    def __init__(
        self,
        base_interrupt_threshold: float = 0.6,
        cognitive_load_factor: float = 1.0,
    ) -> None:
        self._base_threshold = base_interrupt_threshold
        self._cognitive_load = cognitive_load_factor  # 1.0 = normal, >1.0 = overloaded
        self._deferred_queue: list[tuple[float, IncomingEvent]] = []
        self._decisions_made = 0
        self._interrupts_triggered = 0

    # ── Core API ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        event: IncomingEvent,
        current_focus: CurrentFocus | None = None,
    ) -> InterruptDecision:
        """Evaluate whether an event should interrupt the current focus.

        Args:
            event: The incoming event.
            current_focus: What the brain is currently doing (None = idle).

        Returns:
            InterruptDecision with action, score, and reason.
        """
        self._decisions_made += 1

        # Compute urgency
        urgency = self._compute_urgency(event)

        # If idle, always accept
        if current_focus is None or not current_focus.task_id:
            self._interrupts_triggered += 1
            return InterruptDecision(
                action=InterruptAction.INTERRUPT,
                urgency_score=urgency,
                focus_importance=0.0,
                reason="Brain is idle — accepting event.",
            )

        # Non-interruptible tasks block everything below critical
        if not current_focus.interruptible:
            if urgency < 0.9:
                return InterruptDecision(
                    action=InterruptAction.QUEUE,
                    urgency_score=urgency,
                    focus_importance=current_focus.importance,
                    reason="Current task is non-interruptible.",
                )

        # Same user gets priority
        same_user_boost = 0.1 if event.user_id == current_focus.user_id else 0.0
        adjusted_urgency = min(1.0, urgency + same_user_boost)

        # Compute threshold (higher cognitive load = harder to interrupt)
        threshold = current_focus.importance * self._cognitive_load

        # Decision
        if adjusted_urgency > threshold:
            self._interrupts_triggered += 1
            return InterruptDecision(
                action=InterruptAction.INTERRUPT,
                urgency_score=adjusted_urgency,
                focus_importance=current_focus.importance,
                reason=f"Urgency ({adjusted_urgency:.2f}) > threshold ({threshold:.2f}).",
            )
        elif adjusted_urgency > threshold * 0.5:
            return InterruptDecision(
                action=InterruptAction.QUEUE,
                urgency_score=adjusted_urgency,
                focus_importance=current_focus.importance,
                reason=f"Urgency ({adjusted_urgency:.2f}) moderate — queued.",
            )
        else:
            return InterruptDecision(
                action=InterruptAction.DROP,
                urgency_score=adjusted_urgency,
                focus_importance=current_focus.importance,
                reason=f"Urgency ({adjusted_urgency:.2f}) too low — dropped.",
            )

    def set_cognitive_load(self, load: float) -> None:
        """Update the cognitive load factor.

        Higher values make the brain harder to interrupt.

        Args:
            load: 0.5 = relaxed, 1.0 = normal, 1.5 = overloaded.
        """
        self._cognitive_load = max(0.1, min(2.0, load))

    # ── Deferred Queue ───────────────────────────────────────────────────

    def queue_event(self, event: IncomingEvent) -> None:
        """Add an event to the deferred queue."""
        urgency = self._compute_urgency(event)
        self._deferred_queue.append((urgency, event))
        # Keep sorted by urgency (highest first)
        self._deferred_queue.sort(key=lambda x: -x[0])
        # Cap queue size
        if len(self._deferred_queue) > 50:
            self._deferred_queue = self._deferred_queue[:50]

    def pop_next_deferred(self) -> IncomingEvent | None:
        """Pop the highest-urgency deferred event."""
        if self._deferred_queue:
            _, event = self._deferred_queue.pop(0)
            return event
        return None

    @property
    def deferred_count(self) -> int:
        return len(self._deferred_queue)

    # ── Introspection ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "decisions_made": self._decisions_made,
            "interrupts_triggered": self._interrupts_triggered,
            "interrupt_rate": (
                self._interrupts_triggered / self._decisions_made
                if self._decisions_made > 0
                else 0.0
            ),
            "deferred_queue_size": len(self._deferred_queue),
            "cognitive_load": self._cognitive_load,
            "base_threshold": self._base_threshold,
        }

    # ── Internal ─────────────────────────────────────────────────────────

    def _compute_urgency(self, event: IncomingEvent) -> float:
        """Compute urgency score for an event."""
        # Base urgency from source type
        base = _SOURCE_BASE_URGENCY.get(event.source, 0.3)

        # Priority hint from the source itself
        hint_boost = event.priority_hint * 0.2

        # Content signal scanning
        content_boost = 0.0
        if event.content:
            content_lower = event.content.lower()
            for signal, boost in _URGENCY_SIGNALS.items():
                if signal in content_lower:
                    content_boost += boost

        content_boost = min(0.3, content_boost)  # Cap content boost

        urgency = min(1.0, base + hint_boost + content_boost)
        return urgency
