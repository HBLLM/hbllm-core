"""Restraint Engine — cognitive safety gate for autonomous actions.

Evaluates whether the system *should* act, even when it *can*.
Multi-factor assessment prevents over-aggressive autonomy:

    1. Confidence threshold — suppress if brain confidence is too low
    2. Reversibility — defer irreversible actions with medium confidence
    3. Social timing — suppress during user sleep/focus hours
    4. Rejection history — back off if user rejected similar actions recently
    5. Cooldown enforcement — don't repeat same action type too quickly
    6. Rate limiting — cap autonomous actions per hour

Decisions:
    APPROVE  — proceed with the action
    DEFER    — delay and re-evaluate later (not rejected, just not now)
    SUPPRESS — do not perform this action

Bus Topics:
    autonomy.restraint.evaluated  — Published on every evaluation

Usage::

    engine = RestraintEngine()
    decision = engine.evaluate(
        action="proactive.reminder",
        context={"confidence": 0.8, "reversible": True},
    )
    if decision.decision == RestraintDecision.APPROVE:
        # Proceed
        ...
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Decision Types ───────────────────────────────────────────────────────────


class RestraintDecision(str, Enum):
    """Outcome of a restraint evaluation."""

    APPROVE = "approve"
    DEFER = "defer"
    SUPPRESS = "suppress"


@dataclass
class RestraintReason:
    """Full result of a restraint evaluation."""

    decision: RestraintDecision
    confidence: float
    reasons: list[str] = field(default_factory=list)
    cooldown_s: float = 0.0
    action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "cooldown_s": self.cooldown_s,
            "action": self.action,
        }


@dataclass
class RestraintConfig:
    """Tunable parameters for the restraint engine."""

    # Confidence thresholds
    min_confidence_approve: float = 0.4
    min_confidence_irreversible: float = 0.7

    # Social timing (24h format)
    quiet_hours_start: int = 22  # 10 PM
    quiet_hours_end: int = 7  # 7 AM

    # Rate limiting
    max_actions_per_hour: int = 20
    cooldown_per_action_type_s: float = 60.0

    # Rejection backoff
    rejection_window_s: float = 3600.0  # 1 hour
    max_rejections_before_suppress: int = 3
    backoff_multiplier: float = 2.0


# ── Engine ───────────────────────────────────────────────────────────────────


class RestraintEngine:
    """Multi-factor safety gate for autonomous actions.

    Args:
        config: Tunable restraint parameters.
        bus: Optional MessageBus for event publishing.
    """

    def __init__(
        self,
        config: RestraintConfig | None = None,
        bus: Any | None = None,
    ) -> None:
        self.config = config or RestraintConfig()
        self.bus = bus

        # Tracking state
        self._action_timestamps: deque[float] = deque(maxlen=1000)
        self._last_action_by_type: dict[str, float] = {}
        self._rejection_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=50))
        self._approval_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=50))

        # Telemetry
        self._total_evaluations = 0
        self._decisions: dict[str, int] = {d.value: 0 for d in RestraintDecision}

    def evaluate(
        self,
        action: str,
        context: dict[str, Any] | None = None,
    ) -> RestraintReason:
        """Evaluate whether an autonomous action should proceed.

        Args:
            action: Action identifier (e.g. "proactive.reminder", "iot.command").
            context: Evaluation context. Recognised keys:
                - confidence (float): Brain's confidence in this action (0-1)
                - reversible (bool): Whether the action can be undone
                - priority (str): "low", "normal", "high", "critical"
                - hour (int): Override current hour for testing

        Returns:
            RestraintReason with decision, confidence, and explanation.
        """
        self._total_evaluations += 1
        ctx = context or {}
        confidence = float(ctx.get("confidence", 0.5))
        reversible = bool(ctx.get("reversible", True))
        priority = str(ctx.get("priority", "normal"))
        hour = int(ctx.get("hour", _current_hour()))

        reasons: list[str] = []

        # Critical priority bypasses all checks
        if priority == "critical":
            return self._result(
                RestraintDecision.APPROVE,
                confidence,
                ["Critical priority — bypassing restraint checks"],
                action=action,
            )

        # ── Factor 1: Confidence threshold ───────────────────────────
        if confidence < self.config.min_confidence_approve:
            return self._result(
                RestraintDecision.SUPPRESS,
                confidence,
                [
                    f"Confidence {confidence:.2f} below threshold "
                    f"{self.config.min_confidence_approve}"
                ],
                action=action,
            )

        # ── Factor 2: Reversibility ──────────────────────────────────
        if not reversible and confidence < self.config.min_confidence_irreversible:
            return self._result(
                RestraintDecision.DEFER,
                confidence,
                [
                    f"Irreversible action with confidence {confidence:.2f} below "
                    f"{self.config.min_confidence_irreversible} — deferring"
                ],
                cooldown_s=30.0,
                action=action,
            )

        # ── Factor 3: Social timing ──────────────────────────────────
        if self._is_quiet_hours(hour) and priority != "high":
            return self._result(
                RestraintDecision.SUPPRESS,
                confidence,
                [
                    f"Quiet hours ({self.config.quiet_hours_start}:00–"
                    f"{self.config.quiet_hours_end}:00)"
                ],
                action=action,
            )

        # ── Factor 4: Rejection history ──────────────────────────────
        recent_rejections = self._count_recent(
            self._rejection_history.get(action, deque()),
            self.config.rejection_window_s,
        )
        if recent_rejections >= self.config.max_rejections_before_suppress:
            cooldown = self.config.cooldown_per_action_type_s * (
                self.config.backoff_multiplier**recent_rejections
            )
            return self._result(
                RestraintDecision.SUPPRESS,
                confidence,
                [
                    f"User rejected '{action}' {recent_rejections} times recently "
                    f"— backing off {cooldown:.0f}s"
                ],
                cooldown_s=cooldown,
                action=action,
            )

        # ── Factor 5: Cooldown ───────────────────────────────────────
        last_time = self._last_action_by_type.get(action, 0.0)
        elapsed = time.monotonic() - last_time
        if elapsed < self.config.cooldown_per_action_type_s:
            remaining = self.config.cooldown_per_action_type_s - elapsed
            return self._result(
                RestraintDecision.DEFER,
                confidence,
                [f"Cooldown: {remaining:.0f}s remaining for '{action}'"],
                cooldown_s=remaining,
                action=action,
            )

        # ── Factor 6: Rate limiting ──────────────────────────────────
        now = time.monotonic()
        recent_actions = sum(1 for t in self._action_timestamps if now - t < 3600)
        if recent_actions >= self.config.max_actions_per_hour:
            return self._result(
                RestraintDecision.DEFER,
                confidence,
                [
                    f"Rate limit: {recent_actions}/{self.config.max_actions_per_hour} "
                    f"actions per hour"
                ],
                cooldown_s=60.0,
                action=action,
            )

        # ── All checks passed ────────────────────────────────────────
        reasons.append("All restraint checks passed")
        return self._result(
            RestraintDecision.APPROVE,
            confidence,
            reasons,
            action=action,
        )

    # ── Feedback API ─────────────────────────────────────────────────

    def record_rejection(self, action: str) -> None:
        """Record that a user rejected this action type."""
        self._rejection_history[action].append(time.monotonic())
        logger.info("Restraint: user rejected '%s'", action)

    def record_approval(self, action: str) -> None:
        """Record that an action was approved/executed."""
        self._approval_history[action].append(time.monotonic())
        self._action_timestamps.append(time.monotonic())
        self._last_action_by_type[action] = time.monotonic()

    # ── Internal ─────────────────────────────────────────────────────

    def _is_quiet_hours(self, hour: int) -> bool:
        """Check if current hour falls in quiet hours."""
        start = self.config.quiet_hours_start
        end = self.config.quiet_hours_end
        if start > end:
            # Wraps midnight: e.g. 22-7
            return hour >= start or hour < end
        return start <= hour < end

    @staticmethod
    def _count_recent(timestamps: deque[float], window_s: float) -> int:
        """Count events within the recent time window."""
        now = time.monotonic()
        return sum(1 for t in timestamps if now - t < window_s)

    def _result(
        self,
        decision: RestraintDecision,
        confidence: float,
        reasons: list[str],
        cooldown_s: float = 0.0,
        action: str = "",
    ) -> RestraintReason:
        """Build result and update telemetry."""
        self._decisions[decision.value] += 1
        result = RestraintReason(
            decision=decision,
            confidence=confidence,
            reasons=reasons,
            cooldown_s=cooldown_s,
            action=action,
        )
        logger.debug("Restraint [%s] %s: %s", action, decision.value, "; ".join(reasons))
        return result

    def stats(self) -> dict[str, Any]:
        """Engine statistics."""
        return {
            "total_evaluations": self._total_evaluations,
            "decisions": dict(self._decisions),
        }


def _current_hour() -> int:
    """Get the current hour (0-23). Extracted for testability."""
    import datetime

    return datetime.datetime.now().hour
