"""Restraint Engine — cognitive safety gate for autonomous actions.

Evaluates whether the system *should* act, even when it *can*.
Multi-factor assessment prevents over-aggressive autonomy:

    1. Confidence threshold — suppress if brain confidence is too low
    2. Reversibility — defer irreversible actions with medium confidence
    3. Social timing — suppress during user sleep/focus hours
    4. Rejection history — back off if user rejected similar actions recently
    5. Cooldown enforcement — don't repeat same action type too quickly
    6. Rate limiting — cap autonomous actions per hour
    7. Context signals — defer during meetings, suppress during sleep
    8. Suggestion frequency — limit proactive suggestions per hour

Decisions:
    APPROVE  — proceed with the action
    DEFER    — delay and re-evaluate later (not rejected, just not now)
    SUPPRESS — do not perform this action

Bus Topics:
    autonomy.restraint.evaluated  — Published on every evaluation

Usage::

    engine = RestraintEngine()
    decision = engine.evaluate(
        ActionProposal(
            action_type="proactive.reminder",
            category="suggestion",
            confidence=0.8,
        )
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
class ActionProposal:
    """Structured input for restraint evaluation."""

    action_type: str
    category: str = "reflex"  # "reflex", "suggestion", "proactive"
    priority: str = "normal"  # "low", "normal", "high", "critical"
    confidence: float = 0.5
    is_reversible: bool = True
    tenant_id: str = "default"


@dataclass
class RestraintReason:
    """Full result of a restraint evaluation."""

    decision: RestraintDecision
    confidence: float
    reasons: list[str] = field(default_factory=list)
    cooldown_s: float = 0.0
    action: str = ""
    alternative_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "cooldown_s": self.cooldown_s,
            "action": self.action,
            "alternative_action": self.alternative_action,
        }


@dataclass
class RestraintConfig:
    """Tunable parameters for the restraint engine."""

    # Confidence thresholds
    min_confidence_approve: float = 0.7
    min_confidence_irreversible: float = 0.9

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
    backoff_base_s: float = 60.0

    # Suggestion rate limiting
    max_suggestions_per_hour: int = 10


@dataclass
class _TenantContext:
    """Per-tenant context signals."""

    in_meeting: bool = False
    is_sleeping: bool = False


# ── Engine ───────────────────────────────────────────────────────────────────


class RestraintEngine:
    """Multi-factor safety gate for autonomous actions.

    Args:
        config: Tunable restraint parameters. Individual kwargs override config fields.
        bus: Optional MessageBus for event publishing.
        quiet_hours: Tuple of (start, end) hours — overrides config if provided.
        backoff_base_s: Override for config.backoff_base_s.
        max_suggestions_per_hour: Override for config.max_suggestions_per_hour.
    """

    def __init__(
        self,
        config: RestraintConfig | None = None,
        bus: Any | None = None,
        *,
        quiet_hours: tuple[int, int] | None = None,
        backoff_base_s: float | None = None,
        max_suggestions_per_hour: int | None = None,
    ) -> None:
        self.config = config or RestraintConfig()
        self.bus = bus

        # Allow keyword overrides
        if quiet_hours is not None:
            self.config.quiet_hours_start, self.config.quiet_hours_end = quiet_hours
        if backoff_base_s is not None:
            self.config.backoff_base_s = backoff_base_s
        if max_suggestions_per_hour is not None:
            self.config.max_suggestions_per_hour = max_suggestions_per_hour

        # Tracking state (per-tenant)
        self._action_timestamps: deque[float] = deque(maxlen=1000)
        self._last_action_by_type: dict[str, float] = {}
        self._rejection_history: dict[str, dict[str, deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=50))
        )
        self._approval_history: dict[str, dict[str, deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=50))
        )
        self._suggestion_timestamps: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self._tenant_contexts: dict[str, _TenantContext] = defaultdict(_TenantContext)

        # Telemetry
        self._total_evaluations = 0
        self._decisions: dict[str, int] = {d.value: 0 for d in RestraintDecision}

    def evaluate(
        self,
        proposal: ActionProposal,
    ) -> RestraintReason:
        """Evaluate whether an autonomous action should proceed.

        Args:
            proposal: Structured action proposal with metadata.

        Returns:
            RestraintReason with decision, confidence, and explanation.
        """
        self._total_evaluations += 1

        action = proposal.action_type
        confidence = proposal.confidence
        priority = proposal.priority
        is_reversible = proposal.is_reversible
        category = proposal.category
        tenant_id = proposal.tenant_id
        hour = _current_hour()

        # Critical priority bypasses all checks
        if priority == "critical":
            self._track_approval(tenant_id, action)
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

        # ── Factor 2: Context signals (meeting, sleeping) ────────────
        ctx = self._tenant_contexts.get(tenant_id)
        if ctx:
            if ctx.is_sleeping and priority != "critical":
                return self._result(
                    RestraintDecision.SUPPRESS,
                    confidence,
                    ["User is sleeping — suppressing non-critical action"],
                    action=action,
                )
            if ctx.in_meeting and priority not in ("critical", "high"):
                return self._result(
                    RestraintDecision.DEFER,
                    confidence,
                    ["User is in a meeting — deferring non-critical action"],
                    action=action,
                )

        # ── Factor 3: Reversibility ──────────────────────────────────────
        if not is_reversible:
            if confidence <= self.config.min_confidence_irreversible and priority not in (
                "critical",
                "high",
            ):
                return self._result(
                    RestraintDecision.DEFER,
                    confidence,
                    [
                        f"Irreversible action '{action}' with confidence {confidence:.2f} "
                        f"below irreversible threshold {self.config.min_confidence_irreversible} "
                        f"— deferring for confirmation"
                    ],
                    cooldown_s=30.0,
                    action=action,
                    alternative_action=f"confirm:{action}",
                )

        # ── Factor 4: Social timing ──────────────────────────────────
        if self._is_quiet_hours(hour) and priority not in ("high", "critical"):
            return self._result(
                RestraintDecision.SUPPRESS,
                confidence,
                [
                    f"Quiet hours ({self.config.quiet_hours_start}:00–"
                    f"{self.config.quiet_hours_end}:00)"
                ],
                action=action,
            )

        # ── Factor 5: Rejection backoff ──────────────────────────────
        tenant_rejections = self._rejection_history.get(tenant_id, {})
        action_rejections = tenant_rejections.get(action, deque())
        recent_rejections = self._count_recent(
            action_rejections,
            self.config.rejection_window_s,
        )
        if recent_rejections > 0:
            cooldown = self.config.backoff_base_s * (
                self.config.backoff_multiplier**recent_rejections
            )
            return self._result(
                RestraintDecision.SUPPRESS,
                confidence,
                [f"User rejected '{action}' recently — backoff {cooldown:.0f}s"],
                cooldown_s=cooldown,
                action=action,
            )

        # ── Factor 6: Suggestion rate limiting ───────────────────────
        if category == "suggestion":
            tenant_sug = self._suggestion_timestamps[tenant_id]
            now = time.monotonic()
            recent_sug = sum(1 for t in tenant_sug if now - t < 3600)
            if recent_sug >= self.config.max_suggestions_per_hour:
                return self._result(
                    RestraintDecision.DEFER,
                    confidence,
                    [
                        f"Suggestion rate limit: {recent_sug}/"
                        f"{self.config.max_suggestions_per_hour} per hour"
                    ],
                    cooldown_s=60.0,
                    action=action,
                )
            # Track this suggestion
            tenant_sug.append(now)

        # ── Factor 7: Cooldown ───────────────────────────────────────
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

        # ── Factor 8: Global rate limiting ───────────────────────────
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
        self._track_approval(tenant_id, action)
        return self._result(
            RestraintDecision.APPROVE,
            confidence,
            ["All restraint checks passed"],
            action=action,
        )

    # ── Context API ──────────────────────────────────────────────────

    def update_context(
        self,
        tenant_id: str,
        *,
        in_meeting: bool | None = None,
        is_sleeping: bool | None = None,
    ) -> None:
        """Update social context signals for a tenant."""
        ctx = self._tenant_contexts[tenant_id]
        if in_meeting is not None:
            ctx.in_meeting = in_meeting
        if is_sleeping is not None:
            ctx.is_sleeping = is_sleeping

    # ── Feedback API ─────────────────────────────────────────────────

    def record_rejection(self, tenant_id: str, action: str) -> None:
        """Record that a user rejected this action type."""
        self._rejection_history[tenant_id][action].append(time.monotonic())
        logger.info("Restraint: user rejected '%s' (tenant=%s)", action, tenant_id)

    def record_acceptance(self, tenant_id: str, action: str) -> None:
        """Record acceptance — clears rejection backoff for this action."""
        # Clear rejections for this tenant+action to reset backoff
        if tenant_id in self._rejection_history:
            self._rejection_history[tenant_id].pop(action, None)
        self._approval_history[tenant_id][action].append(time.monotonic())
        logger.info("Restraint: user accepted '%s' (tenant=%s)", action, tenant_id)

    # ── Internal ─────────────────────────────────────────────────────

    def _track_approval(self, tenant_id: str, action: str) -> None:
        """Track an approved action for rate limiting."""
        now = time.monotonic()
        self._action_timestamps.append(now)
        self._last_action_by_type[action] = now

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
        alternative_action: str | None = None,
    ) -> RestraintReason:
        """Build result and update telemetry."""
        self._decisions[decision.value] += 1
        result = RestraintReason(
            decision=decision,
            confidence=confidence,
            reasons=reasons,
            cooldown_s=cooldown_s,
            action=action,
            alternative_action=alternative_action,
        )
        logger.debug("Restraint [%s] %s: %s", action, decision.value, "; ".join(reasons))
        return result

    def stats(self) -> dict[str, Any]:
        """Engine statistics."""
        return {
            "total_evaluated": self._total_evaluations,
            "decisions": dict(self._decisions),
        }


def _current_hour() -> int:
    """Get the current hour (0-23). Extracted for testability."""
    import datetime

    return datetime.datetime.now().hour
