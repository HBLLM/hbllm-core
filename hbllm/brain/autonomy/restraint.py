"""Restraint Engine — knowing when NOT to act.

Evaluates every proposed autonomous action through multiple lenses:
    1. Social appropriateness — don't interrupt dinner, don't message at 3am
    2. Confidence threshold — don't act if confidence < 0.7
    3. Reversibility check — prefer reversible actions over irreversible
    4. User preference history — if user previously rejected, don't repeat
    5. Cool-down enforcement — exponential backoff on repeated suggestions

Outputs: APPROVE, DEFER, or SUPPRESS with reasoning.
All decisions are logged for training data.

"The best assistant knows when to shut up."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RestraintDecision(str, Enum):
    """Possible restraint outcomes."""

    APPROVE = "approve"  # Safe to proceed
    DEFER = "defer"  # Hold for a better moment
    SUPPRESS = "suppress"  # Do not execute at all


@dataclass
class RestraintResult:
    """Result of a restraint evaluation."""

    decision: RestraintDecision
    reasons: list[str] = field(default_factory=list)
    confidence: float = 1.0
    defer_until: float | None = None  # Timestamp to retry (for DEFER)
    alternative_action: str | None = None  # Suggested softer alternative

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reasons": self.reasons,
            "confidence": self.confidence,
            "defer_until": self.defer_until,
            "alternative_action": self.alternative_action,
        }


@dataclass
class ActionProposal:
    """A proposed autonomous action to evaluate."""

    action_type: str  # e.g., "notification.send", "iot.light.off"
    category: str  # e.g., "suggestion", "reflex", "proactive"
    confidence: float = 1.0  # How confident the system is
    priority: str = "normal"  # "critical", "high", "normal", "suggestion"
    is_reversible: bool = True
    target: str = ""  # What will be acted upon
    tenant_id: str = "default"
    context: dict[str, Any] = field(default_factory=dict)


# ── Rejection History ────────────────────────────────────────────────────


@dataclass
class _RejectionRecord:
    """Tracks a user rejection of an action type."""

    count: int = 0
    last_rejected: float = 0.0
    backoff_until: float = 0.0


class RestraintEngine:
    """Evaluates whether an autonomous action should proceed.

    Usage::

        restraint = RestraintEngine()

        proposal = ActionProposal(
            action_type="notification.send",
            category="suggestion",
            confidence=0.65,
        )
        result = restraint.evaluate(proposal)

        if result.decision == RestraintDecision.APPROVE:
            execute(proposal)
        elif result.decision == RestraintDecision.DEFER:
            schedule_retry(proposal, result.defer_until)
        else:
            log_suppressed(proposal)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_suggestions_per_hour: int = 3,
        quiet_hours: tuple[int, int] = (23, 7),  # 11pm - 7am
        backoff_base_s: float = 300.0,  # 5 minutes
        backoff_max_s: float = 21600.0,  # 6 hours
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_suggestions_per_hour = max_suggestions_per_hour
        self.quiet_hours = quiet_hours
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s

        # Per-tenant rejection history: tenant → action_type → record
        self._rejections: dict[str, dict[str, _RejectionRecord]] = {}

        # Suggestion timestamps: tenant → [timestamps]
        self._suggestion_times: dict[str, list[float]] = {}

        # Context signals
        self._user_in_meeting: dict[str, bool] = {}
        self._user_is_sleeping: dict[str, bool] = {}

        # Telemetry
        self._total_evaluated = 0
        self._decisions: dict[str, int] = {"approve": 0, "defer": 0, "suppress": 0}

    def evaluate(self, proposal: ActionProposal) -> RestraintResult:
        """Evaluate whether a proposed action should proceed.

        Checks are applied in priority order — first failing check wins.
        """
        self._total_evaluated += 1
        reasons: list[str] = []
        now = time.time()

        # 1. Critical actions always pass
        if proposal.priority == "critical":
            return self._approve("Critical priority bypasses restraint")

        # 2. Confidence threshold
        if proposal.confidence < self.confidence_threshold:
            reasons.append(
                f"Confidence {proposal.confidence:.2f} below threshold "
                f"{self.confidence_threshold:.2f}"
            )
            return self._suppress(reasons)

        # 3. Quiet hours check
        if self._is_quiet_hours():
            if proposal.priority not in ("critical", "high"):
                reasons.append(f"Quiet hours ({self.quiet_hours[0]}:00-{self.quiet_hours[1]}:00)")
                # Defer until quiet hours end
                defer_ts = self._next_quiet_hours_end()
                return self._defer(reasons, defer_until=defer_ts)

        # 4. Meeting check
        if self._user_in_meeting.get(proposal.tenant_id, False):
            if proposal.priority not in ("critical",):
                reasons.append("User is in a meeting")
                return self._defer(reasons, defer_until=now + 1800)

        # 5. Sleep check
        if self._user_is_sleeping.get(proposal.tenant_id, False):
            if proposal.priority not in ("critical",):
                reasons.append("User appears to be sleeping")
                return self._suppress(reasons)

        # 6. Rejection backoff
        record = self._get_rejection_record(proposal.tenant_id, proposal.action_type)
        if record and record.backoff_until > now:
            remaining = record.backoff_until - now
            reasons.append(
                f"Backoff active: {record.count} prior rejections, {remaining:.0f}s remaining"
            )
            return self._suppress(reasons)

        # 7. Suggestion rate limit
        if proposal.category == "suggestion":
            if not self._check_suggestion_rate(proposal.tenant_id, now):
                reasons.append(f"Suggestion rate limit: max {self.max_suggestions_per_hour}/hour")
                return self._defer(reasons, defer_until=now + 900)

        # 8. Reversibility check for irreversible high-risk actions
        if not proposal.is_reversible and proposal.priority not in ("critical", "high"):
            reasons.append(
                "Irreversible action without high priority — consider a reversible alternative"
            )
            return self._defer(
                reasons,
                alternative_action=f"reversible_{proposal.action_type}",
            )

        # All checks passed
        if proposal.category == "suggestion":
            self._record_suggestion(proposal.tenant_id, now)

        return self._approve("All restraint checks passed")

    def record_rejection(self, tenant_id: str, action_type: str) -> None:
        """Record that the user rejected an autonomous action.

        This applies exponential backoff to future proposals of the same type.
        """
        if tenant_id not in self._rejections:
            self._rejections[tenant_id] = {}
        if action_type not in self._rejections[tenant_id]:
            self._rejections[tenant_id][action_type] = _RejectionRecord()

        record = self._rejections[tenant_id][action_type]
        record.count += 1
        record.last_rejected = time.time()

        # Exponential backoff: 5min, 15min, 45min, 2.25h, 6h (capped)
        backoff_s = min(
            self._backoff_max_s,
            self._backoff_base_s * (3 ** (record.count - 1)),
        )
        record.backoff_until = time.time() + backoff_s

        logger.info(
            "Restraint: rejection recorded for tenant=%s action=%s (count=%d, backoff=%ds)",
            tenant_id,
            action_type,
            record.count,
            backoff_s,
        )

    def record_acceptance(self, tenant_id: str, action_type: str) -> None:
        """Record that the user accepted an action. Reduces backoff."""
        record = self._get_rejection_record(tenant_id, action_type)
        if record and record.count > 0:
            record.count = max(0, record.count - 1)
            record.backoff_until = 0.0

    def update_context(
        self,
        tenant_id: str,
        in_meeting: bool | None = None,
        is_sleeping: bool | None = None,
    ) -> None:
        """Update social context for a tenant."""
        if in_meeting is not None:
            self._user_in_meeting[tenant_id] = in_meeting
        if is_sleeping is not None:
            self._user_is_sleeping[tenant_id] = is_sleeping

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _approve(self, reason: str) -> RestraintResult:
        self._decisions["approve"] += 1
        return RestraintResult(
            decision=RestraintDecision.APPROVE,
            reasons=[reason],
        )

    def _defer(
        self,
        reasons: list[str],
        defer_until: float | None = None,
        alternative_action: str | None = None,
    ) -> RestraintResult:
        self._decisions["defer"] += 1
        return RestraintResult(
            decision=RestraintDecision.DEFER,
            reasons=reasons,
            defer_until=defer_until,
            alternative_action=alternative_action,
        )

    def _suppress(self, reasons: list[str]) -> RestraintResult:
        self._decisions["suppress"] += 1
        return RestraintResult(
            decision=RestraintDecision.SUPPRESS,
            reasons=reasons,
        )

    def _is_quiet_hours(self) -> bool:
        """Check if the current time is within quiet hours."""
        from datetime import datetime

        hour = datetime.now().hour
        start, end = self.quiet_hours
        if start <= end:
            return start <= hour < end
        else:
            # Overnight range (e.g., 23-7)
            return hour >= start or hour < end

    def _next_quiet_hours_end(self) -> float:
        """Calculate the timestamp when quiet hours next end."""
        from datetime import datetime, timedelta

        now = datetime.now()
        end_hour = self.quiet_hours[1]

        # If we're past the end hour today, it's tomorrow's end
        target = now.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)

        return target.timestamp()

    def _get_rejection_record(self, tenant_id: str, action_type: str) -> _RejectionRecord | None:
        return self._rejections.get(tenant_id, {}).get(action_type)

    def _check_suggestion_rate(self, tenant_id: str, now: float) -> bool:
        """Check if another suggestion is allowed within rate limits."""
        timestamps = self._suggestion_times.get(tenant_id, [])
        one_hour_ago = now - 3600
        recent = [t for t in timestamps if t > one_hour_ago]
        return len(recent) < self.max_suggestions_per_hour

    def _record_suggestion(self, tenant_id: str, now: float) -> None:
        """Record a suggestion timestamp."""
        if tenant_id not in self._suggestion_times:
            self._suggestion_times[tenant_id] = []
        self._suggestion_times[tenant_id].append(now)
        # Prune old timestamps
        one_hour_ago = now - 3600
        self._suggestion_times[tenant_id] = [
            t for t in self._suggestion_times[tenant_id] if t > one_hour_ago
        ]

    def stats(self) -> dict[str, Any]:
        """Restraint engine statistics."""
        return {
            "total_evaluated": self._total_evaluated,
            "decisions": dict(self._decisions),
            "active_backoffs": sum(
                1
                for tenant in self._rejections.values()
                for record in tenant.values()
                if record.backoff_until > time.time()
            ),
            "confidence_threshold": self.confidence_threshold,
            "quiet_hours": f"{self.quiet_hours[0]}:00-{self.quiet_hours[1]}:00",
        }
