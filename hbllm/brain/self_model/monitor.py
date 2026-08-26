"""Metacognitive Monitor and Event-Driven Self-Correction State Machine for A21.

Implements real-time cognitive monitoring, circular search loop detection,
root-cause failure diagnosis, and strategy switching state machine:
NORMAL -> PREDICTION_ERROR -> RETRY_ALLOWED -> DIAGNOSE -> [PROBE / SPECIALIZE / LEARN / BUDGET] -> RE_EVALUATE.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MetacognitiveState(str, Enum):
    """State machine for metacognitive monitoring and strategic control."""

    NORMAL = "normal"  # Routine nominal execution
    PREDICTION_ERROR = "prediction_error"  # Single prediction error detected
    RETRY_ALLOWED = "retry_allowed"  # Allowed single quick retry if error was minor
    DIAGNOSE = "diagnose"  # Root-cause diagnostic analysis triggered
    PROBE = "probe"  # Switched to A19 epistemic probe
    SPECIALIZE = "specialize"  # Switched to A20 schema boundary specialization
    LEARN = "learn"  # Switched to A14 predictive parameter adaptation
    ADJUST_BUDGET = "adjust_budget"  # Switched to cognitive resource budget expansion
    RE_EVALUATE = "re_evaluate"  # Post-correction verification and resumption


class MetacognitiveEventType(str, Enum):
    """Event taxonomy for metacognitive signals."""

    PREDICTION_ERROR_HIGH = "prediction_error_high"
    REPEATED_FAILURE = "repeated_failure"
    SEARCH_CYCLE_DETECTED = "search_cycle_detected"
    SIMULATION_EXHAUSTED = "simulation_exhausted"
    CONFIDENCE_COLLAPSE = "confidence_collapse"
    SCHEMA_CONTRADICTION = "schema_contradiction"
    UNKNOWN_DOMAIN = "unknown_domain"


class FailureCause(str, Enum):
    """Diagnosed root cause of an action or reasoning failure."""

    INSUFFICIENT_KNOWLEDGE = "insufficient_knowledge"  # Missing properties / ungrounded state
    INCORRECT_SCHEMA = "incorrect_schema"  # Applied wrong analogical schema
    MODEL_INADEQUACY = "model_inadequacy"  # Current representation has flawed physics/assumptions
    BUDGET_EXHAUSTION = "budget_exhaustion"  # Insufficient simulation depth/time
    RANDOM_NOISE = "random_noise"  # Minor aleatoric disturbance


class StrategyAction(str, Enum):
    """Recommended strategic correction action."""

    A19_PROBE = "a19_probe"
    A20_SPECIALIZATION = "a20_specialization"
    A14_LEARNING = "a14_learning"
    ADJUST_BUDGET = "adjust_budget"
    RETRY = "retry"


@dataclass
class MetacognitiveEvent:
    """An event emitted when cognitive monitoring detects anomalies or state shifts."""

    event_id: str = field(default_factory=lambda: f"mce_{uuid.uuid4().hex[:8]}")
    event_type: MetacognitiveEventType = MetacognitiveEventType.PREDICTION_ERROR_HIGH
    domain: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    severity: float = 0.50  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class FailureDiagnosis:
    """The root-cause analysis output and prescribed strategy."""

    domain: str
    cause: FailureCause
    recommended_strategy: StrategyAction
    rationale: str
    consecutive_failures: int


class MetacognitiveMonitor:
    """Monitors cognitive state, detects search cycles, and executes strategy switching state machine."""

    def __init__(self) -> None:
        self.state: MetacognitiveState = MetacognitiveState.NORMAL
        self.events: list[MetacognitiveEvent] = []
        self.consecutive_failures: dict[str, int] = {}  # domain -> count
        self.action_history: list[tuple[str, str]] = []  # (domain, action_str)

    def record_action(self, domain: str, action_repr: str) -> bool:
        """Track action history and detect circular search cycles. Returns True if cycle detected."""
        self.action_history.append((domain, action_repr))
        if len(self.action_history) >= 4:
            # Check for immediate oscillation: A -> B -> A -> B
            last4 = self.action_history[-4:]
            if last4[0] == last4[2] and last4[1] == last4[3]:
                self.emit_event(
                    MetacognitiveEventType.SEARCH_CYCLE_DETECTED,
                    domain=domain,
                    details={"cycle": [a[1] for a in last4]},
                    severity=0.85,
                )
                self.state = MetacognitiveState.DIAGNOSE
                return True
        return False

    def emit_event(
        self,
        event_type: MetacognitiveEventType,
        domain: str,
        details: dict[str, Any] | None = None,
        severity: float = 0.50,
    ) -> MetacognitiveEvent:
        """Emit and store a structured metacognitive event."""
        evt = MetacognitiveEvent(
            event_type=event_type,
            domain=domain,
            details=details or {},
            severity=severity,
        )
        self.events.append(evt)
        logger.info("Metacognitive event emitted: %s in domain '%s'", event_type.value, domain)
        return evt

    def process_prediction_outcome(
        self,
        domain: str,
        predicted_confidence: float,
        actual_success: bool,
        context_details: dict[str, Any] | None = None,
    ) -> tuple[MetacognitiveState, FailureDiagnosis | None]:
        """State machine transition based on prediction and execution outcomes."""
        details = context_details or {}

        if actual_success:
            self.consecutive_failures[domain] = 0
            self.state = MetacognitiveState.NORMAL
            return self.state, None

        # Failure occurred
        self.consecutive_failures[domain] = self.consecutive_failures.get(domain, 0) + 1
        fail_count = self.consecutive_failures[domain]

        # High-confidence surprise failure (predicted >= 0.75, but failed)
        is_surprise = predicted_confidence >= 0.75
        severity = 0.85 if is_surprise else 0.50

        self.emit_event(
            MetacognitiveEventType.PREDICTION_ERROR_HIGH
            if is_surprise
            else MetacognitiveEventType.REPEATED_FAILURE,
            domain=domain,
            details={"fail_count": fail_count, "predicted_confidence": predicted_confidence},
            severity=severity,
        )

        # State transition:
        is_specific_issue = bool(
            details.get("is_unknown_domain")
            or details.get("is_transfer")
            or details.get("schema_id")
            or details.get("missing_property")
            or details.get("truncated_budget")
        )
        if fail_count == 1 and not is_surprise and not is_specific_issue:
            self.state = MetacognitiveState.RETRY_ALLOWED
            diagnosis = FailureDiagnosis(
                domain=domain,
                cause=FailureCause.RANDOM_NOISE,
                recommended_strategy=StrategyAction.RETRY,
                rationale="Minor single failure under moderate confidence: single retry permitted.",
                consecutive_failures=fail_count,
            )
            return self.state, diagnosis

        # 2+ failures, high-confidence surprise, or specific issue -> transition to DIAGNOSE
        self.state = MetacognitiveState.DIAGNOSE
        diagnosis = self._diagnose_failure_cause(domain, fail_count, is_surprise, details)

        # Transition state according to prescribed strategy
        if diagnosis.recommended_strategy == StrategyAction.A19_PROBE:
            self.state = MetacognitiveState.PROBE
        elif diagnosis.recommended_strategy == StrategyAction.A20_SPECIALIZATION:
            self.state = MetacognitiveState.SPECIALIZE
        elif diagnosis.recommended_strategy == StrategyAction.A14_LEARNING:
            self.state = MetacognitiveState.LEARN
        elif diagnosis.recommended_strategy == StrategyAction.ADJUST_BUDGET:
            self.state = MetacognitiveState.ADJUST_BUDGET
        else:
            self.state = MetacognitiveState.RETRY_ALLOWED

        return self.state, diagnosis

    def _diagnose_failure_cause(
        self,
        domain: str,
        fail_count: int,
        is_surprise: bool,
        details: dict[str, Any],
    ) -> FailureDiagnosis:
        """Determine root cause and prescribe optimal corrective strategy."""
        # 1. Unknown domain or missing entity property
        if details.get("is_unknown_domain") or details.get("missing_property"):
            return FailureDiagnosis(
                domain=domain,
                cause=FailureCause.INSUFFICIENT_KNOWLEDGE,
                recommended_strategy=StrategyAction.A19_PROBE,
                rationale="Unknown domain / missing properties: A19 epistemic probe required to collapse uncertainty.",
                consecutive_failures=fail_count,
            )

        # 2. Schema contradiction (e.g. analogical transfer failed)
        if details.get("schema_id") or details.get("is_transfer"):
            return FailureDiagnosis(
                domain=domain,
                cause=FailureCause.INCORRECT_SCHEMA,
                recommended_strategy=StrategyAction.A20_SPECIALIZATION,
                rationale="Analogical schema transfer violated target constraints: A20 boundary specialization required.",
                consecutive_failures=fail_count,
            )

        # 3. Budget / depth truncation
        if details.get("truncated_budget"):
            return FailureDiagnosis(
                domain=domain,
                cause=FailureCause.BUDGET_EXHAUSTION,
                recommended_strategy=StrategyAction.ADJUST_BUDGET,
                rationale="Search/simulation budget exhausted: expand A18 depth/branch allocations.",
                consecutive_failures=fail_count,
            )

        # 4. Repeated high-confidence surprise -> Model structural inadequacy
        if is_surprise or fail_count >= 2:
            return FailureDiagnosis(
                domain=domain,
                cause=FailureCause.MODEL_INADEQUACY,
                recommended_strategy=StrategyAction.A19_PROBE,
                rationale="Structural model mismatch detected: halt blind execution and initiate A19 diagnostic probing.",
                consecutive_failures=fail_count,
            )

        return FailureDiagnosis(
            domain=domain,
            cause=FailureCause.RANDOM_NOISE,
            recommended_strategy=StrategyAction.RETRY,
            rationale="Permitted nominal retry.",
            consecutive_failures=fail_count,
        )
