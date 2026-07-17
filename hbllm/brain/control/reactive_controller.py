"""
Reactive Controller — sub-10ms reflex, interrupt, and safety handler.

Part of the Tripartite Executive (ADR 002 §1):
    - **ReactiveController**: Reflexes, interrupts, urgent events (<10ms).
    - DeliberativeController: Planning, reasoning, SkillGraph execution.
    - ReflectiveController: Post-eval, memory consolidation, self-improvement.

The ReactiveController bypasses the full deliberative pipeline for events
that require immediate response: acoustic wake words, safety interrupts,
user cancellations, and emergency stop signals.

Design invariants:
    - Must complete within 10ms for any single event.
    - Never performs LLM inference — only pattern-matched reflex arcs.
    - Always runs at highest scheduler priority (USER_INTERACTIVE).

Usage::

    from hbllm.brain.control.reactive_controller import ReactiveController

    controller = ReactiveController()
    controller.register_reflex("wake_word", handle_wake_word)
    result = await controller.process(event)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hbllm.brain.core.cognitive_event import CognitiveEvent, CognitiveEventType
from hbllm.brain.core.provenance import ProvenanceMetadata, VerificationState

logger = logging.getLogger(__name__)


class ReflexType(StrEnum):
    """Categories of reactive reflexes."""

    WAKE_WORD = "wake_word"
    SAFETY_INTERRUPT = "safety_interrupt"
    USER_CANCEL = "user_cancel"
    EMERGENCY_STOP = "emergency_stop"
    ATTENTION_SPIKE = "attention_spike"
    TIMEOUT = "timeout"


@dataclass
class ReflexResult:
    """Result of a reactive reflex execution.

    Attributes:
        handled: Whether the event was consumed by a reflex.
        reflex_type: Which reflex matched (if any).
        action_taken: Description of the action performed.
        latency_ms: Execution time in milliseconds.
        provenance: Provenance metadata for the reflex result.
    """

    handled: bool = False
    reflex_type: ReflexType | None = None
    action_taken: str = ""
    latency_ms: float = 0.0
    provenance: ProvenanceMetadata | None = None


# Type alias for reflex handlers
ReflexHandler = Callable[[CognitiveEvent], Awaitable[ReflexResult]]


class ReactiveController:
    """Sub-10ms reflex and interrupt handler.

    Maintains a registry of pattern-matched reflex arcs that fire
    immediately when matching events arrive, bypassing the full
    deliberative planning pipeline.

    Args:
        max_latency_ms: Maximum allowed latency per reflex (default 10ms).
            Events exceeding this are logged as warnings.
    """

    def __init__(self, max_latency_ms: float = 10.0) -> None:
        self._max_latency_ms = max_latency_ms

        # Registry: event type → list of (reflex_type, handler) pairs
        self._reflexes: dict[CognitiveEventType, list[tuple[ReflexType, ReflexHandler]]] = {}

        # Telemetry
        self._total_processed = 0
        self._total_handled = 0
        self._total_latency_violations = 0

        logger.info(
            "ReactiveController initialized (max_latency=%.1fms)",
            max_latency_ms,
        )

    # ── Reflex registration ──────────────────────────────────────────

    def register_reflex(
        self,
        event_type: CognitiveEventType,
        reflex_type: ReflexType,
        handler: ReflexHandler,
    ) -> None:
        """Register a reflex arc for a specific event type.

        Args:
            event_type: The cognitive event type to match.
            reflex_type: Category label for this reflex.
            handler: Async callable that processes the event.
        """
        if event_type not in self._reflexes:
            self._reflexes[event_type] = []
        self._reflexes[event_type].append((reflex_type, handler))
        logger.debug(
            "Registered reflex %s for event type %s",
            reflex_type.value,
            event_type.value,
        )

    # ── Event processing ─────────────────────────────────────────────

    async def process(self, event: CognitiveEvent) -> ReflexResult:
        """Attempt to handle an event via registered reflexes.

        If no reflex matches, returns ``ReflexResult(handled=False)``
        so the event can be forwarded to the DeliberativeController.

        Args:
            event: The incoming cognitive event.

        Returns:
            ReflexResult indicating whether a reflex fired.
        """
        self._total_processed += 1
        handlers = self._reflexes.get(event.type, [])

        if not handlers:
            return ReflexResult(handled=False)

        start = time.monotonic()
        for reflex_type, handler in handlers:
            try:
                result = await asyncio.wait_for(
                    handler(event),
                    timeout=self._max_latency_ms / 1000.0,
                )
                latency_ms = (time.monotonic() - start) * 1000.0

                if result.handled:
                    self._total_handled += 1
                    result.latency_ms = latency_ms
                    result.reflex_type = reflex_type

                    if latency_ms > self._max_latency_ms:
                        self._total_latency_violations += 1
                        logger.warning(
                            "Reflex %s exceeded latency budget: %.2fms > %.1fms",
                            reflex_type.value,
                            latency_ms,
                            self._max_latency_ms,
                        )

                    # Attach provenance
                    result.provenance = ProvenanceMetadata.create(
                        source=f"reactive.{reflex_type.value}",
                        confidence=1.0,
                        correlation_id=(
                            event.provenance.correlation_id
                            if event.provenance
                            else event.correlation_id
                        ),
                        verification_state=VerificationState.VERIFIED,
                    )
                    return result

            except asyncio.TimeoutError:
                self._total_latency_violations += 1
                logger.warning(
                    "Reflex %s timed out (>%.1fms) for event %s",
                    reflex_type.value,
                    self._max_latency_ms,
                    event.type.value,
                )
            except Exception as exc:
                logger.error(
                    "Reflex %s raised exception for event %s: %s",
                    reflex_type.value,
                    event.type.value,
                    exc,
                )

        return ReflexResult(handled=False)

    def is_reactive_event(self, event: CognitiveEvent) -> bool:
        """Check if an event has registered reflex handlers."""
        return event.type in self._reflexes

    # ── Telemetry ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Controller statistics."""
        return {
            "total_processed": self._total_processed,
            "total_handled": self._total_handled,
            "total_latency_violations": self._total_latency_violations,
            "registered_reflexes": {k.value: len(v) for k, v in self._reflexes.items()},
        }
