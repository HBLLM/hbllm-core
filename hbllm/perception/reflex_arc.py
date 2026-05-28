"""
Autonomic Reflex Arc Module.

A sub-millisecond fast-path that routes critical sensor alerts directly
to action executors, bypassing the Global Workspace / LLM reasoning loop entirely.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.perception.event_log import EventLog
from hbllm.perception.reality_bus import (
    EventOrigin,
    PerceptionEvent,
    PerceptionModality,
    RealityEventBus,
)

logger = logging.getLogger(__name__)


class ReflexRule:
    """Represents a rule for the Reflex Arc to trigger immediate action on match."""

    def __init__(
        self,
        trigger: dict[str, Any],
        action_topic: str,
        action_payload: dict[str, Any],
        max_latency_ms: float = 5.0,
    ) -> None:
        self.trigger = trigger
        self.action_topic = action_topic
        self.action_payload = action_payload
        self.max_latency_ms = max_latency_ms

    def matches(self, event: PerceptionEvent) -> bool:
        """Evaluate if the PerceptionEvent matches this rule's trigger."""
        for key, expected_val in self.trigger.items():
            if "__gte" in key:
                field = key.replace("__gte", "")
                val = getattr(event, field, None)
                if val is None or val < expected_val:
                    return False
            elif "__gt" in key:
                field = key.replace("__gt", "")
                val = getattr(event, field, None)
                if val is None or val <= expected_val:
                    return False
            elif "__lte" in key:
                field = key.replace("__lte", "")
                val = getattr(event, field, None)
                if val is None or val > expected_val:
                    return False
            elif "__lt" in key:
                field = key.replace("__lt", "")
                val = getattr(event, field, None)
                if val is None or val >= expected_val:
                    return False
            else:
                # Exact match
                val = getattr(event, key, None)
                if val != expected_val:
                    return False
        return True


class ReflexArc:
    """
    Sub-millisecond fast-path that routes critical sensor alerts directly to action executors,
    bypassing the Global Workspace/LLM reasoning loop entirely.
    """

    def __init__(
        self,
        bus: RealityEventBus,
        rules: list[ReflexRule],
        event_log: EventLog | None = None,
    ) -> None:
        self.bus = bus
        self.rules = rules
        self.event_log = event_log
        self.bus.subscribe_pre(self.handle_event)
        self.fired_actions: list[tuple[str, dict[str, Any]]] = []

    def handle_event(self, event: PerceptionEvent) -> None:
        """Process event and fire matching reflex rules immediately."""
        start_time = time.perf_counter()
        for rule in self.rules:
            if rule.matches(event):
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                if latency_ms > rule.max_latency_ms:
                    logger.warning(
                        "Reflex rule matched but exceeded latency budget: %.3fms > %.3fms",
                        latency_ms,
                        rule.max_latency_ms,
                    )

                logger.info(
                    "Autonomic Reflex Activated! Event %s matched rule for topic %s. Latency: %.3fms",
                    event.event_id,
                    rule.action_topic,
                    latency_ms,
                )

                if self.event_log:
                    try:
                        reflex_event = PerceptionEvent(
                            event_type="reflex_activation",
                            sub_type=rule.action_topic,
                            modality=PerceptionModality.SYSTEM,
                            origin=EventOrigin.SYSTEM,
                            payload={
                                "triggering_event_id": event.event_id,
                                "action_topic": rule.action_topic,
                                "action_payload": rule.action_payload,
                                "latency_ms": latency_ms,
                            },
                        )
                        self.event_log.append(reflex_event)
                    except Exception as e:
                        logger.error("Failed to write reflex activation to EventLog: %s", e)

                self.fired_actions.append((rule.action_topic, rule.action_payload))
                break
