"""
Autonomic Reflex Arc Module.

A sub-millisecond fast-path that routes critical sensor alerts directly
to action executors, bypassing the Global Workspace / LLM reasoning loop entirely.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.brain.snn import LIFConfig, SpikeEvent, SpikingAccumulator
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


class SpikingReflexRule(ReflexRule):
    """
    A rule that uses a SpikingAccumulator to trigger action only on accumulated signal bursts.

    Rather than triggering on a single threshold breach, it accumulates input current from matching events
    and fires a spike when the cumulative intensity/frequency exceeds the threshold.
    """

    def __init__(
        self,
        trigger: dict[str, Any],
        action_topic: str,
        action_payload: dict[str, Any],
        config: LIFConfig | None = None,
        max_latency_ms: float = 5.0,
        current_multiplier: float = 1.0,
    ) -> None:
        super().__init__(trigger, action_topic, action_payload, max_latency_ms)
        # Default config: fast reflex decay (decay half-life of 1.0s), threshold of 1.0,
        # and refractory period of 0.0 (allows consecutive/repeated spiking).
        self.config = config or LIFConfig(
            threshold=1.0,
            decay_half_life=1.0,
            reset_potential=0.0,
            refractory_period=0.0,
        )
        self.accumulator = SpikingAccumulator(self.config)
        self.current_multiplier = current_multiplier
        self.last_spike_event: SpikeEvent | None = None

    def matches(self, event: PerceptionEvent) -> bool:
        """
        Evaluate if the event matches the structural criteria and stimulate the SNN.
        Returns True only if a spike fires.
        """
        if not self.trigger:
            return False

        stimulus = 0.0

        # Verify structural header matches first (e.g. event_type matching)
        for key, expected_val in self.trigger.items():
            if key in ("event_type", "sub_type", "modality", "origin"):
                val = getattr(event, key, None)
                if val != expected_val:
                    return False
            else:
                # Check for numerical metrics inside payload or attributes
                val = getattr(event, key, None)
                if val is None and isinstance(event.payload, dict):
                    val = event.payload.get(key)

                if isinstance(val, (int, float)):
                    stimulus += float(val) * self.current_multiplier
                else:
                    if val != expected_val:
                        return False

        # Apply a default current step if no numerical stimulus was gathered
        if stimulus == 0.0:
            stimulus = 0.25 * self.config.threshold

        now = time.time()
        spike_event = self.accumulator.stimulate(stimulus, timestamp=now)
        self.last_spike_event = spike_event

        if spike_event.fired:
            if isinstance(self.action_payload, dict):
                self.action_payload["spike_strength"] = spike_event.strength
            return True

        return False


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

                strength_info = ""
                if isinstance(rule, SpikingReflexRule) and rule.last_spike_event:
                    strength_info = f" (Spike strength: {rule.last_spike_event.strength:.2f})"

                logger.info(
                    "Autonomic Reflex Activated! Event %s matched rule for topic %s. Latency: %.3fms%s",
                    event.event_id,
                    rule.action_topic,
                    latency_ms,
                    strength_info,
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
                                "spike_strength": getattr(
                                    rule.last_spike_event, "strength", None
                                )
                                if isinstance(rule, SpikingReflexRule)
                                else None,
                            },
                        )
                        self.event_log.append(reflex_event)
                    except Exception as e:
                        logger.error("Failed to write reflex activation to EventLog: %s", e)

                self.fired_actions.append((rule.action_topic, rule.action_payload))
                break

