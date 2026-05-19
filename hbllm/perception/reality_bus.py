"""Reality Event Bus — unified perception ingestion layer.

This module provides the schema and routing for all real-world events
(OS activity, sensors, apps) into the cognitive system.

Key concepts:
- Modality Tiers: Differentiate high-trust system events from noisy sensors.
- Logical Ordering: Vector clocks and ingest timestamps for sync.
- Origin Tagging: Prevent infinite feedback loops from autonomous actions.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class PerceptionModality(StrEnum):
    """Trust tiers for incoming perception events."""

    SYSTEM = "system"      # High trust, unlimited budget (e.g., OS idle state)
    APP = "app"            # Medium trust, throttled (e.g., calendar, VSCode)
    SENSOR = "sensor"      # Low trust/noisy, sampled (e.g., webcam motion)
    INFERRED = "inferred"  # AI-generated, heavily rate-limited


class EventOrigin(StrEnum):
    """Where did this event originate from?"""

    EXTERNAL = "external"    # Physical world / user
    SYSTEM = "system"        # OS / Device
    AUTONOMY = "autonomy"    # HBLLM AutonomyCore (danger: feedback loop risk)


@dataclass
class PerceptionEvent:
    """A standardized event representing a change in real-world reality."""

    # Identity
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    entity_id: str = "unknown"  # What entity does this describe? (e.g. 'user_1', 'display_0')

    # Classification
    event_type: str = ""        # e.g., "activity", "schedule", "motion"
    sub_type: str = ""          # e.g., "window_focus", "meeting_start"
    modality: PerceptionModality = PerceptionModality.APP
    origin: EventOrigin = EventOrigin.EXTERNAL

    # Confidence & Truth
    confidence: float = 1.0     # 0.0 to 1.0 (how sure are we this happened?)
    source_trust: float = 1.0   # 0.0 to 1.0 (how reliable is the sensor?)
    priority_hint: int = 0      # 0 (low) to 100 (critical)

    # Ordering
    event_timestamp: float = field(default_factory=time.time)
    ingest_timestamp: float = 0.0
    logical_clock: int = 0      # Assigned by RealityEventBus

    # Payload
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "entity_id": self.entity_id,
            "event_type": self.event_type,
            "sub_type": self.sub_type,
            "modality": self.modality.value,
            "origin": self.origin.value,
            "confidence": self.confidence,
            "source_trust": self.source_trust,
            "priority_hint": self.priority_hint,
            "event_timestamp": self.event_timestamp,
            "ingest_timestamp": self.ingest_timestamp,
            "logical_clock": self.logical_clock,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PerceptionEvent:
        return cls(
            event_id=d["event_id"],
            entity_id=d.get("entity_id", "unknown"),
            event_type=d.get("event_type", ""),
            sub_type=d.get("sub_type", ""),
            modality=PerceptionModality(d.get("modality", "app")),
            origin=EventOrigin(d.get("origin", "external")),
            confidence=d.get("confidence", 1.0),
            source_trust=d.get("source_trust", 1.0),
            priority_hint=d.get("priority_hint", 0),
            event_timestamp=d.get("event_timestamp", 0.0),
            ingest_timestamp=d.get("ingest_timestamp", 0.0),
            logical_clock=d.get("logical_clock", 0),
            payload=d.get("payload", {}),
        )


class RealityEventBus:
    """Ingestion pipeline for physical and digital reality events.

    This acts as the sensory cortex. It accepts raw events from adapters,
    assigns logical clocks and ingest timestamps, and routes them to
    subscribers (typically the EventNormalizer).
    """

    def __init__(self) -> None:
        self._subscribers: list[Callable[[PerceptionEvent], Any]] = []
        self._logical_clock: int = 0
        self._lock = asyncio.Lock()

    def subscribe(self, callback: Callable[[PerceptionEvent], Any]) -> None:
        """Subscribe to the raw reality stream."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[PerceptionEvent], Any]) -> None:
        """Remove a subscription."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    async def ingest(self, event: PerceptionEvent) -> None:
        """Ingest a new raw event from a sensor/adapter."""
        async with self._lock:
            self._logical_clock += 1
            event.logical_clock = self._logical_clock
            event.ingest_timestamp = time.time()

        # Fire and forget to subscribers (to avoid blocking ingestion)
        for sub in self._subscribers:
            try:
                res = sub(event)
                if asyncio.iscoroutine(res):
                    asyncio.create_task(res)
            except Exception as e:
                logger.error("Error in RealityEventBus subscriber: %s", e)

    def get_current_clock(self) -> int:
        return self._logical_clock
