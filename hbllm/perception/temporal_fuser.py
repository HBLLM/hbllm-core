"""Temporal Fuser — cross-time perception event correlation.

Correlates perception events across sliding time windows to detect
higher-order patterns:

    "door opened → footsteps → voice" = someone entered the house
    "temperature spike → smoke alarm" = potential fire danger
    "motion → no motion → motion (different room)" = person walking through

Architecture:
    1. Maintains a sliding window of recent perception events
    2. Pattern matching via SNN accumulator for multi-event spike detection
    3. Publishes fused narrative events to `perception.fused.sequence`

Usage::

    fuser = TemporalFuser(window_s=60.0, bus=message_bus)
    await fuser.start()
    # Events from perception pipeline are automatically correlated
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


@dataclass
class PerceptionSnapshot:
    """A captured perception event with timestamp."""

    event_type: str  # e.g., "audio.ambient", "iot.door", "iot.motion"
    sub_type: str = ""  # e.g., "doorbell", "opened", "detected"
    source: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    room: str = ""


@dataclass
class FusedSequence:
    """A correlated sequence of perception events."""

    sequence_id: str = ""
    pattern_name: str = ""  # e.g., "person_entered", "fire_danger"
    narrative: str = ""  # Human-readable description
    confidence: float = 0.0
    events: list[PerceptionSnapshot] = field(default_factory=list)
    duration_s: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "pattern_name": self.pattern_name,
            "narrative": self.narrative,
            "confidence": round(self.confidence, 3),
            "event_count": len(self.events),
            "event_types": [e.event_type for e in self.events],
            "duration_s": round(self.duration_s, 1),
            "timestamp": self.timestamp,
        }


# ── Pattern Definitions ──────────────────────────────────────────────────────


@dataclass
class SequencePattern:
    """Definition of a temporal event sequence pattern."""

    name: str
    steps: list[dict[str, str]]  # List of {event_type, sub_type?} to match in order
    max_window_s: float = 60.0  # Maximum time window for the full sequence
    min_confidence: float = 0.6
    narrative_template: str = ""  # Template with {room}, {duration} placeholders


# Built-in sequence patterns
BUILTIN_PATTERNS: list[SequencePattern] = [
    SequencePattern(
        name="person_entered",
        steps=[
            {"event_type": "iot.door", "sub_type": "opened"},
            {"event_type": "iot.motion", "sub_type": "detected"},
        ],
        max_window_s=30.0,
        narrative_template="Someone entered through {room} door ({duration:.0f}s sequence).",
    ),
    SequencePattern(
        name="person_entered_audio",
        steps=[
            {"event_type": "iot.door", "sub_type": "opened"},
            {"event_type": "audio.ambient", "sub_type": "footsteps"},
        ],
        max_window_s=20.0,
        narrative_template="Door opened followed by footsteps in {room} ({duration:.0f}s).",
    ),
    SequencePattern(
        name="fire_danger",
        steps=[
            {"event_type": "iot.temperature", "sub_type": "high"},
            {"event_type": "audio.ambient", "sub_type": "smoke_detector"},
        ],
        max_window_s=120.0,
        min_confidence=0.8,
        narrative_template="Temperature spike followed by smoke alarm in {room}! Possible fire.",
    ),
    SequencePattern(
        name="person_left",
        steps=[
            {"event_type": "iot.motion", "sub_type": "cleared"},
            {"event_type": "iot.door", "sub_type": "opened"},
            {"event_type": "iot.door", "sub_type": "closed"},
        ],
        max_window_s=60.0,
        narrative_template="Motion cleared, door opened and closed — someone likely left via {room}.",
    ),
    SequencePattern(
        name="room_transition",
        steps=[
            {"event_type": "iot.motion", "sub_type": "cleared"},
            {"event_type": "iot.motion", "sub_type": "detected"},
        ],
        max_window_s=30.0,
        narrative_template="Movement from one room to another ({duration:.0f}s transition).",
    ),
    SequencePattern(
        name="glass_break_intrusion",
        steps=[
            {"event_type": "audio.ambient", "sub_type": "glass_breaking"},
            {"event_type": "iot.motion", "sub_type": "detected"},
        ],
        max_window_s=30.0,
        min_confidence=0.9,
        narrative_template="Glass breaking followed by motion — possible intrusion in {room}!",
    ),
    SequencePattern(
        name="appliance_forgotten",
        steps=[
            {"event_type": "iot.motion", "sub_type": "cleared"},
            {"event_type": "iot.power", "sub_type": "high"},
        ],
        max_window_s=300.0,
        narrative_template="Room vacated but appliance still running in {room}.",
    ),
    SequencePattern(
        name="doorbell_visitor",
        steps=[
            {"event_type": "audio.ambient", "sub_type": "doorbell"},
            {"event_type": "iot.motion", "sub_type": "detected"},
        ],
        max_window_s=60.0,
        narrative_template="Doorbell rang and motion detected at {room} — visitor at the door.",
    ),
]


# ── Temporal Fuser ────────────────────────────────────────────────────────────


class TemporalFuser:
    """Correlates perception events across time to detect narrative sequences.

    Maintains a sliding window of recent events and matches them against
    defined sequence patterns.
    """

    def __init__(
        self,
        window_s: float = 120.0,
        patterns: list[SequencePattern] | None = None,
        bus: Any | None = None,
        max_events: int = 500,
    ) -> None:
        self.window_s = window_s
        self.patterns = patterns or list(BUILTIN_PATTERNS)
        self.bus = bus
        self.max_events = max_events

        self._events: deque[PerceptionSnapshot] = deque(maxlen=max_events)
        self._last_fused: dict[str, float] = {}  # pattern_name → last fire time
        self._cooldown_s = 30.0  # Don't fire same pattern within 30s

        # Telemetry
        self._events_processed = 0
        self._sequences_detected = 0

    async def start(self) -> None:
        """Subscribe to perception events on the bus."""
        if self.bus:
            # Subscribe to all perception events
            await self.bus.subscribe("perception.*", self._on_perception_event)
            await self.bus.subscribe("iot.event", self._on_iot_event)
            logger.info(
                "TemporalFuser started (window=%.0fs, patterns=%d)",
                self.window_s,
                len(self.patterns),
            )

    def ingest(self, event: PerceptionSnapshot) -> list[FusedSequence]:
        """Ingest a perception event and check for pattern matches.

        Returns any newly detected sequences.
        """
        self._events_processed += 1
        self._events.append(event)

        # Prune old events
        self._prune_window()

        # Check all patterns
        detected: list[FusedSequence] = []
        for pattern in self.patterns:
            # Cooldown check
            last = self._last_fused.get(pattern.name, 0)
            if time.time() - last < self._cooldown_s:
                continue

            match = self._match_pattern(pattern)
            if match:
                self._last_fused[pattern.name] = time.time()
                self._sequences_detected += 1
                detected.append(match)

        return detected

    async def _on_perception_event(self, msg: Message) -> None:
        """Handle perception events from the bus."""
        snapshot = PerceptionSnapshot(
            event_type=msg.payload.get("event_type", msg.topic.split(".")[-1]),
            sub_type=msg.payload.get("sub_type", msg.payload.get("sound_class", "")),
            source=msg.source_node_id,
            payload=msg.payload,
            room=msg.payload.get("room", ""),
        )

        sequences = self.ingest(snapshot)

        # Publish detected sequences
        for seq in sequences:
            if self.bus:
                await self.bus.publish(
                    "perception.fused.sequence",
                    Message(
                        type=MessageType.EVENT,
                        source_node_id="temporal_fuser",
                        topic="perception.fused.sequence",
                        payload=seq.to_dict(),
                    ),
                )

    async def _on_iot_event(self, msg: Message) -> None:
        """Handle IoT events."""
        payload = msg.payload
        snapshot = PerceptionSnapshot(
            event_type=f"iot.{payload.get('device_type', 'unknown')}",
            sub_type=payload.get("state_change", payload.get("action", "")),
            source=msg.source_node_id,
            payload=payload,
            room=payload.get("room", ""),
        )
        sequences = self.ingest(snapshot)

        for seq in sequences:
            if self.bus:
                await self.bus.publish(
                    "perception.fused.sequence",
                    Message(
                        type=MessageType.EVENT,
                        source_node_id="temporal_fuser",
                        topic="perception.fused.sequence",
                        payload=seq.to_dict(),
                    ),
                )

    def _match_pattern(self, pattern: SequencePattern) -> FusedSequence | None:
        """Check if recent events match a sequence pattern."""
        if len(self._events) < len(pattern.steps):
            return None

        now = time.time()
        window_start = now - pattern.max_window_s

        # Get events within pattern's time window
        window_events = [e for e in self._events if e.timestamp >= window_start]

        if len(window_events) < len(pattern.steps):
            return None

        # Try to match steps in order (greedy forward scan)
        matched_events: list[PerceptionSnapshot] = []
        step_idx = 0

        for event in window_events:
            if step_idx >= len(pattern.steps):
                break

            step = pattern.steps[step_idx]
            if self._event_matches_step(event, step):
                matched_events.append(event)
                step_idx += 1

        # All steps matched?
        if step_idx < len(pattern.steps):
            return None

        # Compute confidence based on timing (tighter = higher confidence)
        duration = matched_events[-1].timestamp - matched_events[0].timestamp
        timing_ratio = duration / pattern.max_window_s
        confidence = max(0.0, 1.0 - timing_ratio * 0.5)  # Tighter timing = higher confidence

        if confidence < pattern.min_confidence:
            return None

        # Build narrative
        room = matched_events[0].room or "unknown area"
        narrative = pattern.narrative_template.format(
            room=room,
            duration=duration,
        )

        return FusedSequence(
            sequence_id=f"fused_{pattern.name}_{int(now)}",
            pattern_name=pattern.name,
            narrative=narrative,
            confidence=confidence,
            events=matched_events,
            duration_s=duration,
        )

    def _event_matches_step(self, event: PerceptionSnapshot, step: dict[str, str]) -> bool:
        """Check if a single event matches a pattern step."""
        if event.event_type != step.get("event_type", ""):
            return False
        if "sub_type" in step and event.sub_type != step["sub_type"]:
            return False
        return True

    def _prune_window(self) -> None:
        """Remove events outside the sliding window."""
        cutoff = time.time() - self.window_s
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()

    def get_recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent events for introspection."""
        events = list(self._events)[-limit:]
        return [
            {
                "event_type": e.event_type,
                "sub_type": e.sub_type,
                "room": e.room,
                "timestamp": e.timestamp,
                "age_s": round(time.time() - e.timestamp, 1),
            }
            for e in reversed(events)
        ]

    def stats(self) -> dict[str, Any]:
        """Fuser statistics."""
        return {
            "events_in_window": len(self._events),
            "events_processed": self._events_processed,
            "sequences_detected": self._sequences_detected,
            "pattern_count": len(self.patterns),
            "cooldown_s": self._cooldown_s,
        }
