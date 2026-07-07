"""Attention System — cognitive event prioritization and interruption control.

Implements a multi-factor attention scoring model that decides:
  - Whether an incoming event deserves cognitive resources
  - Whether it should interrupt the current cognitive state
  - How to debounce / decay repeated low-value events

Scoring Model
─────────────
priority_score =
    urgency
  + user_focus_weight
  + emotional_weight
  + temporal_relevance
  + active_goal_alignment
  - interruption_cost
  - cognitive_load_penalty

Event Decay
───────────
Repeated similar events within a short window receive exponentially
decreasing scores to prevent notification storms and CPU thrashing.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Attention Event ──────────────────────────────────────────────────────────


@dataclass
class AttentionEvent:
    """An event submitted to the attention system for scoring.

    Producers (perception, bus listeners, scheduled tasks) create these
    and submit them via ``AttentionSystem.score_event()``.
    """

    event_id: str
    source: str  # e.g. "user_input", "sensor.temperature", "scheduler"
    category: str  # e.g. "user_action", "system_alert", "background"
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)

    # ── Scoring hints (set by the producer) ──────────────────────────
    urgency: float = 0.5  # 0.0 = not urgent, 1.0 = critical
    emotional_weight: float = 0.0  # 0.0 = neutral, 1.0 = high emotional salience
    goal_alignment: float = 0.0  # 0.0 = unrelated, 1.0 = directly supports active goal
    temporal_relevance: float = 0.5  # 0.0 = stale, 1.0 = time-critical right now


@dataclass
class ScoredEvent:
    """An event after attention scoring, ready for the cognitive loop."""

    event: AttentionEvent
    priority_score: float
    should_interrupt: bool
    decay_applied: float = 0.0  # How much decay was applied
    scored_at: float = field(default_factory=time.monotonic)


# ── Incremental Context Window ───────────────────────────────────────────────


@dataclass
class ContextEntity:
    """A tracked entity in the incremental context window."""

    entity_id: str
    entity_type: str  # "person", "device", "task", "topic"
    salience: float = 0.5  # 0.0–1.0, decays over time
    last_seen: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)
    mention_count: int = 1


class IncrementalContextWindow:
    """Maintains a rolling, salience-weighted view of active context.

    Instead of rebuilding the full context every cycle, this window
    incrementally updates tracked entities. High-salience entities
    persist longer; low-salience ones decay and are pruned.

    This provides the ``AutonomyCore`` with immediate situational
    awareness without expensive context reconstruction.
    """

    def __init__(
        self,
        max_entities: int = 200,
        decay_rate: float = 0.98,
        prune_threshold: float = 0.05,
    ) -> None:
        self._entities: dict[str, ContextEntity] = {}
        self._max_entities = max_entities
        self._decay_rate = decay_rate
        self._prune_threshold = prune_threshold

    def update(
        self,
        entity_id: str,
        entity_type: str = "topic",
        salience_boost: float = 0.3,
        metadata: dict[str, Any] | None = None,
    ) -> ContextEntity:
        """Update or insert an entity in the context window.

        If the entity exists, its salience is boosted and metadata merged.
        If new, it is inserted with the given salience.
        """
        now = time.monotonic()
        if entity_id in self._entities:
            entity = self._entities[entity_id]
            entity.salience = min(1.0, entity.salience + salience_boost)
            entity.last_seen = now
            entity.mention_count += 1
            if metadata:
                entity.metadata.update(metadata)
        else:
            entity = ContextEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                salience=min(1.0, 0.3 + salience_boost),
                last_seen=now,
                metadata=metadata or {},
            )
            self._entities[entity_id] = entity

        # Enforce capacity
        if len(self._entities) > self._max_entities:
            self._prune()

        return entity

    def decay_all(self) -> int:
        """Apply temporal decay to all entities. Returns count of pruned."""
        pruned = 0
        to_remove: list[str] = []
        for eid, entity in self._entities.items():
            entity.salience *= self._decay_rate
            if entity.salience < self._prune_threshold:
                to_remove.append(eid)
        for eid in to_remove:
            del self._entities[eid]
            pruned += 1
        return pruned

    def get_top(self, n: int = 10) -> list[ContextEntity]:
        """Return the top-N most salient entities."""
        return sorted(self._entities.values(), key=lambda e: e.salience, reverse=True)[:n]

    def get_entity(self, entity_id: str) -> ContextEntity | None:
        return self._entities.get(entity_id)

    def remove(self, entity_id: str) -> bool:
        if entity_id in self._entities:
            del self._entities[entity_id]
            return True
        return False

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    def snapshot(self) -> list[dict[str, Any]]:
        """Serializable snapshot of the context window."""
        return [
            {
                "entity_id": e.entity_id,
                "entity_type": e.entity_type,
                "salience": round(e.salience, 4),
                "mention_count": e.mention_count,
            }
            for e in self.get_top(20)
        ]

    def _prune(self) -> None:
        """Remove lowest-salience entities to stay within capacity."""
        sorted_entities = sorted(self._entities.values(), key=lambda e: e.salience)
        remove_count = len(self._entities) - self._max_entities + 10
        for entity in sorted_entities[:remove_count]:
            del self._entities[entity.entity_id]


# ── Attention System ─────────────────────────────────────────────────────────


class AttentionSystem:
    """Multi-factor attention scoring and event prioritization.

    Responsibilities:
      1. Score incoming events against a multi-factor model
      2. Apply event decay / debouncing for repeated similar events
      3. Maintain the incremental context window
      4. Decide whether an event should interrupt the current state
      5. Enforce thought budgets / cooldowns to prevent runaway cognition

    Usage::

        attention = AttentionSystem()
        scored = attention.score_event(event, cognitive_load=0.3, user_active=True)
        if scored.should_interrupt:
            state_machine.transition_to(CognitiveState.INTERRUPTED)

        # Periodic maintenance
        attention.tick()  # decay events, prune context
    """

    def __init__(
        self,
        # Scoring weights
        urgency_weight: float = 0.30,
        user_focus_weight: float = 0.20,
        emotional_weight: float = 0.10,
        temporal_weight: float = 0.15,
        goal_weight: float = 0.15,
        interruption_cost_weight: float = 0.05,
        cognitive_load_weight: float = 0.05,
        # Event decay
        decay_window_s: float = 30.0,
        decay_factor: float = 0.5,
        max_decay: float = 0.8,
        # Thought budgets
        max_thoughts_per_minute: int = 30,
        cooldown_after_burst_s: float = 5.0,
        # Context window
        context_window: IncrementalContextWindow | None = None,
    ) -> None:
        # Scoring weights
        self._w_urgency = urgency_weight
        self._w_user_focus = user_focus_weight
        self._w_emotional = emotional_weight
        self._w_temporal = temporal_weight
        self._w_goal = goal_weight
        self._w_interruption_cost = interruption_cost_weight
        self._w_cognitive_load = cognitive_load_weight

        # Event decay tracking: source → list of timestamps
        self._event_history: dict[str, list[float]] = defaultdict(list)
        self._decay_window_s = decay_window_s
        self._decay_factor = decay_factor
        self._max_decay = max_decay

        # Thought budgets
        self._max_thoughts_per_minute = max_thoughts_per_minute
        self._cooldown_after_burst_s = cooldown_after_burst_s
        self._thought_timestamps: deque[float] = deque(maxlen=max_thoughts_per_minute * 2)
        self._cooldown_until: float = 0.0

        # Context window
        self.context = context_window or IncrementalContextWindow()

        # Telemetry
        self._events_scored: int = 0
        self._events_interrupted: int = 0
        self._events_decayed: int = 0
        self._events_budget_blocked: int = 0

    # ── Core Scoring ──────────────────────────────────────────────────

    def score_event(
        self,
        event: AttentionEvent,
        *,
        cognitive_load: float = 0.0,
        user_active: bool = False,
        interruption_threshold: float = 0.5,
    ) -> ScoredEvent:
        """Score an event and determine whether it should interrupt.

        Args:
            event: The incoming attention event.
            cognitive_load: Current cognitive load (0.0–1.0) from the
                CognitiveStateMachine.
            user_active: Whether the user is currently interacting.
            interruption_threshold: The current state's interruption
                threshold from the CognitiveStateMachine.

        Returns:
            A ``ScoredEvent`` with the computed priority and
            interruption decision.
        """
        # Check thought budget
        if self._is_over_budget():
            self._events_budget_blocked += 1
            return ScoredEvent(
                event=event,
                priority_score=0.0,
                should_interrupt=False,
                decay_applied=0.0,
            )

        # Compute raw score
        user_focus = 0.8 if user_active and event.category == "user_action" else 0.2
        interruption_cost = 0.7 if cognitive_load > 0.6 else 0.3

        raw_score = (
            self._w_urgency * event.urgency
            + self._w_user_focus * user_focus
            + self._w_emotional * event.emotional_weight
            + self._w_temporal * event.temporal_relevance
            + self._w_goal * event.goal_alignment
            - self._w_interruption_cost * interruption_cost
            - self._w_cognitive_load * cognitive_load
        )

        # Apply event decay (debouncing)
        decay = self._compute_decay(event.source)
        final_score = max(0.0, min(1.0, raw_score - decay))

        if decay > 0.01:
            self._events_decayed += 1

        # Record this event for future decay calculations
        self._record_event(event.source)

        # Interruption decision
        should_interrupt = final_score > interruption_threshold

        if should_interrupt:
            self._events_interrupted += 1

        # Update context window with event source
        self.context.update(
            entity_id=event.source,
            entity_type="event_source",
            salience_boost=final_score * 0.5,
        )

        # Record thought
        self._thought_timestamps.append(time.monotonic())

        self._events_scored += 1

        return ScoredEvent(
            event=event,
            priority_score=round(final_score, 4),
            should_interrupt=should_interrupt,
            decay_applied=round(decay, 4),
        )

    # ── Event Decay / Debouncing ──────────────────────────────────────

    def _compute_decay(self, source: str) -> float:
        """Compute decay penalty for repeated events from the same source.

        Each recent occurrence within the decay window adds exponentially
        increasing decay, capped at ``max_decay``.
        """
        now = time.monotonic()
        recent = [t for t in self._event_history[source] if now - t < self._decay_window_s]
        if not recent:
            return 0.0

        # Each occurrence adds decay_factor^(1/count) penalty
        count = len(recent)
        decay = min(self._max_decay, count * self._decay_factor * 0.1)
        return decay

    def _record_event(self, source: str) -> None:
        """Record an event timestamp for decay tracking."""
        now = time.monotonic()
        self._event_history[source].append(now)

        # Prune old entries
        cutoff = now - self._decay_window_s * 2
        self._event_history[source] = [t for t in self._event_history[source] if t > cutoff]

    # ── Thought Budget ────────────────────────────────────────────────

    def _is_over_budget(self) -> bool:
        """Check if we've exceeded the thought-per-minute budget.

        If exceeded, enter a cooldown period to prevent runaway cognition.
        """
        now = time.monotonic()

        # Active cooldown
        if now < self._cooldown_until:
            return True

        # Count thoughts in the last 60 seconds
        cutoff = now - 60.0
        recent_count = sum(1 for t in self._thought_timestamps if t > cutoff)

        if recent_count >= self._max_thoughts_per_minute:
            self._cooldown_until = now + self._cooldown_after_burst_s
            logger.warning(
                "Thought budget exceeded (%d/%d per minute). Entering %.1fs cooldown.",
                recent_count,
                self._max_thoughts_per_minute,
                self._cooldown_after_burst_s,
            )
            return True

        return False

    # ── Periodic Maintenance ──────────────────────────────────────────

    def tick(self) -> dict[str, Any]:
        """Periodic maintenance — call this every cognitive tick.

        - Decays context window salience
        - Prunes stale event history
        - Returns maintenance stats
        """
        pruned_context = self.context.decay_all()

        # Prune very old event history
        now = time.monotonic()
        stale_sources: list[str] = []
        for source, timestamps in self._event_history.items():
            self._event_history[source] = [
                t for t in timestamps if now - t < self._decay_window_s * 2
            ]
            if not self._event_history[source]:
                stale_sources.append(source)
        for source in stale_sources:
            del self._event_history[source]

        return {
            "context_entities_pruned": pruned_context,
            "event_sources_tracked": len(self._event_history),
            "context_entities_active": self.context.entity_count,
        }

    # ── Introspection ─────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Serializable snapshot for telemetry / debugging."""
        now = time.monotonic()
        cutoff = now - 60.0
        recent_thoughts = len([t for t in self._thought_timestamps if t > cutoff])

        return {
            "events_scored": self._events_scored,
            "events_interrupted": self._events_interrupted,
            "events_decayed": self._events_decayed,
            "events_budget_blocked": self._events_budget_blocked,
            "thoughts_last_minute": recent_thoughts,
            "thought_budget": self._max_thoughts_per_minute,
            "in_cooldown": time.monotonic() < self._cooldown_until,
            "event_sources_tracked": len(self._event_history),
            "context_window": self.context.snapshot(),
        }
