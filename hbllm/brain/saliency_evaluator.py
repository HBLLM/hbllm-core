"""
Saliency Evaluator — SNN-based event scoring for the cognitive loop.

Implements ``IAttentionSelector``.  Scores each incoming event's
``snn_saliency`` based on:

    1. **Event type weights**: Some event types are inherently more
       salient (e.g., USER_SPOKE > IDLE_DETECTED).
    2. **Cognitive state**: Current curiosity, stress, arousal amplify
       or dampen specific event types.
    3. **Neuromodulation**: NE (alertness) and ACh (focus) modulate
       overall saliency sensitivity.
    4. **Recency decay**: Events lose saliency as they age.

The evaluator uses lightweight multiplication rather than running
full SNN forward passes per event — this keeps the per-event cost O(1).

Usage::

    from hbllm.brain.saliency_evaluator import SaliencyEvaluator

    evaluator = SaliencyEvaluator(cognitive_state, neuromod_engine)
    scored = await evaluator.evaluate(events)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.brain.cognitive_event import CognitiveEvent, CognitiveEventType
from hbllm.brain.cognitive_interfaces import IAttentionSelector
from hbllm.brain.cognitive_state import CognitiveStateSnapshot
from hbllm.brain.neuromodulation import NeuromodulationEngine

logger = logging.getLogger(__name__)


# ── Base saliency weights per event type ─────────────────────────────────

_BASE_SALIENCY: dict[CognitiveEventType, float] = {
    CognitiveEventType.USER_SPOKE: 0.95,
    CognitiveEventType.MEMORY_CONFLICT: 0.80,
    CognitiveEventType.PREDICTION_FAILED: 0.75,
    CognitiveEventType.GOAL_FAILED: 0.75,
    CognitiveEventType.ATTENTION_SPIKE: 0.70,
    CognitiveEventType.EMOTION_CHANGED: 0.65,
    CognitiveEventType.REWARD_RECEIVED: 0.65,
    CognitiveEventType.GOAL_ADDED: 0.60,
    CognitiveEventType.GOAL_COMPLETED: 0.60,
    CognitiveEventType.MEMORY_UPDATED: 0.50,
    CognitiveEventType.TASK_COMPLETED: 0.50,
    CognitiveEventType.SIMULATION_COMPLETE: 0.45,
    CognitiveEventType.IDLE_DETECTED: 0.20,
}


class SaliencyEvaluator(IAttentionSelector):
    """SNN-inspired saliency scorer for cognitive events.

    Combines base event-type weights with cognitive state modulation
    and neuromodulator levels to produce a per-event saliency score.

    Args:
        cognitive_state: Current immutable cognitive state snapshot.
        neuromod: The global neuromodulation engine.
        recency_half_life: Seconds after which event saliency halves
            due to age.  Default 30.0.
    """

    def __init__(
        self,
        cognitive_state: CognitiveStateSnapshot | None = None,
        neuromod: NeuromodulationEngine | None = None,
        recency_half_life: float = 30.0,
    ) -> None:
        self._state = cognitive_state or CognitiveStateSnapshot()
        self._neuromod = neuromod
        self._recency_half_life = max(1.0, recency_half_life)
        self._eval_count = 0

    def update_cognitive_state(self, state: CognitiveStateSnapshot) -> None:
        """Update the cognitive state used for modulation.

        Called by the executive controller when state changes.
        """
        self._state = state

    async def evaluate(self, events: list[Any]) -> list[Any]:
        """Score a batch of events for saliency.

        For each event:
            1. Look up base saliency from event type
            2. Modulate by cognitive state factors
            3. Modulate by neuromodulator levels
            4. Apply recency decay
            5. Blend with original priority hint

        Args:
            events: List of ``CognitiveEvent`` instances.

        Returns:
            Events with ``snn_saliency`` updated, sorted by
            effective priority descending.
        """
        now = time.time()
        scored: list[CognitiveEvent] = []

        for event in events:
            if not isinstance(event, CognitiveEvent):
                continue

            saliency = self._compute_saliency(event, now)
            scored.append(event.with_saliency(saliency))

        # Sort by effective priority (descending)
        scored.sort(key=lambda e: e.effective_priority, reverse=True)

        self._eval_count += len(scored)
        return scored

    def _compute_saliency(self, event: CognitiveEvent, now: float) -> float:
        """Compute saliency score for a single event."""

        # 1. Base saliency from event type
        base = _BASE_SALIENCY.get(event.type, 0.5)

        # 2. Cognitive state modulation
        state_factor = self._cognitive_state_factor(event.type)

        # 3. Neuromodulation
        neuro_factor = self._neuromodulation_factor()

        # 4. Recency decay
        age = max(0.0, now - event.timestamp)
        recency_factor = 2.0 ** (-age / self._recency_half_life)

        # 5. Combine: base × state × neuro × recency
        raw = base * state_factor * neuro_factor * recency_factor

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, raw))

    def _cognitive_state_factor(self, event_type: CognitiveEventType) -> float:
        """Compute state-based modulation factor for an event type.

        Returns a factor in [0.5, 1.5] based on how cognitive state
        amplifies or dampens this event type.
        """
        s = self._state
        factor = 1.0

        if event_type == CognitiveEventType.USER_SPOKE:
            # Always high — slightly boosted by arousal
            factor += s.arousal * 0.2

        elif event_type in (
            CognitiveEventType.MEMORY_CONFLICT,
            CognitiveEventType.PREDICTION_FAILED,
        ):
            # Surprise events: boosted by curiosity, dampened by fatigue
            factor += s.curiosity * 0.3 - s.fatigue * 0.2

        elif event_type in (
            CognitiveEventType.GOAL_ADDED,
            CognitiveEventType.GOAL_COMPLETED,
            CognitiveEventType.GOAL_FAILED,
        ):
            # Goal events: boosted by motivation
            factor += s.motivation * 0.3

        elif event_type in (
            CognitiveEventType.EMOTION_CHANGED,
            CognitiveEventType.REWARD_RECEIVED,
        ):
            # Emotional events: boosted by arousal
            factor += s.arousal * 0.3

        elif event_type == CognitiveEventType.IDLE_DETECTED:
            # Idle: boosted by fatigue (need rest), dampened by motivation
            factor += s.fatigue * 0.3 - s.motivation * 0.2

        return max(0.5, min(1.5, factor))

    def _neuromodulation_factor(self) -> float:
        """Compute neuromodulator-based global saliency factor.

        Combines NE (alertness) and ACh (attention selectivity).
        Returns a factor in [0.5, 1.5].
        """
        if self._neuromod is None:
            return 1.0

        ne_factor = self._neuromod.get_attention_factor()  # [0.7, 1.5]
        # Normalize to center around 1.0
        return max(0.5, min(1.5, ne_factor))

    def stats(self) -> dict[str, Any]:
        """Evaluator statistics."""
        return {
            "total_evaluated": self._eval_count,
            "recency_half_life": self._recency_half_life,
        }
