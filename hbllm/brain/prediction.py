"""
Prediction Engine — Order-N Markov predictors for cognitive anticipation.

Provides ``MarkovPredictor``, a fast, interpretable, online-trainable
model for predicting next states across multiple cognitive timescales.

The ``CognitivePredictors`` facade bundles domain-specific predictors:

    - **query**: Predicts next user query domain (e.g. "coding" → "testing")
    - **goal**: Predicts next goal state transition
    - **memory**: Predicts next memory retrieval pattern
    - **tool**: Predicts next tool invocation
    - **emotion**: Predicts next emotional state shift
    - **attention**: Predicts next attention focus
    - **action**: Predicts next action type

Architecture::

    User says "write a test"
        → CognitivePredictors.query.observe("testing")
        → CognitivePredictors.query.predict()
            → {"debugging": 0.6, "refactoring": 0.2, ...}
        → PredictiveLoader pre-fetches debugging-related memories

    Implements ``IPredictor`` from cognitive_interfaces.py.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.cognitive_interfaces import IPredictor

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# MarkovPredictor — Order-N transition model
# ═══════════════════════════════════════════════════════════════════════════


class MarkovPredictor(IPredictor):
    """Order-N Markov predictor with online learning.

    Maintains transition counts for sequences of length 1..N and
    predicts the next state using the longest matching context.

    Features:
        - Online training: ``observe()`` updates counts incrementally
        - Multi-order: Falls back from order-N to order-1 gracefully
        - Probability distribution: ``predict()`` returns full distribution
        - Entropy tracking: Measures prediction confidence over time
        - Decay: Optional temporal decay for non-stationary distributions

    Args:
        order: Maximum Markov order (context length).
        smoothing: Laplace smoothing parameter for unseen transitions.
        decay_rate: Exponential decay per observation (0 = no decay).
    """

    def __init__(
        self,
        order: int = 3,
        smoothing: float = 0.01,
        decay_rate: float = 0.0,
    ) -> None:
        self._order = order
        self._smoothing = smoothing
        self._decay_rate = decay_rate

        # Transition counts: context_tuple → {next_state → count}
        self._transitions: dict[tuple[str, ...], dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # History buffer (most recent observations)
        self._history: list[str] = []
        self._observation_count: int = 0
        self._vocabulary: set[str] = set()
        self._correct_predictions: int = 0
        self._total_predictions: int = 0

    # ── IPredictor interface ─────────────────────────────────────────

    async def predict_next(self, context: Any) -> dict[str, float]:
        """Predict next state distribution given context.

        Args:
            context: Current state or context string.

        Returns:
            Probability distribution over next states.
        """
        return self.predict()

    async def observe(self, observation: Any) -> None:
        """Record an observation for online learning.

        Args:
            observation: The observed state (string).
        """
        self.train(str(observation))

    # ── Core API ─────────────────────────────────────────────────────

    def train(self, state: str) -> None:
        """Observe a new state and update transition counts.

        Updates all orders from 1 to min(order, len(history)).

        Args:
            state: The observed state label.
        """
        self._vocabulary.add(state)

        # Apply decay to existing counts
        if self._decay_rate > 0 and self._observation_count > 0:
            factor = 1.0 - self._decay_rate
            for ctx in self._transitions:
                for next_state in self._transitions[ctx]:
                    self._transitions[ctx][next_state] *= factor

        # Update transition counts for each context length
        for n in range(1, min(self._order, len(self._history)) + 1):
            context = tuple(self._history[-n:])
            self._transitions[context][state] += 1.0

        # Also update order-0 (unconditional)
        self._transitions[()][state] += 1.0

        self._history.append(state)
        self._observation_count += 1

        # Trim history buffer to prevent unbounded growth
        max_history = self._order * 10
        if len(self._history) > max_history:
            self._history = self._history[-self._order :]

    def predict(self) -> dict[str, float]:
        """Predict next state using longest matching context.

        Falls back from order-N to order-0 if no context matches.

        Returns:
            Probability distribution {state → probability}.
        """
        if not self._vocabulary:
            return {}

        # Try longest context first, fall back to shorter
        for n in range(min(self._order, len(self._history)), -1, -1):
            if n == 0:
                context = ()
            else:
                context = tuple(self._history[-n:])

            counts = self._transitions.get(context)
            if counts:
                return self._normalize(dict(counts))

        # Uniform distribution as last resort
        uniform_p = 1.0 / len(self._vocabulary)
        return {s: uniform_p for s in self._vocabulary}

    def predict_top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k most likely next states.

        Args:
            k: Number of top predictions to return.

        Returns:
            List of (state, probability) tuples, sorted by probability.
        """
        dist = self.predict()
        sorted_states = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        return sorted_states[:k]

    def evaluate_prediction(self, actual: str) -> bool:
        """Check if the top prediction matches the actual outcome.

        Updates accuracy tracking statistics.

        Args:
            actual: The actual observed state.

        Returns:
            True if the top prediction was correct.
        """
        self._total_predictions += 1
        top = self.predict_top_k(1)
        if top and top[0][0] == actual:
            self._correct_predictions += 1
            return True
        return False

    def entropy(self) -> float:
        """Calculate Shannon entropy of the current prediction distribution.

        Lower entropy = more confident prediction.

        Returns:
            Entropy in bits.
        """
        dist = self.predict()
        if not dist:
            return 0.0
        h = 0.0
        for p in dist.values():
            if p > 0:
                h -= p * math.log2(p)
        return h

    @property
    def accuracy(self) -> float:
        """Prediction accuracy (correct / total)."""
        if self._total_predictions == 0:
            return 0.0
        return self._correct_predictions / self._total_predictions

    def reset(self) -> None:
        """Clear all learned transitions and history."""
        self._transitions.clear()
        self._history.clear()
        self._vocabulary.clear()
        self._observation_count = 0
        self._correct_predictions = 0
        self._total_predictions = 0

    def stats(self) -> dict[str, Any]:
        """Predictor statistics."""
        return {
            "order": self._order,
            "observations": self._observation_count,
            "vocabulary_size": len(self._vocabulary),
            "context_count": len(self._transitions),
            "accuracy": round(self.accuracy, 3),
            "entropy": round(self.entropy(), 3),
        }

    # ── Internal ─────────────────────────────────────────────────────

    def _normalize(self, counts: dict[str, float]) -> dict[str, float]:
        """Normalize counts to a probability distribution with smoothing."""
        vocab_size = max(1, len(self._vocabulary))
        total = sum(counts.values()) + self._smoothing * vocab_size
        dist: dict[str, float] = {}
        for state in self._vocabulary:
            raw = counts.get(state, 0.0) + self._smoothing
            dist[state] = raw / total
        return dist


# ═══════════════════════════════════════════════════════════════════════════
# CognitivePredictors — domain-specific predictor bundle
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CognitivePredictors:
    """Bundles domain-specific Markov predictors for cognitive anticipation.

    Each predictor tracks transitions in a different cognitive domain,
    enabling multi-timescale anticipation.

    Attributes:
        query: Next user query domain ("coding" → "testing").
        goal: Next goal state transition.
        memory: Next memory retrieval pattern.
        tool: Next tool invocation.
        emotion: Next emotional state shift.
        attention: Next attention focus area.
        action: Next action type.
    """

    query: MarkovPredictor = field(default_factory=lambda: MarkovPredictor(order=3))
    goal: MarkovPredictor = field(default_factory=lambda: MarkovPredictor(order=2))
    memory: MarkovPredictor = field(default_factory=lambda: MarkovPredictor(order=3))
    tool: MarkovPredictor = field(default_factory=lambda: MarkovPredictor(order=2))
    emotion: MarkovPredictor = field(default_factory=lambda: MarkovPredictor(order=2))
    attention: MarkovPredictor = field(default_factory=lambda: MarkovPredictor(order=2))
    action: MarkovPredictor = field(default_factory=lambda: MarkovPredictor(order=3))

    def stats(self) -> dict[str, dict[str, Any]]:
        """Statistics for all predictors."""
        return {
            "query": self.query.stats(),
            "goal": self.goal.stats(),
            "memory": self.memory.stats(),
            "tool": self.tool.stats(),
            "emotion": self.emotion.stats(),
            "attention": self.attention.stats(),
            "action": self.action.stats(),
        }
