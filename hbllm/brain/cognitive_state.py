"""
Immutable CognitiveState — event-sourced cognitive variable management.

The system's continuous cognitive variables are represented as frozen
(immutable) snapshots.  No node mutates state directly.  Instead:

    1. Nodes produce ``CognitiveStateDelta`` objects describing changes.
    2. The ``CognitiveStateReducer`` applies deltas to produce new snapshots.
    3. History is retained for replay, debugging, and time-travel.

Architecture::

    CognitiveState v120
            ↓
    Reasoner produces CognitiveStateDelta(confidence=0.8)
            ↓
    CognitiveStateReducer.apply()
            ↓
    CognitiveState v121

Advantages:
    - **Deterministic**: Same deltas → same state
    - **Replayable**: Store all deltas → reconstruct any past state
    - **Debuggable**: Inspect which node changed what and why
    - **Time-travel**: Roll back to any version
    - **Distributed**: Nodes produce deltas independently

Population coding (in ``population.py``) provides the SNN-native encoding
for CognitiveState fields.

Writers (which nodes update which fields):
    confidence      ← CriticNode, ReasoningNetwork
    uncertainty     ← CriticNode, DecisionNode
    relevance       ← AttentionManager, RouterNode
    novelty         ← CuriosityNode, ComprehensionEnsemble
    motivation      ← GoalMemory, RewardEvaluator
    valence         ← EmotionEngine
    arousal         ← EmotionEngine, PerceptionFuser
    intention       ← PlannerNode, ExecutiveController
    fatigue         ← SleepNode, AdaptationTracking
    curiosity       ← CuriosityNode
    stress          ← UserModelEngine, NeuromodulationEngine
    focus_target    ← AttentionManager

Usage::

    from hbllm.brain.cognitive_state import (
        CognitiveState, CognitiveStateDelta, CognitiveStateReducer,
    )

    reducer = CognitiveStateReducer()
    state = CognitiveState()  # v0, all defaults

    delta = CognitiveStateDelta(
        source_node="critic_01",
        changes={"confidence": 0.8, "uncertainty": 0.2},
        reason="High-confidence causal chain found",
    )
    state = reducer.apply(state, delta)  # → v1
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CognitiveState — immutable snapshot
# ═══════════════════════════════════════════════════════════════════════════


# Define the valid field names and their ranges for validation
_COGNITIVE_FIELDS: dict[str, tuple[float, float]] = {
    "confidence": (0.0, 1.0),
    "uncertainty": (0.0, 1.0),
    "relevance": (0.0, 1.0),
    "novelty": (0.0, 1.0),
    "motivation": (0.0, 1.0),
    "valence": (-1.0, 1.0),
    "arousal": (0.0, 1.0),
    "intention_strength": (0.0, 1.0),
    "fatigue": (0.0, 1.0),
    "curiosity": (0.0, 1.0),
    "stress": (0.0, 1.0),
}


@dataclass(frozen=True)
class CognitiveState:
    """Immutable snapshot of the system's cognitive variables.

    Never mutated directly.  New states are produced by applying
    ``CognitiveStateDelta`` through ``CognitiveStateReducer``.

    All numeric fields are continuous variables in their defined ranges.
    ``focus_target`` is the only string field (current attention focus).
    """

    # ── Continuous cognitive variables ──
    confidence: float = 0.5
    uncertainty: float = 0.5
    relevance: float = 0.5
    novelty: float = 0.5
    motivation: float = 0.5
    valence: float = 0.0  # [-1, 1] emotional polarity
    arousal: float = 0.5
    intention_strength: float = 0.5
    fatigue: float = 0.0
    curiosity: float = 0.5
    stress: float = 0.0

    # ── Categorical state ──
    focus_target: str = ""

    # ── Metadata ──
    version: int = 0
    timestamp: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        """Return a plain dict of all cognitive variables (no metadata).

        Useful for logging, population encoding, and comparison.
        """
        return {
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "relevance": self.relevance,
            "novelty": self.novelty,
            "motivation": self.motivation,
            "valence": self.valence,
            "arousal": self.arousal,
            "intention_strength": self.intention_strength,
            "fatigue": self.fatigue,
            "curiosity": self.curiosity,
            "stress": self.stress,
            "focus_target": self.focus_target,
        }

    def numeric_vector(self) -> list[float]:
        """Return numeric fields as a flat vector (for population encoding).

        Order: confidence, uncertainty, relevance, novelty, motivation,
               valence, arousal, intention_strength, fatigue, curiosity, stress
        """
        return [
            self.confidence,
            self.uncertainty,
            self.relevance,
            self.novelty,
            self.motivation,
            self.valence,
            self.arousal,
            self.intention_strength,
            self.fatigue,
            self.curiosity,
            self.stress,
        ]

    def diff(self, other: CognitiveState) -> dict[str, tuple[float, float]]:
        """Compare two states, returning fields that differ.

        Returns:
            Dict of ``{field_name: (self_value, other_value)}`` for
            fields where the values differ.
        """
        changes: dict[str, tuple[float, float]] = {}
        for field_name in _COGNITIVE_FIELDS:
            self_val = getattr(self, field_name)
            other_val = getattr(other, field_name)
            if self_val != other_val:
                changes[field_name] = (self_val, other_val)
        if self.focus_target != other.focus_target:
            changes["focus_target"] = (self.focus_target, other.focus_target)  # type: ignore
        return changes


# ═══════════════════════════════════════════════════════════════════════════
# CognitiveStateDelta — a proposed change
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CognitiveStateDelta:
    """A proposed change to CognitiveState.

    Produced by cognitive nodes.  Consumed by the reducer.
    All changes in a single delta are applied atomically.

    Attributes:
        source_node: ID of the node producing this delta.
        changes: Dict of ``{field_name: new_value}``.  Only fields
            listed here are modified; others remain unchanged.
        reason: Human-readable explanation of why this change was made.
            Used for debugging and explainability.
        timestamp: When this delta was produced.
    """

    source_node: str
    changes: dict[str, Any]
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def validate(self) -> list[str]:
        """Check that all fields and values are valid.

        Returns:
            List of error messages (empty = valid).
        """
        errors: list[str] = []
        for field_name, value in self.changes.items():
            if field_name == "focus_target":
                if not isinstance(value, str):
                    errors.append(f"focus_target must be str, got {type(value).__name__}")
                continue

            if field_name not in _COGNITIVE_FIELDS:
                errors.append(f"Unknown cognitive field: '{field_name}'")
                continue

            if not isinstance(value, (int, float)):
                errors.append(f"Field '{field_name}' must be numeric, got {type(value).__name__}")
                continue

            lo, hi = _COGNITIVE_FIELDS[field_name]
            if not (lo <= value <= hi):
                errors.append(f"Field '{field_name}' value {value} out of range [{lo}, {hi}]")

        return errors


# ═══════════════════════════════════════════════════════════════════════════
# CognitiveStateReducer — applies deltas to produce new states
# ═══════════════════════════════════════════════════════════════════════════


class CognitiveStateReducer:
    """Applies ``CognitiveStateDelta`` to produce new immutable states.

    Maintains a history ring buffer for replay and time-travel.

    Args:
        history_size: Maximum number of past states to retain.
        clamp_values: If True, clamp values to their defined ranges
            (prevents out-of-range from floating-point arithmetic).
        log_deltas: If True, log each applied delta at DEBUG level.
    """

    def __init__(
        self,
        history_size: int = 1000,
        clamp_values: bool = True,
        log_deltas: bool = False,
    ) -> None:
        self._history: deque[CognitiveState] = deque(maxlen=history_size)
        self._delta_log: deque[CognitiveStateDelta] = deque(maxlen=history_size)
        self._clamp = clamp_values
        self._log_deltas = log_deltas

    def apply(
        self,
        state: CognitiveState,
        delta: CognitiveStateDelta,
    ) -> CognitiveState:
        """Apply a delta to produce a new immutable state.

        Args:
            state: The current cognitive state.
            delta: The change to apply.

        Returns:
            A new ``CognitiveState`` with version incremented and
            specified fields updated.

        Raises:
            ValueError: If the delta contains invalid fields or values.
        """
        # Validate delta
        errors = delta.validate()
        if errors:
            raise ValueError(
                f"Invalid CognitiveStateDelta from '{delta.source_node}': " + "; ".join(errors)
            )

        # Build update dict
        updates: dict[str, Any] = {}
        for field_name, value in delta.changes.items():
            if field_name == "focus_target":
                updates["focus_target"] = value
            else:
                # Optionally clamp to valid range
                if self._clamp and field_name in _COGNITIVE_FIELDS:
                    lo, hi = _COGNITIVE_FIELDS[field_name]
                    value = max(lo, min(hi, float(value)))
                updates[field_name] = float(value)

        # Increment version and set timestamp
        updates["version"] = state.version + 1
        updates["timestamp"] = delta.timestamp

        # Create new immutable state using dataclasses.replace
        new_state = replace(state, **updates)

        # Store in history
        self._history.append(state)
        self._delta_log.append(delta)

        if self._log_deltas:
            logger.debug(
                "CognitiveState v%d -> v%d by %s: %s (%s)",
                state.version,
                new_state.version,
                delta.source_node,
                delta.changes,
                delta.reason,
            )

        return new_state

    def apply_batch(
        self,
        state: CognitiveState,
        deltas: list[CognitiveStateDelta],
    ) -> CognitiveState:
        """Apply multiple deltas sequentially.

        Args:
            state: Starting state.
            deltas: Deltas to apply in order.

        Returns:
            Final state after all deltas applied.
        """
        for delta in deltas:
            state = self.apply(state, delta)
        return state

    def get_history(self, last_n: int = 10) -> list[CognitiveState]:
        """Return the last N states from history.

        Args:
            last_n: Number of past states to return.

        Returns:
            List of states, most recent last.
        """
        history_list = list(self._history)
        return history_list[-last_n:]

    def get_delta_log(self, last_n: int = 10) -> list[CognitiveStateDelta]:
        """Return the last N deltas from the log.

        Args:
            last_n: Number of past deltas to return.

        Returns:
            List of deltas, most recent last.
        """
        log_list = list(self._delta_log)
        return log_list[-last_n:]

    def rollback(self, to_version: int) -> CognitiveState | None:
        """Roll back to a specific version from history.

        Args:
            to_version: The version number to roll back to.

        Returns:
            The state at that version, or None if not in history.
        """
        for state in self._history:
            if state.version == to_version:
                return state
        return None

    def replay_from(
        self,
        start_state: CognitiveState,
        deltas: list[CognitiveStateDelta],
    ) -> list[CognitiveState]:
        """Replay a sequence of deltas from a starting state.

        Useful for debugging: reproduce the exact state sequence.

        Args:
            start_state: The initial state.
            deltas: Sequence of deltas to replay.

        Returns:
            List of all intermediate states (including start).
        """
        states = [start_state]
        current = start_state
        for delta in deltas:
            current = self.apply(current, delta)
            states.append(current)
        return states

    @property
    def history_size(self) -> int:
        """Number of states currently in history."""
        return len(self._history)

    @property
    def current_version(self) -> int:
        """Version of the most recent state in history."""
        if self._history:
            return self._history[-1].version
        return -1
