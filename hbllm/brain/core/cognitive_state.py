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

    from hbllm.brain.core.cognitive_state import (
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
import uuid
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
class CognitiveStateSnapshot:
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

    def diff(self, other: CognitiveStateSnapshot) -> dict[str, tuple[float, float]]:
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
        self._history: deque[CognitiveStateSnapshot] = deque(maxlen=history_size)
        self._delta_log: deque[CognitiveStateDelta] = deque(maxlen=history_size)
        self._clamp = clamp_values
        self._log_deltas = log_deltas

    def apply(
        self,
        state: CognitiveStateSnapshot,
        delta: CognitiveStateDelta,
    ) -> CognitiveStateSnapshot:
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
        state: CognitiveStateSnapshot,
        deltas: list[CognitiveStateDelta],
    ) -> CognitiveStateSnapshot:
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

    def get_history(self, last_n: int = 10) -> list[CognitiveStateSnapshot]:
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

    def rollback(self, to_version: int) -> CognitiveStateSnapshot | None:
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
        start_state: CognitiveStateSnapshot,
        deltas: list[CognitiveStateDelta],
    ) -> list[CognitiveStateSnapshot]:
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


# ═══════════════════════════════════════════════════════════════════════════
# Legacy types — backward compatibility
# ═══════════════════════════════════════════════════════════════════════════
# These classes were part of the pre-v3 CognitiveState module and are still
# used by executive_cortex.py, layered_simulation, analogical_planning, and
# related tests.  They remain here for API stability.


@dataclass(frozen=True)
class Evidence:
    """Provenance and context tracking for facts and beliefs."""

    source: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    generated_by: str = ""  # Subsystem or Node ID
    reasoning_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "generated_by": self.generated_by,
            "reasoning_path": self.reasoning_path,
        }


@dataclass(frozen=True)
class CandidatePlan:
    """A first-class candidate execution or reasoning plan."""

    plan_id: str = ""
    graph: dict[str, Any] = field(default_factory=dict)
    origin: str = "planner"  # "planner", "analogy", "fallback"
    confidence: float = 1.0
    predicted_reward: float = 0.0
    predicted_cost: dict[str, float] = field(default_factory=dict)
    analogy_used: str | None = None
    simulation_result: dict[str, Any] | None = None
    execution_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "graph": self.graph,
            "origin": self.origin,
            "confidence": self.confidence,
            "predicted_reward": self.predicted_reward,
            "predicted_cost": self.predicted_cost,
            "analogy_used": self.analogy_used,
            "simulation_result": self.simulation_result,
            "execution_trace": self.execution_trace,
        }


@dataclass(frozen=True)
class CognitiveBudget:
    """Multidimensional resource allocation parameters."""

    attention_budget: float | None = None
    memory_budget: int | None = None
    simulation_budget: int | None = None
    reasoning_budget: int | None = None
    verification_budget: int | None = None
    planning_budget: float | None = None  # seconds
    tool_budget: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attention_budget": self.attention_budget,
            "memory_budget": self.memory_budget,
            "simulation_budget": self.simulation_budget,
            "reasoning_budget": self.reasoning_budget,
            "verification_budget": self.verification_budget,
            "planning_budget": self.planning_budget,
            "tool_budget": self.tool_budget,
        }


DEFAULT_COGNITIVE_BUDGET = CognitiveBudget(
    attention_budget=1.0,
    memory_budget=10,
    simulation_budget=3,
    reasoning_budget=5,
    verification_budget=2,
    planning_budget=5.0,
    tool_budget=5,
)


@dataclass(frozen=True)
class CognitivePolicy:
    """Hierarchical policy rules configuring system behaviors."""

    reasoning_strategy: str | None = None  # "direct", "CoT", "GoT", "analogical"
    simulation_depth: int | None = None
    verification_budget: int | None = None
    retrieval_budget: int | None = None
    planner_type: str | None = None
    memory_budget: int | None = None
    model_selection: str | None = None
    reflection_enabled: bool | None = None
    budget: CognitiveBudget | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning_strategy": self.reasoning_strategy,
            "simulation_depth": self.simulation_depth,
            "verification_budget": self.verification_budget,
            "retrieval_budget": self.retrieval_budget,
            "planner_type": self.planner_type,
            "memory_budget": self.memory_budget,
            "model_selection": self.model_selection,
            "reflection_enabled": self.reflection_enabled,
            "budget": self.budget.to_dict() if self.budget else None,
        }


DEFAULT_COGNITIVE_POLICY = CognitivePolicy(
    reasoning_strategy="direct",
    simulation_depth=1,
    verification_budget=2,
    retrieval_budget=5,
    planner_type="got",
    memory_budget=10,
    model_selection="auto",
    reflection_enabled=True,
    budget=DEFAULT_COGNITIVE_BUDGET,
)


class HierarchicalCognitivePolicy:
    """A cascade of overrides from Global to Task level."""

    def __init__(
        self,
        global_policy: CognitivePolicy | None = None,
        conversation_policy: CognitivePolicy | None = None,
        goal_policy: CognitivePolicy | None = None,
        task_policy: CognitivePolicy | None = None,
    ) -> None:
        self.global_policy = global_policy or DEFAULT_COGNITIVE_POLICY
        self.conversation_policy = conversation_policy
        self.goal_policy = goal_policy
        self.task_policy = task_policy

    def resolve(self) -> CognitivePolicy:
        """Resolve effective CognitivePolicy by traversing overrides."""
        resolved_fields: dict[str, Any] = {}
        for field_name in [
            "reasoning_strategy",
            "simulation_depth",
            "verification_budget",
            "retrieval_budget",
            "planner_type",
            "memory_budget",
            "model_selection",
            "reflection_enabled",
        ]:
            # Safety override: global reflection_enabled=True cannot be disabled
            if (
                field_name == "reflection_enabled"
                and self.global_policy
                and self.global_policy.reflection_enabled
            ):
                resolved_fields[field_name] = True
                continue

            resolved_val = None
            for p in [
                self.task_policy,
                self.goal_policy,
                self.conversation_policy,
                self.global_policy,
                DEFAULT_COGNITIVE_POLICY,
            ]:
                if p is not None:
                    val = getattr(p, field_name)
                    if val is not None:
                        resolved_val = val
                        break
            resolved_fields[field_name] = resolved_val

        # Merge budgets per-field across levels
        budget_fields: dict[str, Any] = {}
        for b_field in [
            "attention_budget",
            "memory_budget",
            "simulation_budget",
            "reasoning_budget",
            "verification_budget",
            "planning_budget",
            "tool_budget",
        ]:
            resolved_val = None
            for p in [
                self.task_policy,
                self.goal_policy,
                self.conversation_policy,
                self.global_policy,
                DEFAULT_COGNITIVE_POLICY,
            ]:
                if p is not None:
                    b_obj = getattr(p, "budget", None)
                    if b_obj is not None:
                        val = getattr(b_obj, b_field, None)
                        if val is not None:
                            resolved_val = val
                            break
            budget_fields[b_field] = resolved_val

        resolved_fields["budget"] = CognitiveBudget(**budget_fields)
        return CognitivePolicy(**resolved_fields)

    @property
    def effective(self) -> CognitivePolicy:
        """Alias for resolve()."""
        return self.resolve()

    def set_task_policy(self, policy: CognitivePolicy) -> None:
        self.task_policy = policy

    def clear_task_policy(self) -> None:
        self.task_policy = None


# ═══════════════════════════════════════════════════════════════════════════
# LegacyCognitiveState — pre-v3 API (goal + policy based)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LegacyCognitiveState:
    """Pre-v3 immutable working memory snapshot with goal and policy fields.

    Used by executive_cortex, layered_simulation, analogical_planning,
    and their tests. Preserved for backward compatibility.
    """

    goal: Any = None
    policy: Any = None
    state_id: str = field(default_factory=lambda: f"state_{uuid.uuid4().hex[:12]}")
    version: int = 1
    parent_state_id: str | None = None
    retrieved_memory: list[dict[str, Any]] = field(default_factory=list)
    simulations: list[dict[str, Any]] = field(default_factory=list)
    candidate_plans: list[CandidatePlan] = field(default_factory=list)
    active_skills: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    beliefs: list[dict[str, Any]] = field(default_factory=list)
    evidence_ledger: dict[str, Evidence] = field(default_factory=dict)
    working_memory: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)

    @property
    def effective_policy(self) -> CognitivePolicy:
        if isinstance(self.policy, HierarchicalCognitivePolicy):
            return self.policy.resolve()
        return HierarchicalCognitivePolicy(global_policy=self.policy).resolve()

    def derive_state(self, **mutations: Any) -> LegacyCognitiveState:
        """Derive a new version with bumped version number."""
        mutations["version"] = self.version + 1
        mutations["parent_state_id"] = self.state_id
        mutations["state_id"] = f"state_{uuid.uuid4().hex[:12]}"
        mutations["created_at"] = time.time()
        return replace(self, **mutations)

    def fork(self) -> LegacyCognitiveState:
        """Create a forked copy with a new state_id and bumped version."""
        return replace(
            self,
            state_id=f"state_{uuid.uuid4().hex[:12]}",
            parent_state_id=self.state_id,
            version=self.version + 1,
        )

    def to_dict(self) -> dict[str, Any]:
        goal_dict = None
        if self.goal is not None:
            goal_dict = (
                self.goal.to_dict()
                if hasattr(self.goal, "to_dict")
                else {
                    "name": getattr(self.goal, "name", ""),
                    "description": getattr(self.goal, "description", ""),
                }
            )
        policy_dict = None
        if self.policy is not None:
            policy_dict = (
                self.policy.to_dict() if hasattr(self.policy, "to_dict") else str(self.policy)
            )
        return {
            "state_id": self.state_id,
            "version": self.version,
            "parent_state_id": self.parent_state_id,
            "goal": goal_dict,
            "policy": policy_dict,
            "confidence": self.confidence,
            "active_skills": self.active_skills,
            "reflections": self.reflections,
            "beliefs": self.beliefs,
            "evidence_ledger": {k: v.to_dict() for k, v in self.evidence_ledger.items()},
            "candidate_plans": [p.to_dict() for p in self.candidate_plans],
            "working_memory": self.working_memory,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CognitiveState — smart constructor for backward compatibility
# ═══════════════════════════════════════════════════════════════════════════
# When called with goal=/policy= kwargs → returns LegacyCognitiveState
# When called with v3 kwargs (confidence, etc.) → returns CognitiveStateSnapshot


class _CognitiveStateMeta(type):
    """Metaclass that dispatches CognitiveState() to the right class."""

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if "goal" in kwargs or "policy" in kwargs:
            return LegacyCognitiveState(*args, **kwargs)
        return CognitiveStateSnapshot(*args, **kwargs)

    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, (CognitiveStateSnapshot, LegacyCognitiveState))


class CognitiveState(metaclass=_CognitiveStateMeta):
    """Factory for cognitive state objects.

    - ``CognitiveState(goal=..., policy=...)`` → ``LegacyCognitiveState``
    - ``CognitiveState(confidence=0.8, ...)``  → ``CognitiveStateSnapshot``
    - ``CognitiveState()``                     → ``CognitiveStateSnapshot``
    """

    pass
