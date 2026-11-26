"""Engine State Helpers, Frozen Working Memory, and Subgoal Snapshots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

_LEGACY_ENGINE_STATE_FIELDS: dict[str, str] = {
    "level_epistemic_probes": "context_epistemic_probes",
    "level_learned_barriers": "context_barriers",
    "level_failed_transitions": "context_failed_transitions",
    "level_lethal_positions": "context_hazards",
    "current_level": "context_epoch",
    "_current_level": "context_epoch",
}


def get_engine_state_attr(target: Any, attr: str, default: Any = None) -> Any:
    """Safely read attribute from target.state, target, or CognitiveContext."""
    if target is None:
        return default
    val = getattr(target, attr, None) if not isinstance(target, (dict, list, tuple)) else None
    if val is not None and (
        not isinstance(val, (dict, set, list, tuple, MappingProxyType)) or len(val) > 0
    ):
        return val
    state = getattr(target, "state", None) if not isinstance(target, (dict, list, tuple)) else None
    if state is not None and hasattr(state, attr):
        s_val = getattr(state, attr)
        if s_val is not None and (
            not isinstance(s_val, (dict, set, list, tuple, MappingProxyType)) or len(s_val) > 0
        ):
            return s_val
        if val is None and s_val is not None:
            return s_val
    if val is not None:
        return val
    if hasattr(target, attr):
        return getattr(target, attr)
    return default


def _apply_target_mutation(
    prefix: str,
    target: Any,
    sub_field: str,
    val: Any,
    op: str,
    before_recorder: Callable[[str, Any], None] | None,
) -> bool:
    if target is None:
        return False
    try:
        prev = getattr(target, sub_field)
    except AttributeError:
        return False
    if before_recorder is not None:
        before_recorder(f"{prefix}{sub_field}", prev)
    if op == "set":
        setattr(target, sub_field, val)
    elif op == "append" and isinstance(prev, list):
        prev.append(val)
    elif op == "add" and isinstance(prev, set):
        prev.add(val)
    return True


def apply_wm_field_delta(
    host: Any,
    working_memory: Any,
    state: Any,
    field: str,
    val: Any,
    op: str,
    before_recorder: Callable[[str, Any], None] | None = None,
) -> None:
    """Apply a declarative field mutation (`set`/`append`/`add`) to the canonical owner."""
    if field.startswith("state."):
        _apply_target_mutation("state.", state, field[6:], val, op, before_recorder)
        return
    if field.startswith("working_memory."):
        _apply_target_mutation(
            "working_memory.", working_memory, field[15:], val, op, before_recorder
        )
        return
    for prefix, target in (("", host), ("working_memory.", working_memory), ("state.", state)):
        if _apply_target_mutation(prefix, target, field, val, op, before_recorder):
            return


@dataclass(frozen=True)
class FrozenSymbolicTheory:
    """Read-only view of symbolic world theory for pure cortical stages."""

    barrier_features: frozenset[int] = frozenset()
    walkable_features: frozenset[int] = frozenset()
    goal_features: frozenset[int] = frozenset()
    candidate_goal_features: frozenset[int] = frozenset()
    cargo_features: frozenset[int] = frozenset()
    receptacle_features: frozenset[int] = frozenset()
    actuator_features: frozenset[int] = frozenset()
    transition_kb: Any | None = None
    tool_barrier_posteriors: MappingProxyType[tuple[Any, Any], tuple[float, float]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def is_barrier(self, feat: int) -> bool:
        return feat in self.barrier_features

    def is_walkable(self, feat: int) -> bool:
        return feat in self.walkable_features

    def evaluate_transition(
        self,
        feature_id: Any,
        *,
        locus: tuple[int, int] | None = None,
        active_state_tokens: frozenset[str] | set[str] | None = None,
        unknown_traversable: bool = True,
        unknown_cost_penalty: float = 0.0,
        risk_weight: float = 50.0,
    ) -> Any:
        """Evaluate edge traversability from embedded TransitionKB or frozen feature sets."""
        if self.transition_kb is not None:
            return self.transition_kb.evaluate_transition(
                feature_id,
                locus=locus,
                active_state_tokens=active_state_tokens,
                unknown_traversable=unknown_traversable,
                unknown_cost_penalty=unknown_cost_penalty,
                risk_weight=risk_weight,
            )
        from hbllm.hcir.world.causal.transition_kb import TransitionEdgeEvaluation

        is_goal = feature_id in self.goal_features
        is_blocked = (feature_id in self.barrier_features) and not is_goal
        is_walk = feature_id in self.walkable_features
        is_trav = not is_blocked and (is_goal or is_walk or unknown_traversable)
        known = (
            is_goal
            or is_blocked
            or is_walk
            or feature_id in self.cargo_features
            or feature_id in self.receptacle_features
            or feature_id in self.actuator_features
        )
        return TransitionEdgeEvaluation(
            traversable=is_trav,
            step_cost=(
                (1.0 + max(0.0, unknown_cost_penalty) if not known else 1.0)
                if is_trav
                else float("inf")
            ),
            risk_cost=0.0,
            goal_value=(1.0 if is_goal else 0.0),
            info_gain=(0.0 if known else 1.0),
            mutates_state=(feature_id in self.actuator_features),
            confidence=(1.0 if known else 0.0),
        )

    def is_tool_barrier_incompatible(self, tool_feat: Any, barrier_feat: Any) -> bool:
        pair = self.tool_barrier_posteriors.get((tool_feat, barrier_feat))
        if pair is None:
            return False
        alpha, beta = pair
        return beta > alpha

    def has_tool_barrier_affinity(self, tool_feat: Any, barrier_feat: Any) -> bool:
        pair = self.tool_barrier_posteriors.get((tool_feat, barrier_feat))
        if pair is None:
            return False
        alpha, beta = pair
        return alpha > beta


@dataclass(frozen=True)
class FrozenBodySchema:
    """Read-only view of extended body schema."""

    held_tools: tuple[Any, ...] = ()
    active_manipulandum: Any | None = None

    def is_holding(self, feat: int | None = None) -> bool:
        if feat is None:
            return bool(self.held_tools) or self.active_manipulandum is not None
        return any(getattr(t, "feature_id", None) == feat for t in self.held_tools)

    def has_manipulandum(self) -> bool:
        return self.active_manipulandum is not None or bool(self.held_tools)


@dataclass(frozen=True)
class FrozenSubgoalStack:
    """Read-only view of frontopolar subgoal stack."""

    has_pending_goals: bool = False
    current_subgoal: Any | None = None


@dataclass(frozen=True)
class FrozenWorkingMemory:
    """Read-only view of prefrontal working memory."""

    body_schema: FrozenBodySchema = field(default_factory=FrozenBodySchema)
    subgoal_stack: FrozenSubgoalStack = field(default_factory=FrozenSubgoalStack)
    focus_switch_actions: frozenset[int] = frozenset()
    controllable_avatar_features: frozenset[int] = frozenset()
    known_objects: MappingProxyType[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    object_trial_counts: MappingProxyType[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
