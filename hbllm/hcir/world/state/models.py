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
    candidates: tuple[tuple[str, Any], ...] = (
        ("", host),
        ("working_memory.", working_memory),
        ("state.", state),
    )
    for prefix, target in candidates:
        if target is None or not hasattr(target, field):
            continue
        if before_recorder is not None:
            before_recorder(f"{prefix}{field}", getattr(target, field, None))
        if op == "set":
            setattr(target, field, val)
        elif op == "append":
            cur = getattr(target, field)
            if isinstance(cur, list):
                cur.append(val)
        elif op == "add":
            cur = getattr(target, field)
            if isinstance(cur, set):
                cur.add(val)
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

    def is_barrier(self, feat: int) -> bool:
        return feat in self.barrier_features

    def is_walkable(self, feat: int) -> bool:
        return feat in self.walkable_features

    def is_tool_barrier_incompatible(self, tool_feat: int, barrier_feat: int) -> bool:
        return False

    def has_tool_barrier_affinity(self, tool_feat: int, barrier_feat: int) -> bool:
        return False


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
