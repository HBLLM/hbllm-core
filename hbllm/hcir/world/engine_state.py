from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.percept import EpistemicPhase


def get_engine_state_attr(target: Any, attr: str, default: Any = None) -> Any:
    """Safely read attribute from target.state, target, or CognitiveContext."""
    if target is None:
        return default
    state = getattr(target, "state", None) if not isinstance(target, (dict, list, tuple)) else None
    if state is not None and hasattr(state, attr):
        return getattr(state, attr)
    if hasattr(target, attr):
        return getattr(target, attr)
    return default


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


@dataclass(frozen=True)
class CognitiveContext:
    """Immutable view of cognitive engine state at a single discrete inference step.

    Faculties receive this frozen context. They cannot mutate engine state directly.
    Signals produced by prior stages in the cortical pipeline are merged via `with_signals`.
    """

    # Episode-Transient Fields
    curr_grid: np.ndarray | None = None
    prev_grid: np.ndarray | None = None
    grid_history: tuple[np.ndarray, ...] = ()
    last_action: int | None = None
    last_action_data: MappingProxyType[str, Any] | None = None
    avatar_pos: tuple[int, int] | None = None
    avatar_feature: int | None = None
    avatar_features: frozenset[int] = frozenset()
    avatar_size: int = 1
    avatar_w: int | None = None
    avatar_h: int | None = None
    prior_avatar_feature: int | None = None
    avatar_controllability_evidence: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    bg_feature: int = 0
    recent_positions: tuple[tuple[int, int], ...] = ()
    recent_actions: tuple[int, ...] = ()
    step_counter: int = 0
    step_count: int = 0
    consecutive_stuck_steps: int = 0
    consecutive_quiescent_actions: int = 0
    consecutive_pacing_steps: int = 0
    pending_phase_wait_steps: int = 0
    consecutive_causal_probes: int = 0
    consecutive_effective_clicks: int = 0
    consecutive_goal_converging_clicks: int = 0
    consecutive_plan_failures: int = 0
    consecutive_simulation_failures: int = 0
    exploration_cooldown: int = 0
    simulation_cooldown: int = 0
    active_probe_id: str | None = None
    active_probe_target: tuple[int, int] | None = None
    probe_target_steps: int = 0
    total_epistemic_probes: int = 0
    level_epistemic_probes: int = 0
    last_predicted_pos: tuple[int, int] | None = None
    last_non_displacement_action: int | None = None
    last_effective_click_coord: tuple[int, int] | None = None
    last_surprise: float = 0.0
    last_surprise_eval: Any | None = None
    latest_lifecycle_result: Any | None = None
    feedback_assimilated: bool = False
    current_actor_pos: tuple[int, int] | None = None

    # Learned World Theory Fields
    learned_barriers: frozenset[tuple[int, int]] = frozenset()
    level_learned_barriers: MappingProxyType[int, frozenset[tuple[int, int]]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    learned_goal_positions: frozenset[tuple[int, int]] = frozenset()
    learned_goal_features: frozenset[int] = frozenset()
    learned_receptacle_positions: frozenset[tuple[int, int]] = frozenset()
    learned_cargo_features: frozenset[int] = frozenset()
    action_dynamics: MappingProxyType[int, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    tested_actions: frozenset[int] = frozenset()
    tested_action_positions: frozenset[tuple[int, tuple[int, int] | None]] = frozenset()
    state_mutations: tuple[Any, ...] = ()
    failed_transitions: tuple[Any, ...] = ()
    level_failed_transitions: MappingProxyType[int, tuple[Any, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    verified_safe_features: frozenset[int] = frozenset()
    mobile_threat_features: frozenset[int] = frozenset()
    level_lethal_positions: MappingProxyType[int, frozenset[tuple[int, int]]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    prev_oriented_threats: tuple[Any, ...] = ()
    quiescent_features: frozenset[int] = frozenset()
    effective_features: frozenset[int] = frozenset()
    action_affordances: MappingProxyType[int, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    click_affordances: MappingProxyType[tuple[int, int], Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    quiescent_click_targets: frozenset[tuple[int, int]] = frozenset()
    effective_click_targets: frozenset[tuple[int, int]] = frozenset()
    spatial_lattice_stride: int = 1
    effector_kinetics_mode: int | None = None
    active_causal_actuator_coord: tuple[int, int] | None = None
    active_goal_converging_coord: tuple[int, int] | None = None
    causal_actuator_streak_limit: int = 4
    entity_action_failed_positions: MappingProxyType[str, frozenset[tuple[int, int]]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    entity_visit_counts: MappingProxyType[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    position_visit_counts: MappingProxyType[tuple[int, int], int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    probed_entity_ids: frozenset[str] = frozenset()
    topology_rooms: MappingProxyType[int, tuple[tuple[int, int], ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    topology_doors: tuple[Any, ...] = ()
    room_adjacency: MappingProxyType[int, tuple[int, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    entity_lineage_graph: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    undo_actions: frozenset[int] = frozenset()
    value_mutator_actions: frozenset[int] = frozenset()
    focus_switch_actions: frozenset[int] = frozenset()
    index_selector_actions: frozenset[int] = frozenset()
    immobile_entity_actions: frozenset[int] = frozenset()

    # Executive & Deliberation Fields
    phase: EpistemicPhase = EpistemicPhase.MOTOR_GROUNDING
    phase_history: tuple[tuple[Any, Any, str], ...] = ()
    mental_plan: tuple[Any, ...] = ()
    current_simulated_goal: tuple[int, int] | None = None
    exhausted_candidate_goals: frozenset[tuple[int, int]] = frozenset()
    active_hypothesis: Any | None = None
    inhibited_actions: MappingProxyType[Any, int] | frozenset[int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    current_room_id: int | None = None
    current_level: int = 0
    current_game_id: str = "default"
    world_id: str = "default"
    bilateral_state: Any | None = None
    entropy_monitor: Any | None = None
    goals: tuple[Any, ...] = ()

    # Memory Stores & Hierarchical Faculties
    symbolic_theory: FrozenSymbolicTheory = field(default_factory=FrozenSymbolicTheory)
    working_memory: FrozenWorkingMemory = field(default_factory=FrozenWorkingMemory)
    subgoal_stack: FrozenSubgoalStack = field(default_factory=FrozenSubgoalStack)

    # Inter-stage signals dictionary
    signals: MappingProxyType[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def is_displacement_action(self, action: Any) -> bool:
        """Check if action causes spatial displacement based on action_dynamics or affordances."""
        if action in self.action_dynamics:
            dyn = self.action_dynamics[action]
            dr = getattr(dyn, "delta_r", 0)
            dc = getattr(dyn, "delta_c", 0)
            if dr != 0 or dc != 0:
                return True
        if action in self.action_affordances:
            aff = self.action_affordances[action]
            if getattr(aff, "is_displacement", False):
                return True
        return False

    def is_spatial_effector(self, action: Any) -> bool:
        """Check if action is a targeted spatial effector / coordinate actuator."""
        if action in self.action_affordances:
            aff = self.action_affordances[action]
            if getattr(aff, "requires_spatial_target", False):
                return True
        return False

    def is_focus_switch_action(self, action: Any) -> bool:
        """Check if action triggers focus switch between entities."""
        return (
            action in self.focus_switch_actions
            or action in self.working_memory.focus_switch_actions
        )

    def has_displacement_capabilities(self, available_actions: Sequence[Any] | None = None) -> bool:
        """Returns True if the environment/agent possesses physical displacement actions."""
        actions = (
            available_actions
            if available_actions is not None
            else list(self.action_dynamics.keys())
        )
        return any(self.is_displacement_action(a) for a in actions)

    def estimate_background(self, curr_grid: np.ndarray) -> int:
        from hbllm.hcir.world.cortex_perception import PerceptionEngine

        return int(PerceptionEngine.estimate_background(curr_grid))

    def extract_entities(self, curr_grid: np.ndarray, bg: int | None = None) -> list[Any]:
        from hbllm.hcir.world.cortex_perception import PerceptionEngine

        bg_val = bg if bg is not None else self.bg_feature
        return PerceptionEngine.extract_entities(
            curr_grid,
            bg_val,
            avatar_features=set(self.avatar_features),
            state_mutations=list(self.state_mutations),
            learned_barriers=set(self.learned_barriers),
            learned_goal_positions=set(self.learned_goal_positions),
            learned_receptacle_positions=set(self.learned_receptacle_positions),
            avatar_pos=self.avatar_pos,
        )

    def with_signals(
        self, new_signals: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> CognitiveContext:
        """Return a copy of this context enriched with signals from downstream cortical stages."""
        merged: dict[str, Any] = dict(self.signals)
        if new_signals:
            merged.update(new_signals)
        if kwargs:
            merged.update(kwargs)
        updates: dict[str, Any] = {"signals": MappingProxyType(merged)}
        for k, v in kwargs.items():
            if hasattr(self, k):
                updates[k] = v
        from dataclasses import replace

        return replace(self, **updates)

    def __getattr__(self, name: str) -> Any:
        """Fallback for duck-typed reads, leading underscores, or signals."""
        if name in self.signals:
            return self.signals[name]
        if name.startswith("_") and hasattr(self, name[1:]):
            return getattr(self, name[1:])
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CognitiveContext):
            return False
        if self.step_counter != other.step_counter or self.avatar_pos != other.avatar_pos:
            return False
        if self.curr_grid is not None:
            if other.curr_grid is None or not np.array_equal(self.curr_grid, other.curr_grid):
                return False
        elif other.curr_grid is not None:
            return False

        if self.prev_grid is not None:
            if other.prev_grid is None or not np.array_equal(self.prev_grid, other.prev_grid):
                return False
        elif other.prev_grid is not None:
            return False
        for f in self.__dataclass_fields__:
            if f in ("curr_grid", "prev_grid", "grid_history"):
                continue
            if getattr(self, f) != getattr(other, f):
                return False
        return True


class EngineState(BaseModel):
    """Authoritative Cognitive Engine State representation.

    Owned by HCIR. Faculties never mutate this directly; all modifications
    flow through TransactionOperations committed by the TransactionManager.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        populate_by_name=True,
    )

    # 1. Episode-Transient State
    curr_grid: np.ndarray | None = None
    prev_grid: np.ndarray | None = None
    grid_history: deque[np.ndarray] = Field(default_factory=lambda: deque(maxlen=6))
    last_action: int | None = None
    last_action_data: dict[str, Any] | None = None
    avatar_pos: tuple[int, int] | None = None
    avatar_feature: int | None = None
    avatar_features: set[int] = Field(default_factory=set)
    avatar_size: int = 1
    avatar_w: int | None = None
    avatar_h: int | None = None
    prior_avatar_feature: int | None = None
    avatar_controllability_evidence: dict[str, Any] = Field(
        default_factory=dict, alias="_avatar_controllability_evidence"
    )
    bg_feature: int = 0
    recent_positions: list[tuple[int, int]] = Field(default_factory=list)
    recent_actions: list[int] = Field(default_factory=list)
    step_counter: int = 0
    step_count: int = 0
    consecutive_stuck_steps: int = 0
    consecutive_quiescent_actions: int = 0
    consecutive_pacing_steps: int = 0
    pending_phase_wait_steps: int = 0
    consecutive_causal_probes: int = 0
    consecutive_effective_clicks: int = 0
    consecutive_goal_converging_clicks: int = 0
    consecutive_plan_failures: int = 0
    consecutive_simulation_failures: int = 0
    exploration_cooldown: int = 0
    simulation_cooldown: int = 0
    active_probe_id: str | None = None
    active_probe_target: tuple[int, int] | None = None
    probe_target_steps: int = 0
    total_epistemic_probes: int = 0
    level_epistemic_probes: int = 0
    last_predicted_pos: tuple[int, int] | None = None
    last_non_displacement_action: int | None = None
    last_effective_click_coord: tuple[int, int] | None = None
    last_surprise: float = 0.0
    last_surprise_eval: Any | None = None
    latest_lifecycle_result: Any | None = None
    feedback_assimilated: bool = Field(default=False, alias="_feedback_assimilated")
    feature_quiescent_counts: dict[int, int] = Field(
        default_factory=dict, alias="_feature_quiescent_counts"
    )
    current_actor_pos: tuple[int, int] | None = None

    # 2. Learned World Theory State
    learned_barriers: set[tuple[int, int]] = Field(default_factory=set)
    level_learned_barriers: dict[int, set[tuple[int, int]]] = Field(default_factory=dict)
    learned_goal_positions: set[tuple[int, int]] = Field(default_factory=set)
    learned_goal_features: set[int] = Field(default_factory=set)
    learned_receptacle_positions: set[tuple[int, int]] = Field(default_factory=set)
    learned_cargo_features: set[int] = Field(default_factory=set)
    action_dynamics: dict[int, Any] = Field(default_factory=dict)
    tested_actions: set[int] = Field(default_factory=set)
    tested_action_positions: set[tuple[int, tuple[int, int] | None]] = Field(default_factory=set)
    state_mutations: list[Any] = Field(default_factory=list)
    failed_transitions: set[tuple[tuple[int, int], int]] | list[Any] = Field(default_factory=set)
    level_failed_transitions: dict[int, set[Any]] = Field(default_factory=dict)
    verified_safe_features: set[int] = Field(default_factory=set)
    mobile_threat_features: set[int] = Field(default_factory=set)
    level_lethal_positions: dict[int, set[tuple[int, int]]] = Field(default_factory=dict)
    prev_oriented_threats: list[Any] = Field(default_factory=list, alias="_prev_oriented_threats")
    quiescent_features: set[int] = Field(default_factory=set)
    effective_features: set[int] = Field(default_factory=set)
    action_affordances: dict[int, Any] = Field(default_factory=dict)
    click_affordances: dict[tuple[int, int], Any] = Field(default_factory=dict)
    quiescent_click_targets: set[tuple[int, int]] = Field(default_factory=set)
    effective_click_targets: set[tuple[int, int]] = Field(default_factory=set)
    spatial_lattice_stride: int = 1
    effector_kinetics_mode: int | None = None
    active_causal_actuator_coord: tuple[int, int] | None = None
    active_goal_converging_coord: tuple[int, int] | None = None
    causal_actuator_streak_limit: int = 4
    entity_action_failed_positions: dict[str, set[tuple[int, int]]] = Field(default_factory=dict)
    entity_visit_counts: dict[str, int] = Field(default_factory=dict)
    position_visit_counts: dict[tuple[int, int], int] = Field(default_factory=dict)
    probed_entity_ids: set[str] = Field(default_factory=set)
    topology_rooms: dict[int, list[tuple[int, int]]] = Field(default_factory=dict)
    topology_doors: list[Any] = Field(default_factory=list)
    room_adjacency: dict[int, list[int]] = Field(default_factory=dict)
    entity_lineage_graph: dict[str, Any] = Field(default_factory=dict)
    undo_actions: set[int] = Field(default_factory=set)
    value_mutator_actions: set[int] = Field(default_factory=set)
    focus_switch_actions: set[int] = Field(default_factory=set)
    index_selector_actions: set[int] = Field(default_factory=set)
    immobile_entity_actions: set[int] = Field(default_factory=set)

    # 3. Executive & Deliberation State
    phase: EpistemicPhase = Field(default=EpistemicPhase.MOTOR_GROUNDING, alias="_phase")
    phase_history: list[tuple[Any, Any, str]] = Field(default_factory=list)
    mental_plan: list[Any] | deque[Any] = Field(default_factory=list)
    current_simulated_goal: tuple[int, int] | None = None
    exhausted_candidate_goals: set[tuple[int, int]] = Field(default_factory=set)
    active_hypothesis: Any | None = None
    inhibited_actions: dict[Any, int] | set[int] = Field(default_factory=dict)
    current_room_id: int | None = None
    current_level: int = Field(default=0, alias="_current_level")
    current_game_id: str = "default"
    world_id: str = "default"
    bilateral_state: Any | None = None
    entropy_monitor: Any | None = None
    goals: list[Any] = Field(default_factory=list)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") and hasattr(self, name[1:]):
            return getattr(self, name[1:])
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") and name[1:] in type(self).model_fields:
            super().__setattr__(name[1:], value)
        else:
            super().__setattr__(name, value)

    @classmethod
    def from_engine(cls, engine: Any) -> EngineState:
        """Construct an EngineState snapshot from an AutonomousEpistemicEngine instance."""
        if hasattr(engine, "state") and isinstance(engine.state, EngineState):
            return engine.state.model_copy(deep=True)
        attrs: dict[str, Any] = {}
        for f_name, f_info in cls.model_fields.items():
            val = None
            alias = f_info.alias or f_name
            for candidate in (alias, f"_{f_name}", f_name):
                val = get_engine_state_attr(engine, candidate)
                if val is not None:
                    break
            if val is not None:
                # Copy mutable structures so EngineState is decoupled
                if isinstance(val, (set, frozenset)):
                    attrs[f_name] = set(val)
                elif isinstance(val, (list, tuple)):
                    attrs[f_name] = list(val)
                elif isinstance(val, dict):
                    attrs[f_name] = dict(val)
                elif isinstance(val, np.ndarray):
                    attrs[f_name] = val.copy()
                elif isinstance(val, deque):
                    attrs[f_name] = deque(val, maxlen=val.maxlen)
                else:
                    attrs[f_name] = val
        return cls(**attrs)

    def freeze(self) -> CognitiveContext:
        """Create an immutable CognitiveContext snapshot from the current mutable state."""
        return CognitiveContext(
            curr_grid=self.curr_grid.copy() if self.curr_grid is not None else None,
            prev_grid=self.prev_grid.copy() if self.prev_grid is not None else None,
            grid_history=tuple(g.copy() for g in self.grid_history),
            last_action=self.last_action,
            last_action_data=MappingProxyType(dict(self.last_action_data))
            if self.last_action_data
            else None,
            avatar_pos=self.avatar_pos,
            avatar_feature=self.avatar_feature,
            avatar_features=frozenset(self.avatar_features),
            avatar_size=self.avatar_size,
            avatar_w=self.avatar_w,
            avatar_h=self.avatar_h,
            prior_avatar_feature=self.prior_avatar_feature,
            avatar_controllability_evidence=MappingProxyType(
                dict(self.avatar_controllability_evidence)
            ),
            bg_feature=self.bg_feature,
            recent_positions=tuple(self.recent_positions),
            recent_actions=tuple(self.recent_actions),
            step_counter=self.step_counter,
            step_count=self.step_count,
            consecutive_stuck_steps=self.consecutive_stuck_steps,
            consecutive_quiescent_actions=self.consecutive_quiescent_actions,
            consecutive_pacing_steps=self.consecutive_pacing_steps,
            pending_phase_wait_steps=self.pending_phase_wait_steps,
            consecutive_causal_probes=self.consecutive_causal_probes,
            consecutive_effective_clicks=self.consecutive_effective_clicks,
            consecutive_goal_converging_clicks=self.consecutive_goal_converging_clicks,
            consecutive_plan_failures=self.consecutive_plan_failures,
            consecutive_simulation_failures=self.consecutive_simulation_failures,
            exploration_cooldown=self.exploration_cooldown,
            simulation_cooldown=self.simulation_cooldown,
            active_probe_id=self.active_probe_id,
            active_probe_target=self.active_probe_target,
            probe_target_steps=self.probe_target_steps,
            total_epistemic_probes=self.total_epistemic_probes,
            level_epistemic_probes=self.level_epistemic_probes,
            last_predicted_pos=self.last_predicted_pos,
            last_non_displacement_action=self.last_non_displacement_action,
            last_effective_click_coord=self.last_effective_click_coord,
            last_surprise=self.last_surprise,
            last_surprise_eval=self.last_surprise_eval,
            latest_lifecycle_result=self.latest_lifecycle_result,
            feedback_assimilated=self.feedback_assimilated,
            current_actor_pos=self.current_actor_pos,
            learned_barriers=frozenset(self.learned_barriers),
            level_learned_barriers=MappingProxyType(
                {k: frozenset(v) for k, v in self.level_learned_barriers.items()}
            ),
            learned_goal_positions=frozenset(self.learned_goal_positions),
            learned_goal_features=frozenset(self.learned_goal_features),
            learned_receptacle_positions=frozenset(self.learned_receptacle_positions),
            learned_cargo_features=frozenset(self.learned_cargo_features),
            action_dynamics=MappingProxyType(dict(self.action_dynamics)),
            tested_actions=frozenset(self.tested_actions),
            tested_action_positions=frozenset(self.tested_action_positions),
            state_mutations=tuple(self.state_mutations),
            failed_transitions=tuple(self.failed_transitions),
            level_failed_transitions=MappingProxyType(
                {k: tuple(v) for k, v in self.level_failed_transitions.items()}
            ),
            verified_safe_features=frozenset(self.verified_safe_features),
            mobile_threat_features=frozenset(self.mobile_threat_features),
            level_lethal_positions=MappingProxyType(
                {k: frozenset(v) for k, v in self.level_lethal_positions.items()}
            ),
            prev_oriented_threats=tuple(self.prev_oriented_threats),
            quiescent_features=frozenset(self.quiescent_features),
            effective_features=frozenset(self.effective_features),
            action_affordances=MappingProxyType(dict(self.action_affordances)),
            click_affordances=MappingProxyType(dict(self.click_affordances)),
            quiescent_click_targets=frozenset(self.quiescent_click_targets),
            effective_click_targets=frozenset(self.effective_click_targets),
            spatial_lattice_stride=self.spatial_lattice_stride,
            effector_kinetics_mode=self.effector_kinetics_mode,
            active_causal_actuator_coord=self.active_causal_actuator_coord,
            active_goal_converging_coord=self.active_goal_converging_coord,
            causal_actuator_streak_limit=self.causal_actuator_streak_limit,
            entity_action_failed_positions=MappingProxyType(
                {k: frozenset(v) for k, v in self.entity_action_failed_positions.items()}
            ),
            entity_visit_counts=MappingProxyType(dict(self.entity_visit_counts)),
            position_visit_counts=MappingProxyType(dict(self.position_visit_counts)),
            probed_entity_ids=frozenset(self.probed_entity_ids),
            topology_rooms=MappingProxyType({k: tuple(v) for k, v in self.topology_rooms.items()}),
            topology_doors=tuple(self.topology_doors),
            room_adjacency=MappingProxyType({k: tuple(v) for k, v in self.room_adjacency.items()}),
            entity_lineage_graph=MappingProxyType(dict(self.entity_lineage_graph)),
            undo_actions=frozenset(self.undo_actions),
            value_mutator_actions=frozenset(self.value_mutator_actions),
            focus_switch_actions=frozenset(self.focus_switch_actions),
            index_selector_actions=frozenset(self.index_selector_actions),
            immobile_entity_actions=frozenset(self.immobile_entity_actions),
            phase=self.phase,
            phase_history=tuple(self.phase_history),
            mental_plan=tuple(self.mental_plan),
            current_simulated_goal=self.current_simulated_goal,
            exhausted_candidate_goals=frozenset(self.exhausted_candidate_goals),
            active_hypothesis=self.active_hypothesis,
            inhibited_actions=(
                MappingProxyType(dict(self.inhibited_actions))
                if isinstance(self.inhibited_actions, dict)
                else frozenset(self.inhibited_actions)
            ),
            current_room_id=self.current_room_id,
            current_level=self.current_level,
            current_game_id=self.current_game_id,
            world_id=self.world_id,
            bilateral_state=self.bilateral_state,
            entropy_monitor=self.entropy_monitor,
            goals=tuple(self.goals),
        )

    def to_transaction_operations(self, tenant_id: str = "default") -> list[TransactionOperation]:
        """Convert engine state summary to atomic HCIR TransactionOperations."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_data={
                    "id": f"engine_state:{self.world_id}",
                    "type": "ObservationNode",
                    "tenant_id": tenant_id,
                    "properties": {
                        "phase": str(self.phase),
                        "step_counter": self.step_counter,
                        "avatar_pos": self.avatar_pos,
                        "avatar_feature": self.avatar_feature,
                        "learned_barriers_count": len(self.learned_barriers),
                        "learned_goals_count": len(self.learned_goal_positions),
                        "plan_length": len(self.mental_plan),
                    },
                },
            )
        ]
