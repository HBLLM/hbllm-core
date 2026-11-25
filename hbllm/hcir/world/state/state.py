"""Authoritative EngineState and Transactional StateDelta."""

from __future__ import annotations

from collections import deque
from types import MappingProxyType
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.percept import EpistemicPhase
from hbllm.hcir.world.state.context import CognitiveContext
from hbllm.hcir.world.state.models import (
    _LEGACY_ENGINE_STATE_FIELDS,
)


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
    last_action: Any | None = None
    last_action_data: dict[str, Any] | None = None
    avatar_pos: tuple[int, int] | None = None
    avatar_feature: Any | None = None
    avatar_features: set[Any] = Field(default_factory=set)
    avatar_size: int = 1
    avatar_w: int | None = None
    avatar_h: int | None = None
    prior_avatar_feature: Any | None = None
    avatar_controllability_evidence: dict[str, Any] = Field(
        default_factory=dict, alias="_avatar_controllability_evidence"
    )
    bg_feature: Any = 0
    recent_positions: list[tuple[int, int]] = Field(default_factory=list)
    recent_actions: list[Any] = Field(default_factory=list)
    step_counter: int = 0
    step_count: int = 0
    consecutive_stuck_steps: int = 0
    consecutive_quiescent_actions: int = 0
    consecutive_pacing_steps: int = 0
    pending_phase_wait_steps: int = 0
    consecutive_causal_probes: int = 0
    consecutive_effective_actuations: int = 0
    consecutive_goal_converging_actuations: int = 0
    consecutive_plan_failures: int = 0
    consecutive_simulation_failures: int = 0
    exploration_cooldown: int = 0
    simulation_cooldown: int = 0
    active_probe_id: str | None = None
    active_probe_target: tuple[int, int] | None = None
    probe_target_steps: int = 0
    total_epistemic_probes: int = 0
    context_epistemic_probes: int = 0
    last_predicted_pos: tuple[int, int] | None = None
    last_non_displacement_action: int | None = None
    last_effective_actuation_coord: tuple[int, int] | None = None
    last_surprise: float = 0.0
    last_surprise_eval: Any | None = None
    latest_lifecycle_result: Any | None = None
    feedback_assimilated: bool = False
    feature_quiescent_counts: dict[int, int] = Field(
        default_factory=dict, alias="_feature_quiescent_counts"
    )
    current_actor_pos: tuple[int, int] | None = None

    # 2. Learned World Theory State
    learned_barriers: set[tuple[int, int]] = Field(default_factory=set)
    context_barriers: dict[int, set[tuple[int, int]]] = Field(default_factory=dict)
    learned_goal_positions: set[tuple[int, int]] = Field(default_factory=set)
    learned_goal_features: set[int] = Field(default_factory=set)
    learned_receptacle_positions: set[tuple[int, int]] = Field(default_factory=set)
    learned_cargo_features: set[int] = Field(default_factory=set)
    action_dynamics: dict[int, Any] = Field(default_factory=dict)
    tested_actions: set[int] = Field(default_factory=set)
    tested_action_positions: set[tuple[int, tuple[int, int] | None]] = Field(default_factory=set)
    state_mutations: list[Any] = Field(default_factory=list)
    failed_transitions: set[tuple[tuple[int, int], int]] | list[Any] = Field(default_factory=set)
    context_failed_transitions: dict[int, set[Any]] = Field(default_factory=dict)
    verified_safe_features: set[int] = Field(default_factory=set)
    mobile_threat_features: set[int] = Field(default_factory=set)
    context_hazards: dict[int, set[tuple[int, int]]] = Field(default_factory=dict)
    prev_oriented_threats: list[Any] = Field(default_factory=list)
    quiescent_features: set[int] = Field(default_factory=set)
    effective_features: set[int] = Field(default_factory=set)
    action_affordances: dict[int, Any] = Field(default_factory=dict)
    actuation_affordances: dict[tuple[int, int], Any] = Field(default_factory=dict)
    quiescent_actuation_targets: set[tuple[int, int]] = Field(default_factory=set)
    effective_actuation_targets: set[tuple[int, int]] = Field(default_factory=set)
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
    context_epoch: int = Field(default=0, alias="_context_epoch")
    current_context_id: str = "default"
    world_id: str = "default"
    bilateral_state: Any | None = None
    entropy_monitor: Any | None = None
    goals: list[Any] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _remap_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            remapped = dict(data)
            for legacy_name, canonical_name in _LEGACY_ENGINE_STATE_FIELDS.items():
                if legacy_name in remapped and canonical_name not in remapped:
                    remapped[canonical_name] = remapped.pop(legacy_name)
            return remapped
        return data

    def __getattr__(self, name: str) -> Any:
        if name in _LEGACY_ENGINE_STATE_FIELDS:
            return getattr(self, _LEGACY_ENGINE_STATE_FIELDS[name])
        if name.startswith("_") and hasattr(self, name[1:]):
            return getattr(self, name[1:])
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _LEGACY_ENGINE_STATE_FIELDS:
            super().__setattr__(_LEGACY_ENGINE_STATE_FIELDS[name], value)
        elif name.startswith("_") and name[1:] in type(self).model_fields:
            super().__setattr__(name[1:], value)
        else:
            super().__setattr__(name, value)

    @property
    def level_epistemic_probes(self) -> int:
        return self.context_epistemic_probes

    @level_epistemic_probes.setter
    def level_epistemic_probes(self, value: int) -> None:
        self.context_epistemic_probes = value

    @property
    def level_learned_barriers(self) -> dict[int, set[tuple[int, int]]]:
        return self.context_barriers

    @level_learned_barriers.setter
    def level_learned_barriers(self, value: dict[int, set[tuple[int, int]]]) -> None:
        self.context_barriers = value

    @property
    def level_failed_transitions(self) -> dict[int, set[Any]]:
        return self.context_failed_transitions

    @level_failed_transitions.setter
    def level_failed_transitions(self, value: dict[int, set[Any]]) -> None:
        self.context_failed_transitions = value

    @property
    def level_lethal_positions(self) -> dict[int, set[tuple[int, int]]]:
        return self.context_hazards

    @level_lethal_positions.setter
    def level_lethal_positions(self, value: dict[int, set[tuple[int, int]]]) -> None:
        self.context_hazards = value

    @property
    def current_level(self) -> int:
        return self.context_epoch

    @current_level.setter
    def current_level(self, value: int) -> None:
        self.context_epoch = value

    @property
    def current_context_hazards(self) -> set[tuple[int, int]]:
        return self.context_hazards.get(self.context_epoch, set())

    @property
    def current_level_lethal_positions(self) -> set[tuple[int, int]]:
        return self.current_context_hazards

    @classmethod
    def from_engine(cls, source: Any) -> EngineState:
        """Construct an EngineState snapshot from an AutonomousEpistemicEngine instance."""
        attrs: dict[str, Any] = {}
        if hasattr(source, "state") and isinstance(source.state, EngineState):
            for f_name in cls.model_fields.keys():
                val = getattr(source.state, f_name, None)
                if val is not None:
                    attrs[f_name] = val
        for f_name, f_info in cls.model_fields.items():
            alias = f_info.alias or f_name
            for candidate in (alias, f"_{f_name}", f_name):
                val = getattr(source, candidate, None)
                if val is not None and (
                    not isinstance(val, (dict, set, list, tuple, MappingProxyType))
                    or len(val) > 0
                    or f_name not in attrs
                ):
                    attrs[f_name] = val
                    break
        for f_name, val in list(attrs.items()):
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
            consecutive_effective_actuations=self.consecutive_effective_actuations,
            consecutive_goal_converging_actuations=self.consecutive_goal_converging_actuations,
            consecutive_plan_failures=self.consecutive_plan_failures,
            consecutive_simulation_failures=self.consecutive_simulation_failures,
            exploration_cooldown=self.exploration_cooldown,
            simulation_cooldown=self.simulation_cooldown,
            active_probe_id=self.active_probe_id,
            active_probe_target=self.active_probe_target,
            probe_target_steps=self.probe_target_steps,
            total_epistemic_probes=self.total_epistemic_probes,
            context_epistemic_probes=self.context_epistemic_probes,
            last_predicted_pos=self.last_predicted_pos,
            last_non_displacement_action=self.last_non_displacement_action,
            last_effective_actuation_coord=self.last_effective_actuation_coord,
            last_surprise=self.last_surprise,
            last_surprise_eval=self.last_surprise_eval,
            latest_lifecycle_result=self.latest_lifecycle_result,
            feedback_assimilated=self.feedback_assimilated,
            current_actor_pos=self.current_actor_pos,
            learned_barriers=frozenset(self.learned_barriers),
            context_barriers=MappingProxyType(
                {k: frozenset(v) for k, v in self.context_barriers.items()}
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
            context_failed_transitions=MappingProxyType(
                {k: tuple(v) for k, v in self.context_failed_transitions.items()}
            ),
            verified_safe_features=frozenset(self.verified_safe_features),
            mobile_threat_features=frozenset(self.mobile_threat_features),
            context_hazards=MappingProxyType(
                {k: frozenset(v) for k, v in self.context_hazards.items()}
            ),
            prev_oriented_threats=tuple(self.prev_oriented_threats),
            quiescent_features=frozenset(self.quiescent_features),
            effective_features=frozenset(self.effective_features),
            action_affordances=MappingProxyType(dict(self.action_affordances)),
            actuation_affordances=MappingProxyType(dict(self.actuation_affordances)),
            quiescent_actuation_targets=frozenset(self.quiescent_actuation_targets),
            effective_actuation_targets=frozenset(self.effective_actuation_targets),
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
            context_epoch=self.context_epoch,
            current_context_id=self.current_context_id,
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


_MISSING: Any = object()


class StateDelta:
    """Write-intent overlay over the engine's state (WP1).

    Faculties propose mutations via ``set``/``mutate``; the engine commits them
    atomically through ``apply_delta`` — the only assignment path.
    """

    def __init__(self) -> None:
        self._pending: dict[str, Any] = {}
        self._cow: dict[str, Any] = {}

    def set(self, field: str, value: Any) -> None:
        self._pending[field] = value

    def get(self, field: str, default: Any = _MISSING) -> Any:
        """Read-through: pending value if set, else the caller supplies state."""
        if field in self._pending:
            return self._pending[field]
        if default is not _MISSING:
            return default
        raise KeyError(field)

    def mutate(self, field: str, current: Any) -> Any:
        """Mutable view of a container field (WP1).

        Returns the live container (marks the field dirty in the delta so
        apply_delta emits it) — copy-on-write would produce a stale snapshot
        that clobbers concurrent in-place edits through owned subsystems.
        """
        if field not in self._cow:
            self._cow[field] = self._pending.get(field, current)
            self._pending[field] = self._cow[field]
        return self._cow[field]

    def items(self) -> dict[str, Any]:
        return dict(self._pending)

    def is_empty(self) -> bool:
        return not self._pending
