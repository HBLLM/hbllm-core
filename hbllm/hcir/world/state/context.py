"""CognitiveContext — Immutable Epistemic View of World State."""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from hbllm.hcir.world.motor_calibration import (
    resolve_calibrated_transition_deltas,
    resolve_effector_coupling_offsets,
)
from hbllm.hcir.world.percept import EpistemicPhase
from hbllm.hcir.world.state.models import (
    _LEGACY_ENGINE_STATE_FIELDS,
    FrozenSubgoalStack,
    FrozenSymbolicTheory,
    FrozenWorkingMemory,
)

_CTX_ENTITY_CACHE: OrderedDict[tuple[Any, ...], list[Any]] = OrderedDict()
_CTX_ENTITY_CACHE_MAX: int = 64


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
    last_action: Any | None = None
    last_action_data: MappingProxyType[str, Any] | None = None
    avatar_pos: tuple[int, int] | None = None
    avatar_feature: Any | None = None
    avatar_features: frozenset[Any] = frozenset()
    avatar_size: int = 1
    avatar_w: int | None = None
    avatar_h: int | None = None
    prior_avatar_feature: Any | None = None
    avatar_controllability_evidence: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    bg_feature: Any = 0
    recent_positions: tuple[tuple[int, int], ...] = ()
    recent_actions: tuple[Any, ...] = ()
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
    current_actor_pos: tuple[int, int] | None = None

    # Learned World Theory Fields
    learned_barriers: frozenset[tuple[int, int]] = frozenset()
    context_barriers: MappingProxyType[int, frozenset[tuple[int, int]]] = field(
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
    context_failed_transitions: MappingProxyType[int, tuple[Any, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    verified_safe_features: frozenset[int] = frozenset()
    mobile_threat_features: frozenset[int] = frozenset()
    context_hazards: MappingProxyType[int, frozenset[tuple[int, int]]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    prev_oriented_threats: tuple[Any, ...] = ()
    quiescent_features: frozenset[int] = frozenset()
    effective_features: frozenset[int] = frozenset()
    action_affordances: MappingProxyType[int, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    actuation_affordances: MappingProxyType[tuple[int, int], Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    quiescent_actuation_targets: frozenset[tuple[int, int]] = frozenset()
    effective_actuation_targets: frozenset[tuple[int, int]] = frozenset()
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
    context_epoch: int = 0
    current_context_id: str = "default"
    world_id: str = "default"
    bilateral_state: Any | None = None
    entropy_monitor: Any | None = None
    goals: tuple[Any, ...] = ()

    @property
    def level_epistemic_probes(self) -> int:
        return self.context_epistemic_probes

    @property
    def level_learned_barriers(self) -> MappingProxyType[int, frozenset[tuple[int, int]]]:
        return self.context_barriers

    @property
    def level_failed_transitions(self) -> MappingProxyType[int, tuple[Any, ...]]:
        return self.context_failed_transitions

    @property
    def level_lethal_positions(self) -> MappingProxyType[int, frozenset[tuple[int, int]]]:
        return self.context_hazards

    @property
    def current_level(self) -> int:
        return self.context_epoch

    @property
    def current_context_hazards(self) -> frozenset[tuple[int, int]]:
        return self.context_hazards.get(self.context_epoch, frozenset())

    @property
    def current_level_lethal_positions(self) -> frozenset[tuple[int, int]]:
        return self.current_context_hazards

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
        if available_actions is not None:
            return any(self.is_displacement_action(a) for a in available_actions)
        actions = list(
            dict.fromkeys([*self.action_affordances.keys(), *self.action_dynamics.keys()])
        )
        return any(self.is_displacement_action(a) for a in actions)

    def _context_action_deltas(self) -> dict[Any, tuple[int, int]]:
        deltas: dict[Any, tuple[int, int]] = {}
        for act, aff in self.action_affordances.items():
            if aff.is_displacement and aff.delta and len(aff.delta) >= 2:
                deltas[act] = (int(round(float(aff.delta[0]))), int(round(float(aff.delta[1]))))
        for act, dyn in self.action_dynamics.items():
            dr = int(dyn.delta_r)
            dc = int(dyn.delta_c)
            if dr != 0 or dc != 0:
                deltas[act] = (dr, dc)
        return deltas

    def get_calibrated_transition_deltas(
        self, available_actions: Sequence[Any] | None = None
    ) -> tuple[tuple[int, int], ...]:
        """Return deduplicated non-zero displacement vectors from calibrated motor dynamics."""
        deltas = self._context_action_deltas()
        return resolve_calibrated_transition_deltas(
            deltas,
            available_actions=available_actions,
            has_displacement=self.has_displacement_capabilities(available_actions),
            has_declared_affordances=bool(self.action_affordances),
            inhibited_actions=self.inhibited_actions,
            action_dynamics=self.action_dynamics,
        )

    def get_effector_coupling_offsets(
        self, available_actions: Sequence[Any] | None = None
    ) -> tuple[tuple[int, int], ...]:
        """Return predecessor coupling offsets {target - delta_a} for active effectors."""
        acts = (
            list(available_actions)
            if available_actions is not None
            else (
                list(self.action_affordances.keys())
                if self.action_affordances
                else list(self.action_dynamics.keys())
            )
        )
        deltas = self._context_action_deltas()
        has_disp = self.has_displacement_capabilities(acts if acts else None)
        has_spatial = any(self.is_spatial_effector(a) for a in acts)
        has_disc = any(
            not self.is_displacement_action(a) and not self.is_spatial_effector(a) for a in acts
        )
        return resolve_effector_coupling_offsets(
            deltas,
            available_actions=available_actions,
            has_displacement=has_disp,
            has_spatial_effector=has_spatial,
            has_discrete_local_interact=has_disc,
            has_declared_affordances=bool(self.action_affordances),
            inhibited_actions=self.inhibited_actions,
            action_dynamics=self.action_dynamics,
        )

    def estimate_background(self, curr_grid: np.ndarray) -> int:
        from hbllm.hcir.world.cortex_perception import PerceptionEngine

        return int(PerceptionEngine.estimate_background(curr_grid))

    def extract_entities(self, curr_grid: np.ndarray, bg: int | None = None) -> list[Any]:
        from hbllm.hcir.world.cortex_perception import PerceptionEngine

        bg_val = bg if bg is not None else self.bg_feature
        # Fast cache keyed on immutable grid bytes and classification inputs (avoids redundant scipy.ndimage.label)
        mut_sig = tuple(
            (getattr(m, "feature", None), getattr(m, "pos", None))
            for m in self.state_mutations[-4:]
        )
        cache_key = (
            curr_grid.shape,
            curr_grid.dtype.str,
            curr_grid.tobytes(),
            int(bg_val) if bg_val is not None else None,
            self.avatar_pos,
            frozenset(self.avatar_features),
            frozenset(self.learned_barriers),
            frozenset(self.learned_goal_positions),
            frozenset(self.learned_receptacle_positions),
            len(self.state_mutations),
            mut_sig,
        )
        cached = _CTX_ENTITY_CACHE.get(cache_key)
        if cached is not None:
            _CTX_ENTITY_CACHE.move_to_end(cache_key)
            return [copy.copy(e) for e in cached]

        entities = PerceptionEngine.extract_entities(
            curr_grid,
            bg_val,
            avatar_features=set(self.avatar_features),
            state_mutations=list(self.state_mutations),
            learned_barriers=set(self.learned_barriers),
            learned_goal_positions=set(self.learned_goal_positions),
            learned_receptacle_positions=set(self.learned_receptacle_positions),
            avatar_pos=self.avatar_pos,
        )
        _CTX_ENTITY_CACHE[cache_key] = [copy.copy(e) for e in entities]
        if len(_CTX_ENTITY_CACHE) > _CTX_ENTITY_CACHE_MAX:
            _CTX_ENTITY_CACHE.popitem(last=False)
        return entities

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
            canonical = _LEGACY_ENGINE_STATE_FIELDS.get(k, k)
            if canonical in self.__dataclass_fields__:
                updates[canonical] = v
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
