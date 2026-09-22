"""CognitiveBlackbox — Self-Contained Cognitive Engine for MIMO Driver Architecture.

The blackbox is the ONLY place where HCIR internals exist. Drivers never see them.
All planning, memory, learning, and reasoning state lives here.

Drivers provide raw observations (grids, pixels, sensor data) via DriverInput.
The blackbox:
  1. Interprets raw data using registered perception lifters
  2. Builds internal entity graphs and workspace state
  3. Plans actions through learned models (not hardcoded rules)
  4. Returns DriverActions to the driver

Architecture:
    DriverManager embeds CognitiveBlackbox
    ├── HCIRWorkspaceState (shared cognitive graph)
    ├── HCIRSpatialEntityPlanner (spatial path & subgoal sequence reasoning)
    ├── HierarchicalGoalDecomposer (recursive obstacle decomposition)
    ├── Registered PerceptionLifters (domain-specific, per source type)
    └── AgentState (learned models, carrying, phases, discovered features)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hbllm.drivers.base import DriverAction, DriverFeedback, DriverInput
from hbllm.hcir.graph import WorldVariableNode
from hbllm.hcir.spatial_planner import (
    EntityGraph,
    HCIRSpatialEntityPlanner,
    SequencePlanStep,
    SpatialEntity,
)
from hbllm.hcir.subgoal_decomposer import HierarchicalGoalDecomposer
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


# ── Perception Lifter Protocol ────────────────────────────────────────────
# A lifter converts raw perception data (from driver.get_perception_data())
# into internal HCIR types (SpatialEntity, barriers). This is the boundary
# where domain-specific interpretation happens. Lifters are registered
# per source type and called ONLY inside the blackbox — never by drivers.

PerceptionLifterFn = Callable[
    [dict[str, Any], "AgentState"],
    tuple[list[SpatialEntity], set[tuple[int, int]]],
]


class AgentPhase(StrEnum):
    """Lifecycle phases of the cognitive agent."""

    EPISTEMIC_LEARNING = "epistemic_learning"  # Exploring to learn world rules
    EXPLOITATION = "exploitation"  # Executing learned plan
    SOFT_RESTART_PENDING = "soft_restart_pending"  # Waiting to restart with accumulated knowledge
    COMPLETED = "completed"


@dataclass
class CarryingState:
    """State of the agent's carried payload."""

    holding: bool = False
    offset: tuple[float, float] = (0.0, 0.0)
    entity_id: str | None = None


class AgentState:
    """All mutable cognitive agent state, owned by the blackbox (not plugins)."""

    def __init__(
        self,
        phase: AgentPhase = AgentPhase.EPISTEMIC_LEARNING,
        carrying: CarryingState | None = None,
        delivered_positions: set[tuple[int, int]] | None = None,
        learned_obstacle_features: set[Any] | None = None,
        learned_target_features: set[Any] | None = None,
        learned_traversable_features: set[Any] | None = None,
        current_plan: list[SequencePlanStep] | None = None,
        step_count: int = 0,
        total_reward: float = 0.0,
        action_models: dict[int, Any] | None = None,
        # Backward-compatibility kwargs
        learned_barrier_colors: set[int] | None = None,
        learned_item_colors: set[int] | None = None,
        learned_walkable_colors: set[int] | None = None,
    ) -> None:
        self.phase = phase
        self.carrying = carrying if carrying is not None else CarryingState()
        self.delivered_positions = set(delivered_positions or [])
        self.learned_obstacle_features = set(learned_obstacle_features or [])
        if learned_barrier_colors:
            self.learned_obstacle_features.update(learned_barrier_colors)
        self.learned_target_features = set(learned_target_features or [])
        if learned_item_colors:
            self.learned_target_features.update(learned_item_colors)
        self.learned_traversable_features = set(learned_traversable_features or [])
        if learned_walkable_colors:
            self.learned_traversable_features.update(learned_walkable_colors)
        self.current_plan = list(current_plan or [])
        self.step_count = step_count
        self.total_reward = total_reward
        self.action_models = dict(action_models or {})

    # Backward-compatible property aliases
    @property
    def learned_barrier_colors(self) -> set[Any]:
        return self.learned_obstacle_features

    @learned_barrier_colors.setter
    def learned_barrier_colors(self, val: set[Any]) -> None:
        self.learned_obstacle_features = set(val)

    @property
    def learned_item_colors(self) -> set[Any]:
        return self.learned_target_features

    @learned_item_colors.setter
    def learned_item_colors(self, val: set[Any]) -> None:
        self.learned_target_features = set(val)

    @property
    def learned_walkable_colors(self) -> set[Any]:
        return self.learned_traversable_features

    @learned_walkable_colors.setter
    def learned_walkable_colors(self, val: set[Any]) -> None:
        self.learned_traversable_features = set(val)


class CognitiveBlackbox:
    """Self-contained cognitive engine. All state lives here, not in plugins.

    Drivers provide raw observations. The blackbox does ALL interpretation,
    learning, and planning. No HCIR types are ever exposed to drivers.

    Usage:
        blackbox = CognitiveBlackbox()
        blackbox.register_lifter("arc_agi", my_arc_lifter_fn)
        blackbox.observe(driver_input, perception_data={...})
        action = blackbox.decide(available_actions)
        blackbox.update(action, feedback)
    """

    def __init__(self, workspace: HCIRWorkspaceState | None = None) -> None:
        self.workspace = workspace or HCIRWorkspaceState()
        self.spatial_planner = HCIRSpatialEntityPlanner()
        self.goal_decomposer = HierarchicalGoalDecomposer()
        self.state = AgentState()

        # Per-source state for MIMO multi-driver support
        self._source_states: dict[str, AgentState] = {}
        self._source_entity_graphs: dict[str, EntityGraph] = {}

        # Registered perception lifters: source_type -> lifter_fn
        # These convert raw perception data into internal HCIR types
        self._lifters: dict[str, PerceptionLifterFn] = {}

    # ── Lifter Registration ───────────────────────────────────────────────

    def register_lifter(
        self,
        source_type: str,
        lifter: PerceptionLifterFn,
    ) -> None:
        """Register a domain-specific perception lifter.

        Lifters convert raw perception data (dicts from driver.get_perception_data())
        into internal SpatialEntity lists and barrier sets. This is the ONLY way
        domain-specific interpretation enters the core.

        Args:
            source_type: Driver source identifier (e.g., "arc_agi", "sokoban")
            lifter: Callable(perception_data, agent_state) -> (entities, barriers)
        """
        self._lifters[source_type] = lifter
        logger.info("Registered perception lifter for source_type='%s'", source_type)

    # ── Source Management (MIMO) ──────────────────────────────────────────

    def get_state(self, source_id: str = "default") -> AgentState:
        """Get or create per-source agent state."""
        if source_id not in self._source_states:
            self._source_states[source_id] = AgentState()
        return self._source_states[source_id]

    # ── Workspace Graph Synchronization ───────────────────────────────────

    def sync_state_to_workspace(self, source_id: str = "default") -> None:
        """Push current agent state into workspace graph variables.

        This enables the spatial planner and subgoal decomposer to read
        agent state from the workspace graph without parameter threading.
        """
        state = self.get_state(source_id)

        # Carrying state
        self._upsert_var(
            "var_carrying_state",
            {
                "holding": state.carrying.holding,
                "offset": list(state.carrying.offset),
                "entity_id": state.carrying.entity_id,
            },
        )

        # Delivered positions
        self._upsert_var("var_delivered_positions", [list(p) for p in state.delivered_positions])

        # Learned target features
        if state.learned_target_features:
            self._upsert_var("var_target_features", list(state.learned_target_features))
            self._upsert_var("var_target_entity_features", list(state.learned_target_features))
            self._upsert_var("var_learned_item_colors", list(state.learned_target_features))

        # Learned obstacle features
        if state.learned_obstacle_features:
            self._upsert_var("var_obstacle_features", list(state.learned_obstacle_features))
            self._upsert_var("var_learned_barrier_colors", list(state.learned_obstacle_features))

        # Learned traversable features
        if state.learned_traversable_features:
            self._upsert_var("var_traversable_features", list(state.learned_traversable_features))
            self._upsert_var(
                "var_learned_walkable_colors", list(state.learned_traversable_features)
            )

        # Action capabilities
        self._upsert_var(
            "var_action_capabilities",
            {
                "pickup_drop": 5 in state.action_models,
            },
        )

    def _upsert_var(self, name: str, value: Any) -> None:
        """Upsert a WorldVariableNode into the workspace graph."""
        existing = self.workspace.graph.get_node(name)
        if isinstance(existing, WorldVariableNode):
            existing.value = value
            self.workspace.upsert_node(existing)
        else:
            node = WorldVariableNode(
                id=name,
                variable_name=name,
                value=value,
            )
            self.workspace.upsert_node(node)

    # ── Observation (Input) ───────────────────────────────────────────────

    def observe(
        self,
        driver_input: DriverInput,
        perception_data: dict[str, Any] | None = None,
    ) -> EntityGraph | None:
        """Ingest observation and build entity graph from raw perception data.

        The blackbox uses registered lifters to interpret the raw perception
        data into internal HCIR types. Drivers never see or construct these.
        """
        source_id = driver_input.source_id or "default"
        state = self.get_state(source_id)
        state.step_count += 1

        if perception_data is None:
            return None

        # Use registered lifter to convert raw data → internal entities
        lifter = self._lifters.get(source_id)
        if lifter is None:
            # Try modality-based fallback
            lifter = self._lifters.get(driver_input.modality.value)

        if lifter is not None:
            entities, barriers = lifter(perception_data, state)
            grid_shape = driver_input.metadata.get("grid_shape", (64, 64))
            step_size = driver_input.metadata.get("step_size", 1)

            eg = self.spatial_planner.construct_entity_graph(
                entities=entities,
                barriers=barriers,
                grid_shape=grid_shape,
                step_size=step_size,
            )
            self._source_entity_graphs[source_id] = eg
            return eg

        return None

    # ── Decision (Output) ─────────────────────────────────────────────────

    def decide(
        self,
        available_actions: list[DriverAction],
        source_id: str = "default",
        entity_graph: EntityGraph | None = None,
    ) -> DriverAction:
        """Plan next action from internal state.

        Uses the entity graph from the most recent observe() call,
        or an explicitly provided one.
        """
        state = self.get_state(source_id)
        eg = entity_graph or self._source_entity_graphs.get(source_id)

        if eg is None:
            # No entity graph available — return first available action
            return available_actions[0] if available_actions else DriverAction(action_id=0)

        # Sync state to workspace so planner can read it
        self.sync_state_to_workspace(source_id)

        # Plan sequence
        if not state.current_plan:
            state.current_plan = self.spatial_planner.plan_sequence(
                eg=eg,
                workspace=self.workspace,
            )

        # Convert plan to action
        if state.current_plan:
            step = state.current_plan[0]
            action = self._plan_step_to_action(step, eg, state, available_actions)
            return action

        # Fallback: return first available action
        return available_actions[0] if available_actions else DriverAction(action_id=0)

    # ── Feedback (Causal Update) ──────────────────────────────────────────

    def update(
        self,
        action: DriverAction,
        feedback: DriverFeedback,
        source_id: str = "default",
    ) -> None:
        """Update internal state from action-outcome pair through trial-and-error.

        This is where the blackbox LEARNS from experience:
        - Motor dynamics (which action causes which displacement)
        - Obstacle / barrier features (collisions)
        - Traversable features (successful steps)
        - Target / receptacle classifications (interaction outcomes)
        """
        state = self.get_state(source_id)
        state.total_reward += feedback.reward

        # Empirical trial-and-error learning from feedback info
        info = feedback.info or {}
        if "observed_delta" in info:
            act_id = action.action_id
            delta = info["observed_delta"]
            if act_id not in state.action_models:
                from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

                state.action_models[act_id] = ActionDynamicsModel(
                    action_id=act_id,
                    delta_r=int(delta[0]) if len(delta) > 0 else 0,
                    delta_c=int(delta[1]) if len(delta) > 1 else 0,
                    confidence=0.6,
                )
            else:
                model = state.action_models[act_id]
                if hasattr(model, "update_from_trial"):
                    model.update_from_trial(delta, success=True)

        if "collision_feature" in info:
            feat = info["collision_feature"]
            state.learned_obstacle_features.add(feat)
            state.learned_traversable_features.discard(feat)

        if "traversed_feature" in info:
            feat = info["traversed_feature"]
            state.learned_traversable_features.add(feat)
            state.learned_obstacle_features.discard(feat)

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self, source_id: str = "default", retain_memory: bool = True) -> None:
        """Reset agent state for a new episode.

        If retain_memory is True, learned knowledge persists
        across episodes (transfer learning from experience).
        """
        state = self.get_state(source_id)

        if retain_memory:
            # Preserve learned knowledge
            saved_obstacles = set(state.learned_obstacle_features)
            saved_targets = set(state.learned_target_features)
            saved_traversable = set(state.learned_traversable_features)
            saved_action_models = dict(state.action_models)

            self._source_states[source_id] = AgentState(
                learned_obstacle_features=saved_obstacles,
                learned_target_features=saved_targets,
                learned_traversable_features=saved_traversable,
                action_models=saved_action_models,
            )
        else:
            self._source_states[source_id] = AgentState()

        self._source_entity_graphs.pop(source_id, None)
        self.spatial_planner.reset()

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _plan_step_to_action(
        self,
        step: SequencePlanStep,
        eg: EntityGraph,
        state: AgentState,
        available_actions: list[DriverAction],
    ) -> DriverAction:
        """Convert a high-level plan step into a concrete driver action."""
        if not eg.avatar:
            return available_actions[0] if available_actions else DriverAction(action_id=0)

        avatar_pos = eg.avatar.grid_pos
        target_pos = step.target_pos

        # Compute delta to target
        dr = target_pos[0] - avatar_pos[0]
        dc = target_pos[1] - avatar_pos[1]

        if dr == 0 and dc == 0:
            # At target — execute the action type
            if step.action_type in ("PICKUP", "DROP", "ACTIVATE"):
                # Find the interact action
                for a in available_actions:
                    if a.semantic_intent in ("interact", "pickup_drop", "activate"):
                        return a
            # Advance plan
            if state.current_plan:
                state.current_plan.pop(0)
            return available_actions[0] if available_actions else DriverAction(action_id=0)

        # Navigate toward target using LEARNED action models (not hardcoded)
        action_id = self.spatial_planner.get_action_for_delta(dr, dc, state.action_models)
        if action_id is not None:
            for a in available_actions:
                if a.action_id == action_id:
                    return a

        # Fallback to any movement action
        return available_actions[0] if available_actions else DriverAction(action_id=0)
