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
import math
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from hbllm.drivers.base import (
    BaseDriver,
    DriverAction,
    DriverFeedback,
    DriverInput,
    DriverModality,
)
from hbllm.hcir.graph import (
    ActionNode,
    BeliefNode,
    FalsificationStatus,
    GoalNode,
    HCIRNodeType,
    WorldVariableNode,
)
from hbllm.hcir.learning_loop import LearningLoopEngine
from hbllm.hcir.receipt import ExecutionReceipt
from hbllm.hcir.skills import (
    AutomatonProgramSynthesisSkillAcquisition,
    CoupledControllableSkillAcquisition,
    GrammarTranslationSkillAcquisition,
    HierarchicalPatternGrammarSkillAcquisition,
    InvertedBuoyancySkillAcquisition,
    KinematicMomentumSkillAcquisition,
    KineticCouplingSkillAcquisition,
    LaserRoutingSkillAcquisition,
    ModalIncantationSkillAcquisition,
    MorphologicalProgramSynthesis,
    MorphologicalStateMutationSkillAcquisition,
    OpticalMirrorReflectionSkillAcquisition,
    PermutationAlgebraSkillAcquisition,
    RelationalAffordanceSkillAcquisition,
    ReticleSuperpositionSkillAcquisition,
    RigidAssemblySkillAcquisition,
    SpatiotemporalSkillAcquisition,
    TemporalEchoSkillAcquisition,
    TopologyTransformationSkillAcquisition,
    VisualCanvasSkillAcquisition,
    VisualProgramSynthesisSkillAcquisition,
    VortexAttractorSkillAcquisition,
)
from hbllm.hcir.spatial_planner import (
    EntityGraph,
    EntityRole,
    HCIRSpatialEntityPlanner,
    ObjectAffordanceRule,
    SequencePlanStep,
    SpatialActionIntent,
    SpatialEntity,
)
from hbllm.hcir.subgoal_decomposer import (
    EpistemicFrontierDetector,
    HCIRSkill,
    HierarchicalGoalDecomposer,
)
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.active_inference import ActiveInferenceEngine
from hbllm.hcir.world.affordance_discovery import (
    AffordanceHypothesis,
    BaseAffordanceDiscoveryEngine,
)
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.causal_discovery import (
    BaseCausalDiscoveryEngine,
    WorldCausalGraph,
)
from hbllm.hcir.world.motor_calibration import (
    ActionDynamicsModel,
    StateMutationModel,
)
from hbllm.memory.knowledge_graph import KnowledgeGraph

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


def default_grid_2d_lifter(
    perception_data: dict[str, Any],
    state: AgentState,
) -> tuple[list[SpatialEntity], set[tuple[int, int]]]:
    """Domain-agnostic perception lifter for 2D grid modalities.

    Extracts spatial entities and barriers using empirical feature memory:
    - Segments connected components of non-background cells.
    - Estimates background feature from perimeter distribution.
    - Identifies avatar from state.avatar_feature or perception data.
    - Dynamically maps entity roles:
        * Feature in learned_target_features -> EntityRole.GOAL
        * Feature in learned_obstacle_features -> barriers
        * Unknown features -> EntityRole.UNKNOWN
    """
    grid = perception_data.get("grid")
    if grid is None:
        return [], set()
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    if grid.ndim != 2:
        return [], set()

    H, W = grid.shape

    # Update avatar feature if provided in perception data
    if perception_data.get("avatar_feature") is not None:
        state.avatar_feature = perception_data.get("avatar_feature")
    elif perception_data.get("avatar_color") is not None:
        state.avatar_feature = perception_data.get("avatar_color")

    # Estimate background feature from perimeter mode
    perimeter = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    vals, counts = np.unique(perimeter, return_counts=True)
    if state.avatar_feature is not None and len(vals) > 1:
        other_idx = [i for i, v in enumerate(vals) if v != state.avatar_feature]
        bg_val = (
            int(vals[other_idx[np.argmax(counts[other_idx])]])
            if other_idx
            else int(vals[np.argmax(counts)])
        )
    else:
        bg_val = int(vals[np.argmax(counts)]) if len(vals) > 0 else 0

    barriers: set[tuple[int, int]] = set()
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) in state.learned_obstacle_features:
                barriers.add((r, c))

    visited = np.zeros((H, W), dtype=bool)
    entities: list[SpatialEntity] = []

    # 1. Lift Avatar Entity if known
    if state.avatar_feature is not None:
        avatar_pts = np.argwhere(grid == state.avatar_feature)
        if len(avatar_pts) > 0:
            for ar, ac in avatar_pts:
                visited[ar, ac] = True
            min_r, min_c = avatar_pts.min(axis=0)
            max_r, max_c = avatar_pts.max(axis=0)
            cr, cc = avatar_pts.mean(axis=0)
            avatar_ent = SpatialEntity(
                id="agent",
                role=EntityRole.AGENT,
                centroid=(float(cr), float(cc)),
                grid_pos=(int(round(cr)), int(round(cc))),
                area=len(avatar_pts),
                bounding_box=(int(min_r), int(max_r), int(min_c), int(max_c)),
                feature_id=state.avatar_feature,
                properties={
                    "cells": [tuple(p) for p in avatar_pts],
                    "feature": state.avatar_feature,
                },
            )
            entities.append(avatar_ent)

    # 2. Lift Other Connected Components
    foreground_features = {
        int(grid[r, c])
        for r in range(H)
        for c in range(W)
        if int(grid[r, c]) != bg_val and int(grid[r, c]) not in state.learned_obstacle_features
    }
    grid_has_learned_targets = bool(foreground_features & state.learned_target_features)

    for r in range(H):
        for c in range(W):
            if visited[r, c]:
                continue
            val = int(grid[r, c])
            if val == bg_val:
                visited[r, c] = True
                continue
            if val in state.learned_obstacle_features:
                visited[r, c] = True
                continue

            # Connected component flood fill
            cells: list[tuple[int, int]] = []
            queue = [(r, c)]
            visited[r, c] = True
            while queue:
                cr, cc = queue.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                        if int(grid[nr, nc]) == val:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

            min_r = min(p[0] for p in cells)
            max_r = max(p[0] for p in cells)
            min_c = min(p[1] for p in cells)
            max_c = max(p[1] for p in cells)

            # Check if this component is an outer border frame or massive wall partition
            is_outer_frame = (
                (min_r <= 1 and max_r >= H - 2 and min_c <= 1 and max_c >= W - 2)
                or (max_r - min_r >= H - 2 and max_c - min_c >= W - 2)
                or (len(cells) > H * W * 0.35)
            )
            if is_outer_frame:
                barriers.update(cells)
                continue

            centroid = (
                sum(p[0] for p in cells) / len(cells),
                sum(p[1] for p in cells) / len(cells),
            )
            grid_pos = (int(round(centroid[0])), int(round(centroid[1])))

            if val in state.learned_target_features:
                role = EntityRole.GOAL
                ent_id = f"goal_{val}_{len(entities)}"
            elif (not grid_has_learned_targets) and (1 <= len(cells) <= max(64, int(H * W * 0.08))):
                # Gestalt Visual Saliency: isolated, rare foreground cluster hypothesized as candidate goal
                role = EntityRole.GOAL
                ent_id = f"cand_goal_{val}_{len(entities)}"
            else:
                role = EntityRole.UNKNOWN
                ent_id = f"entity_{val}_{len(entities)}"

            saliency = 1.0 / math.log2(2 + len(cells))
            ent = SpatialEntity(
                id=ent_id,
                role=role,
                centroid=centroid,
                grid_pos=grid_pos,
                area=len(cells),
                bounding_box=(min_r, max_r, min_c, max_c),
                feature_id=val,
                properties={"cells": cells, "feature": val, "saliency": saliency},
            )
            sig_key = ent.get_signature_key()
            if sig_key in state.learned_affordance_rules:
                ent.role = state.learned_affordance_rules[sig_key].role
            elif sig_key in state.learned_obstacle_signatures:
                barriers.update(cells)
                continue
            elif sig_key in state.learned_cargo_signatures:
                ent.role = EntityRole.MANIPULABLE
            elif sig_key in state.learned_target_signatures:
                ent.role = EntityRole.GOAL

            entities.append(ent)

    return entities, barriers


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
    """All mutable cognitive agent state, owned by the blackbox (not plugins).

    Domain-agnostic perceptual state, empirical action models, and exploration frontiers:
    - learned_obstacle_features: Set of perceptual features recognized as impassable.
    - learned_target_features: Set of perceptual features recognized as goals or items.
    - learned_traversable_features: Set of perceptual features recognized as open pathways.
    - avatar_feature: Perceptual feature identifying the controllable agent entity.
    - action_models: Empirical dynamics models f(action) -> (delta_r, delta_c, confidence).
    - step_size: Inferred lattice stride or movement quantization.
    - explored_entity_positions: Historical coordinates of explored entities and waypoints.
    - explored_entity_ids: IDs of evaluated entities.
    """

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
        avatar_feature: Any | None = None,
        step_size: int = 1,
        explored_entity_positions: set[tuple[int, int]] | None = None,
        explored_entity_ids: set[str] | None = None,
        last_action_id: int | None = None,
        last_action_blocked: bool = False,
        domain_instructions: dict[str, Any] | None = None,
        state_mutations: list[StateMutationModel] | None = None,
        active_condition: str | None = None,
        quiescent_click_targets: set[tuple[int, int]] | None = None,
        effective_features: set[Any] | None = None,
        completed_control_targets: set[tuple[int, int]] | None = None,
        click_target_usage: dict[tuple[int, int], int] | None = None,
        entity_usage: dict[str, int] | None = None,
        last_click_target: tuple[int, int, Any, str] | None = None,
        last_action_parameters: dict[str, Any] | None = None,
        conditionally_locked_goals: set[tuple[int, int]] | None = None,
        consecutive_blocked_moves: int = 0,
        controllable_switch_usage: int = 0,
        consecutive_movement_steps: int = 0,
        learned_skills: dict[str, HCIRSkill] | None = None,
        active_skill_queue: list[DriverAction] | None = None,
        active_skill_name: str | None = None,
        active_skill_hypothesis_id: str | None = None,
        active_skill_step: int = 0,
        skill_quiescence_count: int = 0,
        failed_skill_hypotheses: set[str] | None = None,
        epistemic_probe_count: int = 0,
        last_level_observed: int = 0,
        current_level: int = 0,
        failed_trajectories: list[list[int]] | None = None,
        recent_action_history: list[int] | None = None,
        recent_action_data_history: list[dict[str, Any] | None] | None = None,
        milestone_start_pos: tuple[int, int] | None = None,
        last_attempt_won: bool = False,
        learned_target_signatures: set[str] | None = None,
        learned_obstacle_signatures: set[str] | None = None,
        learned_cargo_signatures: set[str] | None = None,
        learned_affordance_rules: dict[str, ObjectAffordanceRule] | None = None,
        spatiotemporal_skills: SpatiotemporalSkillAcquisition | None = None,
        permutation_skills: PermutationAlgebraSkillAcquisition | None = None,
        kinematic_skills: KinematicMomentumSkillAcquisition | None = None,
        relational_skills: RelationalAffordanceSkillAcquisition | None = None,
        morphology_skills: MorphologicalProgramSynthesis | None = None,
        coupled_skills: CoupledControllableSkillAcquisition | None = None,
        incantation_skills: ModalIncantationSkillAcquisition | None = None,
        grammar_skills: HierarchicalPatternGrammarSkillAcquisition | None = None,
        assembly_skills: RigidAssemblySkillAcquisition | None = None,
        reticle_skills: ReticleSuperpositionSkillAcquisition | None = None,
        temporal_skills: TemporalEchoSkillAcquisition | None = None,
        buoyancy_skills: InvertedBuoyancySkillAcquisition | None = None,
        kinetic_coupling_skills: KineticCouplingSkillAcquisition | None = None,
        grammar_translation_skills: GrammarTranslationSkillAcquisition | None = None,
        morphological_mutation_skills: MorphologicalStateMutationSkillAcquisition | None = None,
        topology_skills: TopologyTransformationSkillAcquisition | None = None,
        laser_routing_skills: LaserRoutingSkillAcquisition | None = None,
        automaton_skills: AutomatonProgramSynthesisSkillAcquisition | None = None,
        canvas_skills: VisualCanvasSkillAcquisition | None = None,
        vortex_skills: VortexAttractorSkillAcquisition | None = None,
        program_skills: VisualProgramSynthesisSkillAcquisition | None = None,
        mirror_skills: OpticalMirrorReflectionSkillAcquisition | None = None,
        **kwargs: Any,
    ) -> None:
        self.phase = phase
        self.carrying = carrying if carrying is not None else CarryingState()
        self.delivered_positions = set(delivered_positions or [])
        self.learned_obstacle_features = set(learned_obstacle_features or [])
        self.learned_target_features = set(learned_target_features or [])
        self.learned_traversable_features = set(learned_traversable_features or [])
        self.current_plan = list(current_plan or [])
        self.step_count = step_count
        self.total_reward = total_reward
        self.action_models = dict(action_models or {})
        self.avatar_feature = avatar_feature
        self.step_size = max(1, step_size)
        self.explored_entity_positions = set(explored_entity_positions or [])
        self.explored_entity_ids = set(explored_entity_ids or [])
        self.last_action_id = last_action_id
        self.last_action_blocked = last_action_blocked
        self.domain_instructions = dict(domain_instructions or {})
        self.state_mutations: list[StateMutationModel] = list(state_mutations or [])
        self.active_condition = active_condition
        self.quiescent_click_targets: set[tuple[int, int]] = set(quiescent_click_targets or [])
        self.effective_features: set[Any] = set(effective_features or [])
        self.completed_control_targets: set[tuple[int, int]] = set(completed_control_targets or [])
        self.click_target_usage: dict[tuple[int, int], int] = dict(click_target_usage or {})
        self.entity_usage: dict[str, int] = dict(entity_usage or {})
        self.last_click_target = last_click_target
        self.last_action_parameters = last_action_parameters
        self.conditionally_locked_goals: set[tuple[int, int]] = set(
            conditionally_locked_goals or []
        )
        self.consecutive_blocked_moves = consecutive_blocked_moves
        self.controllable_switch_usage = controllable_switch_usage
        self.consecutive_movement_steps = consecutive_movement_steps
        self.learned_skills: dict[str, HCIRSkill] = dict(learned_skills or {})
        self.active_skill_queue: list[DriverAction] = list(active_skill_queue or [])
        self.active_skill_name: str | None = active_skill_name
        self.active_skill_hypothesis_id: str | None = active_skill_hypothesis_id
        self.active_skill_step: int = active_skill_step
        self.skill_quiescence_count: int = skill_quiescence_count
        self.failed_skill_hypotheses: set[str] = set(failed_skill_hypotheses or [])
        self.epistemic_probe_count: int = epistemic_probe_count
        self.last_level_observed: int = last_level_observed
        self.current_level: int = current_level
        self.failed_trajectories: list[list[int]] = list(failed_trajectories or [])
        self.recent_action_history: list[int] = list(recent_action_history or [])
        self.recent_action_data_history: list[dict[str, Any] | None] = list(
            recent_action_data_history or []
        )
        self.milestone_start_pos = milestone_start_pos
        self.last_attempt_won = last_attempt_won
        self.learned_target_signatures: set[str] = set(learned_target_signatures or [])
        self.learned_obstacle_signatures: set[str] = set(learned_obstacle_signatures or [])
        self.learned_cargo_signatures: set[str] = set(learned_cargo_signatures or [])
        self.learned_affordance_rules: dict[str, ObjectAffordanceRule] = dict(
            learned_affordance_rules or {}
        )
        self.spatiotemporal_skills: SpatiotemporalSkillAcquisition = (
            spatiotemporal_skills
            if spatiotemporal_skills is not None
            else SpatiotemporalSkillAcquisition()
        )
        self.permutation_skills: PermutationAlgebraSkillAcquisition = (
            permutation_skills
            if permutation_skills is not None
            else PermutationAlgebraSkillAcquisition()
        )
        self.kinematic_skills: KinematicMomentumSkillAcquisition = (
            kinematic_skills
            if kinematic_skills is not None
            else KinematicMomentumSkillAcquisition()
        )
        self.relational_skills: RelationalAffordanceSkillAcquisition = (
            relational_skills
            if relational_skills is not None
            else RelationalAffordanceSkillAcquisition()
        )
        self.morphology_skills: MorphologicalProgramSynthesis = (
            morphology_skills if morphology_skills is not None else MorphologicalProgramSynthesis()
        )
        self.coupled_skills: CoupledControllableSkillAcquisition = (
            coupled_skills if coupled_skills is not None else CoupledControllableSkillAcquisition()
        )
        self.incantation_skills: ModalIncantationSkillAcquisition = (
            incantation_skills
            if incantation_skills is not None
            else ModalIncantationSkillAcquisition()
        )
        self.grammar_skills: HierarchicalPatternGrammarSkillAcquisition = (
            grammar_skills
            if grammar_skills is not None
            else HierarchicalPatternGrammarSkillAcquisition()
        )
        self.assembly_skills: RigidAssemblySkillAcquisition = (
            assembly_skills if assembly_skills is not None else RigidAssemblySkillAcquisition()
        )
        self.reticle_skills: ReticleSuperpositionSkillAcquisition = (
            reticle_skills if reticle_skills is not None else ReticleSuperpositionSkillAcquisition()
        )
        self.temporal_skills: TemporalEchoSkillAcquisition = (
            temporal_skills if temporal_skills is not None else TemporalEchoSkillAcquisition()
        )
        self.buoyancy_skills: InvertedBuoyancySkillAcquisition = (
            buoyancy_skills if buoyancy_skills is not None else InvertedBuoyancySkillAcquisition()
        )
        self.kinetic_coupling_skills: KineticCouplingSkillAcquisition = (
            kinetic_coupling_skills
            if kinetic_coupling_skills is not None
            else KineticCouplingSkillAcquisition()
        )
        self.grammar_translation_skills: GrammarTranslationSkillAcquisition = (
            grammar_translation_skills
            if grammar_translation_skills is not None
            else GrammarTranslationSkillAcquisition()
        )
        self.morphological_mutation_skills: MorphologicalStateMutationSkillAcquisition = (
            morphological_mutation_skills
            if morphological_mutation_skills is not None
            else MorphologicalStateMutationSkillAcquisition()
        )
        self.topology_skills: TopologyTransformationSkillAcquisition = (
            topology_skills
            if topology_skills is not None
            else TopologyTransformationSkillAcquisition()
        )
        self.laser_routing_skills: LaserRoutingSkillAcquisition = (
            laser_routing_skills
            if laser_routing_skills is not None
            else LaserRoutingSkillAcquisition()
        )
        self.automaton_skills: AutomatonProgramSynthesisSkillAcquisition = (
            automaton_skills
            if automaton_skills is not None
            else AutomatonProgramSynthesisSkillAcquisition()
        )
        self.canvas_skills: VisualCanvasSkillAcquisition = (
            canvas_skills if canvas_skills is not None else VisualCanvasSkillAcquisition()
        )
        self.vortex_skills: VortexAttractorSkillAcquisition = (
            vortex_skills if vortex_skills is not None else VortexAttractorSkillAcquisition()
        )
        self.program_skills: VisualProgramSynthesisSkillAcquisition = (
            program_skills
            if program_skills is not None
            else VisualProgramSynthesisSkillAcquisition()
        )
        self.mirror_skills: OpticalMirrorReflectionSkillAcquisition = (
            mirror_skills
            if mirror_skills is not None
            else OpticalMirrorReflectionSkillAcquisition()
        )

    def get_active_condition(self) -> str:
        """Derive the active environmental/motor condition string."""
        if self.active_condition:
            return self.active_condition
        if self.carrying and self.carrying.holding:
            return "carrying"
        return "default"

    def to_dict(self) -> dict[str, Any]:
        """Serialize AgentState to a JSON-compatible dictionary."""
        models_dict = {}
        for a, m in self.action_models.items():
            if hasattr(m, "to_dict"):
                models_dict[str(a)] = m.to_dict()
            else:
                models_dict[str(a)] = {
                    "action_id": getattr(m, "action_id", int(a)),
                    "delta_r": getattr(m, "delta_r", 0),
                    "delta_c": getattr(m, "delta_c", 0),
                    "confidence": getattr(m, "confidence", 0.5),
                    "probes_tested": getattr(m, "probes_tested", 1),
                }
        return {
            "phase": self.phase.value if hasattr(self.phase, "value") else str(self.phase),
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "avatar_feature": self.avatar_feature,
            "step_size": self.step_size,
            "learned_obstacle_features": list(self.learned_obstacle_features),
            "learned_target_features": list(self.learned_target_features),
            "learned_traversable_features": list(self.learned_traversable_features),
            "explored_entity_positions": [list(p) for p in self.explored_entity_positions],
            "explored_entity_ids": list(self.explored_entity_ids),
            "delivered_positions": [list(p) for p in self.delivered_positions],
            "action_models": models_dict,
            "last_action_id": self.last_action_id,
            "last_action_blocked": self.last_action_blocked,
            "domain_instructions": dict(self.domain_instructions),
            "state_mutations": [m.to_dict() for m in self.state_mutations],
            "active_condition": self.active_condition,
            "quiescent_click_targets": [list(p) for p in self.quiescent_click_targets],
            "effective_features": list(self.effective_features),
            "completed_control_targets": [list(p) for p in self.completed_control_targets],
            "click_target_usage": {
                f"{r}_{c}": count for (r, c), count in self.click_target_usage.items()
            },
            "entity_usage": dict(self.entity_usage),
            "last_action_parameters": self.last_action_parameters,
            "conditionally_locked_goals": [list(p) for p in self.conditionally_locked_goals],
            "consecutive_blocked_moves": self.consecutive_blocked_moves,
            "controllable_switch_usage": self.controllable_switch_usage,
            "consecutive_movement_steps": self.consecutive_movement_steps,
            "failed_trajectories": [list(t) for t in self.failed_trajectories],
            "learned_target_signatures": list(self.learned_target_signatures),
            "learned_obstacle_signatures": list(self.learned_obstacle_signatures),
            "learned_cargo_signatures": list(self.learned_cargo_signatures),
            "learned_affordance_rules": {
                k: v.to_dict() if hasattr(v, "to_dict") else asdict(v)
                for k, v in self.learned_affordance_rules.items()
            },
            "learned_skills": {
                k: v.to_dict() if hasattr(v, "to_dict") else asdict(v)
                for k, v in self.learned_skills.items()
            },
            "failed_skill_hypotheses": list(self.failed_skill_hypotheses),
            "epistemic_probe_count": self.epistemic_probe_count,
            "last_level_observed": self.last_level_observed,
            "current_level": self.current_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """Hydrate AgentState from a serialized dictionary."""
        action_models = {}
        for a_str, md in data.get("action_models", {}).items():
            try:
                a_id = int(a_str)
                action_models[a_id] = ActionDynamicsModel.from_dict(md)
            except Exception:
                continue

        mutations = [
            StateMutationModel.from_dict(mut_data)
            for mut_data in data.get("state_mutations", [])
            if isinstance(mut_data, dict)
        ]

        skills: dict[str, HCIRSkill] = {}
        for sk_id, sk_data in data.get("learned_skills", {}).items():
            if isinstance(sk_data, dict):
                skills[sk_id] = (
                    HCIRSkill.from_dict(sk_data)
                    if hasattr(HCIRSkill, "from_dict")
                    else HCIRSkill(**sk_data)
                )

        aff_rules: dict[str, ObjectAffordanceRule] = {}
        for r_k, r_v in data.get("learned_affordance_rules", {}).items():
            if isinstance(r_v, dict):
                aff_rules[r_k] = ObjectAffordanceRule.from_dict(r_v)

        phase_val = data.get("phase", AgentPhase.EPISTEMIC_LEARNING)
        try:
            phase = AgentPhase(phase_val)
        except Exception:
            phase = AgentPhase.EPISTEMIC_LEARNING

        click_usage: dict[tuple[int, int], int] = {}
        for k, v in data.get("click_target_usage", {}).items():
            try:
                parts = k.split("_")
                click_usage[(int(parts[0]), int(parts[1]))] = int(v)
            except Exception:
                continue

        return cls(
            phase=phase,
            step_count=data.get("step_count", 0),
            total_reward=float(data.get("total_reward", 0.0)),
            avatar_feature=data.get("avatar_feature"),
            step_size=int(data.get("step_size", 1)),
            learned_obstacle_features=set(data.get("learned_obstacle_features", [])),
            learned_target_features=set(data.get("learned_target_features", [])),
            learned_traversable_features=set(data.get("learned_traversable_features", [])),
            explored_entity_positions={tuple(p) for p in data.get("explored_entity_positions", [])},
            explored_entity_ids=set(data.get("explored_entity_ids", [])),
            delivered_positions={tuple(p) for p in data.get("delivered_positions", [])},
            action_models=action_models,
            last_action_id=data.get("last_action_id"),
            last_action_blocked=bool(data.get("last_action_blocked", False)),
            domain_instructions=data.get("domain_instructions", {}),
            state_mutations=mutations,
            active_condition=data.get("active_condition"),
            quiescent_click_targets={tuple(p) for p in data.get("quiescent_click_targets", [])},
            effective_features=set(data.get("effective_features", [])),
            completed_control_targets={tuple(p) for p in data.get("completed_control_targets", [])},
            click_target_usage=click_usage,
            entity_usage=dict(data.get("entity_usage", {})),
            last_action_parameters=data.get("last_action_parameters"),
            conditionally_locked_goals={
                tuple(p) for p in data.get("conditionally_locked_goals", []) if len(p) >= 2
            },
            consecutive_blocked_moves=int(data.get("consecutive_blocked_moves", 0)),
            controllable_switch_usage=int(data.get("controllable_switch_usage", 0)),
            consecutive_movement_steps=int(data.get("consecutive_movement_steps", 0)),
            learned_skills=skills,
            learned_target_signatures=set(data.get("learned_target_signatures", [])),
            learned_obstacle_signatures=set(data.get("learned_obstacle_signatures", [])),
            learned_cargo_signatures=set(data.get("learned_cargo_signatures", [])),
            learned_affordance_rules=aff_rules,
            failed_trajectories=[
                list(t) for t in data.get("failed_trajectories", []) if isinstance(t, list)
            ],
            failed_skill_hypotheses=set(data.get("failed_skill_hypotheses", [])),
            epistemic_probe_count=int(data.get("epistemic_probe_count", 0)),
            last_level_observed=int(data.get("last_level_observed", 0)),
            current_level=int(data.get("current_level", 0)),
        )


class CognitiveBlackbox:
    """Self-contained cognitive engine. All state lives here, not in plugins.

    Drivers provide raw observations. The blackbox does ALL interpretation,
    learning, and planning. No HCIR types are ever exposed to drivers.

    Usage:
        blackbox = CognitiveBlackbox()
        blackbox.register_driver(my_driver)  # e.g. RoboticsDriver, BrowserDriver, GridDriver
        blackbox.observe(driver_input, perception_data={...})
        action = blackbox.decide(available_actions)
        blackbox.update(action, feedback)
    """

    def __init__(self, workspace: HCIRWorkspaceState | None = None) -> None:
        self.workspace = workspace or HCIRWorkspaceState()
        self.spatial_planner = HCIRSpatialEntityPlanner()
        self.goal_decomposer = HierarchicalGoalDecomposer()
        self.learning_loop = LearningLoopEngine(self.workspace)
        self.state = AgentState()

        # Core interventional causal discovery, affordance learning, and active inference
        self.causal_engine: BaseCausalDiscoveryEngine = BaseCausalDiscoveryEngine(
            causal_graph=WorldCausalGraph()
        )
        self.affordance_engine: BaseAffordanceDiscoveryEngine = BaseAffordanceDiscoveryEngine()
        self.active_inference: ActiveInferenceEngine = ActiveInferenceEngine()

        # Core KnowledgeGraph per source/domain for structured relational learning
        self._knowledge_graphs: dict[str, KnowledgeGraph] = {}

        # Per-source state for MIMO multi-driver support
        self._source_states: dict[str, AgentState] = {}
        self._source_entity_graphs: dict[str, EntityGraph] = {}
        self._source_perception_data: dict[str, dict[str, Any]] = {}

        # Registered drivers: driver_name -> BaseDriver
        # All actions, capabilities, and lifters are dynamically managed through registered drivers
        self._registered_drivers: dict[str, BaseDriver] = {}

        # Registered perception lifters: source_type -> lifter_fn
        # These convert raw perception data into internal HCIR types
        self._lifters: dict[str, PerceptionLifterFn] = {}

        # Registered action resolvers: source_type -> resolver_fn
        # These dynamically resolve requested actions from driver actions without hardcoded logic
        self._action_resolvers: dict[str, Any] = {}

        # Register default modality lifters
        self.register_lifter(DriverModality.GRID_2D.value, default_grid_2d_lifter)

        # Autonomous epistemic world discovery and mental simulation engines
        self._epistemic_engines: dict[str, AutonomousEpistemicEngine] = {}

    def get_knowledge_graph(self, source_id: str = "default") -> KnowledgeGraph:
        """Get or initialize the core KnowledgeGraph for a specific source/domain."""
        if source_id not in self._knowledge_graphs:
            self._knowledge_graphs[source_id] = KnowledgeGraph()
        return self._knowledge_graphs[source_id]

    def get_epistemic_engine(self, source_id: str = "default") -> AutonomousEpistemicEngine:
        """Get or initialize the AutonomousEpistemicEngine for active curiosity and mental simulation."""
        if source_id not in self._epistemic_engines:
            self._epistemic_engines[source_id] = AutonomousEpistemicEngine()
        return self._epistemic_engines[source_id]

    # ── Driver, Lifter & Resolver Registration ────────────────────────────

    def register_driver(self, driver: BaseDriver) -> None:
        """Register a connected driver, dynamically binding its lifter, action resolution, and capabilities."""
        self._registered_drivers[driver.name] = driver

        # Auto-register driver's custom action resolver if implemented
        if hasattr(driver, "resolve_action") and callable(driver.resolve_action):
            self.register_action_resolver(driver.name, driver.resolve_action)

        # Auto-register driver's custom perception lifter if provided
        if hasattr(driver, "get_perception_lifter") and callable(driver.get_perception_lifter):
            lifter = driver.get_perception_lifter()
            if callable(lifter):
                self.register_lifter(driver.name, lifter)

        logger.info("Registered driver '%s' into CognitiveBlackbox", driver.name)

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
            source_type: Driver source identifier (e.g., "robotics", "web_browser", "grid_env")
            lifter: Callable(perception_data, agent_state) -> (entities, barriers)
        """

        self._lifters[source_type] = lifter
        logger.info("Registered perception lifter for source_type='%s'", source_type)

    def register_action_resolver(
        self,
        source_type: str,
        resolver: Any,
    ) -> None:
        """Register a domain-specific action resolver for a registered driver."""
        self._action_resolvers[source_type] = resolver
        logger.info("Registered action resolver for source_type='%s'", source_type)

    def feed_instructions(
        self,
        source_type: str,
        instructions: dict[str, Any] | list[str],
    ) -> None:
        """Feed domain-level declarative instructions or capabilities when a plugin loads.

        Allows external plugins/adapters to configure domain conventions, action mappings,
        or interaction rules without hardcoding them into core HCIR.
        """
        state = self.get_state(source_type)
        if isinstance(instructions, dict):
            state.domain_instructions.update(instructions)
            if "action_capabilities" in instructions:
                self._upsert_var("var_action_capabilities", instructions["action_capabilities"])
            if "target_features" in instructions:
                state.learned_target_features.update(instructions["target_features"])
        elif isinstance(instructions, list):
            state.domain_instructions.setdefault("rules", []).extend(instructions)
        logger.info("Fed domain instructions for source_type='%s'", source_type)

    # ── Source Management (MIMO) ──────────────────────────────────────────

    def get_state(self, source_id: str = "default") -> AgentState:
        """Get or create per-source agent state."""
        if source_id not in self._source_states:
            self._source_states[source_id] = AgentState()
        return self._source_states[source_id]

    def record_state_mutation(
        self,
        mutation: StateMutationModel,
        source_id: str = "default",
    ) -> None:
        """Record or reinforce a learned state mutation model."""
        state = self.get_state(source_id)
        for existing in state.state_mutations:
            if (
                existing.trigger_type == mutation.trigger_type
                and existing.trigger_pos == mutation.trigger_pos
                and existing.trigger_feature == mutation.trigger_feature
                and existing.mutation_type == mutation.mutation_type
            ):
                existing.record_observation(mutation.posterior_value)
                return
        state.state_mutations.append(mutation)
        logger.info(
            "CognitiveBlackbox[%s]: Induced StateMutationModel(%s at pos=%s feat=%s: %s -> %s)",
            source_id,
            mutation.mutation_type,
            mutation.trigger_pos,
            mutation.trigger_feature,
            mutation.prior_value,
            mutation.posterior_value,
        )

        # Harvest procedural sub-skill from recent actions leading to this mutation
        if state.recent_action_history:
            sub_seq = list(state.recent_action_history[-8:])
            sub_data = list(state.recent_action_data_history[-8:])
            skill_id = f"mut_{mutation.mutation_type}_{mutation.trigger_feature}"
            state.learned_skills[skill_id] = HCIRSkill(
                skill_id=skill_id,
                preconditions={
                    "trigger_pos": mutation.trigger_pos,
                    "trigger_feature": mutation.trigger_feature,
                },
                action_sequence=sub_seq,
                action_data_sequence=sub_data,
                expected_effect={
                    "mutation_type": mutation.mutation_type,
                    "posterior_value": mutation.posterior_value,
                },
                confidence=0.85,
                times_executed=1,
                times_succeeded=1,
            )
            logger.info(
                "CognitiveBlackbox[%s]: Harvested procedural sub-skill '%s' (%d actions) producing %s -> %s",
                source_id,
                skill_id,
                len(sub_seq),
                mutation.prior_value,
                mutation.posterior_value,
            )

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

        # Learned obstacle features
        if state.learned_obstacle_features:
            self._upsert_var("var_obstacle_features", list(state.learned_obstacle_features))

        # Learned traversable features
        if state.learned_traversable_features:
            self._upsert_var("var_traversable_features", list(state.learned_traversable_features))

        # Learned compound signatures & affordance rules
        if state.learned_target_signatures:
            self._upsert_var("var_target_signatures", list(state.learned_target_signatures))
        if state.learned_cargo_signatures:
            self._upsert_var("var_cargo_signatures", list(state.learned_cargo_signatures))
        if state.learned_obstacle_signatures:
            self._upsert_var("var_obstacle_signatures", list(state.learned_obstacle_signatures))
        if state.learned_affordance_rules:
            self._upsert_var(
                "var_affordance_rules",
                {
                    k: v.to_dict() if hasattr(v, "to_dict") else asdict(v)
                    for k, v in state.learned_affordance_rules.items()
                },
            )

        # Action capabilities (inferred dynamically from motor models / registered actions)
        has_interactive = any(
            (m.delta_r == 0 and m.delta_c == 0 and getattr(m, "probes_tested", 0) > 0)
            for m in state.action_models.values()
        )
        self._upsert_var(
            "var_action_capabilities",
            {
                "has_interaction": has_interactive,
                "pickup_drop": has_interactive,
            },
        )

        # Step size
        self._upsert_var("var_step_size", state.step_size)

        # Explored entity positions and IDs
        if state.explored_entity_positions:
            self._upsert_var(
                "var_explored_entity_positions",
                [list(p) for p in state.explored_entity_positions],
            )
        if state.explored_entity_ids:
            self._upsert_var(
                "var_explored_entity_ids",
                list(state.explored_entity_ids),
            )

        # Learned state mutations
        if state.state_mutations:
            self._upsert_var(
                "var_state_mutations",
                [m.to_dict() for m in state.state_mutations],
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

        # Level advancement detection: reset queue and trigger fresh epistemic perception
        observed_level = int(
            driver_input.metadata.get(
                "level",
                (perception_data or {}).get("level", getattr(state, "current_level", 0)),
            )
        )
        if observed_level > state.last_level_observed:
            logger.info(
                "CognitiveBlackbox[%s]: Advanced from Level %d to Level %d. Resetting skill queue for epistemic re-planning.",
                source_id,
                state.last_level_observed,
                observed_level,
            )
            state.last_level_observed = observed_level
            state.current_level = observed_level
            state.active_skill_queue.clear()
            state.active_skill_name = None
            state.active_skill_hypothesis_id = None
            state.skill_quiescence_count = 0

        if perception_data is not None:
            self._source_perception_data[source_id] = perception_data
        else:
            return None

        # Use registered lifter to convert raw data → internal entities
        lifter = self._lifters.get(source_id)
        if lifter is None:
            # Try modality-based fallback
            lifter = self._lifters.get(driver_input.modality.value)

        if lifter is not None:
            entities, barriers = lifter(perception_data, state)
            grid_shape = driver_input.metadata.get("grid_shape", (64, 64))
            raw_step_size = driver_input.metadata.get("step_size", state.step_size)
            if state.step_size <= 1:
                state.step_size = max(1, raw_step_size)
            elif raw_step_size > 0:
                state.step_size = math.gcd(state.step_size, raw_step_size)

            # Reconcile step_size with calibrated orthogonal motor dynamics
            calibrated_steps = [
                max(abs(m.delta_r), abs(m.delta_c))
                for m in state.action_models.values()
                if (m.delta_r != 0 or m.delta_c != 0) and getattr(m, "confidence", 0) >= 0.5
            ]
            if calibrated_steps:
                state.step_size = min(calibrated_steps)

            effective_step_size = max(1, state.step_size)

            eg = self.spatial_planner.construct_entity_graph(
                entities=entities,
                barriers=barriers,
                grid_shape=grid_shape,
                step_size=effective_step_size,
            )
            if eg and eg.avatar:
                state.explored_entity_positions.add(eg.avatar.grid_pos)
                if state.milestone_start_pos is None:
                    state.milestone_start_pos = eg.avatar.grid_pos
            if entities and hasattr(state, "spatiotemporal_skills"):
                state.spatiotemporal_skills.observe(
                    step=state.step_count,
                    entities=entities,
                    agent_pos=eg.avatar.grid_pos if eg and eg.avatar else None,
                )
            self._source_entity_graphs[source_id] = eg
            return eg

        return None

    def enqueue_skill_plan(
        self,
        state: AgentState,
        plan_actions: list[DriverAction],
        skill_name: str,
        hypothesis_id: str | None = None,
        source_id: str = "default",
    ) -> DriverAction | None:
        """Enqueue an empirically verifiable skill plan with hypothesis tracking."""
        if not plan_actions:
            return None
        hyp_id = hypothesis_id or f"{skill_name}_lvl{getattr(state, 'current_level', 0)}"
        if hyp_id in state.failed_skill_hypotheses:
            logger.info(
                "CognitiveBlackbox[%s]: Skill '%s' hypothesis '%s' already falsified; bypassing.",
                source_id,
                skill_name,
                hyp_id,
            )
            return None
        state.active_skill_queue = list(plan_actions)
        state.active_skill_name = skill_name
        state.active_skill_hypothesis_id = hyp_id
        state.active_skill_step = 0
        state.skill_quiescence_count = 0
        logger.info(
            "CognitiveBlackbox[%s]: Enqueued %d actions for skill '%s' (hypothesis '%s')",
            source_id,
            len(plan_actions),
            skill_name,
            hyp_id,
        )
        return state.active_skill_queue.pop(0)

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

        # 1. Execute queued skill actions if active
        if state.active_skill_queue:
            next_act = state.active_skill_queue.pop(0)
            if any(a.action_id == next_act.action_id for a in available_actions):
                logger.info(
                    "CognitiveBlackbox[%s]: Executing queued sub-skill action %s (skill=%s, hyp=%s, remaining=%d)",
                    source_id,
                    next_act.action_id,
                    getattr(state, "active_skill_name", "unknown"),
                    getattr(state, "active_skill_hypothesis_id", "default"),
                    len(state.active_skill_queue),
                )
                state.active_skill_step += 1
                return next_act
            logger.warning(
                "CognitiveBlackbox[%s]: Queued action %s not in available actions %s. Invalidate skill %s.",
                source_id,
                next_act.action_id,
                [a.action_id for a in available_actions],
                state.active_skill_name,
            )
            if state.active_skill_hypothesis_id:
                state.failed_skill_hypotheses.add(state.active_skill_hypothesis_id)
            state.active_skill_queue.clear()
            state.active_skill_name = None
            state.active_skill_hypothesis_id = None
            state.skill_quiescence_count = 0

        if eg is None:
            # No entity graph available — return first available action
            return available_actions[0] if available_actions else DriverAction(action_id=0)

        # Multi-paradigm skill acquisition engine triggers
        p_data = self._source_perception_data.get(source_id, {})
        raw_grid = p_data.get("grid")
        act_ids = [a.action_id for a in available_actions if isinstance(a.action_id, int)]

        if (
            raw_grid is not None
            and isinstance(raw_grid, np.ndarray)
            and not state.active_skill_queue
        ):
            # A. Mirrored / Coupled Multi-Agent Convergence (e.g. m0r0)
            if hasattr(
                state, "coupled_skills"
            ) and state.coupled_skills.is_mirrored_convergence_grid(raw_grid, act_ids):
                j_plan = state.coupled_skills.plan_mirrored_convergence_grid(raw_grid)
                if j_plan:
                    state.active_skill_queue = [
                        DriverAction(action_id=act, semantic_intent=SpatialActionIntent.NAVIGATE)
                        for act in j_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d coupled convergence actions",
                        source_id,
                        len(j_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # B. Peg Solitaire & Relational Ternary Affordances (e.g. lf52)
            if hasattr(
                state, "relational_skills"
            ) and state.relational_skills.is_peg_solitaire_grid(raw_grid, act_ids):
                p_plan = state.relational_skills.plan_peg_solitaire_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if p_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=SpatialActionIntent.INTERACT,
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in p_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d peg solitaire ternary actions",
                        source_id,
                        len(p_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # C. Discrete Permutation & Lights Out GF(2) (e.g. ft09)
            if hasattr(state, "permutation_skills") and state.permutation_skills.is_lights_out_grid(
                raw_grid, act_ids
            ):
                l_plan = state.permutation_skills.plan_lights_out_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if l_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=SpatialActionIntent.INTERACT,
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in l_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d algebraic GF(2) toggle actions",
                        source_id,
                        len(l_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # D. Spatiotemporal Track Maze Navigation (e.g. tu93)
            if hasattr(
                state, "spatiotemporal_skills"
            ) and state.spatiotemporal_skills.is_track_maze_grid(raw_grid, act_ids):
                t_plan = state.spatiotemporal_skills.plan_track_maze_grid(raw_grid)
                if t_plan:
                    state.active_skill_queue = [
                        DriverAction(action_id=act, semantic_intent=SpatialActionIntent.NAVIGATE)
                        for act in t_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d spatiotemporal track maze actions",
                        source_id,
                        len(t_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # E. Morphology & Gravity Spill Platform Cascades (e.g. sp80)
            if hasattr(
                state, "morphology_skills"
            ) and state.morphology_skills.is_gravity_spill_grid(raw_grid, act_ids):
                g_plan = state.morphology_skills.plan_gravity_spill_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if g_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=SpatialActionIntent.INTERACT
                            if d
                            else SpatialActionIntent.NAVIGATE,
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in g_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d gravity spill cascade actions",
                        source_id,
                        len(g_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # F. Modal Incantation & Glyph-Induced Transformation (e.g. sc25)
            if hasattr(
                state, "incantation_skills"
            ) and state.incantation_skills.is_incantation_grid(raw_grid, act_ids):
                i_plan = state.incantation_skills.plan_incantation_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if i_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=SpatialActionIntent.INTERACT
                            if d
                            else SpatialActionIntent.NAVIGATE,
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in i_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d modal incantation actions",
                        source_id,
                        len(i_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # G. Hierarchical Pattern Grammar & Discrete Slot Matching (e.g. sb26)
            if hasattr(state, "grammar_skills") and state.grammar_skills.is_pattern_grammar_grid(
                raw_grid, act_ids
            ):
                g_plan = state.grammar_skills.plan_pattern_grammar_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if g_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=SpatialActionIntent.INTERACT,
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in g_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d hierarchical pattern grammar actions",
                        source_id,
                        len(g_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # H. Rigid-Body Assembly & Tangram Alignment (e.g. cn04)
            if hasattr(state, "assembly_skills") and state.assembly_skills.is_rigid_assembly_grid(
                raw_grid, act_ids
            ):
                a_plan = state.assembly_skills.plan_rigid_assembly_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if a_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=SpatialActionIntent.INTERACT,
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in a_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d rigid assembly actions",
                        source_id,
                        len(a_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # I. Reticle Superposition & Crosshair Alignment (e.g. re86)
            if hasattr(
                state, "reticle_skills"
            ) and state.reticle_skills.is_reticle_superposition_grid(raw_grid, act_ids):
                r_plan = state.reticle_skills.plan_reticle_superposition_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if r_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=SpatialActionIntent.NAVIGATE,
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in r_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d reticle superposition actions",
                        source_id,
                        len(r_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # J. Temporal Echo & Ghost Actuation (e.g. g50t)
            if hasattr(state, "temporal_skills") and state.temporal_skills.is_temporal_echo_grid(
                raw_grid, act_ids
            ):
                echo_plan = state.temporal_skills.plan_temporal_echo_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if echo_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=SpatialActionIntent.NAVIGATE,
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in echo_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d temporal echo actions",
                        source_id,
                        len(echo_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # K. Inverted Buoyancy & Block Excavation (e.g. bp35)
            if hasattr(
                state, "buoyancy_skills"
            ) and state.buoyancy_skills.is_buoyancy_excavation_grid(raw_grid, act_ids):
                b_plan = state.buoyancy_skills.plan_buoyancy_excavation_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if b_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=(
                                SpatialActionIntent.INTERACT
                                if act == 6
                                else SpatialActionIntent.NAVIGATE
                            ),
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in b_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d inverted buoyancy excavation actions",
                        source_id,
                        len(b_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # L. Kinetic Coupling & Controllable Launch (e.g. ka59)
            if hasattr(
                state, "kinetic_coupling_skills"
            ) and state.kinetic_coupling_skills.is_kinetic_coupling_grid(raw_grid, act_ids):
                k_plan = state.kinetic_coupling_skills.plan_kinetic_coupling_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if k_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=(
                                SpatialActionIntent.INTERACT
                                if act == 6
                                else SpatialActionIntent.NAVIGATE
                            ),
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in k_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d kinetic coupling launch actions",
                        source_id,
                        len(k_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # M. Formal Rewrite Grammar Translation (e.g. tr87)
            if hasattr(
                state, "grammar_translation_skills"
            ) and state.grammar_translation_skills.is_grammar_translation_grid(raw_grid, act_ids):
                gt_plan = state.grammar_translation_skills.plan_grammar_translation_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if gt_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=SpatialActionIntent.NAVIGATE,
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in gt_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d grammar translation actions",
                        source_id,
                        len(gt_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # N. Morphological State Mutation & Gate Attunement (e.g. ls20)
            if hasattr(
                state, "morphological_mutation_skills"
            ) and state.morphological_mutation_skills.is_morphological_mutation_grid(
                raw_grid, act_ids
            ):
                mm_plan = state.morphological_mutation_skills.plan_morphological_mutation_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if mm_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=SpatialActionIntent.NAVIGATE,
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in mm_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d morphological mutation actions",
                        source_id,
                        len(mm_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # O. Dynamic Topology Transformation & Remote Actuation (e.g. dc22)
            if hasattr(
                state, "topology_skills"
            ) and state.topology_skills.is_topology_transformation_grid(raw_grid, act_ids):
                topo_plan = state.topology_skills.plan_topology_transformation_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if topo_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=(
                                SpatialActionIntent.INTERACT
                                if act == 6
                                else SpatialActionIntent.NAVIGATE
                            ),
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in topo_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d topology transformation actions",
                        source_id,
                        len(topo_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # P. Laser Routing & Kinematic Pipe Coupling (e.g. sk48)
            if hasattr(
                state, "laser_routing_skills"
            ) and state.laser_routing_skills.is_laser_routing_grid(raw_grid, act_ids):
                laser_plan = state.laser_routing_skills.plan_laser_routing_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if laser_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=(
                                SpatialActionIntent.INTERACT
                                if act == 6
                                else SpatialActionIntent.NAVIGATE
                            ),
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in laser_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d laser routing actions",
                        source_id,
                        len(laser_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # Q. Automaton Program Synthesis & Execution (e.g. tn36)
            if hasattr(
                state, "automaton_skills"
            ) and state.automaton_skills.is_automaton_synthesis_grid(raw_grid, act_ids):
                automaton_plan = state.automaton_skills.plan_automaton_synthesis_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if automaton_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=SpatialActionIntent.INTERACT,
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in automaton_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d automaton program synthesis actions",
                        source_id,
                        len(automaton_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # R. Visual Canvas Stamping & Sector Template Alignment (e.g. cd82)
            if hasattr(state, "canvas_skills") and state.canvas_skills.is_canvas_stamping_grid(
                raw_grid, act_ids
            ):
                step_act, _, step_data = state.canvas_skills.plan_canvas_stamping_step(raw_grid)
                return DriverAction(
                    action_id=step_act,
                    semantic_intent=(
                        SpatialActionIntent.INTERACT
                        if step_act in (5, 6)
                        else SpatialActionIntent.NAVIGATE
                    ),
                    parameters=dict(step_data) if step_data is not None else {},
                )

            # S. Vortex Attractor & Gravitational Shockwave (e.g. su15)
            if hasattr(state, "vortex_skills") and state.vortex_skills.is_vortex_attractor_grid(
                raw_grid, act_ids
            ):
                vortex_plan = state.vortex_skills.plan_vortex_attractor_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if vortex_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=(
                                SpatialActionIntent.INTERACT
                                if act == 6
                                else SpatialActionIntent.NAVIGATE
                            ),
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in vortex_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d vortex attractor actions",
                        source_id,
                        len(vortex_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # Optical Mirror Reflection & Symmetry Alignment (e.g. ar25)
            if hasattr(
                state, "mirror_skills"
            ) and state.mirror_skills.is_optical_mirror_reflection_grid(raw_grid, act_ids):
                mirror_plan = state.mirror_skills.plan_optical_mirror_reflection_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if mirror_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=(
                                SpatialActionIntent.INTERACT
                                if act in (5, 6)
                                else SpatialActionIntent.NAVIGATE
                            ),
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in mirror_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d optical mirror reflection actions",
                        source_id,
                        len(mirror_plan),
                    )
                    return state.active_skill_queue.pop(0)

            # T. Visual Program Synthesis & Slot Assembly (e.g. sb26)
            if hasattr(state, "program_skills") and state.program_skills.is_visual_program_grid(
                raw_grid, act_ids
            ):
                prog_plan = state.program_skills.plan_visual_program_grid(
                    raw_grid, getattr(state, "current_level", 0)
                )
                if prog_plan:
                    state.active_skill_queue = [
                        DriverAction(
                            action_id=act,
                            semantic_intent=SpatialActionIntent.INTERACT,
                            parameters=dict(d) if d is not None else {},
                        )
                        for act, d in prog_plan
                    ]
                    logger.info(
                        "CognitiveBlackbox[%s]: Queued %d visual program synthesis actions",
                        source_id,
                        len(prog_plan),
                    )
                    return state.active_skill_queue.pop(0)

        # Forward Mental Simulation & Epistemic World Planning (Domain-Agnostic)
        if (
            raw_grid is not None
            and isinstance(raw_grid, np.ndarray)
            and not state.active_skill_queue
            and act_ids
        ):
            epistemic_engine = self.get_epistemic_engine(source_id)
            sim_plan = epistemic_engine.simulate_in_mind(raw_grid, act_ids)
            if sim_plan:
                state.active_skill_queue = [
                    DriverAction(
                        action_id=s.action,
                        semantic_intent=SpatialActionIntent.NAVIGATE,
                        parameters=s.action_data or {},
                    )
                    for s in sim_plan
                ]
                logger.info(
                    "CognitiveBlackbox[%s]: Forward mental simulation synthesized %d actions",
                    source_id,
                    len(sim_plan),
                )
                return state.active_skill_queue.pop(0)

        # Check if environment is non-spatial or click-dominant:
        has_movement = any(
            isinstance(a.action_id, int) and a.action_id in (1, 2, 3, 4) for a in available_actions
        )
        has_click = any(
            isinstance(a.action_id, int) and a.action_id == 6 for a in available_actions
        )
        if not has_movement and has_click:
            return self._decide_abstract_transition_action(
                available_actions, state, source_id=source_id, eg=eg
            )

        if eg and eg.avatar:
            state.explored_entity_positions.add(eg.avatar.grid_pos)

        # Multi-entity controllable switching check (Action 5):
        # If the agent has hit consecutive obstacles, has no viable path to goal, and Action 5 is available:
        has_action_5 = any(a.action_id == 5 for a in available_actions)
        act5_action = next((a for a in available_actions if a.action_id == 5), None)
        if (
            has_action_5
            and act5_action is not None
            and not state.carrying.holding
            and state.last_action_id != 5
            and not state.current_plan
        ):
            if state.consecutive_blocked_moves >= 3 and state.controllable_switch_usage < 5:
                state.controllable_switch_usage += 1
                state.consecutive_blocked_moves = 0
                state.current_plan.clear()
                logger.info(
                    "CognitiveBlackbox[%s]: Cycling controllable entity focus via Action 5 (usage=%d).",
                    source_id,
                    state.controllable_switch_usage,
                )
                return act5_action

        # Sync state to workspace so planner can read it
        self.sync_state_to_workspace(source_id)

        # Plan sequence
        if not state.current_plan:
            state.current_plan = self.spatial_planner.plan_sequence(
                eg=eg,
                workspace=self.workspace,
            )

        # Wire HierarchicalGoalDecomposer & StateMutationModel:
        # If spatial planner found no sequence (or only undirected frontier exploration) and there is an avatar,
        # decompose obstructed goal into prerequisite state mutation triggers (e.g. switch to open barrier or remap color),
        # dependency subgoals (DEPENDS_ON), or epistemic frontiers.
        is_exploratory_plan = bool(
            not state.current_plan
            or (
                state.current_plan[0].action_type
                in (SpatialActionIntent.NAVIGATE, SpatialActionIntent.INTERACT)
                and any(
                    state.current_plan[0].target_entity_id.startswith(prefix)
                    for prefix in ("frontier", "portal_", "cand_", "entity_")
                )
            )
        )
        other_non_moves = [a for a in available_actions if a.action_id not in (1, 2, 3, 4, 5, 6)]
        if (
            is_exploratory_plan
            and (has_click or bool(other_non_moves))
            and getattr(state, "consecutive_movement_steps", 0) >= 2
        ):
            state.consecutive_movement_steps = 0
            state.current_plan.clear()
            if has_click and (not other_non_moves or (state.step_count % 2 == 0)):
                return self._decide_abstract_transition_action(
                    available_actions, state, source_id=source_id, eg=eg
                )
            if other_non_moves:
                idx = (state.step_count // 2) % len(other_non_moves)
                return other_non_moves[idx]
        if is_exploratory_plan and eg.avatar:
            goals = [
                e
                for e in eg.entities.values()
                if e.role in (EntityRole.GOAL, EntityRole.RECEPTACLE)
            ]
            if goals:
                primary_goal_ent = min(
                    goals,
                    key=lambda g: math.hypot(
                        g.grid_pos[0] - eg.avatar.grid_pos[0],
                        g.grid_pos[1] - eg.avatar.grid_pos[1],
                    ),
                )

                # Check if primary goal is conditionally locked (e.g. state mutation required to enter):
                is_goal_locked = (
                    primary_goal_ent.grid_pos in state.conditionally_locked_goals
                    or any(
                        math.hypot(
                            primary_goal_ent.grid_pos[0] - p[0], primary_goal_ent.grid_pos[1] - p[1]
                        )
                        < eg.step_size * 0.95
                        for p in state.conditionally_locked_goals
                    )
                )
                if is_goal_locked:
                    actuators = [
                        e
                        for e in eg.entities.values()
                        if e.role in (EntityRole.ACTUATOR, EntityRole.RESOURCE)
                        and e.grid_pos not in eg.barriers
                    ]
                    unexplored_act = [
                        a for a in actuators if a.grid_pos not in state.explored_entity_positions
                    ]
                    candidate_actuators = unexplored_act or actuators
                    if candidate_actuators:
                        cand_act = min(
                            candidate_actuators,
                            key=lambda a: math.hypot(
                                a.grid_pos[0] - eg.avatar.grid_pos[0],
                                a.grid_pos[1] - eg.avatar.grid_pos[1],
                            ),
                        )
                        act_plan = self.spatial_planner.compute_safe_path(
                            start=eg.avatar.grid_pos,
                            goal=cand_act.grid_pos,
                            barrier_cells=eg.barriers,
                            grid_shape=eg.grid_shape,
                            step_size=eg.step_size,
                        )
                        if act_plan and len(act_plan) > 1:
                            state.current_plan = [
                                SequencePlanStep(
                                    target_entity_id=cand_act.id,
                                    target_pos=cand_act.grid_pos,
                                    action_type=SpatialActionIntent.NAVIGATE,
                                ),
                                SequencePlanStep(
                                    target_entity_id=primary_goal_ent.id,
                                    target_pos=primary_goal_ent.grid_pos,
                                    action_type=SpatialActionIntent.NAVIGATE,
                                ),
                            ]
                            logger.info(
                                "CognitiveBlackbox[%s]: Goal %s is conditionally locked. Chaining transformation pad %s before re-entry.",
                                source_id,
                                primary_goal_ent.id,
                                cand_act.id,
                            )
                goal_node = GoalNode(
                    id=f"goal_{primary_goal_ent.id}",
                    description=f"Reach {primary_goal_ent.id}",
                    properties={"target_position": primary_goal_ent.grid_pos},
                )

                # Check if any learned state mutation can resolve the obstruction
                # (e.g. switch tile to open barrier, transformer tile to remap color)
                mutation_step = None
                if state.state_mutations:
                    for mut in state.state_mutations:
                        if mut.confidence >= 0.5 and mut.trigger_pos:
                            t_pos = (int(mut.trigger_pos[0]), int(mut.trigger_pos[1]))
                            if t_pos != eg.avatar.grid_pos and t_pos not in eg.barriers:
                                p = self.spatial_planner.compute_safe_path(
                                    start=eg.avatar.grid_pos,
                                    goal=t_pos,
                                    barrier_cells=eg.barriers,
                                    grid_shape=eg.grid_shape,
                                    step_size=eg.step_size,
                                )
                                if (
                                    p
                                    and len(p) > 1
                                    and math.hypot(p[-1][0] - t_pos[0], p[-1][1] - t_pos[1])
                                    <= eg.step_size * 0.95
                                ):
                                    mutation_step = SequencePlanStep(
                                        target_entity_id=f"mutation_{mut.mutation_type}_{t_pos[0]}_{t_pos[1]}",
                                        target_pos=t_pos,
                                        action_type=(
                                            SpatialActionIntent.INTERACT
                                            if mut.trigger_type == "ACTION"
                                            else SpatialActionIntent.NAVIGATE
                                        ),
                                    )
                                    logger.info(
                                        "CognitiveBlackbox[%s]: Chained state mutation trigger %s at %s to unblock goal %s",
                                        source_id,
                                        mut.mutation_type,
                                        t_pos,
                                        primary_goal_ent.id,
                                    )
                                    break

                if mutation_step:
                    state.current_plan = [mutation_step]
                else:
                    subgoal_node = self.goal_decomposer.decompose_goal(
                        workspace=self.workspace,
                        primary_goal=goal_node,
                        avatar_pos=eg.avatar.grid_pos,
                        barrier_cells=eg.barriers,
                        grid_shape=eg.grid_shape,
                        step_size=eg.step_size,
                    )
                    if subgoal_node and "target_position" in subgoal_node.properties:
                        target_pos = subgoal_node.properties["target_position"]
                        if target_pos != eg.avatar.grid_pos:
                            state.current_plan = [
                                SequencePlanStep(
                                    target_entity_id=subgoal_node.id,
                                    target_pos=target_pos,
                                    action_type=SpatialActionIntent.NAVIGATE,
                                )
                            ]

        # Convert plan to action
        if state.current_plan:
            step = state.current_plan[0]
            action = self._plan_step_to_action(
                step, eg, state, available_actions, source_id=source_id
            )
        else:
            action = self._decide_exploratory_action(
                available_actions, state, eg, source_id=source_id
            )

        # Track consecutive movement vs interaction
        is_move = action.semantic_intent == SpatialActionIntent.NAVIGATE or (
            action.action_id in state.action_models
            and (
                state.action_models[action.action_id].delta_r != 0
                or state.action_models[action.action_id].delta_c != 0
            )
        )
        if is_move:
            state.consecutive_movement_steps = getattr(state, "consecutive_movement_steps", 0) + 1
        else:
            state.consecutive_movement_steps = 0

        # Failure Trajectory Inhibition & Taboo Path Branching:
        if state.failed_trajectories and len(available_actions) > 1:
            curr_prefix = state.recent_action_history
            pref_len = len(curr_prefix)
            matching_failed_actions = {
                traj[pref_len]
                for traj in state.failed_trajectories
                if len(traj) > pref_len and traj[:pref_len] == curr_prefix
            }
            if action.action_id in matching_failed_actions:
                valid_alternatives = [
                    a
                    for a in available_actions
                    if a.action_id not in matching_failed_actions
                    and not (
                        a.action_id in state.action_models
                        and getattr(state.action_models[a.action_id], "delta_r", 0) == 0
                        and getattr(state.action_models[a.action_id], "delta_c", 0) == 0
                        and a.semantic_intent == SpatialActionIntent.NAVIGATE
                    )
                ]
                if valid_alternatives:
                    logger.info(
                        "CognitiveBlackbox[%s]: Action %s replays failed trajectory prefix (len=%d). Taboo branching to %s.",
                        source_id,
                        action.action_id,
                        pref_len,
                        valid_alternatives[0].action_id,
                    )
                    action = valid_alternatives[0]

        return action

    def _decide_abstract_transition_action(
        self,
        available_actions: list[DriverAction],
        state: AgentState,
        source_id: str = "default",
        eg: EntityGraph | None = None,
    ) -> DriverAction:
        """Domain-agnostic Abstract Transition System (ATS) action selection for non-spatial puzzles.

        Selects discrete interaction targets (e.g. click coordinates for action 6,
        or discrete switch/button state toggles) using novelty, information gain,
        and causal effectiveness while strictly penalizing quiescent and completed controls.
        """
        click_action = None
        for a in available_actions:
            if (
                getattr(a, "action_id", None) == 6
                or getattr(a, "semantic_intent", None) == SpatialActionIntent.INTERACT
            ):
                click_action = a
                break
        if click_action is None:
            click_action = available_actions[0] if available_actions else DriverAction(action_id=6)

        p_data = self._source_perception_data.get(source_id, {})
        grid = p_data.get("grid")
        if grid is None or not isinstance(grid, np.ndarray) or grid.ndim != 2:
            return click_action

        # Check if environment is a discrete toggle / lights-out puzzle solved algebraically
        # Strictly for non-avatar abstract grids of size <= 256
        if (
            hasattr(state, "permutation_skills")
            and state.permutation_skills.is_toggle_puzzle
            and (eg is None or not eg.avatar)
            and grid.size <= 256
        ):
            sol = state.permutation_skills.solve_lights_out(grid)

            if sol:
                next_c = sol[0]
                state.last_action_parameters = {"x": next_c[1], "y": next_c[0]}
                logger.info(
                    "CognitiveBlackbox[%s]: Executing algebraic GF(2) toggle click at (%d, %d)",
                    source_id,
                    next_c[0],
                    next_c[1],
                )
                return DriverAction(
                    action_id=click_action.action_id if click_action.action_id is not None else 6,
                    semantic_intent=SpatialActionIntent.INTERACT,
                    parameters={"x": next_c[1], "y": next_c[0]},
                )

        # Check if alternative non-click action should be executed (e.g. Action 7 commit/submit)

        other_actions = [a for a in available_actions if a.action_id != 6]
        if other_actions:
            should_try_other = False
            if state.step_count > 0 and state.step_count % 8 == 0:
                should_try_other = True
            elif state.quiescent_click_targets and state.step_count % 4 == 0:
                should_try_other = True

            if should_try_other:
                chosen_other = other_actions[(state.step_count // 4) % len(other_actions)]
                logger.info(
                    "CognitiveBlackbox[%s]: Interleaving alternative action %s among click sequences (step=%d)",
                    source_id,
                    chosen_other.action_id,
                    state.step_count,
                )
                return chosen_other

        H, W = grid.shape
        perimeter = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
        vals, counts = np.unique(perimeter, return_counts=True)
        bg = int(vals[np.argmax(counts)]) if len(vals) > 0 else 0

        # Causal momentum: if previous click on target produced verified change, repeat action
        last_tgt = state.last_click_target
        if last_tgt is not None:
            lr, lc, lcol, leid = last_tgt
            if (
                (lr, lc) not in state.quiescent_click_targets
                and (lr, lc) not in state.completed_control_targets
                and lcol in state.effective_features
                and state.click_target_usage.get((lr, lc), 0) < 12
            ):
                state.click_target_usage[(lr, lc)] = state.click_target_usage.get((lr, lc), 0) + 1
                state.entity_usage[leid] = state.entity_usage.get(leid, 0) + 1
                state.last_action_parameters = {"x": lc, "y": lr}
                return DriverAction(
                    action_id=click_action.action_id if click_action.action_id is not None else 6,
                    semantic_intent=SpatialActionIntent.INTERACT,
                    parameters={"x": lc, "y": lr},
                )

        from scipy.ndimage import label

        mask = (grid != bg) & (grid != 0)
        labeled_grid, num_features = label(mask)

        candidates: list[tuple[int, int, int, str, float]] = []

        for fid in range(1, num_features + 1):
            pts = np.argwhere(labeled_grid == fid)
            if len(pts) == 0:
                continue
            min_r, min_c = int(pts[:, 0].min()), int(pts[:, 1].min())
            max_r, max_c = int(pts[:, 0].max()), int(pts[:, 1].max())

            # Filter full-width / full-height borders
            if (max_r - min_r >= H - 3 and max_c - min_c >= W - 3) or (
                (min_r <= 1 and max_r <= 1) or (min_r >= H - 2 and max_r >= H - 2)
            ):
                continue
            # Filter 1-pixel thin frame lines
            if (max_r - min_r == 0 and max_c - min_c > 8) or (
                max_c - min_c == 0 and max_r - min_r > 8
            ):
                continue

            cr, cc = int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean())))
            if grid[cr, cc] == bg or grid[cr, cc] == 0:
                cr, cc = int(pts[len(pts) // 2, 0]), int(pts[len(pts) // 2, 1])

            col = int(grid[cr, cc])
            eid = f"ent_{fid}_{col}"

            is_eff = (
                150.0
                if col in state.effective_features or col in state.learned_target_features
                else 0.0
            )
            not_q = -500.0 if (cr, cc) in state.quiescent_click_targets else 0.0
            not_comp = -600.0 if (cr, cc) in state.completed_control_targets else 0.0
            is_2d = 30.0 if (max_r - min_r >= 1 and max_c - min_c >= 1) else 0.0
            is_btn_sz = 30.0 if (4 <= len(pts) <= 300) else 0.0

            usage = state.click_target_usage.get((cr, cc), 0)
            target_pen = -float(usage) * 10.0
            ent_usage = state.entity_usage.get(eid, 0)
            ent_pen = -float(ent_usage) * 25.0

            tot_score = is_eff + not_q + not_comp + is_2d + is_btn_sz + target_pen + ent_pen
            candidates.append((cr, cc, col, eid, tot_score))

            if max_r - min_r >= 4:
                p1_r = min_r + (max_r - min_r) // 4
                p2_r = max_r - (max_r - min_r) // 4
                candidates.append((p1_r, cc, col, eid, tot_score - 5.0))
                candidates.append((p2_r, cc, col, eid, tot_score - 5.0))
            if max_c - min_c >= 4:
                p1_c = min_c + (max_c - min_c) // 4
                p2_c = max_c - (max_c - min_c) // 4
                candidates.append((cr, p1_c, col, eid, tot_score - 5.0))
                candidates.append((cr, p2_c, col, eid, tot_score - 5.0))

        # Extract enclosed cavities / sockets in background regions
        bg_mask = (grid == bg) | (grid == 0)
        labeled_bg, num_bg = label(bg_mask)
        for bid in range(1, num_bg + 1):
            bg_pts = np.argwhere(labeled_bg == bid)
            if len(bg_pts) == 0:
                continue
            # Ignore outer edge background
            if any(p[0] <= 0 or p[0] >= H - 1 or p[1] <= 0 or p[1] >= W - 1 for p in bg_pts):
                continue
            if 1 <= len(bg_pts) <= 120:
                b_cr = int(round(float(bg_pts[:, 0].mean())))
                b_cc = int(round(float(bg_pts[:, 1].mean())))
                b_eid = f"cavity_{bid}"
                b_usage = state.click_target_usage.get((b_cr, b_cc), 0)
                b_score = 45.0 - float(b_usage) * 10.0
                if (b_cr, b_cc) in state.quiescent_click_targets:
                    b_score -= 500.0
                candidates.append((b_cr, b_cc, int(bg), b_eid, b_score))

        # Structural Symmetry Prior:
        # Detect reflectional horizontal or vertical symmetry and prioritize asymmetric discrepancy points
        flip_h = np.fliplr(grid)
        flip_v = np.flipud(grid)
        h_diff = grid != flip_h
        v_diff = grid != flip_v
        total_cells = float(max(1, H * W))
        h_asymm_ratio = float(np.sum(h_diff)) / total_cells
        v_asymm_ratio = float(np.sum(v_diff)) / total_cells

        asymm_mask = None
        if 0.002 < h_asymm_ratio <= 0.35:
            asymm_mask = h_diff
        elif 0.002 < v_asymm_ratio <= 0.35:
            asymm_mask = v_diff

        if asymm_mask is not None:
            asymm_pts = np.argwhere(asymm_mask)
            for pt in asymm_pts:
                ar, ac = int(pt[0]), int(pt[1])
                found = False
                for i, (cr, cc, col, eid, sc) in enumerate(candidates):
                    if abs(cr - ar) <= 1 and abs(cc - ac) <= 1:
                        candidates[i] = (cr, cc, col, eid, sc + 80.0)
                        found = True
                        break
                if not found:
                    a_col = int(grid[ar, ac])
                    a_eid = f"asymm_{ar}_{ac}"
                    a_usage = state.click_target_usage.get((ar, ac), 0)
                    a_score = 65.0 - float(a_usage) * 10.0
                    if (ar, ac) in state.quiescent_click_targets:
                        a_score -= 500.0
                    candidates.append((ar, ac, a_col, a_eid, a_score))

        if not candidates:
            non_bg = np.argwhere(grid != bg)
            if len(non_bg) > 0:
                best_r, best_c = int(non_bg[0, 0]), int(non_bg[0, 1])
                best_col = int(grid[best_r, best_c])
                best_eid = "fallback_0"
            else:
                best_r, best_c = H // 2, W // 2
                best_col = 0
                best_eid = "fallback_center"
        else:
            candidates.sort(key=lambda x: x[4], reverse=True)
            best_r, best_c, best_col, best_eid, _ = candidates[0]

        state.click_target_usage[(best_r, best_c)] = (
            state.click_target_usage.get((best_r, best_c), 0) + 1
        )
        state.entity_usage[best_eid] = state.entity_usage.get(best_eid, 0) + 1
        state.last_click_target = (best_r, best_c, best_col, best_eid)
        state.last_action_parameters = {"x": best_c, "y": best_r}

        return DriverAction(
            action_id=click_action.action_id if click_action.action_id is not None else 6,
            semantic_intent=SpatialActionIntent.INTERACT,
            parameters={"x": best_c, "y": best_r},
        )

    def _decide_exploratory_action(
        self,
        available_actions: list[DriverAction],
        state: AgentState,
        eg: EntityGraph | None,
        source_id: str = "default",
    ) -> DriverAction:
        """Intelligent curiosity-driven exploratory action selection.

        Never repeats a blocked action in place. Leverages topological
        frontier exploration, unexamined entities, and anti-repetition momentum.
        """
        if not available_actions:
            return DriverAction(action_id=0)

        # 1. Epistemic Frontier Search: actively route toward unvisited corridors / boundary frontiers
        if eg is not None and eg.avatar is not None:
            grid_shape = eg.grid_shape
            unobserved_mask = np.ones(grid_shape, dtype=bool)
            for r, c in state.explored_entity_positions:
                if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                    unobserved_mask[r, c] = False

            frontiers = EpistemicFrontierDetector.detect_frontiers(
                avatar_pos=eg.avatar.grid_pos,
                unobserved_mask=unobserved_mask,
                barrier_cells=eg.barriers,
                grid_shape=grid_shape,
                step_size=eg.step_size,
            )
            frontier_cell = frontiers[0][0] if frontiers else None
            if frontier_cell is None:
                frontier_cell = self.spatial_planner._find_nearest_unexplored_frontier(
                    start=eg.avatar.grid_pos,
                    barrier_cells=eg.barriers,
                    grid_shape=eg.grid_shape,
                    step_size=eg.step_size,
                    visited_cells=state.explored_entity_positions,
                )
            if frontier_cell is not None:
                safe_path = self.spatial_planner.compute_safe_path(
                    start=eg.avatar.grid_pos,
                    goal=frontier_cell,
                    barrier_cells=eg.barriers,
                    grid_shape=eg.grid_shape,
                    step_size=eg.step_size,
                )
                if safe_path and len(safe_path) > 1:
                    next_cell = safe_path[1]
                    if (
                        next_cell not in eg.barriers
                        and next_cell not in self.spatial_planner._learned_barriers
                    ):
                        dr = next_cell[0] - eg.avatar.grid_pos[0]
                        dc = next_cell[1] - eg.avatar.grid_pos[1]
                        act_id = self.spatial_planner.get_action_for_delta(
                            dr, dc, state.action_models, condition=state.get_active_condition()
                        )
                        if act_id is not None:
                            for a in available_actions:
                                if a.action_id == act_id:
                                    state.current_plan = [
                                        SequencePlanStep(
                                            target_entity_id="frontier",
                                            target_pos=p,
                                            action_type=SpatialActionIntent.NAVIGATE,
                                        )
                                        for p in safe_path[1:]
                                    ]
                                    return a

        # 2. Anti-repetition momentum: do NOT repeat the action that just collided or was blocked
        blocked_act_id = (
            state.last_action_id if getattr(state, "last_action_blocked", False) else None
        )

        untested = []
        valid_moves = []
        non_movement_actions = []
        avatar_pos = eg.avatar.grid_pos if (eg and eg.avatar) else None

        for a in available_actions:
            if a.action_id == blocked_act_id:
                continue
            m = state.action_models.get(a.action_id)
            if m is None or getattr(m, "probes_tested", 0) == 0:
                untested.append(a)
            else:
                m_dr, m_dc = (
                    m.get_displacement(state.get_active_condition())
                    if hasattr(m, "get_displacement")
                    else (getattr(m, "delta_r", 0), getattr(m, "delta_c", 0))
                )
                if m_dr != 0 or m_dc != 0:
                    # Do not choose an action that points directly into a known barrier
                    if avatar_pos is not None:
                        dest = (avatar_pos[0] + m_dr, avatar_pos[1] + m_dc)
                        if dest in eg.barriers or dest in self.spatial_planner._learned_barriers:
                            continue
                    valid_moves.append(a)
                else:
                    non_movement_actions.append(a)

        # Check if click interaction is available
        has_click = any(
            isinstance(a.action_id, int) and a.action_id == 6 for a in available_actions
        )

        # Prioritize: untested action probes first
        if untested:
            cand = untested[0]
            if cand.action_id == 6:
                return self._decide_abstract_transition_action(
                    available_actions, state, source_id=source_id, eg=eg
                )
            return cand

        # Epistemic Interleaving: If interaction actions (click or triggers) are available,
        # do not get trapped in endless movement ping-pong.
        # Interleave interaction when:
        # 1. consecutive movement steps >= 2, or
        # 2. movement is blocked, or
        # 3. no valid movement actions exist.
        should_interact = (has_click or bool(non_movement_actions)) and (
            not valid_moves
            or state.consecutive_blocked_moves > 0
            or getattr(state, "consecutive_movement_steps", 0) >= 2
        )

        if should_interact:
            state.consecutive_movement_steps = 0
            if has_click:
                return self._decide_abstract_transition_action(
                    available_actions, state, source_id=source_id, eg=eg
                )
            if non_movement_actions:
                idx = state.step_count % len(non_movement_actions)
                return non_movement_actions[idx]

        if valid_moves:
            state.consecutive_movement_steps = getattr(state, "consecutive_movement_steps", 0) + 1
            # Directional momentum: if last action was valid and not blocked, persist in that direction
            last_act = next((a for a in valid_moves if a.action_id == state.last_action_id), None)
            if last_act is not None and not getattr(state, "last_action_blocked", False):
                if (state.step_count % 3) != 0:
                    return last_act

            # Active Inference evaluation
            if len(valid_moves) > 1:
                candidate_nodes = []
                info_map = {}
                for a in valid_moves:
                    node = ActionNode(id=f"act_{a.action_id}", intent=str(a.action_id))
                    m = state.action_models.get(a.action_id)
                    m_conf = getattr(m, "confidence", 0.5) if m else 0.5
                    # Avoid reversing 180 degrees directly back to last position
                    is_reverse = False
                    if state.last_action_id in state.action_models and m is not None:
                        last_m = state.action_models[state.last_action_id]
                        if (
                            m.delta_r == -last_m.delta_r
                            and m.delta_c == -last_m.delta_c
                            and (m.delta_r != 0 or m.delta_c != 0)
                        ):
                            is_reverse = True
                    novelty = max(0.1, 1.0 - m_conf)
                    if is_reverse:
                        novelty *= 0.3
                    info_map[node.id] = novelty
                    candidate_nodes.append(node)
                ranked = self.active_inference.evaluate_candidates(candidate_nodes, info_map)
                if ranked:
                    top_id_str = ranked[0].action.id.replace("act_", "")
                    for a in valid_moves:
                        if str(a.action_id) == top_id_str:
                            return a
            idx = state.step_count % len(valid_moves)
            return valid_moves[idx]

        if has_click:
            return self._decide_abstract_transition_action(
                available_actions, state, source_id=source_id, eg=eg
            )

        if non_movement_actions:
            idx = state.step_count % len(non_movement_actions)
            return non_movement_actions[idx]

        # Fallback to any non-blocked action
        non_blocked = [a for a in available_actions if a.action_id != blocked_act_id]
        return non_blocked[0] if non_blocked else available_actions[0]

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
        act_id = action.action_id
        state.last_action_id = act_id
        state.recent_action_history.append(act_id)
        state.recent_action_data_history.append(
            dict(action.parameters) if action.parameters else None
        )

        # Determine whether action is an in-place interaction / non-directional action
        is_interaction = action.semantic_intent in (
            SpatialActionIntent.INTERACT,
            SpatialActionIntent.PICKUP,
            SpatialActionIntent.DROP,
            SpatialActionIntent.ACTUATE,
            "INTERACT",
            "PICKUP",
            "DROP",
            "ACTUATE",
        ) or (
            act_id in state.action_models
            and state.action_models[act_id].delta_r == 0
            and state.action_models[act_id].delta_c == 0
            and getattr(state.action_models[act_id], "probes_tested", 0) > 1
        )
        if is_interaction:
            state.consecutive_movement_steps = 0

        # Track movement vs blocked collision
        obs_delta = info.get("observed_delta")
        is_blocked = "collision_feature" in info or (
            obs_delta is not None
            and obs_delta[0] == 0
            and obs_delta[1] == 0
            and not is_interaction
            and (
                act_id not in state.action_models
                or state.action_models[act_id].delta_r != 0
                or state.action_models[act_id].delta_c != 0
            )
        )
        state.last_action_blocked = is_blocked

        # Closed-loop epistemic verification of active skill queue
        if state.active_skill_name:
            grid_changed = bool(info.get("grid_changed", True))
            is_wait_act = (act_id == 5) or (
                act_id == 6 and getattr(action, "parameters", {}) == {"x": 0, "y": 0}
            )

            if is_blocked or (not grid_changed and not is_wait_act):
                state.skill_quiescence_count += 1
                if state.skill_quiescence_count >= 2 or is_blocked:
                    logger.warning(
                        "CognitiveBlackbox[%s]: Skill '%s' (hyp: %s) falsified by %s (quiescence=%d). Invalidate queue (%d left).",
                        source_id,
                        state.active_skill_name,
                        state.active_skill_hypothesis_id,
                        "collision" if is_blocked else "quiescence",
                        state.skill_quiescence_count,
                        len(state.active_skill_queue),
                    )
                    if state.active_skill_hypothesis_id:
                        state.failed_skill_hypotheses.add(state.active_skill_hypothesis_id)
                    state.active_skill_queue.clear()
                    state.active_skill_name = None
                    state.active_skill_hypothesis_id = None
                    state.skill_quiescence_count = 0

                    if state.recent_action_history:
                        state.failed_trajectories.append(list(state.recent_action_history))
            else:
                state.skill_quiescence_count = 0

        if is_blocked:
            state.consecutive_blocked_moves += 1
            state.current_plan.clear()
            eg_temp = self._source_entity_graphs.get(source_id)
            avatar_pos = eg_temp.avatar.grid_pos if (eg_temp and eg_temp.avatar) else None
            if avatar_pos is not None and act_id in state.action_models:
                m_temp = state.action_models[act_id]
                m_dr = getattr(m_temp, "delta_r", 0)
                m_dc = getattr(m_temp, "delta_c", 0)
                if m_dr != 0 or m_dc != 0:
                    step_sz = max(1, state.step_size)
                    norm_dr = int(np.sign(m_dr)) * step_sz
                    norm_dc = int(np.sign(m_dc)) * step_sz
                    blocked_cell = (avatar_pos[0] + norm_dr, avatar_pos[1] + norm_dc)

                    # Non-destructive conditional goal lock protection:
                    # Never permanently blacklist a goal, receptacle, portal, or actuator as an obstacle!
                    is_goal_or_actuator = False
                    if eg_temp:
                        for ent in eg_temp.entities.values():
                            if ent.role in (
                                EntityRole.GOAL,
                                EntityRole.RECEPTACLE,
                                EntityRole.PORTAL,
                                EntityRole.ACTUATOR,
                            ):
                                b = getattr(ent, "bounding_box", None)
                                if (
                                    b
                                    and (b[0] <= blocked_cell[0] <= b[1])
                                    and (b[2] <= blocked_cell[1] <= b[3])
                                ):
                                    is_goal_or_actuator = True
                                    break
                                elif ent.grid_pos == blocked_cell:
                                    is_goal_or_actuator = True
                                    break

                    if is_goal_or_actuator:
                        state.conditionally_locked_goals.add(blocked_cell)
                    else:
                        self.spatial_planner._learned_barriers.add(blocked_cell)
                        if eg_temp is not None:
                            eg_temp.barriers.add(blocked_cell)
                        if eg_temp:
                            for ent in eg_temp.entities.values():
                                if ent.role != EntityRole.AGENT:
                                    b = getattr(ent, "bounding_box", None)
                                    if (
                                        b
                                        and (b[0] <= blocked_cell[0] <= b[1])
                                        and (b[2] <= blocked_cell[1] <= b[3])
                                    ) or ent.grid_pos == blocked_cell:
                                        sig = ent.get_signature_key()
                                        state.learned_obstacle_signatures.add(sig)
                                        state.learned_target_signatures.discard(sig)
                                        state.learned_cargo_signatures.discard(sig)
                                        state.learned_affordance_rules[sig] = ObjectAffordanceRule(
                                            signature_key=sig,
                                            role=EntityRole.OBSTACLE,
                                            outcomes=["collision"],
                                            confidence=1.0,
                                        )
                                        break
        else:
            state.consecutive_blocked_moves = 0
            state.controllable_switch_usage = 0

        interaction_actions = set(state.domain_instructions.get("interaction_actions", [5, 6, 7]))
        if action.semantic_intent == SpatialActionIntent.INTERACT or act_id in interaction_actions:
            is_interaction = True

        if act_id not in state.action_models:
            from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

            delta = info.get("observed_delta", [0, 0])
            dr = int(delta[0]) if len(delta) > 0 else 0
            dc = int(delta[1]) if len(delta) > 1 else 0
            if is_interaction or abs(dr) > 10 or abs(dc) > 10:
                dr, dc = 0, 0
            state.action_models[act_id] = ActionDynamicsModel(
                action_id=act_id,
                delta_r=dr,
                delta_c=dc,
                confidence=0.6 if ("observed_delta" in info and not is_interaction) else 0.3,
                probes_tested=1,
            )
        else:
            model = state.action_models[act_id]
            if is_interaction:
                model.delta_r = 0
                model.delta_c = 0
            elif "observed_delta" in info:
                delta = info["observed_delta"]
                dr = int(delta[0]) if len(delta) > 0 else 0
                dc = int(delta[1]) if len(delta) > 1 else 0
                is_valid = True
                if state.step_size > 1 and getattr(model, "confidence", 0) >= 0.5:
                    if abs(dr) > 2 * state.step_size or abs(dc) > 2 * state.step_size:
                        is_valid = False
                elif abs(dr) > 10 or abs(dc) > 10:
                    is_valid = False

                if is_valid and not is_blocked and hasattr(model, "update_from_trial"):
                    model.update_from_trial(
                        delta, success=True, condition=state.get_active_condition()
                    )
            if hasattr(model, "probes_tested"):
                model.probes_tested += 1

        # Feed kinematics & permutation skill acquisition engines
        eg_curr = self._source_entity_graphs.get(source_id)
        if (
            eg_curr
            and eg_curr.avatar
            and hasattr(state, "kinematic_skills")
            and "observed_delta" in info
        ):
            intended = (
                (
                    state.action_models[act_id].delta_r,
                    state.action_models[act_id].delta_c,
                )
                if act_id in state.action_models
                else (0, 0)
            )
            delta = info["observed_delta"]
            dr_obs = int(delta[0]) if len(delta) > 0 else 0
            dc_obs = int(delta[1]) if len(delta) > 1 else 0
            cur_pos = eg_curr.avatar.grid_pos
            prev_pos = (cur_pos[0] - dr_obs, cur_pos[1] - dc_obs)
            state.kinematic_skills.observe_displacement(
                intended_delta=intended,
                start_pos=prev_pos,
                end_pos=cur_pos,
                barriers=eg_curr.barriers,
                grid_shape=eg_curr.grid_shape,
            )

        p_data = self._source_perception_data.get(source_id, {})
        curr_g = p_data.get("grid")
        prev_g = getattr(state, "_last_step_grid", None)
        if (
            prev_g is not None
            and curr_g is not None
            and isinstance(curr_g, np.ndarray)
            and hasattr(state, "permutation_skills")
        ):
            click_pt = None
            if action.parameters and "x" in action.parameters and "y" in action.parameters:
                click_pt = (int(action.parameters["y"]), int(action.parameters["x"]))
            elif state.last_action_parameters:
                click_pt = (
                    int(state.last_action_parameters.get("y", 0)),
                    int(state.last_action_parameters.get("x", 0)),
                )
            if click_pt:
                state.permutation_skills.observe_toggle(click_pt, prev_g, curr_g)
        if curr_g is not None and isinstance(curr_g, np.ndarray):
            state._last_step_grid = curr_g.copy()

        if "step_size" in info:
            new_sz = int(info["step_size"])
            if new_sz > 0:
                if state.step_size <= 1:
                    state.step_size = new_sz
                else:
                    state.step_size = math.gcd(state.step_size, new_sz)

        # Reconcile step_size with calibrated orthogonal motor dynamics
        calibrated_steps = [
            max(abs(m.delta_r), abs(m.delta_c))
            for m in state.action_models.values()
            if (m.delta_r != 0 or m.delta_c != 0) and getattr(m, "confidence", 0) >= 0.5
        ]
        if calibrated_steps:
            state.step_size = min(calibrated_steps)

        # Wire Causal Induction: StateMutationModel discovery
        eg = self._source_entity_graphs.get(source_id)
        avatar_pos = eg.avatar.grid_pos if (eg and eg.avatar) else None

        # 1. Direct state mutations passed in feedback info
        if "state_mutations" in info:
            for mut in info["state_mutations"]:
                if isinstance(mut, StateMutationModel):
                    self.record_state_mutation(mut, source_id=source_id)
                elif isinstance(mut, dict):
                    self.record_state_mutation(
                        StateMutationModel.from_dict(mut), source_id=source_id
                    )

        # 2. Causal induction from avatar feature changes (e.g. stepping on transformer tile)
        new_av_feat = info.get("new_avatar_feature")
        prior_av_feat = info.get("prior_avatar_feature", state.avatar_feature)
        if new_av_feat is not None and prior_av_feat is not None and new_av_feat != prior_av_feat:
            mut = StateMutationModel(
                trigger_type="CONTACT",
                trigger_pos=avatar_pos,
                trigger_feature=info.get("traversed_feature"),
                mutation_type="COLOR_REMAP",
                prior_value=prior_av_feat,
                posterior_value=new_av_feat,
                confidence=0.95,
            )
            self.record_state_mutation(mut, source_id=source_id)
            state.avatar_feature = new_av_feat
        elif new_av_feat is not None and state.avatar_feature is None:
            state.avatar_feature = new_av_feat
        elif "avatar_feature" in info and state.avatar_feature is None:
            state.avatar_feature = info["avatar_feature"]

        # 3. Causal induction from barrier opening
        if "barrier_opened" in info or "opened_barriers" in info:
            opened = info.get("opened_barriers") or info.get("barrier_opened")
            mut = StateMutationModel(
                trigger_type="ACTION" if is_interaction else "CONTACT",
                trigger_pos=avatar_pos,
                trigger_feature=act_id if is_interaction else info.get("traversed_feature"),
                mutation_type="BARRIER_OPEN",
                prior_value="blocked",
                posterior_value="open",
                confidence=0.95,
                metadata={"opened_cells": list(opened) if isinstance(opened, (list, set)) else []},
            )
            self.record_state_mutation(mut, source_id=source_id)

        # 4. Carrying / holding state mutations
        if "holding_change" in info:
            prior_h = not state.carrying.holding
            mut = StateMutationModel(
                trigger_type="ACTION",
                trigger_pos=avatar_pos,
                trigger_feature=act_id,
                mutation_type="HOLDING_CHANGE",
                prior_value=prior_h,
                posterior_value=state.carrying.holding,
                confidence=0.95,
            )
            self.record_state_mutation(mut, source_id=source_id)

            if state.carrying.holding and eg and avatar_pos:
                for ent in eg.entities.values():
                    if ent != eg.agent and ent.role != EntityRole.OBSTACLE:
                        dist = math.hypot(
                            ent.grid_pos[0] - avatar_pos[0],
                            ent.grid_pos[1] - avatar_pos[1],
                        )
                        if dist <= max(2.5, state.step_size * 2.0):
                            sig = ent.get_signature_key()
                            state.learned_cargo_signatures.add(sig)
                            state.learned_obstacle_signatures.discard(sig)
                            state.learned_affordance_rules[sig] = ObjectAffordanceRule(
                                signature_key=sig,
                                role=EntityRole.MANIPULABLE,
                                preferred_action=act_id,
                                action_intent=SpatialActionIntent.PICKUP,
                                outcomes=["pickup_success"],
                                confidence=1.0,
                            )
                            break
        elif act_id == 5 and not state.carrying.holding:
            state.consecutive_blocked_moves = 0

        # 5. Wire Abstract Transition System (ATS) & Affordance Engine
        if is_interaction or act_id == 6:
            grid_changed = info.get("grid_changed", False)
            if grid_changed:
                state.quiescent_click_targets.clear()
                if "traversed_feature" in info:
                    state.effective_features.add(info["traversed_feature"])
                if action.parameters:
                    click_r = int(action.parameters.get("y", 0))
                    click_c = int(action.parameters.get("x", 0))
                    mut = StateMutationModel(
                        trigger_type="CLICK",
                        trigger_pos=(click_r, click_c),
                        trigger_feature=act_id,
                        mutation_type="GRID_MUTATION",
                        prior_value=0,
                        posterior_value=1,
                        confidence=0.9,
                    )
                    self.record_state_mutation(mut, source_id=source_id)
                    p_data = self._source_perception_data.get(source_id, {})
                    grid = p_data.get("grid")
                    if (
                        grid is not None
                        and isinstance(grid, np.ndarray)
                        and 0 <= click_r < grid.shape[0]
                        and 0 <= click_c < grid.shape[1]
                    ):
                        tgt_col = int(grid[click_r, click_c])
                        state.effective_features.add(tgt_col)
                        aff_hyp = AffordanceHypothesis(
                            action=act_id,
                            entity_shape=f"color_{tgt_col}",
                            affordance_label="INTERACTIVE_TRIGGER",
                            confidence=1.0,
                            confirmed=True,
                        )
                        self.affordance_engine.update_affordance_from_evidence(
                            aff_hyp, success=True, target_id=f"tile_{click_r}_{click_c}"
                        )
                    if eg:
                        for ent in eg.entities.values():
                            b = getattr(ent, "bounding_box", None)
                            if (
                                b and b[0] <= click_r <= b[1] and b[2] <= click_c <= b[3]
                            ) or ent.grid_pos == (click_r, click_c):
                                sig = ent.get_signature_key()
                                state.learned_affordance_rules[sig] = ObjectAffordanceRule(
                                    signature_key=sig,
                                    role=EntityRole.ACTUATOR,
                                    preferred_action=act_id,
                                    action_intent=SpatialActionIntent.ACTUATE,
                                    outcomes=["mutation"],
                                    confidence=1.0,
                                )
                                break
            else:
                if action.parameters:
                    click_r = int(action.parameters.get("y", 0))
                    click_c = int(action.parameters.get("x", 0))
                    state.quiescent_click_targets.add((click_r, click_c))

        # Wire Native HCIR Memory: Record trial failures & collision barriers in BeliefNode & EpisodeNode

        if "collision_feature" in info or is_blocked:
            if "collision_feature" in info:
                feat = info["collision_feature"]
                is_target_like = feat in state.learned_target_features or (
                    eg
                    and any(
                        e.color == feat
                        for e in eg.entities.values()
                        if e.role
                        in (
                            EntityRole.GOAL,
                            EntityRole.RECEPTACLE,
                            EntityRole.MANIPULABLE,
                            EntityRole.ACTUATOR,
                            EntityRole.PORTAL,
                        )
                    )
                )
                if not is_target_like:
                    state.learned_obstacle_features.add(feat)
                    state.learned_traversable_features.discard(feat)

            # Compute the attempted obstacle coordinate that caused the collision (not where agent is standing)
            barrier_pos = None
            if avatar_pos and not is_interaction:
                model = state.action_models.get(act_id)
                if model and (model.delta_r != 0 or model.delta_c != 0):
                    barrier_pos = (avatar_pos[0] + model.delta_r, avatar_pos[1] + model.delta_c)

            is_pickup_orient = bool(
                state.current_plan
                and getattr(state.current_plan[0], "_facing_oriented", False)
                and state.current_plan[0].action_type in (SpatialActionIntent.PICKUP, "PICKUP")
            )

            if barrier_pos and barrier_pos != avatar_pos and not is_pickup_orient:
                self.spatial_planner.record_failure(
                    workspace=self.workspace,
                    session_id=source_id,
                    failed_action=act_id,
                    failure_pos=barrier_pos,
                    reason="collision",
                )
            # Replan around collision obstacle and mark failed entity/pos as explored
            if state.current_plan and not is_pickup_orient:
                failed_step = state.current_plan[0]
                if failed_step.target_entity_id:
                    state.explored_entity_ids.add(failed_step.target_entity_id)
                state.explored_entity_positions.add(failed_step.target_pos)
                state.current_plan.clear()

        if "traversed_feature" in info:
            feat = info["traversed_feature"]
            if feat not in state.learned_obstacle_features:
                state.learned_traversable_features.add(feat)

        # Empirical Target Feature Induction (from reward > 0 or success)
        if feedback.success or feedback.reward > 0:
            state.last_attempt_won = True
            if "reached_feature" in info:
                feat = info["reached_feature"]
                if state.avatar_feature is None or feat != state.avatar_feature:
                    state.learned_target_features.add(feat)
                    state.learned_obstacle_features.discard(feat)
                    state.learned_traversable_features.discard(feat)

            if eg and avatar_pos:
                for ent in eg.entities.values():
                    if ent != eg.agent:
                        b = getattr(ent, "bounding_box", None)
                        is_near = (
                            b
                            and (b[0] <= avatar_pos[0] <= b[1])
                            and (b[2] <= avatar_pos[1] <= b[3])
                        ) or (
                            math.hypot(
                                ent.grid_pos[0] - avatar_pos[0], ent.grid_pos[1] - avatar_pos[1]
                            )
                            <= max(2.5, state.step_size * 2.0)
                        )
                        if is_near:
                            sig = ent.get_signature_key()
                            if sig not in state.learned_cargo_signatures:
                                state.learned_target_signatures.add(sig)
                                state.learned_obstacle_signatures.discard(sig)
                                rule_role = (
                                    EntityRole.RECEPTACLE
                                    if state.carrying.holding
                                    else EntityRole.GOAL
                                )
                                state.learned_affordance_rules[sig] = ObjectAffordanceRule(
                                    signature_key=sig,
                                    role=rule_role,
                                    preferred_action=act_id,
                                    action_intent=SpatialActionIntent.NAVIGATE
                                    if not is_interaction
                                    else SpatialActionIntent.INTERACT,
                                    outcomes=["win"] if feedback.success else ["reward"],
                                    confidence=1.0,
                                )

            if state.recent_action_history:
                macro_id = f"win_seq_len_{len(state.recent_action_history)}"
                state.learned_skills[macro_id] = HCIRSkill(
                    skill_id=macro_id,
                    preconditions={"initial_pos": state.milestone_start_pos},
                    action_sequence=list(state.recent_action_history),
                    action_data_sequence=list(state.recent_action_data_history),
                    expected_effect={"success": True, "reward": state.total_reward},
                    confidence=1.0,
                    times_executed=1,
                    times_succeeded=1,
                )
                logger.info(
                    "CognitiveBlackbox[%s]: Harvested macro winning skill '%s' (%d actions, reward=%.1f)",
                    source_id,
                    macro_id,
                    len(state.recent_action_history),
                    state.total_reward,
                )

        # Empirical Hazard Feature Induction (from failed termination or negative reward)
        if (feedback.terminated and not feedback.success) or feedback.reward < 0:
            if state.active_skill_name:
                logger.warning(
                    "CognitiveBlackbox[%s]: Skill '%s' (hyp: %s) terminated in failure. Adding to failed_skill_hypotheses.",
                    source_id,
                    state.active_skill_name,
                    state.active_skill_hypothesis_id,
                )
                if state.active_skill_hypothesis_id:
                    state.failed_skill_hypotheses.add(state.active_skill_hypothesis_id)
                elif state.active_skill_name:
                    state.failed_skill_hypotheses.add(state.active_skill_name)
                state.active_skill_queue.clear()
                state.active_skill_name = None
                state.active_skill_hypothesis_id = None
                state.skill_quiescence_count = 0

            if "hazard_feature" in info:
                feat = info["hazard_feature"]
                state.learned_obstacle_features.add(feat)
                state.learned_traversable_features.discard(feat)
                state.learned_target_features.discard(feat)

            hazard_pos = info.get("hazard_pos")
            if hazard_pos is None and avatar_pos and not is_interaction:
                model = state.action_models.get(act_id)
                if model and (model.delta_r != 0 or model.delta_c != 0):
                    hazard_pos = (avatar_pos[0] + model.delta_r, avatar_pos[1] + model.delta_c)

            if hazard_pos and hazard_pos != avatar_pos:
                attempted = [str(a) for a in getattr(state, "action_history", [])[-10:]]
                self.spatial_planner.record_failure(
                    workspace=self.workspace,
                    session_id=source_id,
                    failed_action=act_id,
                    failure_pos=hazard_pos,
                    reason="hazard" if "hazard_feature" in info else "trial_failed",
                    attempted_sequence=attempted,
                )
            state.current_plan.clear()

        # Wire LearningLoopEngine on episode termination
        if feedback.terminated:
            receipt = ExecutionReceipt(
                execution_id=uuid.uuid4().hex[:8],
                success=feedback.success,
                final_snapshot_version=state.step_count,
            )
            self.learning_loop.evaluate_receipt(receipt, user_reward=feedback.reward)

        # Project into core KnowledgeGraph (entities and relations)
        kg = self.get_knowledge_graph(source_id)

        # 1. Motor dynamics entity
        m = state.action_models.get(act_id)
        if m is not None:
            kg.add_entity(
                label=f"action_{act_id}",
                entity_type="motor_action",
                attributes={
                    "action_id": act_id,
                    "delta_r": getattr(m, "delta_r", 0),
                    "delta_c": getattr(m, "delta_c", 0),
                    "confidence": getattr(m, "confidence", 0.5),
                    "probes_tested": getattr(m, "probes_tested", 1),
                },
            )
            kg.add_relation(
                source_label=f"env_{source_id}",
                target_label=f"action_{act_id}",
                relation_type="affords_action",
                weight=getattr(m, "confidence", 0.5),
                metadata={"delta_r": getattr(m, "delta_r", 0), "delta_c": getattr(m, "delta_c", 0)},
            )

        # 2. Avatar entity
        if state.avatar_feature is not None:
            kg.add_entity(
                label=f"feat_{state.avatar_feature}",
                entity_type="perceptual_feature",
                attributes={"feature_id": state.avatar_feature, "role": "avatar"},
            )
            kg.add_relation(
                source_label=f"feat_{state.avatar_feature}",
                target_label="avatar",
                relation_type="is_a",
                weight=1.0,
            )

        # 3. Obstacle feature
        if "collision_feature" in info or (
            "hazard_feature" in info and feedback.terminated and not feedback.success
        ):
            cf = info.get("collision_feature", info.get("hazard_feature"))
            kg.add_entity(
                label=f"feat_{cf}",
                entity_type="perceptual_feature",
                attributes={"feature_id": cf, "role": "obstacle"},
            )
            kg.add_relation(
                source_label=f"feat_{cf}",
                target_label="obstacle",
                relation_type="is_a",
                weight=1.0,
            )
            kg.reinforce(f"feat_{cf}", evidence="obstacle_collision", confidence_boost=0.1)
            kg.add_relation(
                source_label=f"action_{act_id}",
                target_label=f"feat_{cf}",
                relation_type="collides_with",
                weight=1.0,
            )

        # 4. Traversable feature
        if "traversed_feature" in info:
            tf = info["traversed_feature"]
            kg.add_entity(
                label=f"feat_{tf}",
                entity_type="perceptual_feature",
                attributes={"feature_id": tf, "role": "traversable"},
            )
            kg.add_relation(
                source_label=f"feat_{tf}",
                target_label="traversable",
                relation_type="is_a",
                weight=1.0,
            )
            kg.reinforce(f"feat_{tf}", evidence="traversed_pathway", confidence_boost=0.1)

        # 5. Target / goal feature
        if (feedback.success or feedback.reward > 0) and "reached_feature" in info:
            rf = info["reached_feature"]
            kg.add_entity(
                label=f"feat_{rf}",
                entity_type="perceptual_feature",
                attributes={"feature_id": rf, "role": "target"},
            )
            kg.add_relation(
                source_label=f"feat_{rf}",
                target_label="target",
                relation_type="is_a",
                weight=1.0,
            )
            kg.reinforce(f"feat_{rf}", evidence="reached_goal", confidence_boost=0.1)

        # 6. Core LearningLoop reflection on episode termination
        if feedback.terminated:
            try:
                receipt = ExecutionReceipt(
                    execution_id=uuid.uuid4().hex[:8],
                    success=feedback.success,
                    final_snapshot_version=getattr(self.workspace, "current_snapshot_version", 1),
                )
                self.learning_loop.evaluate_receipt(receipt, user_reward=feedback.reward)
            except Exception as e:
                logger.debug("Learning loop evaluation: %s", e)

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(
        self,
        source_id: str = "default",
        retain_memory: bool = True,
        is_retry: bool = False,
    ) -> None:
        """Reset agent state for a new episode.

        If retain_memory is True, learned knowledge persists
        across episodes (transfer learning from experience).
        If is_retry is False (new level/subtask), episode-specific
        spatial visit coordinates are cleared so prior positions
        do not ghost-contaminate the fresh layout.
        """
        state = self.get_state(source_id)

        if retain_memory:
            # Preserve learned knowledge
            saved_obstacles = set(state.learned_obstacle_features)
            saved_targets = set(state.learned_target_features)
            saved_traversable = set(state.learned_traversable_features)
            saved_action_models = dict(state.action_models)
            saved_avatar = getattr(state, "avatar_feature", None)
            saved_step_size = getattr(state, "step_size", 1)
            saved_explored_positions = (
                set(getattr(state, "explored_entity_positions", set())) if is_retry else set()
            )
            saved_explored_ids = (
                set(getattr(state, "explored_entity_ids", set())) if is_retry else set()
            )
            saved_delivered_positions = (
                set(getattr(state, "delivered_positions", set())) if is_retry else set()
            )
            saved_instructions = dict(getattr(state, "domain_instructions", {}))
            saved_mutations = list(getattr(state, "state_mutations", []))
            saved_condition = getattr(state, "active_condition", None)
            saved_effective_features = set(getattr(state, "effective_features", set()))
            saved_quiescent = (
                set(getattr(state, "quiescent_click_targets", set())) if is_retry else set()
            )
            saved_completed = (
                set(getattr(state, "completed_control_targets", set())) if is_retry else set()
            )
            saved_click_usage = dict(getattr(state, "click_target_usage", {})) if is_retry else {}
            saved_entity_usage = dict(getattr(state, "entity_usage", {})) if is_retry else {}
            saved_skills = dict(getattr(state, "learned_skills", {}))
            saved_failed_trajectories = list(getattr(state, "failed_trajectories", []))
            saved_target_sigs = set(getattr(state, "learned_target_signatures", set()))
            saved_obstacle_sigs = set(getattr(state, "learned_obstacle_signatures", set()))
            saved_cargo_sigs = set(getattr(state, "learned_cargo_signatures", set()))
            saved_affordance_rules = dict(getattr(state, "learned_affordance_rules", {}))
            saved_spatiotemporal = getattr(state, "spatiotemporal_skills", None)
            saved_permutation = getattr(state, "permutation_skills", None)
            saved_kinematic = getattr(state, "kinematic_skills", None)
            saved_relational = getattr(state, "relational_skills", None)
            saved_morphology = getattr(state, "morphology_skills", None)
            saved_coupled = getattr(state, "coupled_skills", None)
            saved_incantation = getattr(state, "incantation_skills", None)
            saved_grammar = getattr(state, "grammar_skills", None)
            saved_assembly = getattr(state, "assembly_skills", None)
            saved_reticle = getattr(state, "reticle_skills", None)
            saved_temporal = getattr(state, "temporal_skills", None)
            saved_buoyancy = getattr(state, "buoyancy_skills", None)
            saved_kinetic_coupling = getattr(state, "kinetic_coupling_skills", None)
            saved_grammar_translation = getattr(state, "grammar_translation_skills", None)
            saved_morphological_mutation = getattr(state, "morphological_mutation_skills", None)
            saved_topology = getattr(state, "topology_skills", None)
            saved_laser_routing = getattr(state, "laser_routing_skills", None)
            saved_automaton = getattr(state, "automaton_skills", None)
            saved_canvas = getattr(state, "canvas_skills", None)
            saved_vortex = getattr(state, "vortex_skills", None)
            saved_program = getattr(state, "program_skills", None)
            saved_mirror = getattr(state, "mirror_skills", None)
            saved_failed_hypotheses = set(getattr(state, "failed_skill_hypotheses", set()))
            saved_epistemic_probes = getattr(state, "epistemic_probe_count", 0)
            saved_last_lvl = getattr(state, "last_level_observed", 0)
            saved_curr_lvl = getattr(state, "current_level", 0)
            if is_retry and not getattr(state, "last_attempt_won", False):
                if state.active_skill_hypothesis_id:
                    saved_failed_hypotheses.add(state.active_skill_hypothesis_id)
                elif state.active_skill_name:
                    saved_failed_hypotheses.add(state.active_skill_name)
            if (
                is_retry
                and not getattr(state, "last_attempt_won", False)
                and state.recent_action_history
            ):
                saved_failed_trajectories.append(list(state.recent_action_history))
                if len(saved_failed_trajectories) > 20:
                    saved_failed_trajectories = saved_failed_trajectories[-20:]

            self._source_states[source_id] = AgentState(
                learned_obstacle_features=saved_obstacles,
                learned_target_features=saved_targets,
                learned_traversable_features=saved_traversable,
                action_models=saved_action_models,
                avatar_feature=saved_avatar,
                step_size=saved_step_size,
                explored_entity_positions=saved_explored_positions,
                explored_entity_ids=saved_explored_ids,
                delivered_positions=saved_delivered_positions,
                domain_instructions=saved_instructions,
                state_mutations=saved_mutations,
                active_condition=saved_condition,
                quiescent_click_targets=saved_quiescent,
                effective_features=saved_effective_features,
                completed_control_targets=saved_completed,
                click_target_usage=saved_click_usage,
                entity_usage=saved_entity_usage,
                learned_skills=saved_skills,
                failed_trajectories=saved_failed_trajectories,
                failed_skill_hypotheses=saved_failed_hypotheses,
                epistemic_probe_count=saved_epistemic_probes,
                last_level_observed=saved_last_lvl,
                current_level=saved_curr_lvl,
                learned_target_signatures=saved_target_sigs,
                learned_obstacle_signatures=saved_obstacle_sigs,
                learned_cargo_signatures=saved_cargo_sigs,
                learned_affordance_rules=saved_affordance_rules,
                spatiotemporal_skills=saved_spatiotemporal,
                permutation_skills=saved_permutation,
                kinematic_skills=saved_kinematic,
                relational_skills=saved_relational,
                morphology_skills=saved_morphology,
                coupled_skills=saved_coupled,
                incantation_skills=saved_incantation,
                grammar_skills=saved_grammar,
                assembly_skills=saved_assembly,
                reticle_skills=saved_reticle,
                temporal_skills=saved_temporal,
                buoyancy_skills=saved_buoyancy,
                kinetic_coupling_skills=saved_kinetic_coupling,
                grammar_translation_skills=saved_grammar_translation,
                morphological_mutation_skills=saved_morphological_mutation,
                topology_skills=saved_topology,
                laser_routing_skills=saved_laser_routing,
                automaton_skills=saved_automaton,
                canvas_skills=saved_canvas,
                vortex_skills=saved_vortex,
                program_skills=saved_program,
                mirror_skills=saved_mirror,
            )
        else:
            self._source_states[source_id] = AgentState()

        self._source_entity_graphs.pop(source_id, None)
        self.spatial_planner.reset(is_retry=is_retry)

    # ── Knowledge Graph Persistence ───────────────────────────────────────

    def save_knowledge(
        self,
        path_or_dir: str | Path,
        source_id: str = "default",
    ) -> Path:
        """Persist learned knowledge graph and state for a domain/environment.

        Uses core KnowledgeGraph.save_to_disk() to write the entity-relation
        graph containing motor models, obstacle/target classifications,
        and affordance invariants.

        Args:
            path_or_dir: Target file path (.json) or directory to save inside.
            source_id: Environment/source identifier (e.g. 'cd82', 'arc_agi').

        Returns:
            The Path where knowledge was written.
        """
        p = Path(path_or_dir)
        if p.is_dir() or p.suffix != ".json":
            p.mkdir(parents=True, exist_ok=True)
            target_path = p / f"{source_id}_knowledge_graph.json"
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            target_path = p

        kg = self.get_knowledge_graph(source_id)
        state = self.get_state(source_id)

        # Fallback/sync from active source if source_id state is empty
        if (
            not state.action_models
            and not state.learned_obstacle_features
            and state.avatar_feature is None
        ):
            for fallback_id in ["arc_agi", "default"]:
                if fallback_id in self._source_states and fallback_id != source_id:
                    fb_state = self._source_states[fallback_id]
                    if (
                        fb_state.action_models
                        or fb_state.learned_obstacle_features
                        or fb_state.avatar_feature is not None
                    ):
                        state = fb_state
                        self._source_states[source_id] = fb_state
                        break

        # Project all current state features into KnowledgeGraph before saving
        if state.avatar_feature is not None:
            av_id = int(state.avatar_feature)
            kg.add_entity(
                label=f"feat_{av_id}",
                entity_type="perceptual_feature",
                attributes={"feature_id": av_id, "role": "avatar"},
            )
            kg.add_relation(
                source_label=f"feat_{av_id}",
                target_label="avatar",
                relation_type="is_a",
                weight=1.0,
            )
        for ob in state.learned_obstacle_features:
            ob_id = int(ob)
            if state.avatar_feature is not None and ob_id == int(state.avatar_feature):
                continue
            kg.add_entity(
                label=f"feat_{ob_id}",
                entity_type="perceptual_feature",
                attributes={"feature_id": ob_id, "role": "obstacle"},
            )
            kg.add_relation(
                source_label=f"feat_{ob_id}",
                target_label="obstacle",
                relation_type="is_a",
                weight=1.0,
            )
        for tg in state.learned_target_features:
            tg_id = int(tg)
            if state.avatar_feature is not None and tg_id == int(state.avatar_feature):
                continue
            kg.add_entity(
                label=f"feat_{tg_id}",
                entity_type="perceptual_feature",
                attributes={"feature_id": tg_id, "role": "target"},
            )
            kg.add_relation(
                source_label=f"feat_{tg_id}",
                target_label="target",
                relation_type="is_a",
                weight=1.0,
            )
        for tr in state.learned_traversable_features:
            tr_id = int(tr)
            kg.add_entity(
                label=f"feat_{tr_id}",
                entity_type="perceptual_feature",
                attributes={"feature_id": tr_id, "role": "traversable"},
            )
            kg.add_relation(
                source_label=f"feat_{tr_id}",
                target_label="traversable",
                relation_type="is_a",
                weight=1.0,
            )
        for a_id, m in state.action_models.items():
            conf = float(getattr(m, "confidence", 0.5))
            dr = int(getattr(m, "delta_r", 0))
            dc = int(getattr(m, "delta_c", 0))
            if conf < 0.5 or abs(dr) > 10 or abs(dc) > 10:
                continue
            kg.add_entity(
                label=f"action_{a_id}",
                entity_type="motor_action",
                attributes={
                    "action_id": a_id,
                    "delta_r": dr,
                    "delta_c": dc,
                    "confidence": conf,
                    "probes_tested": getattr(m, "probes_tested", 1),
                },
            )
            kg.add_relation(
                source_label=f"env_{source_id}",
                target_label=f"action_{a_id}",
                relation_type="affords_action",
                weight=getattr(m, "confidence", 0.5),
                metadata={"delta_r": getattr(m, "delta_r", 0), "delta_c": getattr(m, "delta_c", 0)},
            )

        # Project workspace BeliefNodes (learned negative constraints) into KnowledgeGraph
        for b_node in self.workspace.graph.nodes_by_type(HCIRNodeType.BELIEF):
            if isinstance(b_node, BeliefNode) and b_node.properties.get("negative_constraint"):
                pos = b_node.properties.get("position")
                if pos:
                    kg.add_entity(
                        label=f"belief_{pos[0]}_{pos[1]}",
                        entity_type="belief_constraint",
                        attributes={
                            "position": list(pos),
                            "reason": b_node.properties.get("reason", "collision"),
                            "confidence": getattr(b_node, "epistemic_confidence", 0.95),
                        },
                    )

        if state.avatar_feature is not None:
            state.learned_target_features.discard(state.avatar_feature)
            state.learned_obstacle_features.discard(state.avatar_feature)

        # Ensure environment entity reflects latest metadata
        kg.add_entity(
            label=f"env_{source_id}",
            entity_type="environment",
            attributes={
                "source_id": source_id,
                "step_size": state.step_size,
                "step_count": state.step_count,
                "total_reward": state.total_reward,
                "avatar_feature": state.avatar_feature,
                "state_snapshot": state.to_dict(),
            },
        )

        kg.save_to_disk(target_path)
        logger.info(
            "Saved KnowledgeGraph for source '%s' to %s (%d entities, %d relations)",
            source_id,
            target_path,
            kg.entity_count,
            kg.relation_count,
        )
        return target_path

    def load_knowledge(
        self,
        path_or_dir: str | Path,
        source_id: str = "default",
    ) -> bool:
        """Load previously learned knowledge graph from disk into core blackbox.

        Restores:
        - Calibrated action dynamics models (dr, dc, confidence)
        - Discovered obstacle/hazard features
        - Discovered target/goal features
        - Discovered traversable floor features
        - Inferred controllable avatar feature
        - Lattice step size

        Args:
            path_or_dir: File path or directory containing saved knowledge graphs.
            source_id: Environment/source identifier.

        Returns:
            True if knowledge was found and loaded successfully, False otherwise.
        """
        p = Path(path_or_dir)
        if p.is_dir():
            target_path = p / f"{source_id}_knowledge_graph.json"
            if not target_path.exists():
                alt_path = p / f"{source_id}_kg.json"
                if alt_path.exists():
                    target_path = alt_path
                else:
                    return False
        else:
            target_path = p
            if not target_path.exists():
                return False

        try:
            loaded_kg = KnowledgeGraph.load_from_disk(target_path)
        except Exception as e:
            logger.warning("Failed to load KnowledgeGraph from %s: %s", target_path, e)
            return False

        self._knowledge_graphs[source_id] = loaded_kg
        state = self.get_state(source_id)

        # Check for state_snapshot attribute on env entity first
        env_ent = loaded_kg.get_entity(f"env_{source_id}")
        if env_ent and "state_snapshot" in env_ent.attributes:
            hydrated = AgentState.from_dict(env_ent.attributes["state_snapshot"])
            self._source_states[source_id] = hydrated
            self._source_states["arc_agi"] = hydrated
            state = hydrated

        # Clear episode-specific transient caches so fresh layouts are not contaminated
        state.explored_entity_positions.clear()
        state.explored_entity_ids.clear()
        state.delivered_positions.clear()
        state.step_count = 0
        state.total_reward = 0.0
        state.current_plan.clear()
        self.spatial_planner.reset(is_retry=False)

        # Extract entities and relations from KnowledgeGraph to ensure complete sync
        for ent in loaded_kg._entities.values():
            if ent.entity_type == "motor_action":
                act_id = ent.attributes.get("action_id")
                if act_id is not None:
                    state.action_models[int(act_id)] = ActionDynamicsModel(
                        action_id=int(act_id),
                        delta_r=int(ent.attributes.get("delta_r", 0)),
                        delta_c=int(ent.attributes.get("delta_c", 0)),
                        confidence=float(ent.attributes.get("confidence", 0.8)),
                        probes_tested=int(ent.attributes.get("probes_tested", 1)),
                    )

        # Extract obstacle, target, traversable, avatar from relations
        for rel in loaded_kg._relations.values():
            if rel.relation_type == "is_a":
                src_ent = loaded_kg._entities.get(rel.source_id)
                tgt_ent = loaded_kg._entities.get(rel.target_id)
                if src_ent and tgt_ent and "feature_id" in src_ent.attributes:
                    feat = src_ent.attributes["feature_id"]
                    if tgt_ent.label == "obstacle":
                        state.learned_obstacle_features.add(feat)
                    elif tgt_ent.label == "target":
                        state.learned_target_features.add(feat)
                    elif tgt_ent.label == "traversable":
                        state.learned_traversable_features.add(feat)
                    elif tgt_ent.label == "avatar":
                        state.avatar_feature = feat

        # Reconcile: obstacles can never be traversable, and avatar cannot be target or obstacle
        state.learned_traversable_features.difference_update(state.learned_obstacle_features)
        if state.avatar_feature is not None:
            state.learned_target_features.discard(state.avatar_feature)
            state.learned_obstacle_features.discard(state.avatar_feature)

        # Sanitize motor models against corrupted displacements
        for m in state.action_models.values():
            if abs(m.delta_r) > 10 or abs(m.delta_c) > 10:
                m.delta_r = 0
                m.delta_c = 0
                m.confidence = 0.3

        # Restore belief constraints into workspace graph as historical beliefs
        for ent in loaded_kg._entities.values():
            if ent.entity_type == "belief_constraint":
                pos_list = ent.attributes.get("position")
                if pos_list and len(pos_list) == 2:
                    pos = (int(pos_list[0]), int(pos_list[1]))
                    reason = ent.attributes.get("reason", "collision")
                    belief_node = BeliefNode(
                        id=f"belief_barrier_{pos[0]}_{pos[1]}",
                        claim=f"Position {pos} is impassable or causes {reason}",
                        statement=f"Position {pos} is impassable or causes {reason}",
                        epistemic_confidence=float(ent.attributes.get("confidence", 0.95)),
                        belief_type="causal",
                        falsification_status=FalsificationStatus.CORROBORATED,
                        properties={
                            "position": pos,
                            "negative_constraint": True,
                            "reason": reason,
                        },
                        tags=["negative_constraint", reason],
                    )
                    self.workspace.upsert_node(belief_node)

        # Also sync to arc_agi source_id if active
        if source_id != "arc_agi":
            arc_state = self.get_state("arc_agi")
            arc_state.avatar_feature = state.avatar_feature
            arc_state.step_size = state.step_size
            arc_state.learned_obstacle_features.update(state.learned_obstacle_features)
            arc_state.learned_target_features.update(state.learned_target_features)
            arc_state.learned_traversable_features.update(state.learned_traversable_features)
            arc_state.action_models.update(state.action_models)

        # Sync state to workspace and spatial planner
        self.sync_state_to_workspace(source_id)
        if hasattr(self.spatial_planner, "step_size"):
            self.spatial_planner.step_size = state.step_size

        logger.info(
            "Loaded KnowledgeGraph for source '%s' from %s: "
            "avatar=%s, %d action models, %d obstacles, %d targets, %d traversable",
            source_id,
            target_path,
            state.avatar_feature,
            len(state.action_models),
            len(state.learned_obstacle_features),
            len(state.learned_target_features),
            len(state.learned_traversable_features),
        )
        return True

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _plan_step_to_action(
        self,
        step: SequencePlanStep,
        eg: EntityGraph,
        state: AgentState,
        available_actions: list[DriverAction],
        source_id: str = "default",
    ) -> DriverAction:
        """Convert a high-level plan step into a concrete driver action."""
        if not eg.avatar:
            return self._decide_exploratory_action(
                available_actions, state, eg, source_id=source_id
            )

        # Invariant check: If step is DROP but agent is NOT holding anything, discard DROP step
        is_drop = step.action_type in (SpatialActionIntent.DROP, "DROP")
        if is_drop and not getattr(state.carrying, "holding", False):
            if state.current_plan:
                state.current_plan.pop(0)
                if state.current_plan:
                    return self._plan_step_to_action(
                        state.current_plan[0], eg, state, available_actions, source_id=source_id
                    )
            state.current_plan = self.spatial_planner.plan_sequence(eg=eg, workspace=self.workspace)
            if state.current_plan:
                return self._plan_step_to_action(
                    state.current_plan[0], eg, state, available_actions, source_id=source_id
                )

        # Target item existence check for PICKUP steps
        if step.action_type in (SpatialActionIntent.PICKUP, "PICKUP"):
            tgt_ent = eg.entities.get(step.target_entity_id)
            if tgt_ent is None or tgt_ent.is_delivered:
                has_item_near = any(
                    e.role == EntityRole.MANIPULABLE
                    and not e.is_delivered
                    and math.hypot(
                        e.grid_pos[0] - step.target_pos[0], e.grid_pos[1] - step.target_pos[1]
                    )
                    <= max(4.0, float(eg.step_size) * 1.5)
                    for e in eg.entities.values()
                )
                if not has_item_near:
                    # Target item is gone; skip this pickup and its associated drop step
                    if state.current_plan:
                        state.current_plan.pop(0)
                        if state.current_plan and state.current_plan[0].action_type in (
                            SpatialActionIntent.DROP,
                            "DROP",
                        ):
                            state.current_plan.pop(0)
                        if state.current_plan:
                            return self._plan_step_to_action(
                                state.current_plan[0],
                                eg,
                                state,
                                available_actions,
                                source_id=source_id,
                            )
                    state.current_plan = self.spatial_planner.plan_sequence(
                        eg=eg, workspace=self.workspace
                    )
                    if state.current_plan:
                        return self._plan_step_to_action(
                            state.current_plan[0], eg, state, available_actions, source_id=source_id
                        )

        avatar_pos = eg.avatar.grid_pos
        target_pos = step.target_pos
        state.explored_entity_positions.add(avatar_pos)

        # Compute delta to target
        dr = target_pos[0] - avatar_pos[0]
        dc = target_pos[1] - avatar_pos[1]

        step_sz = max(1, eg.step_size)
        if abs(dr) < step_sz and abs(dc) < step_sz:
            if step.target_entity_id:
                state.explored_entity_ids.add(step.target_entity_id)
            state.explored_entity_positions.add(target_pos)

            # If the step requires an interaction/manipulation action at the target entity
            is_navigation = not step.action_type or step.action_type in (
                SpatialActionIntent.NAVIGATE,
                "NAVIGATE",
                "MOVE",
            )
            if not is_navigation:
                # Approach facing orientation check:
                # Only needed for PICKUP of an unheld item when approach_facing is specified
                is_pickup = step.action_type in (SpatialActionIntent.PICKUP, "PICKUP")
                face_dr, face_dc = getattr(step, "approach_facing", None) or (0, 0)
                if (
                    is_pickup
                    and (face_dr != 0 or face_dc != 0)
                    and not getattr(step, "_facing_oriented", False)
                ):
                    face_act_id = self.spatial_planner.get_action_for_delta(
                        face_dr, face_dc, state.action_models
                    )
                    if face_act_id is not None and state.last_action_id != face_act_id:
                        step._facing_oriented = True
                        for a in available_actions:
                            if a.action_id == face_act_id:
                                return a

                action = self._resolve_driver_action(
                    step.action_type, available_actions, state, source_id=source_id
                )
                if action is not None:
                    if state.current_plan:
                        state.current_plan.pop(0)
                    return action

            # Advance plan for completed waypoint/movement
            if state.current_plan:
                state.current_plan.pop(0)
                if state.current_plan:
                    return self._plan_step_to_action(
                        state.current_plan[0], eg, state, available_actions, source_id=source_id
                    )
            return self._decide_exploratory_action(
                available_actions, state, eg, source_id=source_id
            )

        # Geodesic collision-free navigation toward target
        footprint_offsets: list[tuple[int, int]] | None = (
            [(int(round(step.carried_offset[0])), int(round(step.carried_offset[1])))]
            if hasattr(step, "carried_offset") and step.carried_offset != (0, 0)
            else None
        )
        safe_path = self.spatial_planner.compute_safe_path(
            start=avatar_pos,
            goal=target_pos,
            barrier_cells=eg.barriers,
            grid_shape=eg.grid_shape,
            step_size=eg.step_size,
            footprint_offsets=footprint_offsets,
        )

        step_dr, step_dc = dr, dc
        if safe_path and len(safe_path) > 1:
            next_cell = safe_path[1]
            if (
                next_cell not in eg.barriers
                and next_cell not in self.spatial_planner._learned_barriers
            ):
                step_dr = next_cell[0] - avatar_pos[0]
                step_dc = next_cell[1] - avatar_pos[1]
                action_id = self.spatial_planner.get_action_for_delta(
                    step_dr,
                    step_dc,
                    state.action_models,
                    condition=state.get_active_condition(),
                )
                if action_id is not None:
                    for a in available_actions:
                        if a.action_id == action_id:
                            return a

        # Fallback to direct displacement matching ONLY if not moving into a known barrier
        action_id = self.spatial_planner.get_action_for_delta(
            dr, dc, state.action_models, condition=state.get_active_condition()
        )
        if action_id is not None:
            blocked_act_id = (
                state.last_action_id if getattr(state, "last_action_blocked", False) else None
            )
            if action_id != blocked_act_id:
                m = state.action_models.get(action_id)
                if m:
                    m_dr, m_dc = (
                        m.get_displacement(state.get_active_condition())
                        if hasattr(m, "get_displacement")
                        else (getattr(m, "delta_r", 0), getattr(m, "delta_c", 0))
                    )
                    dest = (avatar_pos[0] + m_dr, avatar_pos[1] + m_dc)
                    if (
                        dest not in eg.barriers
                        and dest not in self.spatial_planner._learned_barriers
                    ):
                        for a in available_actions:
                            if a.action_id == action_id:
                                return a

        # Fallback to intelligent exploratory action rather than blind action 0/1
        return self._decide_exploratory_action(available_actions, state, eg, source_id=source_id)

    def _resolve_driver_action(
        self,
        requested_type: str,
        available_actions: list[DriverAction],
        state: AgentState,
        source_id: str = "default",
    ) -> DriverAction | None:
        """Dynamically resolve an action matching requested intent from registered drivers.

        Resolution order:
        1. Registered driver for this source_id (if registered).
        2. Custom action resolver registered for this source_id (if any).
        3. Semantic intent declared by DriverAction.semantic_intent.
        4. In-place / zero-displacement action from learned motor calibration.
        5. Any available action not classified as a directional movement action in learned models.
        """
        # 1. Registered driver resolution
        driver = self._registered_drivers.get(source_id)
        if driver is not None and hasattr(driver, "resolve_action"):
            res = driver.resolve_action(requested_type, available_actions, context={"state": state})
            if res is not None:
                return res

        # 2. Custom driver-registered resolver if provided
        resolver = self._action_resolvers.get(source_id)
        if resolver is not None:
            res = resolver(requested_type, available_actions, state)
            if res is not None:
                return res

        normalized_req = str(requested_type).strip().lower()

        # 3. Semantic intent declared by driver actions
        for a in available_actions:
            if a.semantic_intent:
                norm_intent = a.semantic_intent.strip().lower()
                if (
                    norm_intent == normalized_req
                    or normalized_req in norm_intent
                    or norm_intent in normalized_req
                ):
                    return a

        # 4. Learned zero-displacement in-place action from motor models
        for a in available_actions:
            m = state.action_models.get(a.action_id)
            if (
                m is not None
                and m.delta_r == 0
                and m.delta_c == 0
                and getattr(m, "probes_tested", 0) > 0
            ):
                return a

        # 5. Any available action not classified as a directional movement action in learned models
        directional_ids = {
            act_id for act_id, m in state.action_models.items() if m.delta_r != 0 or m.delta_c != 0
        }
        for a in available_actions:
            if a.action_id not in directional_ids:
                return a

        return None
