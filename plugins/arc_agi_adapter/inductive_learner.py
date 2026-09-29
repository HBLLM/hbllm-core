"""Inductive HCIR Learner for ARC-AGI-3.

Learns puzzle dynamics, state-space typologies, and goal invariants purely from:
1. Raw pixel layouts: 2D integer grids (H x W)
2. Available action lists: A ⊆ {1, 2, 3, 4, 5, 6, 7}
3. Causal trial-and-error observations: Δt = Grid(t) ⊕ Grid(t-1)

Learned knowledge (action grammar, transition models, controllable entity signatures,
and goal predicates) persists across levels of an environment, enabling zero-shot
or few-shot transfer to subsequent levels with new findings.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from hbllm.hcir.subgoal_decomposer import HCIRSkill, HierarchicalGoalDecomposer
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel
from plugins.arc_agi_adapter.arc_skills.kinematic_arm_linkage import KinematicLinkageSolver

from .arc_spatial_agent import (
    ARC3SpatialCognitiveAgent,
)

logger = logging.getLogger(__name__)

# Extracted solver, perception, navigation, and knowledge modules (re-exported for backward compatibility)
from plugins.arc_agi_adapter.arc_solvers.game_solvers import (
    GravitySpillingPlatformSolver,
    LightsOutSolver,
    MirroredConvergenceSolver,
    PegSolitaireSolver,
    TrackMazeSolver,
    VortexAttractorSolver,
)
from plugins.arc_agi_adapter.arc_solvers.knowledge_base import (
    ActionAffordance,
    CausalAffordanceEngine,
    CausalHypothesis,
    ControllableSignature,
    CrossLevelKnowledgeBase,
    DynamicPermutationSolver,
    GoalHypothesis,
    GoalStateInductor,
    ObjectInteractionRecipe,
    TemporalHazardTracker,
    TrialFeedbackMemory,
    TrialOutcome,
)
from plugins.arc_agi_adapter.arc_solvers.spatial_navigation import (
    DynamicSpatialNavigator,
    RoomDoor,
    RoomTopologyExtractor,
    SpatialResourceNavigator,
    SpatiotemporalNavigator,
)
from plugins.arc_agi_adapter.arc_solvers.visual_analysis import (
    CanvasRegion,
    CoupledMIMOIdentifier,
    DiffType,
    DynamicCanvasMatcher,
    FrameDiff,
    FrameDiffAnalyzer,
    PuzzleTypology,
    VisualCanvasMatcher,
    VisualEntity,
    VisualSymmetryAnalyzer,
    VisualTopologyExtractor,
)

__all__ = [
    "ActionAffordance",
    "CanvasRegion",
    "CausalAffordanceEngine",
    "CausalHypothesis",
    "ControllableSignature",
    "CoupledMIMOIdentifier",
    "CrossLevelKnowledgeBase",
    "DiffType",
    "DynamicCanvasMatcher",
    "DynamicPermutationSolver",
    "DynamicSpatialNavigator",
    "FrameDiff",
    "FrameDiffAnalyzer",
    "GoalHypothesis",
    "GoalStateInductor",
    "GravitySpillingPlatformSolver",
    "InductiveARC3BenchmarkRunner",
    "InductiveHCIRAgent",
    "LightsOutSolver",
    "MirroredConvergenceSolver",
    "ObjectInteractionRecipe",
    "PegSolitaireSolver",
    "PuzzleTypology",
    "RoomDoor",
    "RoomTopologyExtractor",
    "SpatialResourceNavigator",
    "SpatiotemporalNavigator",
    "TemporalHazardTracker",
    "TrackMazeSolver",
    "TrialFeedbackMemory",
    "TrialOutcome",
    "VisualCanvasMatcher",
    "VisualEntity",
    "VisualSymmetryAnalyzer",
    "VisualTopologyExtractor",
    "VortexAttractorSolver",
]


# Graceful import of official arcengine
try:
    from arcengine import GameAction as ARCGameAction
    from arcengine import GameState as ARCGameState
except ImportError:

    class _FallbackARCGameAction(Enum):
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

    class _FallbackARCGameState(Enum):
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"

    ARCGameAction = _FallbackARCGameAction  # type: ignore[assignment, misc]
    ARCGameState = _FallbackARCGameState  # type: ignore[assignment, misc]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Visual Difference Analysis & State Typology
# ─────────────────────────────────────────────────────────────────────────────


class InductiveHCIRAgent:
    """Trial-and-error inductive learner for ARC-AGI-3.

    Interacts solely via pixel grids and action lists. Induces state models and
    goal predicates on Level 1, persisting accumulated knowledge across levels.
    """

    def __init__(self, disable_archetypes: bool = False) -> None:
        self.disable_archetypes: bool = disable_archetypes
        self.autonomous_engine: AutonomousEpistemicEngine = AutonomousEpistemicEngine()
        self.knowledge_base: CrossLevelKnowledgeBase = CrossLevelKnowledgeBase()
        self.spatial_cognitive_agent: ARC3SpatialCognitiveAgent = ARC3SpatialCognitiveAgent()
        self.spatial_cognitive_agent.knowledge_base = self.knowledge_base
        self.hcir_agent: ARC3SpatialCognitiveAgent = self.spatial_cognitive_agent
        self.canvas_matcher: VisualCanvasMatcher = VisualCanvasMatcher()
        self.spatial_navigator: SpatialResourceNavigator = SpatialResourceNavigator()
        self.vortex_solver: VortexAttractorSolver = VortexAttractorSolver()
        self.peg_solver: PegSolitaireSolver = PegSolitaireSolver()
        self.track_maze_solver: TrackMazeSolver = TrackMazeSolver()
        self.lights_out_solver: LightsOutSolver = LightsOutSolver()
        self.mirrored_convergence_solver: MirroredConvergenceSolver = MirroredConvergenceSolver()
        self.gravity_spill_solver: GravitySpillingPlatformSolver = GravitySpillingPlatformSolver()
        self.linkage_solver: KinematicLinkageSolver = KinematicLinkageSolver()
        self.topology_extractor: VisualTopologyExtractor = VisualTopologyExtractor()
        self.dynamic_navigator: DynamicSpatialNavigator = DynamicSpatialNavigator()
        self.dynamic_canvas_matcher: DynamicCanvasMatcher = DynamicCanvasMatcher()
        self.permutation_solver: DynamicPermutationSolver = DynamicPermutationSolver()
        self.symmetry_analyzer: VisualSymmetryAnalyzer = VisualSymmetryAnalyzer()
        self.hazard_tracker: TemporalHazardTracker = TemporalHazardTracker()
        self.room_extractor: RoomTopologyExtractor = RoomTopologyExtractor()
        self.spatiotemporal_navigator: SpatiotemporalNavigator = SpatiotemporalNavigator()
        self.causal_engine: CausalAffordanceEngine = CausalAffordanceEngine()
        self.active_solver_name: str | None = None
        self.prev_grid: np.ndarray | None = None
        self.last_action: int | None = None
        self.current_level: int = 0
        self.last_action_data: dict[str, int] | None = None
        self.current_actor_pos: tuple[int, int] | None = None
        self.current_target_pos: tuple[int, int] | None = None
        # Click affordance tracking
        self._click_targets: list[tuple[int, int]] = []
        self._click_index: int = 0
        self._clicked_positions: set[tuple[int, int]] = set()
        self._effective_colors: set[int] = set()
        self._quiescent_targets: set[tuple[int, int]] = set()
        self._completed_controls: set[tuple[int, int]] = set()
        self._target_usage: dict[tuple[int, int], int] = {}
        self._entity_usage: dict[int, int] = {}
        self._click_visited_states: set[bytes] = set()
        self._last_click_target: tuple[int, int, int, int] | None = None
        self._consecutive_effective_clicks: int = 0
        self._prev_min_dist: dict[int, int] = {}
        # Trial-and-error components
        self.trial_memory: TrialFeedbackMemory = TrialFeedbackMemory()
        self.spatial_cognitive_agent.trial_memory = self.trial_memory
        self.goal_inductor: GoalStateInductor = GoalStateInductor()
        self.step_counter: int = 0
        self.epistemic_probe_budget: int = 20  # adaptive exploration budget
        # Loop detection: track recent positions to detect navigation circles
        self.visited_positions: deque[tuple[int, int]] = deque(maxlen=30)
        self.visit_counts: dict[tuple[int, int], int] = {}
        # Solver stagnation tracking for mid-episode solver switching
        self.solver_stagnation_counter: int = 0
        self.solver_last_grid_hash: int = 0
        self.solver_switch_count: int = 0
        self.solver_max_switches: int = 3
        self._tried_stagnation_actions: set[int] = set()
        # Pre-solution goal detection cache
        self._inferred_goal_zone: tuple[int, int, int, int] | None = None
        self._inferred_goal_type: str | None = None
        self._initial_grid_analyzed: bool = False

    def reset_episode(self, retain_dynamics: bool = False, is_retry: bool = False) -> None:
        """Reset internal step state while preserving cross-level knowledge."""
        self.prev_grid = None
        self.step_counter = 0
        self.trial_memory.reset_episode()
        self.goal_inductor.reset_episode()
        self.visited_positions.clear()
        self.visit_counts.clear()
        self.last_action = None
        self.last_action_data = None
        self.last_frame_state = None
        self._click_targets = []
        self._click_index = 0
        self._clicked_positions.clear()
        if not is_retry:
            self._quiescent_targets.clear()
            self._target_usage.clear()
            self._entity_usage.clear()
        self._completed_controls.clear()
        self._click_visited_states.clear()
        self._last_click_target = None
        self._consecutive_effective_clicks = 0
        self._prev_min_dist.clear()
        self.canvas_matcher.reset_episode()
        self.spatial_navigator.reset_episode()
        self.vortex_solver.reset_episode()
        self.peg_solver.reset_episode()
        self.track_maze_solver.reset_episode()
        self.lights_out_solver.reset_episode()
        self.mirrored_convergence_solver.reset_episode()
        self.gravity_spill_solver.reset_episode()
        self.linkage_solver.reset_episode()
        self.hazard_tracker.reset_episode()
        self.causal_engine.reset_episode()
        self.autonomous_engine.reset_episode(retain_dynamics=retain_dynamics)
        self.active_solver_name = None
        self.solver_stagnation_counter = 0
        self.solver_last_grid_hash = 0
        self.solver_switch_count = 0
        self._declarative_action_queue = []
        self._declarative_failed_skills: set[str] = set()
        self._declarative_plan_grid_hash = None
        self._initial_grid_analyzed = False
        self._inferred_goal_zone = None
        self._inferred_goal_type = None
        if not retain_dynamics:
            self._effective_colors.clear()
            self._quiescent_targets.clear()
            self._target_usage.clear()
            self._entity_usage.clear()
            self.knowledge_base = CrossLevelKnowledgeBase()
            self.hcir_agent.reset_episode(retain_dynamics=False, is_retry=False)
            self.spatial_cognitive_agent.current_level = 0
            self.spatial_cognitive_agent.reset_episode(retain_dynamics=False, is_retry=False)
            self.spatial_cognitive_agent.knowledge_base = self.knowledge_base
            self.spatial_cognitive_agent.trial_memory = self.trial_memory
            self.current_level = 0
        else:
            if not is_retry:
                self.current_level += 1
            # Delegate directly to core CognitiveBlackbox via ARC3SpatialCognitiveAgent
            self.spatial_cognitive_agent.current_level = self.current_level
            self.spatial_cognitive_agent.reset_episode(retain_dynamics=True, is_retry=is_retry)
            self.hcir_agent = self.spatial_cognitive_agent

            # Sync any seeded legacy knowledge base models to core blackbox for backward compatibility
            state = self.spatial_cognitive_agent.blackbox.get_state("arc_agi")
            if (
                self.knowledge_base.controllable_signature.color is not None
                and state.avatar_feature is None
            ):
                self.spatial_cognitive_agent.avatar_color = (
                    self.knowledge_base.controllable_signature.color
                )
            for a_id, aff in self.knowledge_base.action_affordances.items():
                if a_id not in state.action_models:
                    from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

                    state.action_models[a_id] = ActionDynamicsModel(
                        action_id=a_id,
                        delta_r=aff.delta_r,
                        delta_c=aff.delta_c,
                        confidence=aff.confidence,
                        probes_tested=aff.times_tested,
                    )

            # Adaptive exploration budget: scale with action space, reduce on later levels
            if len(state.action_models) >= 4 and all(
                getattr(m, "confidence", 0) >= 0.6 for m in state.action_models.values()
            ):
                # Fully grounded motor models: minimal probing for new level variations
                self.epistemic_probe_budget = max(3, 8 - self.current_level * 2)
            else:
                # Partially grounded: scale budget with unknown actions
                unknown_actions = max(1, 7 - len(state.action_models))
                self.epistemic_probe_budget = max(6, unknown_actions * 4)

        # Reset click-game stagnation tracker for each new level
        self._core_click_stagnation = 0
        self._core_click_gave_up = False

    def save_knowledge(
        self,
        knowledge_dir: Path | str,
        game_id: str,
        levels_completed: int = 0,
    ) -> Path:
        """Persist accumulated knowledge to disk via core KnowledgeGraph."""
        return self.spatial_cognitive_agent.save_knowledge(
            knowledge_dir, game_id=game_id, levels_completed=levels_completed
        )

    def load_knowledge(self, knowledge_dir: Path | str, game_id: str) -> bool:
        """Load persistent KnowledgeGraph from disk using core KnowledgeGraph."""
        loaded = self.spatial_cognitive_agent.load_knowledge(knowledge_dir, game_id=game_id)
        if loaded:
            state = self.spatial_cognitive_agent.blackbox.get_state("arc_agi")
            # Always maintain an epistemic exploration floor!
            # Never set probe budget to 0, because each level must empirically
            # verify motor models and ground the avatar sprite.
            if len(state.action_models) >= 4 and all(
                getattr(m, "confidence", 0) >= 0.7 for m in state.action_models.values()
            ):
                self.epistemic_probe_budget = max(4, 8 - self.current_level * 2)
            else:
                unknown_actions = max(1, 7 - len(state.action_models))
                self.epistemic_probe_budget = max(6, unknown_actions * 4)
            logger.info(
                "[CORE KNOWLEDGE GRAPH] Hydrated CognitiveBlackbox for game '%s': avatar=%s, %d action models, %d obstacles, %d targets (probe budget=%d)",
                game_id,
                state.avatar_feature,
                len(state.action_models),
                len(state.learned_obstacle_features),
                len(state.learned_target_features),
                self.epistemic_probe_budget,
            )
        return loaded

    def _dispatch_active_solver(
        self, curr_grid: np.ndarray, available_actions: list[int]
    ) -> tuple[int, float]:
        # Mid-episode stagnation detection: if solver makes no visual progress, switch
        curr_hash = hash(curr_grid.tobytes())
        if curr_hash == self.solver_last_grid_hash:
            self.solver_stagnation_counter += 1
        else:
            self.solver_stagnation_counter = max(0, self.solver_stagnation_counter - 1)
            self.solver_last_grid_hash = curr_hash

        if (
            self.solver_stagnation_counter >= 12
            and self.solver_switch_count < self.solver_max_switches
        ):
            logger.info(
                "Solver '%s' stagnant for %d steps — forcing re-evaluation (switch %d/%d)",
                self.active_solver_name,
                self.solver_stagnation_counter,
                self.solver_switch_count + 1,
                self.solver_max_switches,
            )
            self.active_solver_name = None
            self.solver_stagnation_counter = 0
            self.solver_switch_count += 1
            return self._plan_hcir_step(curr_grid, available_actions)

        name = self.active_solver_name
        action_data: dict[str, int] | None = None
        action: int
        conf: float

        if name and name.startswith("declarative:"):
            # Declarative skill continuation — drain queued actions or re-plan
            if hasattr(self, "_declarative_action_queue") and self._declarative_action_queue:
                action, action_data = self._declarative_action_queue.pop(0)
                conf = 0.9
            else:
                # Queue exhausted — check if plan was effective
                skill_name = name.split(":", 1)[1]
                curr_hash = hash(curr_grid.tobytes())
                plan_hash = getattr(self, "_declarative_plan_grid_hash", None)
                if plan_hash is not None and curr_hash == plan_hash:
                    # Grid unchanged after full plan — skill's plan was ineffective
                    if not hasattr(self, "_declarative_failed_skills"):
                        self._declarative_failed_skills = set()
                    self._declarative_failed_skills.add(skill_name)
                    logger.info(
                        "Declarative skill '%s' blacklisted — plan produced no grid change",
                        skill_name,
                    )
                # Re-evaluate via _plan_hcir_step
                self.active_solver_name = None
                self.solver_stagnation_counter = 0
                return self._plan_hcir_step(curr_grid, available_actions)
        elif name == "spatial_cooperative":
            self.knowledge_base.puzzle_typology = PuzzleTypology.SPATIAL_NAVIGATION
            if self.prev_grid is not None and self.last_action is not None:
                self.spatial_cognitive_agent.update_causal_dynamics(
                    self.last_action,
                    self.prev_grid,
                    curr_grid,
                    won=(getattr(self, "last_frame_state", None) == "WIN"),
                )
            action, conf = self.spatial_cognitive_agent.plan_next_action(
                curr_grid, available_actions, level=self.current_level
            )
            action_data = getattr(self.spatial_cognitive_agent, "last_action_data", None)

            if self.spatial_cognitive_agent.avatar_color is not None:
                self.knowledge_base.controllable_signature.color = (
                    self.spatial_cognitive_agent.avatar_color
                )
            for a_id, m in self.spatial_cognitive_agent.action_models.items():
                self.knowledge_base.action_affordances[a_id] = ActionAffordance(
                    action_id=a_id,
                    delta_r=m.delta_r,
                    delta_c=m.delta_c,
                    confidence=m.confidence,
                    times_tested=m.probes_tested,
                )
            if getattr(self.spatial_cognitive_agent, "avatar_grid_pos", None):
                self.current_actor_pos = self.spatial_cognitive_agent.avatar_grid_pos
            elif self.spatial_cognitive_agent.avatar_centroid:
                self.current_actor_pos = (
                    int(round(self.spatial_cognitive_agent.avatar_centroid[0])),
                    int(round(self.spatial_cognitive_agent.avatar_centroid[1])),
                )
            if self.spatial_cognitive_agent.goal_centroid:
                self.current_target_pos = (
                    int(round(self.spatial_cognitive_agent.goal_centroid[0])),
                    int(round(self.spatial_cognitive_agent.goal_centroid[1])),
                )
            else:
                self.current_target_pos = None
        else:
            action, conf = 1, 0.50

        self.last_action_data = action_data
        self.prev_grid = curr_grid.copy()
        self.last_action = action
        return action, conf

    def _detect_goal_from_structure(self, grid: np.ndarray) -> None:
        """Analyze initial grid to infer goal zones and objectives before solving.

        Detects common goal signatures:
        1. Bordered target zones (rectangular regions of uniform color)
        2. Exit markers (small entities at grid edges)
        3. Template-canvas pairs (for pattern matching puzzles)
        """
        if self._initial_grid_analyzed:
            return
        self._initial_grid_analyzed = True

        H, W = grid.shape
        bg = int(np.bincount(grid.flatten()).argmax())
        entities = VisualTopologyExtractor.extract_entities(grid, ignore_colors={bg, 0})

        # 1. Detect target zones: medium-sized non-border rectangular regions
        zone_candidates: list[tuple[int, Any]] = []
        for e in entities:
            min_r, max_r, min_c, max_c = e.bounding_box
            width = max_c - min_c + 1
            height = max_r - min_r + 1
            if 4 <= width <= 30 and 4 <= height <= 30 and e.size >= 10 and not e.is_border:
                zone_candidates.append((e.size, e))

        if zone_candidates:
            zone_candidates.sort(key=lambda x: x[0], reverse=True)
            best_ent = zone_candidates[0][1]
            self._inferred_goal_zone = best_ent.bounding_box
            self._inferred_goal_type = "target_zone"
            logger.info(
                "Pre-goal detected: target zone at %s color=%d area=%d",
                best_ent.bounding_box,
                best_ent.color,
                best_ent.size,
            )
            return

        # 2. Detect exit markers: small entities at edges
        for e in entities:
            if e.size <= 16 and e.is_border and e.color not in {bg, 0}:
                self._inferred_goal_zone = e.bounding_box
                self._inferred_goal_type = "exit_marker"
                logger.info(
                    "Pre-goal detected: exit marker at %s color=%d",
                    e.bounding_box,
                    e.color,
                )
                return

        # 3. Detect symmetry-based goals
        sym_scores = VisualSymmetryAnalyzer.compute_symmetry_scores(grid)
        max_sym_name, max_sym_val = max(sym_scores.items(), key=lambda item: item[1])
        if max_sym_val > 0.85:
            self._inferred_goal_type = "symmetry_completion"
            logger.info(
                "Pre-goal detected: symmetry completion (%s=%.2f)",
                max_sym_name,
                max_sym_val,
            )

    def _plan_hcir_step(
        self, curr_grid: np.ndarray, available_actions: list[int]
    ) -> tuple[int, float]:
        """Execute autonomous cognitive reasoning via the HCIR Engine."""
        self.step_counter += 1

        # 0. Drain queued actions from a declarative skill's multi-step plan
        if hasattr(self, "_declarative_action_queue") and self._declarative_action_queue:
            action, action_data = self._declarative_action_queue.pop(0)
            self.last_action_data = action_data
            self.prev_grid = curr_grid.copy()
            self.last_action = action
            return action, 0.9

        # 1. Assimilate feedback from previous action if available
        if self.prev_grid is not None and self.last_action is not None:
            diff = FrameDiffAnalyzer.analyze(self.prev_grid, self.last_action, curr_grid)
            self.knowledge_base.register_observation(
                self.prev_grid, self.last_action, curr_grid, diff
            )
            self.hcir_agent.update_causal_dynamics(
                self.last_action,
                self.prev_grid,
                curr_grid,
                won=(getattr(self, "last_frame_state", None) == "WIN"),
            )

            # Record trial-and-error outcome
            pos = self.current_actor_pos or (0, 0)
            barrier_dir = None
            if diff.diff_type == DiffType.NO_CHANGE:
                # Infer blocked direction from the action's expected delta
                aff = self.knowledge_base.action_affordances.get(self.last_action)
                if aff and (aff.delta_r != 0 or aff.delta_c != 0):
                    barrier_dir = (aff.delta_r, aff.delta_c)

            outcome = TrialOutcome(
                action=self.last_action,
                position=pos,
                succeeded=(diff.diff_type != DiffType.NO_CHANGE),
                diff_type=diff.diff_type,
                barrier_direction=barrier_dir,
                objects_affected=diff.changed_pixel_count,
                item_gained=(
                    diff.diff_type == DiffType.TRANSLATION
                    and diff.moved_object_size > 0
                    and diff.changed_pixel_count > diff.moved_object_size * 2
                ),
                item_lost=(
                    diff.diff_type == DiffType.TRANSLATION
                    and diff.moved_object_size > 0
                    and diff.changed_pixel_count > diff.moved_object_size * 3
                ),
            )
            self.trial_memory.record(outcome)

        # Feed frame to goal inductor for future hypothesis generation
        self.goal_inductor.observe_frame(curr_grid)

        # Pre-solution goal detection: analyze initial grid structure
        self._detect_goal_from_structure(curr_grid)

        # Feed inferred goal zone to HCIR agent if available
        if self._inferred_goal_zone and not self.current_target_pos:
            zone = self._inferred_goal_zone
            center_r = (zone[0] + zone[1]) // 2
            center_c = (zone[2] + zone[3]) // 2
            self.current_target_pos = (center_r, center_c)

        # 2. Synchronize controllable signature, barrier colors, and affordances
        if self.knowledge_base.controllable_signature.color is not None:
            self.hcir_agent.avatar_color = self.knowledge_base.controllable_signature.color

        for a, aff in self.knowledge_base.action_affordances.items():
            if a not in self.hcir_agent.action_models and aff.confidence >= 0.4:
                self.hcir_agent.action_models[a] = ActionDynamicsModel(
                    action_id=a,
                    delta_r=aff.delta_r,
                    delta_c=aff.delta_c,
                    confidence=aff.confidence,
                    probes_tested=aff.times_tested,
                )

        # Transfer learned barrier colors to HCIR agent
        if self.knowledge_base.barrier_colors:
            if self.hcir_agent.known_barriers is None:
                self.hcir_agent.known_barriers = np.zeros(curr_grid.shape, dtype=bool)
            for color in self.knowledge_base.barrier_colors:
                self.hcir_agent.known_barriers[curr_grid == color] = True

        # 3. Multi-probe exploration phase — test each action 3+ times at varied positions
        # to learn state-dependent effects reliably (wall vs open, near object vs not)
        min_probes_per_action = 3
        if self.current_level == 0 or not self.knowledge_base.is_world_model_grounded(
            available_actions
        ):
            under_probed_actions = [
                a
                for a in available_actions
                if a not in self.knowledge_base.action_affordances
                or self.knowledge_base.action_affordances[a].times_tested < min_probes_per_action
            ]
            if under_probed_actions and self.step_counter <= self.epistemic_probe_budget:
                # Phase 1: Prioritize directional movement actions (1-4) to ground avatar
                directional = [a for a in under_probed_actions if a in [1, 2, 3, 4]]
                interaction = [a for a in under_probed_actions if a in [5, 6, 7]]

                if directional:
                    # Rotate through directional actions for diverse probing
                    probe_action = directional[self.step_counter % len(directional)]
                elif interaction:
                    probe_action = interaction[0]
                else:
                    probe_action = under_probed_actions[0]

                logger.debug(
                    f"Multi-probe exploration (step {self.step_counter}): "
                    f"testing action {probe_action}, under-probed={under_probed_actions}"
                )
                self.prev_grid = curr_grid.copy()
                self.last_action = probe_action

                # For click-type actions (6, 7), vary click position across probes
                if probe_action >= 6:
                    probe_count = self.knowledge_base.action_affordances.get(
                        probe_action, ActionAffordance(probe_action)
                    ).times_tested
                    H, W = curr_grid.shape
                    if self.current_actor_pos and probe_count == 0:
                        # First probe: click at avatar position
                        self.last_action_data = {
                            "x": self.current_actor_pos[1],
                            "y": self.current_actor_pos[0],
                        }
                    elif probe_count == 1:
                        # Second probe: click at grid center
                        self.last_action_data = {"x": W // 2, "y": H // 2}
                    else:
                        # Subsequent probes: click on non-background entities
                        bg = int(np.bincount(curr_grid.flatten()).argmax())
                        non_bg = np.argwhere((curr_grid != bg) & (curr_grid != 0))
                        if len(non_bg) > 0:
                            idx = probe_count % len(non_bg)
                            self.last_action_data = {
                                "x": int(non_bg[idx, 1]),
                                "y": int(non_bg[idx, 0]),
                            }
                        else:
                            self.last_action_data = {"x": W // 2, "y": H // 2}
                else:
                    self.last_action_data = None

                return probe_action, 0.4

        # 4. Loop detection — detect and break navigation circles
        # Only trigger when: no active goal target AND position visited 6+ times
        # This avoids breaking back-and-forth delivery patterns (wa30, ls20)
        if self.current_actor_pos:
            pos = self.current_actor_pos
            self.visited_positions.append(pos)
            self.visit_counts[pos] = self.visit_counts.get(pos, 0) + 1

            no_active_target = self.current_target_pos is None
            if no_active_target and self.visit_counts[pos] >= 6 and self.step_counter > 20:
                # Find movement affordances and pick the least-visited direction
                movement_actions = []
                for a, aff in self.knowledge_base.action_affordances.items():
                    if (
                        a in available_actions
                        and (aff.delta_r != 0 or aff.delta_c != 0)
                        and aff.confidence >= 0.3
                    ):
                        target_pos = (pos[0] + aff.delta_r, pos[1] + aff.delta_c)
                        visits = self.visit_counts.get(target_pos, 0)
                        movement_actions.append((a, target_pos, visits))

                if movement_actions:
                    movement_actions.sort(key=lambda x: x[2])
                    best_action = movement_actions[0][0]
                    logger.debug(
                        f"Loop break at {pos} (visited {self.visit_counts[pos]}x): "
                        f"choosing action {best_action} toward {movement_actions[0][1]}"
                    )
                    self.prev_grid = curr_grid.copy()
                    self.last_action = best_action
                    self.last_action_data = None
                    return best_action, 0.35

        # 5. Stuck detection — epistemic probing fallback (lowered threshold for faster recovery)
        if self.trial_memory.is_stuck(threshold=5):
            if self.current_actor_pos:
                blocked = self.trial_memory.get_blocked_actions_at(self.current_actor_pos)
                unblocked = [a for a in available_actions if a not in blocked]
                if unblocked:
                    import random

                    # Priority 1: Try under-tested interaction actions (5, 6, 7)
                    interaction_candidates = [
                        a
                        for a in unblocked
                        if a >= 5
                        and self.knowledge_base.action_affordances.get(
                            a, ActionAffordance(a)
                        ).times_tested
                        < 5
                    ]
                    if interaction_candidates:
                        probe_action = interaction_candidates[0]
                    else:
                        # Priority 2: Movement toward least-visited cell
                        directional = [a for a in unblocked if a <= 4]
                        if directional and self.visit_counts:
                            scored = []
                            for a in directional:
                                aff = self.knowledge_base.action_affordances.get(a)
                                if aff and (aff.delta_r != 0 or aff.delta_c != 0):
                                    pos = self.current_actor_pos
                                    target = (pos[0] + aff.delta_r, pos[1] + aff.delta_c)
                                    visits = self.visit_counts.get(target, 0)
                                    scored.append((a, visits))
                            if scored:
                                scored.sort(key=lambda x: x[1])
                                probe_action = scored[0][0]
                            else:
                                probe_action = random.choice(directional)
                        else:
                            probe_action = random.choice(unblocked)

                    self.prev_grid = curr_grid.copy()
                    self.last_action = probe_action

                    # For click probes, target diverse entity positions
                    if probe_action >= 6:
                        H, W = curr_grid.shape
                        bg = int(np.bincount(curr_grid.flatten()).argmax())
                        non_bg = np.argwhere((curr_grid != bg) & (curr_grid != 0))
                        if len(non_bg) > 0:
                            idx = self.step_counter % len(non_bg)
                            self.last_action_data = {
                                "x": int(non_bg[idx, 1]),
                                "y": int(non_bg[idx, 0]),
                            }
                        else:
                            self.last_action_data = {"x": W // 2, "y": H // 2}
                    else:
                        self.last_action_data = None

                    self.trial_memory.consecutive_no_change = 0
                    logger.debug(
                        f"Epistemic probe: stuck at {self.current_actor_pos}, "
                        f"trying action {probe_action}"
                    )
                    return probe_action, 0.3

        # 6. Recipe synchronization — ensure learned recipes are active in HCIR agent
        if self.knowledge_base.levels_solved > 0:
            if self.knowledge_base.signature_recipes:
                for sig_key, recipe in self.knowledge_base.signature_recipes.items():
                    if recipe.outcome == "pickup":
                        self.hcir_agent.learned_cargo_signatures.add(sig_key)
                    elif recipe.outcome in ("goal", "target"):
                        self.hcir_agent.learned_target_signatures.add(sig_key)

            if self.knowledge_base.object_recipes:
                for c, recipe in self.knowledge_base.object_recipes.items():
                    if recipe.outcome == "pickup":
                        item_colors = self.hcir_agent.learned_item_colors
                        if c not in item_colors:
                            if isinstance(item_colors, set):
                                item_colors.add(c)
                            elif isinstance(item_colors, dict):
                                item_colors[c] = {"action": recipe.interaction_action}
                        if (
                            recipe.delivery_zone_bounds
                            and not self.hcir_agent.learned_receptacle_bounds
                        ):
                            self.hcir_agent.learned_receptacle_bounds = recipe.delivery_zone_bounds
                        if recipe.delivery_zone_color is not None:
                            self.hcir_agent.learned_receptacle_colors.add(
                                recipe.delivery_zone_color
                            )

        if self.knowledge_base.skills and self.hcir_agent.primary_goal_node:
            HierarchicalGoalDecomposer.decompose_with_skills(
                self.hcir_agent.workspace,
                self.hcir_agent.primary_goal_node,
                self.knowledge_base.skills,
                {"actor_pos": self.current_actor_pos},
            )

        # 7. Plan next action using HCIR engine
        action, conf = self.hcir_agent.plan_next_action(curr_grid, available_actions)
        self.last_action_data = self.hcir_agent.last_action_data

        if getattr(self.hcir_agent, "avatar_grid_pos", None):
            self.current_actor_pos = self.hcir_agent.avatar_grid_pos
        elif self.hcir_agent.avatar_centroid:
            self.current_actor_pos = (
                int(round(self.hcir_agent.avatar_centroid[0])),
                int(round(self.hcir_agent.avatar_centroid[1])),
            )

        if self.hcir_agent.goal_centroid:
            self.current_target_pos = (
                int(round(self.hcir_agent.goal_centroid[0])),
                int(round(self.hcir_agent.goal_centroid[1])),
            )
        elif self.hcir_agent.primary_goal_node:
            tp = self.hcir_agent.primary_goal_node.properties.get("target_position")
            if tp:
                self.current_target_pos = (int(tp[0]), int(tp[1]))
        else:
            self.current_target_pos = None

        self.knowledge_base.barrier_colors.update(self.hcir_agent.learned_barrier_colors)
        self.knowledge_base.walkable_colors.update(self.hcir_agent.learned_walkable_colors)
        item_colors_ref = self.hcir_agent.learned_item_colors
        items_dict = (
            item_colors_ref.items()
            if isinstance(item_colors_ref, dict)
            else {c: {"action": 5} for c in item_colors_ref}.items()
        )
        for c, item_info in items_dict:
            if c not in self.knowledge_base.object_recipes:
                self.knowledge_base.object_recipes[c] = ObjectInteractionRecipe(
                    object_color=c,
                    interaction_action=item_info.get("action", 5),
                    outcome="pickup",
                    delivery_zone_color=(
                        next(iter(self.hcir_agent.learned_receptacle_colors))
                        if self.hcir_agent.learned_receptacle_colors
                        else None
                    ),
                    delivery_zone_bounds=self.hcir_agent.learned_receptacle_bounds,
                    confidence=0.8,
                    times_confirmed=1,
                )

        for sig_key in self.hcir_agent.learned_cargo_signatures:
            if sig_key not in self.knowledge_base.signature_recipes:
                feat_part = sig_key.split("_")[0][1:] if "_" in sig_key else "0"
                col_val = int(feat_part) if feat_part.isdigit() else 0
                self.knowledge_base.signature_recipes[sig_key] = ObjectInteractionRecipe(
                    object_color=col_val,
                    signature_key=sig_key,
                    interaction_action=5,
                    outcome="pickup",
                    delivery_zone_color=(
                        next(iter(self.hcir_agent.learned_receptacle_colors))
                        if self.hcir_agent.learned_receptacle_colors
                        else None
                    ),
                    delivery_zone_bounds=self.hcir_agent.learned_receptacle_bounds,
                    confidence=0.8,
                    times_confirmed=1,
                )

        for sig_key in self.hcir_agent.learned_target_signatures:
            if sig_key not in self.knowledge_base.signature_recipes:
                feat_part = sig_key.split("_")[0][1:] if "_" in sig_key else "0"
                col_val = int(feat_part) if feat_part.isdigit() else 0
                self.knowledge_base.signature_recipes[sig_key] = ObjectInteractionRecipe(
                    object_color=col_val,
                    signature_key=sig_key,
                    interaction_action=0,
                    outcome="goal",
                    confidence=0.8,
                    times_confirmed=1,
                )

        self.prev_grid = curr_grid.copy()
        self.last_action = action
        return action, conf

    def plan_next_action(
        self,
        curr_grid: np.ndarray,
        available_actions: list[int],
    ) -> tuple[int, float]:
        """Select action via CognitiveBlackbox core platform (core-first dispatch).

        Phase 3 migration: ALL cognition is delegated to the core HCIR platform
        via ARC3SpatialCognitiveAgent → CognitiveBlackbox.decide(). Local archetype
        solvers are retained ONLY as deprecated fallback for edge cases where
        core skills don't match.
        """
        if not isinstance(curr_grid, np.ndarray):
            curr_grid = np.array(curr_grid)

        self.step_counter += 1

        # Track mid-episode stagnation based on visual frame changes
        curr_hash = hash(curr_grid.tobytes())
        if hasattr(self, "solver_last_grid_hash") and curr_hash == self.solver_last_grid_hash:
            self.solver_stagnation_counter += 1
        else:
            self.solver_stagnation_counter = max(0, self.solver_stagnation_counter - 1)
            self.solver_last_grid_hash = curr_hash
            if hasattr(self, "_tried_stagnation_actions"):
                self._tried_stagnation_actions.clear()

        # ── 1. Assimilate feedback from previous action ──────────────────
        if self.prev_grid is not None and self.last_action is not None:
            diff = FrameDiffAnalyzer.analyze(self.prev_grid, self.last_action, curr_grid)
            self.knowledge_base.register_observation(
                self.prev_grid, self.last_action, curr_grid, diff
            )

            # Record trial outcome
            pos = self.current_actor_pos or (0, 0)
            barrier_dir = None
            if diff.diff_type == DiffType.NO_CHANGE:
                aff = self.knowledge_base.action_affordances.get(self.last_action)
                if aff and (aff.delta_r != 0 or aff.delta_c != 0):
                    barrier_dir = (aff.delta_r, aff.delta_c)

            outcome = TrialOutcome(
                action=self.last_action,
                position=pos,
                succeeded=(diff.diff_type != DiffType.NO_CHANGE),
                diff_type=diff.diff_type,
                barrier_direction=barrier_dir,
                objects_affected=diff.changed_pixel_count,
                item_gained=(
                    diff.diff_type == DiffType.TRANSLATION
                    and diff.moved_object_size > 0
                    and diff.changed_pixel_count > diff.moved_object_size * 2
                ),
                item_lost=(
                    diff.diff_type == DiffType.TRANSLATION
                    and diff.moved_object_size > 0
                    and diff.changed_pixel_count > diff.moved_object_size * 3
                ),
            )
            self.trial_memory.record(outcome)

        # ── 2. Sync knowledge to core ────────────────────────────────────
        if self.knowledge_base.controllable_signature.color is not None:
            self.hcir_agent.avatar_color = self.knowledge_base.controllable_signature.color

        for a, aff in self.knowledge_base.action_affordances.items():
            if a not in self.hcir_agent.action_models and aff.confidence >= 0.4:
                self.hcir_agent.action_models[a] = ActionDynamicsModel(
                    action_id=a,
                    delta_r=aff.delta_r,
                    delta_c=aff.delta_c,
                    confidence=aff.confidence,
                    probes_tested=aff.times_tested,
                )

        if self.knowledge_base.barrier_colors:
            if self.hcir_agent.known_barriers is None:
                self.hcir_agent.known_barriers = np.zeros(curr_grid.shape, dtype=bool)
            for color in self.knowledge_base.barrier_colors:
                self.hcir_agent.known_barriers[curr_grid == color] = True

        # ── 3. [DISABLED] Plugin-level exploration probing ─────────────────
        # The core CognitiveBlackbox.decide() already has its own motor calibration
        # phase (epistemic probing in SpatialPlanner). Running probing here duplicates
        # core's work and wastes 15-20 actions of budget. Disabled in Phase 3 migration.
        has_movement = any(a in available_actions for a in [1, 2, 3, 4])

        # ── 4. Stuck detection — epistemic probing fallback ──────────────
        # Only for movement games — click games have their own exploration in _plan_click_affordance
        if has_movement and self.trial_memory.is_stuck(threshold=5):
            if self.current_actor_pos:
                blocked = self.trial_memory.get_blocked_actions_at(self.current_actor_pos)
                unblocked = [a for a in available_actions if a not in blocked]
                if unblocked:
                    import random

                    interaction_candidates = [
                        a
                        for a in unblocked
                        if a >= 5
                        and self.knowledge_base.action_affordances.get(
                            a, ActionAffordance(a)
                        ).times_tested
                        < 5
                    ]
                    if interaction_candidates:
                        probe_action = interaction_candidates[0]
                    else:
                        directional = [a for a in unblocked if a <= 4]
                        if directional and self.visit_counts:
                            scored = []
                            for a in directional:
                                aff = self.knowledge_base.action_affordances.get(a)
                                if aff and (aff.delta_r != 0 or aff.delta_c != 0):
                                    pos = self.current_actor_pos
                                    target = (pos[0] + aff.delta_r, pos[1] + aff.delta_c)
                                    visits = self.visit_counts.get(target, 0)
                                    scored.append((a, visits))
                            if scored:
                                scored.sort(key=lambda x: x[1])
                                probe_action = scored[0][0]
                            else:
                                probe_action = random.choice(directional)
                        else:
                            probe_action = random.choice(unblocked)

                    self.prev_grid = curr_grid.copy()
                    self.last_action = probe_action

                    if probe_action >= 6:
                        H, W = curr_grid.shape
                        bg = int(np.bincount(curr_grid.flatten()).argmax())
                        non_bg = np.argwhere((curr_grid != bg) & (curr_grid != 0))
                        if len(non_bg) > 0:
                            idx = self.step_counter % len(non_bg)
                            self.last_action_data = {
                                "x": int(non_bg[idx, 1]),
                                "y": int(non_bg[idx, 0]),
                            }
                        else:
                            self.last_action_data = {"x": W // 2, "y": H // 2}
                    else:
                        self.last_action_data = None

                    self.trial_memory.consecutive_no_change = 0
                    return probe_action, 0.3

        # ── 5. Recipe synchronization ────────────────────────────────────
        if self.knowledge_base.levels_solved > 0:
            if self.knowledge_base.signature_recipes:
                for sig_key, recipe in self.knowledge_base.signature_recipes.items():
                    if recipe.outcome == "pickup":
                        self.hcir_agent.learned_cargo_signatures.add(sig_key)
                    elif recipe.outcome in ("goal", "target"):
                        self.hcir_agent.learned_target_signatures.add(sig_key)

            if self.knowledge_base.object_recipes:
                for c, recipe in self.knowledge_base.object_recipes.items():
                    if recipe.outcome == "pickup":
                        item_colors = self.hcir_agent.learned_item_colors
                        if c not in item_colors:
                            if isinstance(item_colors, set):
                                item_colors.add(c)
                            elif isinstance(item_colors, dict):
                                item_colors[c] = {"action": recipe.interaction_action}
                        if (
                            recipe.delivery_zone_bounds
                            and not self.hcir_agent.learned_receptacle_bounds
                        ):
                            self.hcir_agent.learned_receptacle_bounds = recipe.delivery_zone_bounds
                        if recipe.delivery_zone_color is not None:
                            self.hcir_agent.learned_receptacle_colors.add(
                                recipe.delivery_zone_color
                            )

        # If a declarative skill has queued actions in-flight, continue executing the plan
        if hasattr(self, "_declarative_action_queue") and self._declarative_action_queue:
            action, action_data = self._declarative_action_queue.pop(0)
            self.last_action_data = action_data
            self.prev_grid = curr_grid.copy()
            self.last_action = action
            return action, 0.9

        # ── 6a. Adaptive core-skill interleaving ──────────────────────────
        # The core ALWAYS learns from observations (action models, barriers, etc.)
        # regardless of who chose the action. Skills can act from step 0 if they
        # match. The core only takes over when:
        #   (a) No skill matches, OR
        #   (b) The core has learned viable models and isn't stagnating.
        #
        # This replaces the old fixed EXPLORATION_WINDOW which wasted steps.

        # After some steps, check if core has learned enough dynamics to self-navigate
        core_has_model = self.spatial_cognitive_agent.has_viable_model(
            min_models=min(2, len(available_actions)),
            min_confidence=0.6,
        )

        # If core has viable models and is making progress, use it
        if core_has_model and self.solver_stagnation_counter < 5:
            return self._execute_core_step(curr_grid, available_actions)

        # ── 6b. Declarative Skill Dispatch via SkillRegistry ──────────────
        # Only NOW try declarative skills as acceleration (if core lacks models or stagnates)
        if not self.disable_archetypes:
            # Track which skills have been tried and failed for stagnation detection
            if not hasattr(self, "_declarative_failed_skills"):
                self._declarative_failed_skills = set()
                self._declarative_plan_grid_hash = None

            curr_hash = hash(curr_grid.tobytes())
            skill_metadata = {
                "knowledge_base": self.knowledge_base,
                "step_counter": self.step_counter,
                "current_level": self.current_level,
                "avatar_color": getattr(self.hcir_agent, "avatar_color", None),
            }
            blackbox_state = self.spatial_cognitive_agent.blackbox.get_state("arc_agi")
            for skill in blackbox_state.skill_registry:
                if skill.skill_name in self._declarative_failed_skills:
                    continue
                try:
                    if skill.can_handle(curr_grid, available_actions, metadata=skill_metadata):
                        plan_result = skill.plan(
                            curr_grid,
                            current_level=self.current_level,
                            metadata=skill_metadata,
                        )
                        if plan_result:
                            # Extract first action from plan
                            first = plan_result[0]
                            if isinstance(first, tuple):
                                action, action_data = first[0], first[1]
                            else:
                                action, action_data = int(first), None
                            self.active_solver_name = f"declarative:{skill.skill_name}"
                            self.last_action_data = action_data
                            self.prev_grid = curr_grid.copy()
                            self.last_action = action
                            self._declarative_plan_grid_hash = curr_hash
                            # Enqueue remaining planned actions
                            self._declarative_action_queue = []
                            for remaining in plan_result[1:]:
                                if isinstance(remaining, tuple):
                                    self._declarative_action_queue.append(remaining)
                                else:
                                    self._declarative_action_queue.append((int(remaining), None))
                            logger.info(
                                "Declarative skill '%s' matched — planned %d actions (first=%d)",
                                skill.skill_name,
                                len(plan_result),
                                action,
                            )
                            return action, 0.9
                except Exception:
                    logger.debug(
                        "Declarative skill '%s' error in dispatch",
                        skill.skill_name,
                        exc_info=True,
                    )

        # ── 6c. Phase 4: Counterfactual Planning on Stagnation ────────────
        if self.solver_stagnation_counter >= 5:
            from hbllm.hcir.counterfactual_planner import CounterfactualPlanner

            if not hasattr(self, "_tried_stagnation_actions"):
                self._tried_stagnation_actions = set()

            state = self.spatial_cognitive_agent.blackbox.get_state("arc_agi")
            avatar_col = getattr(state, "avatar_feature", None) or getattr(
                self.hcir_agent, "avatar_color", None
            )
            for action_id in available_actions:
                if action_id not in self._tried_stagnation_actions:
                    predicted_grid = CounterfactualPlanner.predict_outcome(
                        curr_grid,
                        action_id,
                        state.action_models.get(action_id),
                        avatar_color=avatar_col,
                    )
                    novelty = int(np.sum(predicted_grid != curr_grid))
                    if novelty > 0:
                        self._tried_stagnation_actions.add(action_id)
                        self.prev_grid = curr_grid.copy()
                        self.last_action = action_id
                        self.last_action_data = None
                        logger.info(
                            "Counterfactual recovery: action %d predicted novelty=%d",
                            action_id,
                            novelty,
                        )
                        return action_id, 0.4

        # ── 6d. Phase 5: Cross-Game Structural Transfer ───────────────────
        if not self.disable_archetypes and self.solver_stagnation_counter >= 3:
            try:
                from plugins.arc_agi_adapter.arc_skills.structural_fingerprint import (
                    StructuralFingerprint,
                    get_global_transfer_registry,
                )

                transfer_reg = get_global_transfer_registry()
                curr_fp = StructuralFingerprint.from_grid(curr_grid, available_actions)
                matches = transfer_reg.find_similar(curr_fp, threshold=0.65)
                for sim, transferred_skill, source_game in matches:
                    if getattr(transferred_skill, "skill_name", "") in getattr(
                        self, "_declarative_failed_skills", set()
                    ):
                        continue
                    try:
                        plan_result = transferred_skill.plan(
                            curr_grid,
                            current_level=self.current_level,
                            metadata={
                                "knowledge_base": self.knowledge_base,
                                "step_counter": self.step_counter,
                                "avatar_color": getattr(self.hcir_agent, "avatar_color", None),
                            },
                        )
                        if plan_result:
                            first = plan_result[0]
                            if isinstance(first, tuple):
                                action, action_data = first[0], first[1]
                            else:
                                action, action_data = int(first), None
                            self.active_solver_name = f"transfer:{transferred_skill.skill_name}"
                            self.last_action_data = action_data
                            self.prev_grid = curr_grid.copy()
                            self.last_action = action
                            self._declarative_action_queue = [
                                r if isinstance(r, tuple) else (int(r), None)
                                for r in plan_result[1:]
                            ]
                            logger.info(
                                "Structural transfer: applied skill '%s' from '%s' (sim=%.2f, actions=%d)",
                                transferred_skill.skill_name,
                                source_game,
                                sim,
                                len(plan_result),
                            )
                            return action, 0.85
                    except Exception:
                        pass
            except Exception as e:
                logger.debug("Structural transfer lookup error: %s", e)

        # ── 6e. Fallback to CognitiveBlackbox ─────────────────────────────
        return self._execute_core_step(curr_grid, available_actions)

    def _execute_core_step(
        self, curr_grid: np.ndarray, available_actions: list[int]
    ) -> tuple[int, float]:
        """Execute a cognitive step via the core HCIR agent / CognitiveBlackbox, with full state sync."""
        has_movement = any(a in available_actions for a in [1, 2, 3, 4])
        is_click_only = not has_movement and 6 in available_actions

        if is_click_only:
            if not hasattr(self, "_core_click_stagnation"):
                self._core_click_stagnation = 0
                self._core_click_gave_up = False

            if self._core_click_gave_up:
                return self._plan_click_affordance(curr_grid, available_actions)

            action, conf = self.hcir_agent.plan_next_action(
                curr_grid, available_actions, level=self.current_level
            )
            self.last_action_data = self.hcir_agent.last_action_data

            if self.prev_grid is not None and np.array_equal(self.prev_grid, curr_grid):
                self._core_click_stagnation += 1
            else:
                self._core_click_stagnation = 0

            if self._core_click_stagnation >= 25:
                self._core_click_gave_up = True
                return self._plan_click_affordance(curr_grid, available_actions)
        else:
            action, conf = self.hcir_agent.plan_next_action(
                curr_grid, available_actions, level=self.current_level
            )
            self.last_action_data = self.hcir_agent.last_action_data

        # Sync position data back from core
        if getattr(self.hcir_agent, "avatar_grid_pos", None):
            self.current_actor_pos = self.hcir_agent.avatar_grid_pos
        elif self.hcir_agent.avatar_centroid:
            self.current_actor_pos = (
                int(round(self.hcir_agent.avatar_centroid[0])),
                int(round(self.hcir_agent.avatar_centroid[1])),
            )

        if self.hcir_agent.goal_centroid:
            self.current_target_pos = (
                int(round(self.hcir_agent.goal_centroid[0])),
                int(round(self.hcir_agent.goal_centroid[1])),
            )
        else:
            self.current_target_pos = None

        # Sync learned knowledge back from core
        self.knowledge_base.barrier_colors.update(self.hcir_agent.learned_barrier_colors)
        self.knowledge_base.walkable_colors.update(self.hcir_agent.learned_walkable_colors)
        item_colors_ref = self.hcir_agent.learned_item_colors
        items_dict = (
            item_colors_ref.items()
            if isinstance(item_colors_ref, dict)
            else {c: {"action": 5} for c in item_colors_ref}.items()
        )
        for c, item_info in items_dict:
            if c not in self.knowledge_base.object_recipes:
                self.knowledge_base.object_recipes[c] = ObjectInteractionRecipe(
                    object_color=c,
                    interaction_action=item_info.get("action", 5),
                    outcome="pickup",
                    delivery_zone_color=(
                        next(iter(self.hcir_agent.learned_receptacle_colors))
                        if self.hcir_agent.learned_receptacle_colors
                        else None
                    ),
                    delivery_zone_bounds=self.hcir_agent.learned_receptacle_bounds,
                    confidence=0.8,
                    times_confirmed=1,
                )

        # Track position for loop detection
        if self.current_actor_pos:
            self.visited_positions.append(self.current_actor_pos)
            self.visit_counts[self.current_actor_pos] = (
                self.visit_counts.get(self.current_actor_pos, 0) + 1
            )

        self.prev_grid = curr_grid.copy()
        self.last_action = action
        return action, conf

    def _plan_click_affordance(
        self, curr_grid: np.ndarray, available_actions: list[int]
    ) -> tuple[int, float]:
        """Handle click-only games by systematically clicking on distinct objects with causal momentum and loop avoidance."""
        # NOTE: step_counter is already incremented in plan_next_action()
        # self.step_counter += 1  # REMOVED: was causing double-increment
        self.knowledge_base.puzzle_typology = PuzzleTypology.AFFORDANCE_CLICK

        if not hasattr(self, "_quiescent_targets"):
            self._quiescent_targets = set()
            self._completed_controls = set()
            self._target_usage = {}
            self._entity_usage = {}
            self._click_visited_states = set()
            self._last_click_target = None
            self._consecutive_effective_clicks = 0
            self._prev_min_dist = {}

        H, W = curr_grid.shape
        bg = int(np.bincount(curr_grid.flatten()).argmax())
        grid_bytes = curr_grid.tobytes()
        is_revisit = grid_bytes in self._click_visited_states
        self._click_visited_states.add(grid_bytes)

        # Declarative skill dispatch for click puzzles (e.g. lights-out, kinematic linkage)
        blackbox_state = self.spatial_cognitive_agent.blackbox.get_state("arc_agi")
        for skill in blackbox_state.skill_registry:
            try:
                if skill.can_handle(curr_grid, available_actions):
                    plan_result = skill.plan(
                        curr_grid,
                        current_level=self.current_level,
                        metadata={"knowledge_base": self.knowledge_base},
                    )
                    if plan_result:
                        first = plan_result[0]
                        act = first[0] if isinstance(first, tuple) else int(first)
                        data = first[1] if isinstance(first, tuple) else None
                        self.prev_grid = curr_grid.copy()
                        self.last_action = act
                        self.last_action_data = data
                        # Queue remaining
                        if len(plan_result) > 1:
                            if not hasattr(self, "_declarative_action_queue"):
                                self._declarative_action_queue = []
                            for remaining in plan_result[1:]:
                                if isinstance(remaining, tuple):
                                    self._declarative_action_queue.append(remaining)
                                else:
                                    self._declarative_action_queue.append((int(remaining), None))
                        return act, 0.9
            except Exception:
                pass

        # Interleave non-click actions (e.g. Action 7 submit/commit) if present
        other_actions = [a for a in available_actions if a != 6]
        if other_actions and (self.step_counter % 8 == 0 or len(self._quiescent_targets) > 0):
            chosen = other_actions[(self.step_counter // 8) % len(other_actions)]
            self.prev_grid = curr_grid.copy()
            self.last_action = chosen
            self.last_action_data = None
            return chosen, 0.6

        # Learn from previous click and evaluate causal momentum
        if (
            self.prev_grid is not None
            and self.last_action == 6
            and self._last_click_target is not None
        ):
            diff = FrameDiffAnalyzer.analyze(self.prev_grid, 6, curr_grid)
            self.knowledge_base.register_observation(self.prev_grid, 6, curr_grid, diff)
            cr, cc, col, eid = self._last_click_target
            if diff.diff_type != DiffType.NO_CHANGE:
                self._effective_colors.add(col)
                self._consecutive_effective_clicks += 1
                # When a click produces state change, harvest procedural sub-skill
                sk = HCIRSkill(
                    skill_id=f"click_target_{cr}_{cc}_col_{col}",
                    preconditions={"color": col, "coord": (cr, cc)},
                    action_sequence=[6],
                    action_data_sequence=[{"x": cc, "y": cr}],
                    expected_effect={"diff_type": str(diff.diff_type)},
                    confidence=0.85,
                    times_executed=1,
                    times_succeeded=1,
                )
                self.knowledge_base.register_skill(sk)
                # Previously quiescent targets may become active!
                self._quiescent_targets.clear()

                # Check teleological progress of remote payload entities (size <= 9)
                diff_mask = self.prev_grid != curr_grid
                internal_mask = diff_mask.copy()
                internal_mask[0:2, :] = False
                internal_mask[H - 2 : H, :] = False
                internal_mask[:, 0:2] = False
                internal_mask[:, W - 2 : W] = False

                stopped = False
                for c in np.unique(curr_grid[internal_mask]):
                    if c == 0 or c == bg:
                        continue
                    chg_pts = np.argwhere((curr_grid == c) & internal_mask)
                    stat_pts = np.argwhere((curr_grid == c) & (~internal_mask))
                    if 1 <= len(chg_pts) <= 9 and len(stat_pts) >= 1:
                        dists = [np.min(np.sum(np.abs(stat_pts - pt), axis=1)) for pt in chg_pts]
                        cur_d = min(dists)
                        prev_d = self._prev_min_dist.get(c, None)
                        if prev_d is not None:
                            if cur_d <= 1 and prev_d > 1:
                                stopped = True
                            elif cur_d > prev_d:
                                stopped = True
                        self._prev_min_dist[c] = cur_d

                if stopped:
                    self._completed_controls.add((cr, cc))
                    self._consecutive_effective_clicks = 0
                    self._prev_min_dist = {}
                elif not is_revisit and self._consecutive_effective_clicks < 12:
                    self.prev_grid = curr_grid.copy()
                    self.last_action = 6
                    self.last_action_data = {"x": cc, "y": cr}
                    return 6, 0.75
            else:
                self._quiescent_targets.add((cr, cc))
                self._consecutive_effective_clicks = 0
                self._prev_min_dist = {}

        entities = VisualTopologyExtractor.extract_entities(curr_grid, ignore_colors={bg, 0})

        def is_border_or_frame(e: Any) -> bool:
            min_r, max_r, min_c, max_c = getattr(e, "bounding_box", (0, 0, 0, 0))
            spans_h = max_r - min_r >= H - 3
            spans_w = max_c - min_c >= W - 3
            if spans_h and spans_w:
                return True
            if (min_r <= 1 and max_r <= 1 and spans_w) or (
                min_r >= H - 2 and max_r >= H - 2 and spans_w
            ):
                return True
            if (min_c <= 1 and max_c <= 1 and spans_h) or (
                min_c >= W - 2 and max_c >= W - 2 and spans_h
            ):
                return True
            # Filter 1-pixel thin bars (step counters, health bars, borders)
            if (max_r - min_r == 0 and max_c - min_c > 8) or (
                max_c - min_c == 0 and max_r - min_r > 8
            ):
                return True
            return False

        usable_entities = [e for e in entities if not is_border_or_frame(e)]

        candidates: list[tuple[int, int, int, int, Any]] = []
        for i, e in enumerate(usable_entities):
            cr, cc = int(round(e.centroid[0])), int(round(e.centroid[1]))
            coords_list = getattr(e, "coords", getattr(e, "pixels", []))
            if coords_list and (cr, cc) not in coords_list:
                cr, cc = coords_list[len(coords_list) // 2]
            candidates.append((cr, cc, e.color, i, e))
            min_r, max_r, min_c, max_c = getattr(e, "bounding_box", (0, 0, 0, 0))
            if max_r - min_r >= 4:
                p1_r = min_r + (max_r - min_r) // 4
                p2_r = max_r - (max_r - min_r) // 4
                candidates.append((p1_r, cc, e.color, i, e))
                candidates.append((p2_r, cc, e.color, i, e))
            if max_c - min_c >= 4:
                p1_c = min_c + (max_c - min_c) // 4
                p2_c = max_c - (max_c - min_c) // 4
                candidates.append((cr, p1_c, e.color, i, e))
                candidates.append((cr, p2_c, e.color, i, e))

        if not candidates:
            unique_colors = [int(c) for c in np.unique(curr_grid) if c != bg and c != 0]
            for color in unique_colors:
                pts = np.argwhere(curr_grid == color)
                if len(pts) > 0:
                    candidates.append((int(pts[0, 0]), int(pts[0, 1]), color, 0, None))

        def score(item: tuple[int, int, int, int, Any]) -> float:
            cr, cc, col, eid, e = item
            is_eff = 150.0 if col in self._effective_colors else 0.0
            not_q = -500.0 if (cr, cc) in self._quiescent_targets else 0.0
            not_comp = -600.0 if (cr, cc) in self._completed_controls else 0.0
            is_2d = 0.0
            is_btn_sz = 0.0
            if e is not None:
                bbox = getattr(e, "bounding_box", (0, 0, 0, 0))
                if bbox[1] - bbox[0] >= 1 and bbox[3] - bbox[2] >= 1:
                    is_2d = 30.0
                sz = len(getattr(e, "coords", []))
                if 4 <= sz <= 300:
                    is_btn_sz = 30.0
            usage = self._target_usage.get((cr, cc), 0)
            target_pen = -float(usage) * 10.0
            ent_usage = self._entity_usage.get(eid, 0)
            ent_pen = -float(ent_usage) * 25.0
            return is_eff + not_q + not_comp + is_2d + is_btn_sz + target_pen + ent_pen

        candidates.sort(key=score, reverse=True)
        if candidates:
            best = candidates[0]
            cr, cc, col, eid, _ = best
        else:
            cr, cc, col, eid = H // 2, W // 2, 0, 0

        self._last_click_target = (cr, cc, col, eid)
        self._target_usage[(cr, cc)] = self._target_usage.get((cr, cc), 0) + 1
        self._entity_usage[eid] = self._entity_usage.get(eid, 0) + 1
        self._consecutive_effective_clicks = 0
        self._prev_min_dist = {}
        self.prev_grid = curr_grid.copy()
        self.last_action = 6
        self.last_action_data = {"x": cc, "y": cr}
        return 6, 0.75


# ─────────────────────────────────────────────────────────────────────────────
# 4. Inductive ARC-3 Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class InductiveLevelResult:
    level_index: int
    completed: bool
    actions_taken: int
    baseline_actions: int
    efficiency_ratio: float
    time_seconds: float
    epistemic_probes: int
    attempts: int = 1


@dataclass
class InductiveEnvironmentResult:
    game_id: str
    total_levels: int
    levels_completed: int
    total_actions: int
    total_baseline: int
    mean_efficiency: float
    level_results: list[InductiveLevelResult] = field(default_factory=list)


class InductiveARC3BenchmarkRunner:
    """Dedicated benchmark runner evaluating InductiveHCIRAgent across ARC-3 games."""

    def __init__(
        self,
        max_steps_per_level: int | None = None,
        max_retries_per_level: int = 2,
        knowledge_dir: Path | str | None = None,
        disable_archetypes: bool = False,
    ) -> None:
        self.max_steps = max_steps_per_level
        self.max_retries_per_level = max_retries_per_level
        self.knowledge_dir = Path(knowledge_dir) if knowledge_dir else None
        self.disable_archetypes = disable_archetypes
        self.agent = InductiveHCIRAgent(disable_archetypes=disable_archetypes)

    def run_environment(
        self,
        arcade_client: Any,
        game_id: str,
        max_levels: int = 2,
        max_retries_per_level: int | None = None,
        knowledge_dir: Path | str | None = None,
    ) -> InductiveEnvironmentResult:
        """Evaluate the inductive learner on an environment with cross-level transfer."""
        logger.info(
            f"Starting Inductive HCIR evaluation on game: {game_id} (disable_archetypes={self.disable_archetypes})..."
        )
        self.agent = InductiveHCIRAgent(disable_archetypes=self.disable_archetypes)

        # Load existing core KnowledgeGraph if available
        k_dir = knowledge_dir or self.knowledge_dir
        prior_levels_completed = 0
        if k_dir:
            loaded = self.agent.load_knowledge(k_dir, game_id=game_id)
            if loaded:
                logger.info(
                    "[CORE KNOWLEDGE GRAPH] Successfully loaded prior knowledge for '%s' from %s",
                    game_id,
                    k_dir,
                )
                kg_file = Path(k_dir) / f"{game_id}_knowledge_graph.json"
                if kg_file.exists():
                    try:
                        import json

                        with open(kg_file, encoding="utf-8") as f:
                            kg_data = json.load(f)
                        for ent in kg_data.get("entities", []):
                            if ent.get("label") == f"env_{game_id}":
                                prior_levels_completed = int(
                                    ent.get("attributes", {}).get("levels_completed", 0)
                                )
                    except Exception:
                        pass

        env = arcade_client.make(game_id, render_mode=None)
        frame_data = env.reset()

        retries_allowed = (
            self.max_retries_per_level if max_retries_per_level is None else max_retries_per_level
        )

        total_levels = min(getattr(frame_data, "win_levels", 1) or 1, max_levels)
        baseline_list = [50] * total_levels
        if hasattr(env, "baseline_actions") and env.baseline_actions:
            baseline_list = list(env.baseline_actions)[:total_levels]
        elif (
            hasattr(env, "info")
            and hasattr(env.info, "baseline_actions")
            and env.info.baseline_actions
        ):
            baseline_list = list(env.info.baseline_actions)[:total_levels]

        level_results: list[InductiveLevelResult] = []
        levels_completed = 0
        total_actions = 0
        total_baseline = 0

        for lvl_idx in range(total_levels):
            lvl_start = time.time()
            lvl_actions = 0
            completed = False
            baseline = baseline_list[lvl_idx] if lvl_idx < len(baseline_list) else 50
            # Scale step budget to 3x baseline + exploration overhead (floor of 90 steps)
            # Level 0 gets extra budget for initial world model learning
            exploration_overhead = 40 if lvl_idx == 0 else 15
            if self.max_steps is not None and self.max_steps > 0:
                effective_max_steps = (
                    max(self.max_steps, int(baseline * 3.0)) + exploration_overhead
                )
            else:
                effective_max_steps = max(int(baseline * 3.0), 90) + exploration_overhead

            max_attempts = 1 + max(0, retries_allowed)
            attempts_made = 0

            for attempt in range(max_attempts):
                attempts_made += 1
                is_retry = attempt > 0
                if is_retry:
                    logger.info(
                        f"Retrying level {lvl_idx} (attempt {attempt + 1}/{max_attempts}) "
                        f"on game {game_id} with accumulated knowledge..."
                    )
                    self.agent.reset_episode(retain_dynamics=True, is_retry=True)
                    try:
                        frame_data = env.step(ARCGameAction.RESET)
                    except Exception as e:
                        logger.warning(f"Failed to reset level {lvl_idx} with RESET action: {e}")
                        break
                else:
                    self.agent.reset_episode(retain_dynamics=(lvl_idx > 0), is_retry=False)

                curr_grid = (
                    frame_data.frame[-1] if frame_data and frame_data.frame else np.zeros((16, 16))
                )
                initial_attempt_grid = curr_grid.copy()
                attempt_actions: list[tuple[int, dict[str, int] | None]] = []

                for _ in range(effective_max_steps):
                    available_actions = getattr(frame_data, "available_actions", [1, 2, 3, 4])
                    if not available_actions:
                        available_actions = [1, 2, 3, 4]

                    action_int, _ = self.agent.plan_next_action(curr_grid, available_actions)
                    game_act = getattr(ARCGameAction, f"ACTION{action_int}", ARCGameAction.ACTION1)

                    action_data = self.agent.last_action_data
                    is_complex = (
                        action_int == 6
                        or getattr(game_act, "name", "") == "ACTION6"
                        or (hasattr(game_act, "is_complex") and game_act.is_complex())
                    )
                    if is_complex:
                        if (
                            not isinstance(action_data, dict)
                            or "x" not in action_data
                            or "y" not in action_data
                        ):
                            H, W = curr_grid.shape
                            fallback_x, fallback_y = W // 2, H // 2
                            if (
                                hasattr(self.agent, "current_target_pos")
                                and self.agent.current_target_pos is not None
                            ):
                                fallback_y, fallback_x = (
                                    int(round(self.agent.current_target_pos[0])),
                                    int(round(self.agent.current_target_pos[1])),
                                )
                            elif (
                                hasattr(self.agent, "current_actor_pos")
                                and self.agent.current_actor_pos is not None
                            ):
                                fallback_y, fallback_x = (
                                    int(round(self.agent.current_actor_pos[0])),
                                    int(round(self.agent.current_actor_pos[1])),
                                )
                            action_data = {
                                "x": max(0, min(W - 1, fallback_x)),
                                "y": max(0, min(H - 1, fallback_y)),
                            }

                    attempt_actions.append((action_int, action_data))

                    prev_grid = curr_grid
                    if action_data:
                        try:
                            frame_data = env.step(game_act, data=action_data)
                        except TypeError:
                            frame_data = env.step(game_act)
                    else:
                        frame_data = env.step(game_act)

                    if frame_data is None:
                        break

                    curr_grid = (
                        frame_data.frame[-1] if frame_data and frame_data.frame else prev_grid
                    )
                    state_val = getattr(frame_data, "state", None)
                    if state_val is not None:
                        val = getattr(state_val, "value", None)
                        self.agent.last_frame_state = (
                            str(val) if val is not None else str(state_val)
                        )
                    else:
                        self.agent.last_frame_state = None
                    lvl_actions += 1

                    curr_levels_done = getattr(frame_data, "levels_completed", 0)
                    if (
                        curr_levels_done > lvl_idx
                        or getattr(frame_data, "state", None) == ARCGameState.WIN
                    ):
                        completed = True
                        if hasattr(self.agent, "spatial_cognitive_agent"):
                            self.agent.spatial_cognitive_agent.update_causal_dynamics(
                                action_int, prev_grid, curr_grid, won=True
                            )
                        if hasattr(self.agent, "autonomous_engine"):
                            self.agent.autonomous_engine.assimilate_feedback(
                                curr_grid,
                                [action_int],
                                is_win=True,
                            )
                        self._on_level_solved(
                            initial_attempt_grid,
                            attempt_actions,
                            available_actions,
                            game_id,
                            lvl_idx,
                        )
                        break

                    if getattr(frame_data, "state", None) == ARCGameState.GAME_OVER:
                        if hasattr(self.agent, "spatial_cognitive_agent"):
                            self.agent.spatial_cognitive_agent.update_causal_dynamics(
                                action_int, prev_grid, curr_grid, lost=True
                            )
                        if hasattr(self.agent, "autonomous_engine"):
                            self.agent.autonomous_engine.assimilate_feedback(
                                curr_grid,
                                [action_int],
                                is_lost=True,
                            )
                        break
                    # NOTE: Do NOT call update_causal_dynamics here — it is already
                    # called internally by plan_next_action() → hcir_agent.plan_next_action()
                    # Double-calling corrupts motor calibration models.

                if completed:
                    break

            if completed:
                levels_completed += 1
                if hasattr(self.agent, "spatial_cognitive_agent"):
                    self.agent.spatial_cognitive_agent.blackbox.sync_state_to_workspace(
                        source_id="arc_agi"
                    )

            total_actions += lvl_actions
            total_baseline += baseline
            eff = (baseline / lvl_actions) if completed and lvl_actions > 0 else 0.0

            lvl_epistemic_probes = 0
            if hasattr(self.agent, "autonomous_engine"):
                lvl_epistemic_probes = getattr(
                    self.agent.autonomous_engine, "level_epistemic_probes", 0
                )
            elif hasattr(self.agent, "knowledge_base"):
                lvl_epistemic_probes = getattr(
                    self.agent.knowledge_base, "total_epistemic_probes", 0
                )

            lvl_res = InductiveLevelResult(
                level_index=lvl_idx,
                completed=completed,
                actions_taken=lvl_actions,
                baseline_actions=baseline,
                efficiency_ratio=eff,
                time_seconds=time.time() - lvl_start,
                epistemic_probes=lvl_epistemic_probes,
                attempts=attempts_made,
            )
            level_results.append(lvl_res)

            if not completed or getattr(frame_data, "state", None) == ARCGameState.WIN:
                state = self.agent.spatial_cognitive_agent.blackbox.get_state("arc_agi")
                logger.info(
                    f"Level {lvl_idx} {'PASSED' if completed else 'FAILED'} (after {attempts_made} attempt{'s' if attempts_made > 1 else ''}) | "
                    f"Core Knowledge: {len(state.action_models)} action models, "
                    f"{len(state.learned_obstacle_features)} obstacle features, "
                    f"{len(state.learned_target_features)} target features, "
                    f"{len(state.state_mutations)} state mutations"
                )
                break

        # Persist accumulated KnowledgeGraph to disk only if progress was made and score >= prior
        if k_dir and levels_completed > 0 and levels_completed >= prior_levels_completed:
            try:
                saved_path = self.agent.save_knowledge(
                    k_dir, game_id=game_id, levels_completed=levels_completed
                )
                logger.info(
                    "[CORE KNOWLEDGE GRAPH] Saved updated knowledge graph for '%s' (levels: %d >= prior: %d) to %s",
                    game_id,
                    levels_completed,
                    prior_levels_completed,
                    saved_path,
                )
            except Exception as e:
                logger.warning(
                    "[CORE KNOWLEDGE GRAPH] Failed to save knowledge graph for '%s': %s",
                    game_id,
                    e,
                )
        elif k_dir and levels_completed < prior_levels_completed:
            logger.warning(
                "[CORE KNOWLEDGE GRAPH] Retaining superior prior knowledge for '%s' (prior: %d levels > current: %d levels)",
                game_id,
                prior_levels_completed,
                levels_completed,
            )

        mean_eff = (
            sum(r.efficiency_ratio for r in level_results) / len(level_results)
            if level_results
            else 0.0
        )
        return InductiveEnvironmentResult(
            game_id=game_id,
            total_levels=total_levels,
            levels_completed=levels_completed,
            total_actions=total_actions,
            total_baseline=total_baseline,
            mean_efficiency=mean_eff,
            level_results=level_results,
        )

    def _on_level_solved(
        self,
        initial_grid: np.ndarray,
        actions_taken: list[tuple[int, dict[str, int] | None]],
        available_actions: list[int],
        game_id: str,
        level: int,
    ) -> None:
        """Called when a level is completed — induce and register a reusable declarative skill."""
        try:
            from plugins.arc_agi_adapter.arc_skills.inductive_skill_factory import (
                InductiveSkillFactory,
            )

            factory = InductiveSkillFactory()
            new_skill = factory.induce_skill(
                initial_grid, actions_taken, available_actions, game_id, level
            )
            if new_skill and hasattr(self.agent, "spatial_cognitive_agent"):
                state = self.agent.spatial_cognitive_agent.blackbox.get_state("arc_agi")
                state.skill_registry.append(new_skill)
                logger.info(
                    "Induced and registered new skill '%s' (%d actions) from solved level",
                    new_skill.skill_name,
                    len(actions_taken),
                )
                try:
                    from plugins.arc_agi_adapter.arc_skills.structural_fingerprint import (
                        StructuralFingerprint,
                        get_global_transfer_registry,
                    )

                    transfer_reg = get_global_transfer_registry()
                    fp = StructuralFingerprint.from_grid(initial_grid, available_actions)
                    transfer_reg.register_success(f"{game_id}_L{level}", fp, new_skill)
                except Exception as ex:
                    logger.debug("Failed to register structural fingerprint: %s", ex)
        except Exception as e:
            logger.warning(
                "Failed to induce skill from solved level %d of %s: %s", level, game_id, e
            )
