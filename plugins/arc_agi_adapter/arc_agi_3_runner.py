"""Official ARC-AGI-3 Interactive Reasoning Benchmark Runner for HBLLM.

Evaluates HBLLM's developmental cognitive architecture on the official
ARC-AGI-3 benchmark (interactive turn-based reasoning challenge):
1. Active Motor Exploration & Causal Grounding: Infers action direction vectors
   and dynamics for available GameActions (ACTION1..ACTION7) via interventional probes.
2. Topological Scene Modeling: Lifts 2D visual frames into CognitiveGraphs
   identifying the controllable avatar, immovable barrier obstacles, and target objectives.
3. Goal-Directed Planning: Synthesizes obstacle-clearing navigation paths to targets.
4. Metacognitive Calibration & Scorecard Auditing: Records completion, action efficiency
   relative to human baselines, and calibrated Brier uncertainty scores.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from hbllm.hcir.counterfactual_planner import CounterfactualPlanner, MCTSConfig
from hbllm.hcir.graph import (
    ActionNode,
    BeliefNode,
    GoalNode,
    PhysicalEntityNode,
    PredictionNode,
    WorldVariableNode,
)
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.scheduler import KernelInstructionScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.subgoal_decomposer import HierarchicalGoalDecomposer
from hbllm.hcir.topological_cut_set import TopologicalCutSetAnalyzer
from hbllm.hcir.types import UncertaintyVector
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import InterruptionCheckpoint
from hbllm.hcir.world.predictors.physics import PhysicsPredictor
from hbllm.hcir.world_kernel import WorldKernel

from .arc_agi_runner import ARCGrid, GridTopologyExtractor
from .control_mode import (
    ActionObservation,
    ControlContext,
    EntityId,
    ModeConditionedDynamics,
    ModeSwitchDetector,
)

logger = logging.getLogger(__name__)

# Graceful import of official arcengine and arc_agi packages
try:
    from arcengine import GameAction as ARCGameAction
    from arcengine import GameState as ARCGameState
except ImportError:

    class ARCGameAction(Enum):  # type: ignore[no-redef]
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

    class ARCGameState(Enum):  # type: ignore[no-redef]
        NOT_PLAYED = "NOT_PLAYED"
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"


try:
    from arc_agi import Arcade
except ImportError:
    Arcade = None  # type: ignore[misc,assignment]


# ─────────────────────────────────────────────────────────────────────────────
# 1. ARC-3 Data Structures & Reports
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ActionDynamicsModel:
    """Empirical causal model mapping GameAction to displacement vector (dr, dc)."""

    action_id: int
    delta_r: int = 0
    delta_c: int = 0
    confidence: float = 0.5
    probes_tested: int = 0

    def describe(self) -> str:
        return f"Action({self.action_id}) -> (Δr={self.delta_r:+d}, Δc={self.delta_c:+d}) [conf={self.confidence:.2f}]"


@dataclass
class StateMutationModel:
    """Discrete causal rule mapping an environmental trigger to an observable state mutation."""

    trigger_type: str  # "TILE_CONTACT", "ACTION"
    trigger_pos: tuple[int, int] | None = None
    trigger_color: int | None = None
    mutation_type: str = "COLOR_REMAP"  # "COLOR_REMAP", "BARRIER_OPEN", "ROTATION"
    prior_value: Any = None
    posterior_value: Any = None
    confidence: float = 0.5
    occurrences: int = 1


class TopologicalPathPlanner:
    """Computes obstacle-clearing shortest paths over 2D visual grids using core HCIR PhysicsPredictor."""

    @staticmethod
    def find_shortest_path(
        start: tuple[int, int],
        goal: tuple[int, int],
        grid_shape: tuple[int, int],
        barrier_mask: np.ndarray,
        step_size: int = 1,
        footprint_offsets: list[tuple[int, int]] | set[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        """Breadth-first search for shortest path avoiding barrier cells via native HCIR PhysicsPredictor."""
        barrier_cells = set(zip(*np.where(barrier_mask)))
        return PhysicsPredictor.compute_geodesic_path(
            start=start,
            goal=goal,
            barrier_cells=barrier_cells,
            grid_shape=grid_shape,
            step_size=step_size,
            footprint_offsets=footprint_offsets,
        )


class CornerDeadlockDetector:
    """Detects irreversible corner deadlocks for pushable objects via core HCIR PhysicsPredictor."""

    @staticmethod
    def is_corner_deadlock(
        box_pos: tuple[int, int],
        barrier_mask: np.ndarray,
        target_positions: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int = 1,
    ) -> bool:
        """Returns True if box_pos is in a corner of barriers and not on a target."""
        barrier_cells = set(zip(*np.where(barrier_mask)))
        return PhysicsPredictor.is_corner_deadlock(
            box_pos=box_pos,
            barrier_cells=barrier_cells,
            target_positions=target_positions,
            grid_shape=grid_shape,
            step_size=step_size,
        )


@dataclass
class ARC3LevelResult:
    """Evaluation result for a single level in an ARC-AGI-3 environment."""

    level_index: int
    completed: bool
    actions_taken: int
    baseline_actions: int
    efficiency_ratio: float  # baseline_actions / actions_taken
    brier_uncertainty: float
    time_seconds: float
    discovered_dynamics: dict[int, str] = field(default_factory=dict)


@dataclass
class ARC3EnvironmentResult:
    """Comprehensive performance across all levels of an ARC-AGI-3 game."""

    game_id: str
    total_levels: int
    levels_completed: int
    win_rate: float
    total_actions: int
    total_baseline_actions: int
    mean_efficiency: float
    mean_brier: float
    duration_seconds: float
    level_results: list[ARC3LevelResult] = field(default_factory=list)


@dataclass
class ARC3BenchmarkReport:
    """Official ARC-AGI-3 benchmark evaluation summary across multiple environments."""

    benchmark_title: str
    total_environments: int
    environments_completed: int
    total_levels: int
    levels_completed: int
    overall_completion_rate: float
    mean_action_efficiency: float
    mean_brier_score: float
    total_actions_taken: int
    total_time_seconds: float
    environment_results: list[ARC3EnvironmentResult] = field(default_factory=list)
    raw_scorecard: dict[str, Any] = field(default_factory=dict)

    def format_markdown(self) -> str:
        """Render publication-grade Markdown benchmark report."""
        lines = [
            "# Official ARC-AGI-3 Interactive Reasoning Benchmark Report",
            f"**Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Overall Level Completion Rate**: **{self.levels_completed}/{self.total_levels} ({self.overall_completion_rate * 100:.1f}%)**",
            f"**Mean Fluid Action Efficiency**: **{self.mean_action_efficiency * 100:.1f}%** (vs Human Baseline)",
            f"**Mean Epistemic Brier Uncertainty**: **{self.mean_brier_score:.4f}**",
            f"**Total Actions Executed**: {self.total_actions_taken} across {self.total_environments} environment(s)",
            f"**Total Evaluation Time**: {self.total_time_seconds:.2f}s",
            "",
            "## 1. Environment Performance Breakdown",
            "| Environment | Levels Completed | Win Rate | Actions Taken | Human Baseline | Efficiency | Brier Error |",
            "|---|---|---|---|---|---|---|",
        ]
        for env in self.environment_results:
            lines.append(
                f"| `{env.game_id}` | {env.levels_completed}/{env.total_levels} | "
                f"{env.win_rate * 100:.1f}% | {env.total_actions} | {env.total_baseline_actions} | "
                f"**{env.mean_efficiency * 100:.1f}%** | {env.mean_brier:.4f} |"
            )

        lines.extend(
            [
                "",
                "## 2. Level-by-Level Trace & Causal Dynamics",
            ]
        )
        for env in self.environment_results:
            lines.append(f"### Environment: `{env.game_id}`")
            lines.append(
                "| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for lvl in env.level_results:
                status = "PASSED" if lvl.completed else "ACTIVE"
                dyn_str = ", ".join(f"A{k}:({v})" for k, v in lvl.discovered_dynamics.items())
                lines.append(
                    f"| Level {lvl.level_index + 1} | **{status}** | {lvl.actions_taken} | "
                    f"{lvl.baseline_actions} | {lvl.efficiency_ratio * 100:.1f}% | {lvl.brier_uncertainty:.4f} | `{dyn_str}` |"
                )
            lines.append("")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2. HBLLM Interactive ARC-3 Agent
# ─────────────────────────────────────────────────────────────────────────────


class ARC3InteractiveAgent:
    """Autonomous agent solving ARC-AGI-3 interactive environments.

    Fuses active causal probing, topological graph extraction, and goal-directed
    A* path planning to solve turn-based abstract puzzles without human guidance.
    """

    def __init__(self) -> None:
        self.control_context: ControlContext = ControlContext()
        self.mode_switch_detector: ModeSwitchDetector = ModeSwitchDetector()
        self.action_models: ModeConditionedDynamics = ModeConditionedDynamics()
        self.state_mutations: list[StateMutationModel] = []
        self.avatar_centroid: tuple[float, float] | None = None
        self.avatar_color: int | None = None
        self.goal_centroid: tuple[float, float] | None = None
        self.last_action_data: dict[str, Any] | None = None
        self.blocked_actions: set[int] = set()
        self.available_actions: list[int] = []
        self.stuck_counter: int = 0
        self.step_size: int = 1
        self.known_barriers: np.ndarray | None = None
        self.target_zones: set[tuple[int, int]] = set()
        self.pushable_colors: set[int] = set()
        self.decomposer: HierarchicalGoalDecomposer = HierarchicalGoalDecomposer()
        self.workspace: HCIRWorkspaceState = HCIRWorkspaceState()
        self.world_kernel: WorldKernel = WorldKernel(self.workspace)
        self.visited_positions: list[tuple[int, int]] = []
        self.holding_item: bool = False
        self.walkable_colors: set[int] = set()
        self.primary_goal_node: GoalNode | None = None
        self.active_goal_node: GoalNode | None = None
        self.target_zone_bounds: tuple[int, int, int, int] | None = None
        self.target_zone_base_colors: set[int] = set()
        self.delivered_positions: set[tuple[int, int]] = set()
        self.carried_offset: tuple[float, float] = (0.0, 0.0)
        self.current_facing: tuple[int, int] = (0, 0)
        # Pure Click (Action 6) Affordance Discovery & Cycle State
        self.click_active_controls: list[tuple[int, int]] = []
        self.click_inert_targets: set[tuple[int, int]] = set()
        self.click_visited_states: set[bytes] = set()
        self.click_candidate_targets: list[tuple[int, int]] = []
        self.last_click_target: tuple[int, int] | None = None
        self.click_step_counter: int = 0
        self.interruption_stack: list[InterruptionCheckpoint] = []
        self.level_transition_pending: bool = False
        self.gate_target_cell: tuple[int, int] | None = None
        self.gate_approach_cell: tuple[int, int] | None = None
        # Cross-Level Object & Spatial Behavior Memory (retained across levels)
        self.learned_barrier_colors: set[int] = set()
        self.learned_walkable_colors: set[int] = set()
        self.learned_item_colors: dict[int, dict[str, Any]] = {}
        self.learned_receptacle_colors: set[int] = set()
        self.learned_receptacle_bounds: tuple[int, int, int, int] | None = None
        self.probe_step_counter: int = 0
        self.has_spatial_avatar: bool | None = None
        self.discrete_state_visits: dict[bytes, int] = {}
        self.recent_discrete_actions: list[int] = []
        self.action_state_transitions: dict[tuple[int, bytes], bytes] = {}
        self._clicked_positions: set[tuple[int, int]] = set()
        self._last_discrete_state: bytes | None = None
        self._last_discrete_action: int | None = None
        self.prev_grid: np.ndarray | None = None
        self.last_action: int | None = None

    def reset_episode(self, retain_dynamics: bool = False) -> None:
        """Reset internal agent hypothesis state for a new level/episode."""
        if not retain_dynamics:
            self.action_models.clear()
            self.control_context = ControlContext()
            self.mode_switch_detector = ModeSwitchDetector()
            self.state_mutations.clear()
            self.step_size = 1
            self.avatar_color = None
            self.learned_barrier_colors.clear()
            self.learned_walkable_colors.clear()
            self.learned_item_colors.clear()
            self.learned_receptacle_colors.clear()
            self.learned_receptacle_bounds = None
            self.has_spatial_avatar = None
        self.level_transition_pending = retain_dynamics
        self.avatar_centroid = None
        self.goal_centroid = None
        self.last_action_data = None
        self.prev_grid = None
        self.last_action = None
        self.blocked_actions.clear()
        self.available_actions.clear()
        self.stuck_counter = 0
        self.known_barriers = None
        self.target_zones.clear()
        self.pushable_colors.clear()
        self.decomposer = HierarchicalGoalDecomposer()
        self.workspace = HCIRWorkspaceState()
        self.interruption_stack.clear()
        if not retain_dynamics:
            self.world_kernel = WorldKernel(self.workspace)
        else:
            old_latents = (
                self.world_kernel.belief_graph.get_latent_beliefs()
                if hasattr(self, "world_kernel") and self.world_kernel
                else []
            )
            self.world_kernel = WorldKernel(self.workspace)
            for lb in old_latents:
                self.world_kernel.belief_graph.add_belief(lb)
        self.visited_positions.clear()
        self.holding_item = False
        self.walkable_colors = set(self.learned_walkable_colors)
        self.primary_goal_node = None
        self.active_goal_node = None
        self.target_zone_bounds = None
        self.target_zone_base_colors = set(self.learned_receptacle_colors)
        self.delivered_positions.clear()
        self.picked_up_source_position = None
        self.carried_offset = (0.0, 0.0)
        self.current_facing = (0, 0)
        self._clicked_positions.clear()
        self.click_active_controls.clear()
        self.click_inert_targets.clear()
        self.click_visited_states.clear()
        self.click_candidate_targets.clear()
        self.last_click_target = None
        self.click_step_counter = 0
        self.probe_step_counter = 0
        self.discrete_state_visits.clear()
        self.action_state_transitions.clear()
        self._last_discrete_state = None
        self._last_discrete_action = None
        self.recent_discrete_actions.clear()
        self.gate_target_cell = None
        self.probe_step_counter = 0
        self.discrete_state_visits.clear()
        self.recent_discrete_actions.clear()
        self.action_state_transitions.clear()
        self._clicked_positions.clear()

    def active_probe_action(self, available_actions: list[int]) -> int:
        """Select exploratory action to maximize causal information gain on motor dynamics."""
        # Check calibrated directional models
        calibrated_dirs = [
            a
            for a in (1, 2, 3, 4)
            if a in self.action_models and self.action_models[a].confidence >= 0.8
        ]
        dirs = [a for a in available_actions if a in (1, 2, 3, 4)]
        if len(calibrated_dirs) < len(dirs):
            # Prioritize uncalibrated directional movement actions
            uncalibrated = [
                a
                for a in dirs
                if a not in self.action_models or self.action_models[a].confidence < 0.8
            ]
            if uncalibrated:
                return min(
                    uncalibrated,
                    key=lambda a: (
                        self.action_models[a].probes_tested if a in self.action_models else 0
                    ),
                )

        # Proactively probe interaction affordances (ACTION5 / ACTION6)
        if 5 in available_actions and (
            5 not in self.action_models or self.action_models[5].probes_tested < 2
        ):
            return 5

        pool = dirs if dirs else available_actions

        def probe_priority(a: int) -> tuple[int, float]:
            if a not in self.action_models:
                return (0, 0.0)
            m = self.action_models[a]
            return (m.probes_tested, m.confidence)

        return min(pool, key=probe_priority)

    def _plan_click_affordance_step(
        self, curr_grid: np.ndarray, available_actions: list[int]
    ) -> tuple[int, float]:
        """Execute structured click affordance on internal objects/buttons."""
        H, W = curr_grid.shape
        bg = int(np.bincount(curr_grid.flatten()).argmax())
        from plugins.arc_agi_adapter.inductive_learner import (
            DiffType,
            FrameDiffAnalyzer,
            VisualTopologyExtractor,
        )

        if not hasattr(self, "_quiescent_targets"):
            self._quiescent_targets: set[tuple[int, int]] = set()
            self._completed_controls: set[tuple[int, int]] = set()
            self._target_usage: dict[tuple[int, int], int] = {}
            self._entity_usage: dict[int, int] = {}
            self._effective_colors: set[int] = set()
            self._click_visited_states: set[bytes] = set()
            self._last_click_target: tuple[int, int, int, int] | None = None
            self._consecutive_effective_clicks: int = 0
            self._prev_min_dist: dict[int, int] = {}

        grid_bytes = curr_grid.tobytes()
        is_revisit = grid_bytes in self._click_visited_states
        self._click_visited_states.add(grid_bytes)

        # Learn from previous click and evaluate causal momentum
        if (
            hasattr(self, "prev_grid")
            and self.prev_grid is not None
            and getattr(self, "last_action", None) == 6
            and self._last_click_target is not None
        ):
            diff = FrameDiffAnalyzer.analyze(self.prev_grid, 6, curr_grid)
            cr, cc, col, eid = self._last_click_target
            if diff.diff_type != DiffType.NO_CHANGE:
                self._effective_colors.add(col)
                self._consecutive_effective_clicks += 1

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
            if (max_r - min_r == 0 and max_c - min_c > 8) or (
                max_c - min_c == 0 and max_r - min_r > 8
            ):
                return True
            return False

        usable = [e for e in entities if not is_border_or_frame(e)]

        candidates: list[tuple[int, int, int, int, Any]] = []
        for i, e in enumerate(usable):
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

    def _plan_discrete_state_step(
        self, curr_grid: np.ndarray, available_actions: list[int]
    ) -> tuple[int, float]:
        """Perform heuristic discrete state search with cycle avoidance and state novelty."""
        grid_bytes = curr_grid.tobytes()
        self.discrete_state_visits[grid_bytes] = self.discrete_state_visits.get(grid_bytes, 0) + 1

        if hasattr(self, "_last_discrete_state") and hasattr(self, "_last_discrete_action"):
            self.action_state_transitions[
                (self._last_discrete_action, self._last_discrete_state)
            ] = grid_bytes

        best_action = available_actions[0]
        best_score = -999999.0

        for a in available_actions:
            score = 0.0

            predicted_state = self.action_state_transitions.get((a, grid_bytes))
            if predicted_state is not None:
                if predicted_state == grid_bytes:
                    score -= 500.0  # Known NO_CHANGE
                else:
                    visits = self.discrete_state_visits.get(predicted_state, 0)
                    score -= visits * 50.0
            else:
                score += 100.0  # Novel untried transition from this state

            if self.recent_discrete_actions:
                last_a = self.recent_discrete_actions[-1]
                if (a, last_a) in [(1, 2), (2, 1), (3, 4), (4, 3)]:
                    score -= 80.0  # Discourage immediate reversal

            recent_uses = self.recent_discrete_actions[-6:].count(a)
            score -= recent_uses * 15.0

            if a == 5:
                score += 30.0  # Affordance action (interact / submit)

            if score > best_score:
                best_score = score
                best_action = a

        self._last_discrete_state = grid_bytes
        self._last_discrete_action = best_action
        self.recent_discrete_actions.append(best_action)
        if len(self.recent_discrete_actions) > 20:
            self.recent_discrete_actions.pop(0)

        self.last_action_data = None
        return best_action, 0.60

    def _on_subgoal_resolved(self, subgoal_id: str) -> None:
        """Handle subgoal resolution: update HCIR, clear local gate barriers, and reset transient blocks."""
        self.decomposer.resolve_subgoal(self.workspace, subgoal_id)
        self.blocked_actions.clear()
        self.visited_positions.clear()
        self.stuck_counter = 0
        if self.known_barriers is not None:
            H, W = self.known_barriers.shape
            # Clear barriers around primary goal (e.g. unlocked exit)
            if self.primary_goal_node:
                t_pos = self.primary_goal_node.properties.get("target_position")
                if t_pos:
                    tr, tc = int(round(t_pos[0])), int(round(t_pos[1]))
                    rad = max(2, int(self.step_size * 2.5))
                    r_min, r_max = max(0, tr - rad), min(H, tr + rad + 1)
                    c_min, c_max = max(0, tc - rad), min(W, tc + rad + 1)
                    self.known_barriers[r_min:r_max, c_min:c_max] = False
            # Clear barriers around resolved subgoal position (e.g. collected item)
            node = self.workspace.graph.get_node(subgoal_id)
            if isinstance(node, GoalNode):
                s_pos = node.properties.get("target_position")
                if s_pos:
                    sr, sc = int(round(s_pos[0])), int(round(s_pos[1]))
                    rad = max(2, int(self.step_size * 2.0))
                    r_min, r_max = max(0, sr - rad), min(H, sr + rad + 1)
                    c_min, c_max = max(0, sc - rad), min(W, sc + rad + 1)
                    self.known_barriers[r_min:r_max, c_min:c_max] = False

    def update_causal_dynamics(
        self,
        action_id: int,
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
    ) -> None:
        """Infer avatar identity, motor displacement, barriers, and state mutations."""
        if prev_grid.shape != curr_grid.shape:
            return

        # 0. Inter-level transition: do not diff across level boundary
        if getattr(self, "level_transition_pending", False):
            self.level_transition_pending = False
            if self.avatar_color is not None:
                p_curr = np.where(curr_grid == self.avatar_color)
                if len(p_curr[0]) > 0:
                    self.avatar_centroid = (float(np.mean(p_curr[0])), float(np.mean(p_curr[1])))
            return

        # Actions 5, 6, 7 are non-translating interaction actions (delta_r=0, delta_c=0)
        if action_id in (5, 6, 7):
            if action_id not in self.action_models:
                self.action_models[action_id] = ActionDynamicsModel(
                    action_id=action_id, delta_r=0, delta_c=0, confidence=0.99, probes_tested=1
                )
            else:
                m = self.action_models[action_id]
                m.delta_r = 0
                m.delta_c = 0
                m.confidence = 0.99
                m.probes_tested += 1

            # Detect other entities moving or changing state (e.g. piece switch or character handoff)
            other_entities_moved: list[EntityId] = []
            for col in np.unique(prev_grid):
                if col == 0 or (self.avatar_color is not None and col == self.avatar_color):
                    continue
                p_p = np.where(prev_grid == col)
                p_c = np.where(curr_grid == col)
                if len(p_p[0]) > 0 and len(p_c[0]) > 0 and abs(len(p_p[0]) - len(p_c[0])) <= 2:
                    dr_other = float(np.mean(p_c[0]) - np.mean(p_p[0]))
                    dc_other = float(np.mean(p_c[1]) - np.mean(p_p[1]))
                    if abs(dr_other) > 0.5 or abs(dc_other) > 0.5:
                        other_eid = EntityId(f"entity_{col}")
                        other_entities_moved.append(other_eid)
                        if other_eid not in self.control_context.entities:
                            self.control_context.register_entity(
                                other_eid,
                                (float(np.mean(p_c[0])), float(np.mean(p_c[1]))),
                                int(col),
                                controllable=True,
                            )

            pre_pos = self.avatar_centroid or (0.0, 0.0)
            active_eid = self.control_context.active_entity or EntityId(
                f"entity_{self.avatar_color}" if self.avatar_color is not None else "avatar_0"
            )
            self.mode_switch_detector.record(
                ActionObservation(
                    action=action_id,
                    pre_active_entity=active_eid,
                    pre_centroid=pre_pos,
                    post_centroid=pre_pos,
                    other_entities_moved=other_entities_moved,
                    action_data=self.last_action_data,
                )
            )

            # If confirmed as a candidate mode switch, transition active control focus
            # Mode switches only apply to localized, singular controllable entities (not passive multi-item sets or affordance interactions)
            if (
                action_id in self.mode_switch_detector.candidate_mode_switches()
                and not self.holding_item
                and action_id != 5
            ):
                target_eid = other_entities_moved[0] if other_entities_moved else None
                if not target_eid:
                    available_eids = [
                        eid for eid in self.control_context.entities if eid != active_eid
                    ]
                    if available_eids:
                        target_eid = available_eids[0]

                if target_eid and target_eid in self.control_context.entities:
                    target_ent = self.control_context.entities[target_eid]
                    p_target = np.where(curr_grid == target_ent.color)
                    if 0 < len(p_target[0]) <= 25:
                        self.control_context.switch_to(target_eid)
                        self.action_models.set_active_mode(target_eid)
                        self.avatar_centroid = target_ent.centroid
                        self.avatar_color = target_ent.color
                        logger.info(
                            "ARC-3 ModeSwitch: Switched active control to %s (color=%d)",
                            target_eid.label,
                            target_ent.color,
                        )

            # Pure click environments have no translating avatar
            if action_id == 6 and not any(a in [1, 2, 3, 4] for a in self.available_actions):
                if self.last_click_target is not None:
                    changed = not np.array_equal(prev_grid, curr_grid)
                    if changed:
                        if self.last_click_target not in self.click_active_controls:
                            self.click_active_controls.append(self.last_click_target)
                    else:
                        self.click_inert_targets.add(self.last_click_target)
            return

        # 1. Initialize or maintain barrier mask
        if self.known_barriers is None or self.known_barriers.shape != curr_grid.shape:
            self.known_barriers = np.zeros(curr_grid.shape, dtype=bool)

        # 2. Case 1: Avatar color is already tracked
        if self.avatar_color is not None:
            p_prev = np.where(prev_grid == self.avatar_color)
            p_curr = np.where(curr_grid == self.avatar_color)

            # Check for discrete state mutation: avatar color changed upon stepping on tile
            if len(p_prev[0]) > 0 and len(p_curr[0]) == 0 and self.avatar_centroid is not None:
                r_c, c_c = int(round(self.avatar_centroid[0])), int(round(self.avatar_centroid[1]))
                if 0 <= r_c < curr_grid.shape[0] and 0 <= c_c < curr_grid.shape[1]:
                    new_c = curr_grid[r_c, c_c]
                    new_c_count = int(np.count_nonzero(curr_grid == new_c))
                    if (
                        new_c != 0
                        and new_c != self.avatar_color
                        and new_c_count <= max(25, int(curr_grid.size * 0.15))
                        and len(np.unique(curr_grid)) > 1
                    ):
                        self.state_mutations.append(
                            StateMutationModel(
                                trigger_type="TILE_CONTACT",
                                trigger_pos=(r_c, c_c),
                                mutation_type="COLOR_REMAP",
                                prior_value=self.avatar_color,
                                posterior_value=int(new_c),
                                confidence=0.95,
                            )
                        )
                        self.avatar_color = int(new_c)
                        p_curr = np.where(curr_grid == self.avatar_color)
                        if self.active_goal_node and not getattr(
                            self.active_goal_node, "resolved", False
                        ):
                            self._on_subgoal_resolved(self.active_goal_node.id)

            if len(p_prev[0]) == 0 and len(p_curr[0]) > 0:
                # Avatar reappeared after flash/hazard/respawn
                self.avatar_centroid = (float(np.mean(p_curr[0])), float(np.mean(p_curr[1])))
                return

            if len(p_prev[0]) > 0 and len(p_curr[0]) > 0:
                old_r, old_c = float(np.mean(p_prev[0])), float(np.mean(p_prev[1]))
                new_r, new_c = float(np.mean(p_curr[0])), float(np.mean(p_curr[1]))
                dr = int(round(new_r - old_r))
                dc = int(round(new_c - old_c))

                # Motor Invariant Sanity Check:
                # If this action is already calibrated with high confidence (>= 0.9),
                # any observed displacement that deviates significantly from the expected motor model
                # indicates a respawn, teleport, or environmental artifact, NOT a valid motor update.
                if (
                    action_id in self.action_models
                    and self.action_models[action_id].confidence >= 0.9
                ):
                    expected = self.action_models[action_id]
                    if (dr != 0 or dc != 0) and (
                        abs(dr - expected.delta_r) > 2 or abs(dc - expected.delta_c) > 2
                    ):
                        # Spurious displacement / teleport: update centroid but do not corrupt motor dynamics
                        self.avatar_centroid = (new_r, new_c)
                        if self.control_context.active_entity:
                            self.control_context.update_position(
                                self.control_context.active_entity, (new_r, new_c)
                            )
                        return

                self.avatar_centroid = (new_r, new_c)
                if self.control_context.active_entity:
                    self.control_context.update_position(
                        self.control_context.active_entity, (new_r, new_c)
                    )
                self.visited_positions.append((int(round(new_r)), int(round(new_c))))
                if len(self.visited_positions) > 30:
                    self.visited_positions.pop(0)

                # Track walkable colors that appear where avatar was
                for c in np.unique(curr_grid[p_prev]):
                    if c != self.avatar_color and c != 0:
                        self.walkable_colors.add(int(c))

                # Check if active subgoal is completed by reaching position
                if (
                    self.active_goal_node
                    and self.primary_goal_node
                    and self.active_goal_node.id != self.primary_goal_node.id
                    and not getattr(self.active_goal_node, "resolved", False)
                ):
                    if action_id == 5:
                        if np.array_equal(prev_grid, curr_grid):
                            if self.holding_item:
                                self.holding_item = False
                                self.carried_offset = (0.0, 0.0)
                                self.attempted_pickup_item_id = None
                        else:
                            attempted_id = getattr(self, "attempted_pickup_item_id", None)
                            if attempted_id is not None:
                                self._on_subgoal_resolved(str(attempted_id))
                                self.attempted_pickup_item_id = None

                    if HierarchicalGoalDecomposer.check_subgoal_completion(
                        self.active_goal_node,
                        (int(round(new_r)), int(round(new_c))),
                        tolerance=self.step_size * 1.2,
                    ):
                        self._on_subgoal_resolved(self.active_goal_node.id)

                if dr != 0 or dc != 0:
                    # Update step size estimate and facing orientation
                    self.step_size = max(self.step_size, abs(dr), abs(dc))
                    self.current_facing = (int(np.sign(dr)), int(np.sign(dc)))

                    # Check if an adjacent object was pushed
                    for col in np.unique(prev_grid):
                        if col == 0 or col == self.avatar_color:
                            continue
                        box_prev = np.where(prev_grid == col)
                        box_curr = np.where(curr_grid == col)
                        if (
                            0 < len(box_prev[0]) < 100
                            and 0 < len(box_curr[0]) < 100
                            and abs(len(box_prev[0]) - len(box_curr[0])) <= 1
                        ):
                            b_dr = int(round(np.mean(box_curr[0]) - np.mean(box_prev[0])))
                            b_dc = int(round(np.mean(box_curr[1]) - np.mean(box_prev[1])))
                            if b_dr == dr and b_dc == dc:
                                self.pushable_colors.add(int(col))

                    if action_id not in self.action_models:
                        self.action_models[action_id] = ActionDynamicsModel(
                            action_id=action_id,
                            delta_r=dr,
                            delta_c=dc,
                            confidence=0.95,
                            probes_tested=1,
                        )
                    else:
                        m = self.action_models[action_id]
                        if (
                            self.world_kernel
                            and m.confidence >= 0.8
                            and (m.delta_r != dr or m.delta_c != dc)
                        ):
                            self.world_kernel.observe_and_update(
                                action=ActionNode(
                                    id=f"act_{action_id}", intent=f"ACTION{action_id}"
                                ),
                                actual_state={"delta_r": dr, "delta_c": dc},
                                prediction=PredictionNode(
                                    properties={
                                        "predicted_state": {
                                            "delta_r": m.delta_r,
                                            "delta_c": m.delta_c,
                                        }
                                    }
                                ),
                                prediction_source="arc3_motor",
                            )
                        m.delta_r = dr
                        m.delta_c = dc
                        m.confidence = min(0.99, m.confidence + 0.1)
                        m.probes_tested += 1
                    self.blocked_actions.clear()
                    self.stuck_counter = 0
                    return
                else:
                    # Avatar failed to move: collision with obstacle/wall, or interaction action
                    if action_id in self.action_models:
                        m = self.action_models[action_id]
                        m.probes_tested += 1
                        if m.delta_r != 0 or m.delta_c != 0:
                            if self.world_kernel and m.confidence >= 0.8:
                                self.world_kernel.observe_and_update(
                                    action=ActionNode(
                                        id=f"act_{action_id}", intent=f"ACTION{action_id}"
                                    ),
                                    actual_state={"delta_r": 0, "delta_c": 0},
                                    prediction=PredictionNode(
                                        properties={
                                            "predicted_state": {
                                                "delta_r": m.delta_r,
                                                "delta_c": m.delta_c,
                                            }
                                        }
                                    ),
                                    prediction_source="arc3_motor",
                                )
                            # Mark the blocked destination cell as barrier with step footprint (unless occupied by candidate item)
                            dest_r = int(round(old_r + m.delta_r))
                            dest_c = int(round(old_c + m.delta_c))
                            half_w = max(0, (self.step_size - 1) // 2)
                            H, W = curr_grid.shape
                            max_item_area = max(36, int(self.step_size * self.step_size * 2.5))
                            arc_grid_dyn = ARCGrid.from_list(curr_grid.tolist())
                            dyn_objs = GridTopologyExtractor.extract_objects(arc_grid_dyn)
                            is_item_collision = any(
                                o.area <= max_item_area
                                and o.color not in (self.avatar_color, 0)
                                and o.min_r <= dest_r <= o.max_r
                                and o.min_c <= dest_c <= o.max_c
                                for o in dyn_objs
                            )
                            if not is_item_collision:
                                for b_dr in range(-half_w, half_w + 1):
                                    for b_dc in range(-half_w, half_w + 1):
                                        br, bc = dest_r + b_dr, dest_c + b_dc
                                        if 0 <= br < H and 0 <= bc < W:
                                            self.known_barriers[br, bc] = True

                            # If collides with an immovable wall color, mark that whole color as a barrier
                            if 0 <= dest_r < H and 0 <= dest_c < W:
                                obs_col = int(curr_grid[dest_r, dest_c])
                                if (
                                    obs_col not in self.walkable_colors
                                    and obs_col != self.avatar_color
                                    and obs_col != 0
                                ):
                                    if np.count_nonzero(curr_grid == obs_col) > max(
                                        60, int(curr_grid.size * 0.015)
                                    ):
                                        self.known_barriers[curr_grid == obs_col] = True
                                        if self.target_zone_bounds:
                                            tz_min_r, tz_max_r, tz_min_c, tz_max_c = (
                                                self.target_zone_bounds
                                            )
                                            self.known_barriers[
                                                max(0, tz_min_r - 1) : min(H, tz_max_r + 2),
                                                max(0, tz_min_c - 1) : min(W, tz_max_c + 2),
                                            ] = False

                            # If this blocked collision is near the primary exit, mark exit obstructed
                            if self.primary_goal_node:
                                t_pos = self.primary_goal_node.properties.get("target_position")
                                if (
                                    t_pos
                                    and math.hypot(t_pos[0] - dest_r, t_pos[1] - dest_c)
                                    <= self.step_size * 2.0
                                ):
                                    tr, tc = int(round(t_pos[0])), int(round(t_pos[1]))
                                    for b_dr in range(-half_w, half_w + 1):
                                        for b_dc in range(-half_w, half_w + 1):
                                            br, bc = tr + b_dr, tc + b_dc
                                            if 0 <= br < H and 0 <= bc < W:
                                                self.known_barriers[br, bc] = True
                    else:
                        # Unmoving action with no prior direction is an interaction action (confidence 0.85)
                        self.action_models[action_id] = ActionDynamicsModel(
                            action_id=action_id,
                            delta_r=0,
                            delta_c=0,
                            confidence=0.85,
                            probes_tested=1,
                        )
                    self.blocked_actions.add(action_id)
                    self.stuck_counter += 1
                    return

        # 3. Case 2: Avatar not yet identified. Find rigid moving color cluster
        candidates = []
        for col in np.unique(prev_grid):
            if col == 0:
                continue
            p_prev = np.where(prev_grid == col)
            p_curr = np.where(curr_grid == col)
            n_prev, n_curr = len(p_prev[0]), len(p_curr[0])
            if 0 < n_prev < 300 and 0 < n_curr < 300 and abs(n_prev - n_curr) <= 2:
                dr_f = float(np.mean(p_curr[0]) - np.mean(p_prev[0]))
                dc_f = float(np.mean(p_curr[1]) - np.mean(p_prev[1]))
                if abs(dr_f) > 0.5 or abs(dc_f) > 0.5:
                    candidates.append(
                        (
                            int(col),
                            int(round(dr_f)),
                            int(round(dc_f)),
                            (float(np.mean(p_curr[0])), float(np.mean(p_curr[1]))),
                            n_curr,
                        )
                    )

        if candidates:
            candidates.sort(key=lambda x: x[4])
            best_col, best_dr, best_dc, best_pos, _ = candidates[0]
            self.avatar_color = best_col
            self.avatar_centroid = best_pos
            self.step_size = max(self.step_size, abs(best_dr), abs(best_dc))
            self.action_models[action_id] = ActionDynamicsModel(
                action_id=action_id,
                delta_r=best_dr,
                delta_c=best_dc,
                confidence=0.95,
                probes_tested=1,
            )
            self.blocked_actions.discard(action_id)
            self.stuck_counter = 0

            # Register/update controllable entity in control context
            best_eid = EntityId(f"entity_{best_col}")
            if best_eid not in self.control_context.entities:
                self.control_context.register_entity(
                    best_eid,
                    best_pos,
                    int(best_col),
                    controllable=True,
                    set_active=True,
                )
            else:
                self.control_context.update_position(best_eid, best_pos)
                if self.control_context.active_entity is None:
                    self.control_context.switch_to(best_eid)
            self.action_models.set_active_mode(best_eid)
        else:
            if action_id not in self.action_models:
                self.action_models[action_id] = ActionDynamicsModel(
                    action_id=action_id, delta_r=0, delta_c=0, confidence=0.3, probes_tested=1
                )
            else:
                self.action_models[action_id].probes_tested += 1
            self.blocked_actions.add(action_id)

    def lift_to_hcir(
        self,
        curr_grid: np.ndarray,
        chosen_goal: tuple[int, int] | None = None,
        candidate_items: list[Any] | None = None,
    ) -> tuple[HCIRWorkspaceState, GoalNode, list[ActionNode]]:
        """Lift 2D visual sensory observation into native HCIR CognitiveGraph & Workspace."""
        ws = (
            self.workspace
            if hasattr(self, "workspace") and self.workspace is not None
            else HCIRWorkspaceState()
        )
        H, W = curr_grid.shape

        # 1. Controllable Avatar PhysicalEntityNode
        if self.avatar_centroid is not None:
            ar, ac = int(round(self.avatar_centroid[0])), int(round(self.avatar_centroid[1]))
            active_eid = (
                self.control_context.active_entity.label
                if self.control_context.active_entity
                else "avatar"
            )
            ws.upsert_node(
                PhysicalEntityNode(
                    id="avatar",
                    entity_name="avatar",
                    entity_type="agent",
                    status="active",
                    properties={
                        "position": (ar, ac),
                        "color": int(self.avatar_color) if self.avatar_color is not None else -1,
                        "is_avatar": True,
                        "movable": True,
                        "passable": False,
                        "controlled_entity_id": active_eid,
                    },
                )
            )

        # 2. Pushable blocks / Movable Objects
        for p_col in self.pushable_colors:
            pts = np.where(curr_grid == p_col)
            for r, c in zip(pts[0], pts[1]):
                box_id = f"box_{r}_{c}"
                ws.upsert_node(
                    PhysicalEntityNode(
                        id=box_id,
                        entity_name="pushable_block",
                        entity_type="movable_object",
                        status="active",
                        properties={
                            "position": (int(r), int(c)),
                            "color": int(p_col),
                            "movable": True,
                            "passable": False,
                            "affordances": ["PUSHABLE"],
                        },
                    )
                )

        # 3. Target Goal Zones
        goal_node = GoalNode(id="goal_arc3", description="Reach target location or deliver object")
        if chosen_goal is not None:
            gr, gc = chosen_goal
            goal_node.properties = {"target_position": (gr, gc), "target_entity": "goal_primary"}
            ws.upsert_node(
                PhysicalEntityNode(
                    id="goal_primary",
                    entity_name="goal_zone",
                    entity_type="target_zone",
                    status="active",
                    properties={
                        "position": (gr, gc),
                        "is_goal": True,
                        "movable": False,
                        "passable": True,
                    },
                )
            )
        ws.upsert_node(goal_node)

        # 4. Barriers & Environment Variables
        barrier_cells: list[tuple[int, int]] = []
        if self.known_barriers is not None:
            b_coords = np.where(self.known_barriers)
            barrier_cells = [(int(r), int(c)) for r, c in zip(b_coords[0], b_coords[1])]

        target_positions = list(self.target_zones)
        if chosen_goal is not None and chosen_goal not in target_positions:
            target_positions.append(chosen_goal)

        ws.upsert_node(
            WorldVariableNode(
                id="var_grid_shape",
                variable_name="grid_shape",
                value=[H, W],
            )
        )
        ws.upsert_node(
            WorldVariableNode(
                id="var_barrier_cells",
                variable_name="barrier_cells",
                value=barrier_cells,
            )
        )
        ws.upsert_node(
            WorldVariableNode(
                id="var_target_positions",
                variable_name="target_positions",
                value=target_positions,
            )
        )
        ws.upsert_node(
            WorldVariableNode(
                id="var_control_context",
                variable_name="control_context",
                value=self.control_context.to_dict(),
            )
        )
        ws.upsert_node(
            WorldVariableNode(
                id="var_learned_item_colors",
                variable_name="learned_item_colors",
                value=list(self.learned_item_colors.keys()),
            )
        )
        ws.upsert_node(
            WorldVariableNode(
                id="var_learned_barrier_colors",
                variable_name="learned_barrier_colors",
                value=list(self.learned_barrier_colors),
            )
        )
        ws.upsert_node(
            WorldVariableNode(
                id="var_learned_receptacle_colors",
                variable_name="learned_receptacle_colors",
                value=list(self.learned_receptacle_colors),
            )
        )
        ws.upsert_node(
            WorldVariableNode(
                id="var_walkable_colors",
                variable_name="walkable_colors",
                value=list(self.walkable_colors),
            )
        )

        # 4b. Collectible / Interactable items in HCIR Scene Graph
        if candidate_items:
            for itm in candidate_items:
                i_col = getattr(itm, "color", -1)
                i_pos = getattr(itm, "centroid", (0, 0))
                ir, ic = int(round(i_pos[0])), int(round(i_pos[1]))
                is_known = i_col in self.learned_item_colors
                ws.upsert_node(
                    PhysicalEntityNode(
                        id=f"item_{i_col}_{ir}_{ic}",
                        entity_name=f"item_c{i_col}",
                        entity_type="collectible_item",
                        status="active",
                        properties={
                            "position": (ir, ic),
                            "color": int(i_col),
                            "is_learned_target": is_known,
                            "affordance": self.learned_item_colors.get(i_col, {}).get("action", 5),
                            "movable": True,
                            "passable": True,
                        },
                    )
                )

        # 5. Build candidate ActionNodes from calibrated motor models
        candidate_actions: list[ActionNode] = []
        for a_id, model in self.action_models.items():
            if model.confidence >= 0.8 and (model.delta_r != 0 or model.delta_c != 0):
                act_node = ActionNode(
                    id=f"act_{a_id}",
                    intent=f"MOVE_A{a_id}",
                    properties={
                        "action_id": a_id,
                        "delta_r": model.delta_r,
                        "delta_c": model.delta_c,
                        "grid_shape": [H, W],
                        "barrier_cells": barrier_cells,
                        "target_positions": target_positions,
                    },
                )
                candidate_actions.append(act_node)

        # 6. Sync active latent beliefs into the scene graph
        if getattr(self, "world_kernel", None):
            for lb in self.world_kernel.belief_graph.get_latent_beliefs():
                ws.upsert_node(
                    BeliefNode(
                        id=lb.belief_id,
                        claim=f"Latent confounder for {lb.subject}",
                        belief_type="causal",
                        uncertainty=UncertaintyVector(confidence=lb.confidence),
                        properties={
                            "subject": lb.subject,
                            "value": lb.value,
                            "is_latent": True,
                            "distribution": dict(lb.distribution),
                        },
                    )
                )

        return ws, goal_node, candidate_actions

    async def plan_next_action_counterfactual(
        self,
        curr_grid: np.ndarray,
        available_actions: list[int],
    ) -> tuple[int, float]:
        """Synthesize next action via native HCIR CounterfactualPlanner branch evaluation."""
        if not available_actions:
            return 0, 0.0

        target_pos = (
            (int(round(self.goal_centroid[0])), int(round(self.goal_centroid[1])))
            if self.goal_centroid
            else None
        )
        ws, goal_node, candidate_actions = self.lift_to_hcir(curr_grid, target_pos)
        valid_candidates = [
            a for a in candidate_actions if a.properties.get("action_id") in available_actions
        ]
        if not valid_candidates:
            return available_actions[0], 0.50

        services = KernelServices(
            workspace=ws,
            transaction_manager=TransactionManager(ws),
            capability_resolver=CapabilityResolver(),
            scheduler=KernelInstructionScheduler(),
        )
        planner = CounterfactualPlanner(ws, services)
        best_plan = await planner.evaluate_and_select(
            goal_node, valid_candidates, horizon=2, mcts_config=MCTSConfig(causal_pruning=True)
        )
        best_a = best_plan.action.properties.get("action_id", available_actions[0])
        return best_a, float(best_plan.utility_score)

    def _plan_pure_click_action(
        self,
        curr_grid: np.ndarray,
    ) -> tuple[int, float]:
        """Synthesize coordinate click action for pure-click environments using causal affordance search."""
        return self._plan_click_affordance_step(curr_grid, [6])

    def plan_next_action(
        self,
        curr_grid: np.ndarray,
        available_actions: list[int],
        tags: list[str] | None = None,
    ) -> tuple[int, float]:
        """Synthesize next action using native HCIR scene lifting, physics simulation and path planning."""
        self.available_actions = list(available_actions)
        # 1. Coordinate Click Interaction (Pure Click Isolation)
        is_pure_click = bool(
            6 in available_actions and not any(a in available_actions for a in [1, 2, 3, 4])
        )
        if is_pure_click:
            return self._plan_pure_click_action(curr_grid)

        has_click = bool(
            tags and any("click" in str(t).lower() for t in tags) and 6 in available_actions
        )

        if has_click and not is_pure_click and self.stuck_counter >= 3:
            arc_grid = ARCGrid.from_list(curr_grid.tolist())
            objs = GridTopologyExtractor.extract_objects(arc_grid)
            clickable = [
                o
                for o in objs
                if o.color != 0 and o.color != self.avatar_color and o.area < curr_grid.size * 0.15
            ]
            if clickable:
                target = clickable[self.stuck_counter % len(clickable)]
                self.last_action_data = {
                    "x": int(round(target.centroid[1])),
                    "y": int(round(target.centroid[0])),
                }
                self.stuck_counter = 0
                return 6, 0.85

        # 2. Motor Model Calibration Check
        if not hasattr(self, "probe_step_counter"):
            self.probe_step_counter = 0

        calibrated_models = [
            m
            for a, m in self.action_models.items()
            if a in available_actions and m.confidence >= 0.8 and (m.delta_r != 0 or m.delta_c != 0)
        ]
        directional_avail = [a for a in available_actions if a in [1, 2, 3, 4]]

        probe_limit = max(8, len(directional_avail) * 2) if directional_avail else 6
        if self.avatar_color is None and self.probe_step_counter >= probe_limit:
            self.has_spatial_avatar = False

        if getattr(self, "has_spatial_avatar", True) is False:
            if 6 in available_actions:
                return self._plan_click_affordance_step(curr_grid, available_actions)
            return self._plan_discrete_state_step(curr_grid, available_actions)

        if len(calibrated_models) < min(4, len(directional_avail)):
            self.probe_step_counter += 1
            probe = self.active_probe_action(available_actions)
            self.last_action_data = None
            return probe, 0.50

        # 3. Locate Avatar
        if self.avatar_color is not None:
            avatar_mask = np.where(curr_grid == self.avatar_color)
            if len(avatar_mask[0]) > 0:
                curr_r = int(round(np.mean(avatar_mask[0])))
                curr_c = int(round(np.mean(avatar_mask[1])))
                self.avatar_centroid = (float(curr_r), float(curr_c))
            else:
                self.probe_step_counter += 1
                probe = self.active_probe_action(available_actions)
                self.last_action_data = None
                return probe, 0.40
        else:
            self.probe_step_counter += 1
            probe = self.active_probe_action(available_actions)
            self.last_action_data = None
            return probe, 0.40

        # 4. Extract Objects & Decompose Hierarchical Subgoals
        H, W = curr_grid.shape

        arc_grid = ARCGrid.from_list(curr_grid.tolist())
        objs = GridTopologyExtractor.extract_objects(arc_grid)

        # Detect level-transition UI overlay frames via general background uniformity
        bg_counts = np.bincount(curr_grid.flatten())
        dom_color = int(bg_counts.argmax())
        dom_frac = float(bg_counts[dom_color]) / curr_grid.size
        self._is_transition_frame = bool(dom_frac > 0.95 and len(objs) <= 1)

        candidate_goals = [
            o
            for o in objs
            if o.color != self.avatar_color
            and o.color != 0
            and (o.color not in self.walkable_colors or o.color in self.learned_receptacle_colors)
            and np.count_nonzero(curr_grid == o.color) < (curr_grid.size * 0.30)
            and (H <= 20 or (2 <= o.centroid[0] < (H - 3) and 2 <= o.centroid[1] < (W - 3)))
            and not (H > 30 and o.centroid[0] > (H - 14) and o.centroid[1] < 15)
            and not (H > 30 and o.centroid[0] > (H - 14) and o.centroid[1] > (W - 15))
        ]

        if not candidate_goals:
            unblocked = [a for a in available_actions if a not in self.blocked_actions]
            act = unblocked[0] if unblocked else available_actions[0]
            self.last_action_data = None
            return act, 0.40

        if self.avatar_color is not None and self.avatar_centroid is None:
            p_avatar = np.where(curr_grid == self.avatar_color)
            if len(p_avatar[0]) > 0:
                self.avatar_centroid = (float(np.mean(p_avatar[0])), float(np.mean(p_avatar[1])))

        if self.known_barriers is None:
            self.known_barriers = np.zeros(curr_grid.shape, dtype=bool)
            if self.learned_barrier_colors:
                for b_col in self.learned_barrier_colors:
                    if (
                        b_col != 0
                        and b_col != self.avatar_color
                        and b_col not in self.walkable_colors
                    ):
                        self.known_barriers[curr_grid == b_col] = True

        barrier_cells: set[tuple[int, int]] = set()
        if self.known_barriers is not None:
            # Macro linear partition detection: thin structures spanning >= 40% of grid dimension
            for o in objs:
                if o.color in (0, self.avatar_color) or o.color in self.walkable_colors:
                    continue
                span_r = o.max_r - o.min_r
                span_c = o.max_c - o.min_c
                if (span_r >= int(H * 0.4) and span_c <= max(4, self.step_size * 2)) or (
                    span_c >= int(W * 0.4) and span_r <= max(4, self.step_size * 2)
                ):
                    for cr, cc in o.coords:
                        self.known_barriers[cr, cc] = True

            if self.target_zone_bounds and not self.holding_item:
                tz_min_r, tz_max_r, tz_min_c, tz_max_c = self.target_zone_bounds
                self.known_barriers[
                    max(0, tz_min_r - 1) : min(H, tz_max_r + 2),
                    max(0, tz_min_c - 1) : min(W, tz_max_c + 2),
                ] = False
            barrier_cells = set(zip(*np.where(self.known_barriers)))

        # Augment barrier_cells with physical objects that obstruct spatial traversal
        tz_base = getattr(self, "target_zone_base_colors", set()) | self.learned_receptacle_colors
        for o in objs:
            if (
                o.color != self.avatar_color
                and o.color != 0
                and o.color not in self.walkable_colors
                and (self.holding_item or o.color not in tz_base)
            ):
                if self.holding_item:
                    cur_off = getattr(self, "carried_offset", (0.0, 0.0))
                    c_ir = curr_r + cur_off[0]
                    c_ic = curr_c + cur_off[1]
                    item_is_delivered = any(
                        math.hypot(o.centroid[0] - dp[0], o.centroid[1] - dp[1])
                        < max(2.5, self.step_size * 1.5)
                        for dp in self.delivered_positions
                    )
                    if not item_is_delivered and math.hypot(
                        o.centroid[0] - c_ir, o.centroid[1] - c_ic
                    ) < max(self.step_size * 1.0, 3.0):
                        continue
                    if getattr(self, "picked_up_source_position", None):
                        p_pos = self.picked_up_source_position
                        if math.hypot(o.centroid[0] - p_pos[0], o.centroid[1] - p_pos[1]) < max(
                            self.step_size * 1.0, 3.0
                        ):
                            continue
                # Exempt goal targets (primary exit or active subgoals) from barrier_cells
                if self.active_goal_node:
                    t_pos = self.active_goal_node.properties.get("target_position")
                    if t_pos and math.hypot(
                        o.centroid[0] - t_pos[0], o.centroid[1] - t_pos[1]
                    ) < max(2.5, self.step_size * 1.5):
                        continue
                    i_pos = self.active_goal_node.properties.get("item_position")
                    if i_pos and math.hypot(
                        o.centroid[0] - i_pos[0], o.centroid[1] - i_pos[1]
                    ) < max(2.5, self.step_size * 1.5):
                        continue
                if self.primary_goal_node:
                    pt_pos = self.primary_goal_node.properties.get("target_position")
                    if pt_pos and math.hypot(
                        o.centroid[0] - pt_pos[0], o.centroid[1] - pt_pos[1]
                    ) < max(2.5, self.step_size * 1.5):
                        continue
                for cr, cc in o.coords:
                    barrier_cells.add((cr, cc))

        # Add delivered item footprints as solid obstacles
        for dp in self.delivered_positions:
            d_ir, d_ic = int(round(dp[0])), int(round(dp[1]))
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if 0 <= d_ir + dr < H and 0 <= d_ic + dc < W:
                        barrier_cells.add((d_ir + dr, d_ic + dc))

        # Identify primary destination: prioritized by learned receptacle colors, or distinct target zone
        target_zones = []
        if self.learned_receptacle_colors:
            receptacle_objs = [
                o
                for o in candidate_goals
                if o.color in self.learned_receptacle_colors
                and not (
                    self.known_barriers is not None
                    and self.known_barriers[int(round(o.centroid[0])), int(round(o.centroid[1]))]
                )
            ]
            if receptacle_objs:
                target_zones = receptacle_objs

        if not target_zones:
            target_zones = [
                o
                for o in candidate_goals
                if (getattr(o, "is_frame", False) and o.area > 16)
                or (
                    16 <= o.area <= 160
                    and (o.max_r - o.min_r >= 4)
                    and (o.max_c - o.min_c >= 4)
                    and o.color not in (self.avatar_color, 0)
                    and (
                        o.color not in self.walkable_colors
                        or o.color in self.learned_receptacle_colors
                    )
                )
            ]
        if target_zones:
            # Pick the best distinct target zone entity (prefer learned receptacle colors, then enclosed frames, then area)
            best_tz = max(
                target_zones,
                key=lambda o: (
                    2
                    if o.color in self.learned_receptacle_colors
                    else (1 if getattr(o, "is_frame", False) else 0),
                    o.area,
                ),
            )
            fresh_tz_min_r = best_tz.min_r
            fresh_tz_max_r = best_tz.max_r
            fresh_tz_min_c = best_tz.min_c
            fresh_tz_max_c = best_tz.max_c
            fresh_bounds = (fresh_tz_min_r, fresh_tz_max_r, fresh_tz_min_c, fresh_tz_max_c)
            fresh_area = (fresh_tz_max_r - fresh_tz_min_r + 1) * (
                fresh_tz_max_c - fresh_tz_min_c + 1
            )

            # Update target_zone_bounds if:
            # - Not yet set
            # - Previous bounds were set during a transition frame (stale)
            # - A genuine target zone is found that is significantly different/larger
            should_update = False
            if self.target_zone_bounds is None:
                should_update = True
            elif not getattr(self, "_is_transition_frame", False):
                old_area = (self.target_zone_bounds[1] - self.target_zone_bounds[0] + 1) * (
                    self.target_zone_bounds[3] - self.target_zone_bounds[2] + 1
                )
                # If a target zone is found that is significantly different or relocated, re-detect
                has_receptacle = any(
                    (
                        o.color in self.learned_receptacle_colors
                        or getattr(o, "is_frame", False)
                        or o.area >= 24
                    )
                    for o in target_zones
                )
                bounds_differ = (
                    abs(fresh_bounds[0] - self.target_zone_bounds[0]) > self.step_size * 2
                    or abs(fresh_bounds[2] - self.target_zone_bounds[2]) > self.step_size * 2
                )
                if (has_receptacle or self.learned_receptacle_colors) and (
                    bounds_differ or fresh_area > old_area * 1.5
                ):
                    should_update = True

            if should_update and not getattr(self, "_is_transition_frame", False):
                self.target_zone_bounds = fresh_bounds
                tz_patch = curr_grid[
                    fresh_tz_min_r : fresh_tz_max_r + 1, fresh_tz_min_c : fresh_tz_max_c + 1
                ]
                self.target_zone_base_colors = {
                    int(c)
                    for c in np.unique(tz_patch)
                    if c not in (0, self.avatar_color)
                    and c not in self.walkable_colors
                    and c not in self.learned_barrier_colors
                    and np.count_nonzero(tz_patch == c) >= max(16, int(tz_patch.size * 0.20))
                }
                self.learned_receptacle_colors.update(self.target_zone_base_colors)
                self.learned_receptacle_bounds = self.target_zone_bounds

            closest_tz = min(
                target_zones,
                key=lambda o: math.hypot(o.centroid[0] - curr_r, o.centroid[1] - curr_c),
            )
            target_pos = (int(round(closest_tz.centroid[0])), int(round(closest_tz.centroid[1])))
            self.primary_goal_node = GoalNode(
                id="g_primary_exit",
                description="Reach exit or deliver to target zone",
                properties={"target_position": target_pos},
            )
        elif self.primary_goal_node is None:
            furthest_target = max(
                candidate_goals,
                key=lambda o: math.hypot(o.centroid[0] - curr_r, o.centroid[1] - curr_c),
            )
            target_pos = (
                int(round(furthest_target.centroid[0])),
                int(round(furthest_target.centroid[1])),
            )
            self.primary_goal_node = GoalNode(
                id="g_primary_exit",
                description="Reach exit or deliver to target zone",
                properties={"target_position": target_pos},
            )
        else:
            t_prop = self.primary_goal_node.properties.get("target_position")
            target_pos = (int(t_prop[0]), int(t_prop[1])) if t_prop else (curr_r, curr_c)

        # Snapshot current target zone bounds AFTER detection/re-detection
        tz_bounds = self.target_zone_bounds

        def is_in_zone(o: Any) -> bool:
            if tz_bounds:
                return bool(
                    (tz_bounds[0] - 1) <= o.centroid[0] <= (tz_bounds[1] + 1)
                    and (tz_bounds[2] - 1) <= o.centroid[1] <= (tz_bounds[3] + 1)
                )
            return bool(
                math.hypot(o.centroid[0] - target_pos[0], o.centroid[1] - target_pos[1])
                <= max(2.0, self.step_size * 2.2)
            )

        def is_delivered(o: Any) -> bool:
            if is_in_zone(o):
                return True
            return any(
                math.hypot(o.centroid[0] - dp[0], o.centroid[1] - dp[1])
                < max(2.5, self.step_size * 1.1)
                for dp in self.delivered_positions
            )

        tz_colors = getattr(self, "target_zone_base_colors", set())
        if not tz_colors and target_zones:
            tz_colors = {tz.color for tz in target_zones}

        max_item_area = max(36, int(self.step_size * self.step_size * 2.5))

        def is_barrier_obj(o: Any) -> bool:
            span_r = o.max_r - o.min_r
            span_c = o.max_c - o.min_c
            if (span_r >= int(H * 0.4) and span_c <= max(4, self.step_size * 2)) or (
                span_c >= int(W * 0.4) and span_r <= max(4, self.step_size * 2)
            ):
                return True
            if o.area <= max_item_area:
                return False
            if self.known_barriers is None:
                return False
            r, c = int(round(o.centroid[0])), int(round(o.centroid[1]))
            if 0 <= r < H and 0 <= c < W and self.known_barriers[r, c]:
                return True
            return False

        # Filter candidate items: prioritize free uncarried, undelivered, and non-barrier items
        item_excluded = (
            {self.avatar_color} | tz_colors | self.learned_barrier_colors | self.walkable_colors
        )
        uncarried_items = [
            o
            for o in candidate_goals
            if not is_delivered(o)
            and not is_barrier_obj(o)
            and o.area <= max_item_area
            and o.color not in item_excluded
        ]
        # Prioritize objects matching learned item colors from earlier versions/levels
        if self.learned_item_colors:
            known_item_objs = [o for o in uncarried_items if o.color in self.learned_item_colors]
            if known_item_objs:
                uncarried_items = known_item_objs

        if uncarried_items:
            candidate_items = uncarried_items
        else:
            fallback_excluded = (
                {self.avatar_color} | tz_colors | self.learned_barrier_colors | self.walkable_colors
            )
            candidate_items = [
                o
                for o in candidate_goals
                if not is_delivered(o)
                and not is_barrier_obj(o)
                and o.area <= max_item_area
                and o.color not in fallback_excluded
            ]
            if self.learned_item_colors:
                known_fallback = [o for o in candidate_items if o.color in self.learned_item_colors]
                if known_fallback:
                    candidate_items = known_fallback

        # Gestalt grouping: unify concentric / co-located multi-color components into single composite entities
        deduped_items = []
        for o in candidate_items:
            matched = False
            for idx, existing in enumerate(deduped_items):
                if math.hypot(
                    o.centroid[0] - existing.centroid[0], o.centroid[1] - existing.centroid[1]
                ) < max(2.5, self.step_size * 0.6):
                    matched = True
                    if o.area > existing.area:
                        deduped_items[idx] = o
                    break
            if not matched:
                deduped_items.append(o)
        candidate_items = deduped_items

        gate_dir: tuple[int, int] | None = None
        if self.primary_goal_node:
            t_prop = self.primary_goal_node.properties.get("target_position")
            if t_prop:
                cut_res = TopologicalCutSetAnalyzer.analyze_cut_set(
                    (curr_r, curr_c),
                    (int(t_prop[0]), int(t_prop[1])),
                    barrier_cells,
                    (H, W),
                    self.step_size,
                )
                if cut_res.is_partitioned and cut_res.best_gate_cell and cut_res.approach_cell:
                    dr_g = int(np.sign(cut_res.best_gate_cell[0] - cut_res.approach_cell[0]))
                    dc_g = int(np.sign(cut_res.best_gate_cell[1] - cut_res.approach_cell[1]))
                    if dr_g != 0 or dc_g != 0:
                        gate_dir = (dr_g, dc_g)

        affordance_type = "INTERACTION" if 5 in available_actions else "CONTACT"
        candidate_subgoals = []
        for o in candidate_items:
            o_r, o_c = int(round(o.centroid[0])), int(round(o.centroid[1]))
            # Determine best adjacent approach cell (stand position)
            adj_cells = [
                (o_r - int(self.step_size), o_c),
                (o_r + int(self.step_size), o_c),
                (o_r, o_c - int(self.step_size)),
                (o_r, o_c + int(self.step_size)),
            ]
            valid_adj = [
                (ar, ac)
                for (ar, ac) in adj_cells
                if 0 <= ar < H and 0 <= ac < W and (ar, ac) not in barrier_cells
            ]
            if valid_adj:
                if gate_dir is not None:
                    preferred = (
                        o_r - gate_dir[0] * int(self.step_size),
                        o_c - gate_dir[1] * int(self.step_size),
                    )
                    if preferred in valid_adj:
                        sub_pos = preferred
                    else:
                        sub_pos = min(
                            valid_adj, key=lambda p: math.hypot(p[0] - curr_r, p[1] - curr_c)
                        )
                else:
                    sub_pos = min(valid_adj, key=lambda p: math.hypot(p[0] - curr_r, p[1] - curr_c))
            else:
                sub_pos = (o_r, o_c)

            candidate_subgoals.append(
                {
                    "id": f"sub_{o.color}_{o_r}_{o_c}",
                    "position": sub_pos,
                    "item_position": (o_r, o_c),
                    "color": int(o.color),
                    "area": o.area,
                    "affordance": affordance_type,
                    "description": f"Prerequisite element color={o.color} at ({o_r}, {o_c})",
                }
            )

        # Item holding / delivery state machine
        if self.holding_item:
            offset = getattr(self, "carried_offset", (0.0, 0.0))
            max_off = max(1.0, float(self.step_size * 1.5))
            offset = (
                float(np.clip(offset[0], -max_off, max_off)),
                float(np.clip(offset[1], -max_off, max_off)),
            )
            item_r = curr_r + offset[0]
            item_c = curr_c + offset[1]

            # Find open delivery slot in target zone bounds
            open_target: tuple[int, int] | None = None
            if tz_bounds:
                tz_min_r, tz_max_r, tz_min_c, tz_max_c = tz_bounds
                # Adjust slot boundaries if the zone border is walled with barriers
                start_r = (
                    tz_min_r + 1
                    if any((tz_min_r, c) in barrier_cells for c in range(tz_min_c, tz_max_c + 1))
                    else tz_min_r
                )
                start_c = (
                    tz_min_c + 1
                    if any((r, tz_min_c) in barrier_cells for r in range(tz_min_r, tz_max_r + 1))
                    else tz_min_c
                )
                end_r = (
                    tz_max_r - 1
                    if any((tz_max_r, c) in barrier_cells for c in range(tz_min_c, tz_max_c + 1))
                    else tz_max_r
                )
                end_c = (
                    tz_max_c - 1
                    if any((r, tz_max_c) in barrier_cells for r in range(tz_min_r, tz_max_r + 1))
                    else tz_max_c
                )

                slot_step = max(1, (self.step_size + 1) // 2)
                slots: list[tuple[int, int]] = []
                for sr in range(start_r, end_r + 1, slot_step):
                    for sc in range(start_c, end_c + 1, slot_step):
                        if (sr, sc) not in barrier_cells:
                            slots.append((sr, sc))
                if not slots:
                    slots.append(((tz_min_r + tz_max_r) // 2, (tz_min_c + tz_max_c) // 2))

                def is_slot_occupied(sr: int, sc: int) -> bool:
                    if (sr, sc) in barrier_cells:
                        return True
                    if (
                        self.known_barriers is not None
                        and (0 <= sr < H and 0 <= sc < W)
                        and self.known_barriers[sr, sc]
                    ):
                        return True
                    for dr, dc in self.delivered_positions:
                        if math.hypot(sr - dr, sc - dc) < slot_step * 0.8:
                            return True
                    # Check for non-zone objects blocking the slot.
                    tz_base = getattr(self, "target_zone_base_colors", set())
                    occupancy_colors = [c for c in [4, 5] if c not in tz_base]
                    if occupancy_colors and 0 <= sr < H and 0 <= sc < W:
                        patch = curr_grid[
                            max(0, sr - 1) : min(H, sr + 2), max(0, sc - 1) : min(W, sc + 2)
                        ]
                        if np.any(np.isin(patch, occupancy_colors)):
                            return True
                    return False

                unoccupied_slots: list[tuple[int, int]] = [
                    s for s in slots if not is_slot_occupied(s[0], s[1])
                ]

                # Prefer slots where the avatar's delivery position
                # (slot - offset) is OUTSIDE the zone.  This lets the
                # agent approach from the side without walking through
                # previously delivered packages inside the zone.
                def delivery_pos_outside_zone(slot: tuple[int, int]) -> bool:
                    dr = slot[0] - offset[0]
                    dc = slot[1] - offset[1]
                    return not (tz_min_r <= dr <= tz_max_r and tz_min_c <= dc <= tz_max_c)

                candidates = unoccupied_slots if unoccupied_slots else slots
                outside_slots: list[tuple[int, int]] = [
                    s for s in candidates if delivery_pos_outside_zone(s)
                ]
                pref_delivery_slots: list[tuple[int, int]] = (
                    outside_slots if outside_slots else candidates
                )

                if pref_delivery_slots:

                    def slot_geodesic_dist(s: tuple[int, int]) -> float:
                        deliv_stand_r = max(0, min(H - 1, int(round(s[0] - offset[0]))))
                        deliv_stand_c = max(0, min(W - 1, int(round(s[1] - offset[1]))))
                        if (deliv_stand_r, deliv_stand_c) in barrier_cells:
                            return 99999.0
                        if (
                            tz_bounds is not None
                            and tz_min_r <= deliv_stand_r <= tz_max_r
                            and tz_min_c <= deliv_stand_c <= tz_max_c
                        ):
                            return 99999.0
                        carried_offsets = (
                            [(0, 0), (int(round(offset[0])), int(round(offset[1])))]
                            if self.holding_item and (offset[0] != 0 or offset[1] != 0)
                            else [(0, 0)]
                        )
                        p = PhysicsPredictor.compute_geodesic_path(
                            (int(round(curr_r)), int(round(curr_c))),
                            (deliv_stand_r, deliv_stand_c),
                            barrier_cells,
                            (H, W),
                            self.step_size,
                            footprint_offsets=carried_offsets,
                        )
                        if p and len(p) > 1:
                            return float(len(p))
                        elif p and (deliv_stand_r, deliv_stand_c) == (
                            int(round(curr_r)),
                            int(round(curr_c)),
                        ):
                            return 0.0
                        return 9999.0 + math.hypot(
                            s[0] - offset[0] - curr_r, s[1] - offset[1] - curr_c
                        )

                    reachable_preferred = [
                        s for s in pref_delivery_slots if slot_geodesic_dist(s) < 9000.0
                    ]
                    target_slots = (
                        reachable_preferred if reachable_preferred else pref_delivery_slots
                    )
                    open_target = min(target_slots, key=slot_geodesic_dist)
                else:
                    open_target = ((tz_min_r + tz_max_r) // 2, (tz_min_c + tz_max_c) // 2)
            else:
                open_target = (int(target_pos[0]), int(target_pos[1]))

            # Topologically verify accessibility of delivery destination
            if getattr(self, "gate_target_cell", None) is not None:
                open_target = (
                    self.gate_approach_cell
                    if getattr(self, "gate_approach_cell", None) is not None
                    else self.gate_target_cell
                )
            elif open_target is not None:
                cut_res = TopologicalCutSetAnalyzer.analyze_cut_set(
                    (int(round(curr_r)), int(round(curr_c))),
                    open_target,
                    barrier_cells,
                    (H, W),
                    self.step_size,
                )
                if cut_res.is_partitioned and cut_res.approach_cell:
                    self.gate_target_cell = cut_res.best_gate_cell
                    self.gate_approach_cell = cut_res.approach_cell
                    open_target = cut_res.approach_cell
                else:
                    self.gate_target_cell = None
                    self.gate_approach_cell = None

            in_delivery_zone = False
            if tz_bounds:
                # When target zone bounds are known, always verify item is in the zone.
                # Gate is only a navigation aid, not a delivery trigger.
                tz_min_r, tz_max_r, tz_min_c, tz_max_c = tz_bounds
                # Include zone border (1 cell margin around the detected interior).
                in_delivery_zone = (tz_min_r - 1.0 <= item_r <= tz_max_r + 1.0) and (
                    tz_min_c - 1.0 <= item_c <= tz_max_c + 1.0
                )
            elif self.gate_target_cell:
                item_dist_to_gate = math.hypot(
                    self.gate_target_cell[0] - item_r, self.gate_target_cell[1] - item_c
                )
                player_dist_to_appr = math.hypot(open_target[0] - curr_r, open_target[1] - curr_c)
                in_delivery_zone = (item_dist_to_gate <= max(2.5, self.step_size * 0.9)) or (
                    player_dist_to_appr <= max(1.5, self.step_size * 0.9)
                )
            else:
                deliv_r = int(round(open_target[0] - offset[0]))
                deliv_c = int(round(open_target[1] - offset[1]))
                in_delivery_zone = math.hypot(deliv_r - curr_r, deliv_c - curr_c) <= max(
                    1.5, self.step_size * 0.6
                )

            if in_delivery_zone:
                # If approaching a gate bottleneck, face the gate before releasing / handoff
                if self.gate_target_cell:
                    dr_gate = int(np.sign(self.gate_target_cell[0] - curr_r))
                    dc_gate = int(np.sign(self.gate_target_cell[1] - curr_c))
                    if abs(self.gate_target_cell[0] - curr_r) >= abs(
                        self.gate_target_cell[1] - curr_c
                    ):
                        target_gate_facing = (dr_gate, 0)
                    else:
                        target_gate_facing = (0, dc_gate)

                    facing = getattr(self, "current_facing", (0, 0))
                    if target_gate_facing != (0, 0) and facing != target_gate_facing:
                        for a in available_actions:
                            if a in self.action_models:
                                m = self.action_models[a]
                                if (
                                    target_gate_facing[0] != 0
                                    and np.sign(m.delta_r) == target_gate_facing[0]
                                ) or (
                                    target_gate_facing[1] != 0
                                    and np.sign(m.delta_c) == target_gate_facing[1]
                                ):
                                    self.current_facing = target_gate_facing
                                    return a, 0.98

                if 5 in available_actions:
                    self.holding_item = False
                    self.gate_target_cell = None
                    self.gate_approach_cell = None
                    self.delivered_positions.add((int(round(item_r)), int(round(item_c))))
                    if getattr(self, "picked_up_source_position", None) is not None:
                        p_src = self.picked_up_source_position
                        if p_src is not None:
                            self.delivered_positions.add(
                                (int(round(p_src[0])), int(round(p_src[1])))
                            )
                        self.picked_up_source_position = None
                    if getattr(self, "attempted_pickup_item_id", None) is not None:
                        a_id = self.attempted_pickup_item_id
                        if a_id is not None:
                            self._on_subgoal_resolved(str(a_id))
                        self.attempted_pickup_item_id = None
                    if self.target_zone_base_colors:
                        self.learned_receptacle_colors.update(self.target_zone_base_colors)
                        self.learned_receptacle_bounds = self.target_zone_bounds
                    self.carried_offset = (0.0, 0.0)
                    self.blocked_actions.clear()
                    self.visited_positions.clear()
                    self.last_action_data = None
                    return 5, 0.99

            if self.gate_target_cell:
                # Check if we've reached the gate approach cell
                appr = getattr(self, "gate_approach_cell", self.gate_target_cell)
                dist_to_appr = math.hypot(appr[0] - curr_r, appr[1] - curr_c)
                if dist_to_appr <= max(1.5, self.step_size * 0.6):
                    # Reached the gate — clear it and navigate directly to zone
                    self.gate_target_cell = None
                    self.gate_approach_cell = None
                    deliv_r = max(0, min(H - 1, int(round(open_target[0] - offset[0]))))
                    deliv_c = max(0, min(W - 1, int(round(open_target[1] - offset[1]))))
                    goal_r, goal_c = deliv_r, deliv_c
                else:
                    goal_r, goal_c = int(round(open_target[0])), int(round(open_target[1]))
            else:
                deliv_r = max(0, min(H - 1, int(round(open_target[0] - offset[0]))))
                deliv_c = max(0, min(W - 1, int(round(open_target[1] - offset[1]))))
                goal_r, goal_c = deliv_r, deliv_c
        else:
            unobserved_mask = None

            # Pre-lift scene & candidate entities to HCIR workspace before goal decomposition
            self.lift_to_hcir(curr_grid, target_pos, candidate_items=candidate_items)

            force_subgoals = bool(5 in available_actions and candidate_subgoals)
            self.active_goal_node = self.decomposer.decompose_goal(
                workspace=self.workspace,
                primary_goal=self.primary_goal_node,
                avatar_pos=(curr_r, curr_c),
                barrier_cells=barrier_cells,
                grid_shape=(H, W),
                candidate_subgoals=candidate_subgoals,
                step_size=self.step_size,
                force_subgoals=force_subgoals,
                unobserved_mask=unobserved_mask,
            )
            t_pos = self.active_goal_node.properties.get("target_position", target_pos)
            item_pos = self.active_goal_node.properties.get("item_position", t_pos)
            goal_r, goal_c = int(t_pos[0]), int(t_pos[1])
            item_r, item_c = int(item_pos[0]), int(item_pos[1])

            # Check if reached approach stand position for prerequisite item with affordance
            is_at_stand = math.hypot(goal_r - curr_r, goal_c - curr_c) <= max(
                1.5, self.step_size * 0.75
            )
            is_item_goal = self.active_goal_node.id.startswith("subgoal_sub_")
            if is_at_stand and is_item_goal:
                dr_dir = int(np.sign(item_r - curr_r))
                dc_dir = int(np.sign(item_c - curr_c))
                if abs(item_r - curr_r) >= abs(item_c - curr_c):
                    target_facing = (dr_dir, 0)
                else:
                    target_facing = (0, dc_dir)

                facing = getattr(self, "current_facing", (0, 0))
                if target_facing != (0, 0) and facing != target_facing:
                    for a in available_actions:
                        if a in self.action_models:
                            facing_m = self.action_models[a]
                            if (
                                target_facing[0] != 0
                                and np.sign(facing_m.delta_r) == target_facing[0]
                            ) or (
                                target_facing[1] != 0
                                and np.sign(facing_m.delta_c) == target_facing[1]
                            ):
                                self.current_facing = target_facing
                                return a, 0.98

                if 5 in available_actions:
                    self.holding_item = True
                    dr_off = float(np.clip(item_r - curr_r, -self.step_size, self.step_size))
                    dc_off = float(np.clip(item_c - curr_c, -self.step_size, self.step_size))
                    self.carried_offset = (dr_off, dc_off)
                    self.picked_up_source_position = (item_r, item_c)
                    self.attempted_pickup_item_id = self.active_goal_node.id
                    i_col = self.active_goal_node.properties.get("item_color")
                    if i_col is not None:
                        self.learned_item_colors[int(i_col)] = {"action": 5}
                    self.last_action_data = None
                    return 5, 0.99

        self.goal_centroid = (float(goal_r), float(goal_c))

        # 5. Native HCIR Scene Lifting & Topological Geodesic Search
        _ws, _goal_node, _candidates = self.lift_to_hcir(
            curr_grid, (goal_r, goal_c), candidate_items=candidate_items
        )

        carried_offsets = (
            [(0, 0), (int(round(self.carried_offset[0])), int(round(self.carried_offset[1])))]
            if self.holding_item and (self.carried_offset[0] != 0 or self.carried_offset[1] != 0)
            else [(0, 0)]
        )
        shortest_path = PhysicsPredictor.compute_geodesic_path(
            start=(curr_r, curr_c),
            goal=(goal_r, goal_c),
            barrier_cells=barrier_cells,
            grid_shape=(H, W),
            step_size=self.step_size,
            footprint_offsets=carried_offsets,
        )

        dr_des = goal_r - curr_r
        dc_des = goal_c - curr_c
        if len(shortest_path) > 1:
            next_waypoint = shortest_path[1]
            dr_des = next_waypoint[0] - curr_r
            dc_des = next_waypoint[1] - curr_c

        # 6. Action Scoring with Native HCIR Physics Deadlock Avoidance & Loop Prevention
        best_action = available_actions[0]
        best_score = -999999.0

        for a in available_actions:
            m = self.action_models.get(a)
            if not m or (m.delta_r == 0 and m.delta_c == 0):
                continue

            # Check if this action pushes a box into a corner deadlock via HCIR PhysicsPredictor
            deadlock_penalty = 0.0
            if self.pushable_colors:
                dest_r = curr_r + m.delta_r
                dest_c = curr_c + m.delta_c
                if 0 <= dest_r < H and 0 <= dest_c < W:
                    if curr_grid[dest_r, dest_c] in self.pushable_colors:
                        box_next_r = dest_r + m.delta_r
                        box_next_c = dest_c + m.delta_c
                        if PhysicsPredictor.is_corner_deadlock(
                            box_pos=(box_next_r, box_next_c),
                            barrier_cells=barrier_cells,
                            target_positions=self.target_zones,
                            grid_shape=(H, W),
                            step_size=self.step_size,
                        ):
                            deadlock_penalty = 50000.0

            # Alignment with BFS shortest path waypoint
            alignment = (m.delta_r * dr_des) + (m.delta_c * dc_des)
            penalty = 10000.0 if a in self.blocked_actions else 0.0

            # Oscillation penalty: penalize stepping into cells visited recently (exempt if following geodesic path)
            dest_r = curr_r + m.delta_r
            dest_c = curr_c + m.delta_c
            dest_ir = int(round(dest_r))
            dest_ic = int(round(dest_c))

            recents = self.visited_positions[-8:]
            is_geodesic_step = (
                (dr_des != 0 or dc_des != 0)
                and np.sign(m.delta_r) == np.sign(dr_des)
                and np.sign(m.delta_c) == np.sign(dc_des)
            )
            # Immediate 2-step bounce penalty (stepping back to position from 2 steps ago)
            is_2step_bounce = (
                len(self.visited_positions) >= 2
                and math.hypot(
                    dest_r - self.visited_positions[-2][0], dest_c - self.visited_positions[-2][1]
                )
                < self.step_size * 0.7
            )
            bounce_penalty = 800.0 if is_2step_bounce else 0.0

            loop_penalty = (
                bounce_penalty
                if is_geodesic_step
                else bounce_penalty
                + sum(
                    35.0
                    for vr, vc in recents
                    if math.hypot(dest_r - vr, dest_c - vc) < self.step_size * 0.9
                )
            )

            # Barrier penalty: exempt if destination cell reaches the goal
            is_dest_goal = (dest_ir, dest_ic) == (goal_r, goal_c) or math.hypot(
                dest_r - goal_r, dest_c - goal_c
            ) <= self.step_size * 0.95
            dest_in_barrier = (dest_ir, dest_ic) in barrier_cells
            if self.holding_item and (self.carried_offset[0] != 0 or self.carried_offset[1] != 0):
                carried_ir = int(round(dest_r + self.carried_offset[0]))
                carried_ic = int(round(dest_c + self.carried_offset[1]))
                is_carried_goal = (
                    self.target_zone_bounds is not None
                    and self.target_zone_bounds[0] <= carried_ir <= self.target_zone_bounds[1]
                    and self.target_zone_bounds[2] <= carried_ic <= self.target_zone_bounds[3]
                )
                if not (0 <= carried_ir < H and 0 <= carried_ic < W) or (
                    (carried_ir, carried_ic) in barrier_cells and not is_carried_goal
                ):
                    dest_in_barrier = True
            barrier_penalty = 0.0 if is_dest_goal else (20000.0 if dest_in_barrier else 0.0)

            score = float(alignment - penalty - deadlock_penalty - loop_penalty - barrier_penalty)

            if score > best_score:
                best_score = score
                best_action = a

        # If all actions are blocked or deadlocked, clear blocked set to allow detour
        if best_score < -5000.0:
            if self.active_goal_node:
                self.interruption_stack.append(
                    InterruptionCheckpoint(
                        parent_goal_id=self.active_goal_node.id,
                        parent_frame_id="frame_active",
                        interrupt_goal_id="goal_detour",
                        interrupt_frame_id="frame_detour",
                        in_flight_action=f"ACTION{best_action}",
                        step_index=self.stuck_counter,
                        context_data={
                            "reason": f"arc3_detour_barrier_{self.stuck_counter}",
                            "avatar_pos": (curr_r, curr_c),
                            "holding_item": self.holding_item,
                        },
                    )
                )
            self.blocked_actions.clear()
            for a in available_actions:
                m = self.action_models.get(a)
                if m and (m.delta_r != 0 or m.delta_c != 0):
                    alignment = (m.delta_r * dr_des) + (m.delta_c * dc_des)
                    if alignment > best_score:
                        best_score = float(alignment)
                        best_action = a

        confidence = 0.94 if best_score > 0 else 0.65
        self.last_action_data = None
        return best_action, confidence


# ─────────────────────────────────────────────────────────────────────────────
# 3. ARC-AGI-3 Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────


class ARC3BenchmarkRunner:
    """Executes the official ARC-AGI-3 interactive benchmark evaluation."""

    def __init__(self, max_steps_per_level: int = 150) -> None:
        self.max_steps = max_steps_per_level
        self.agent = ARC3InteractiveAgent()

    def run_environment(
        self,
        arcade_client: Any,
        game_id: str,
        max_levels: int | None = None,
    ) -> ARC3EnvironmentResult:
        """Run the HBLLM interactive agent through an ARC-AGI-3 game environment."""
        logger.info(f"Starting ARC-AGI-3 evaluation on game: {game_id}...")
        start_time = time.time()

        env = arcade_client.make(game_id, render_mode=None)
        frame_data = env.reset()

        tags: list[str] = []
        if hasattr(env, "info") and hasattr(env.info, "tags") and env.info.tags:
            tags = list(env.info.tags)

        total_levels = getattr(frame_data, "win_levels", 1) or 1
        if max_levels is not None and max_levels > 0:
            total_levels = min(total_levels, max_levels)

        level_results: list[ARC3LevelResult] = []
        levels_completed = 0
        total_actions = 0
        total_baseline = 0

        # Baseline actions per level if available
        baseline_actions_list = [50] * total_levels
        if hasattr(env, "baseline_actions") and env.baseline_actions:
            baseline_actions_list = list(env.baseline_actions)
        elif hasattr(env, "info") and hasattr(env.info, "baseline_actions"):
            baseline_actions_list = list(env.info.baseline_actions)

        for lvl_idx in range(total_levels):
            lvl_start = time.time()
            # Retain learned motor dynamics across levels of the same environment
            self.agent.reset_episode(retain_dynamics=(lvl_idx > 0))
            lvl_actions = 0
            lvl_confidences: list[float] = []

            baseline = (
                baseline_actions_list[lvl_idx] if lvl_idx < len(baseline_actions_list) else 50
            )
            completed = False

            curr_grid = (
                frame_data.frame[0] if frame_data and frame_data.frame else np.zeros((16, 16))
            )

            for step in range(self.max_steps):
                available_actions = getattr(frame_data, "available_actions", [1, 2, 3, 4])
                if not available_actions:
                    available_actions = [1, 2, 3, 4]

                # Synthesize action via HBLLM agent
                action_int, conf = self.agent.plan_next_action(
                    curr_grid, available_actions, tags=tags
                )
                lvl_confidences.append(conf)

                # Convert to GameAction or pass data
                game_act = getattr(ARCGameAction, f"ACTION{action_int}", ARCGameAction.ACTION1)
                action_data = self.agent.last_action_data

                prev_grid = curr_grid
                if action_data:
                    try:
                        frame_data = env.step(game_act, data=action_data)
                    except TypeError:
                        frame_data = env.step(game_act)
                else:
                    frame_data = env.step(game_act)
                curr_grid = frame_data.frame[0] if frame_data and frame_data.frame else prev_grid
                lvl_actions += 1

                # Update causal motor models
                self.agent.update_causal_dynamics(action_int, prev_grid, curr_grid)

                # Check level completion
                curr_levels_done = getattr(frame_data, "levels_completed", 0)
                if (
                    curr_levels_done > lvl_idx
                    or getattr(frame_data, "state", None) == ARCGameState.WIN
                ):
                    completed = True
                    break

                if getattr(frame_data, "state", None) == ARCGameState.GAME_OVER:
                    # Reset current level
                    env.reset()
                    break

            if completed:
                levels_completed += 1
                if lvl_idx + 1 < total_levels:
                    # Advance environment to render the fresh frame of the new level
                    try:
                        advance_act = getattr(ARCGameAction, "ACTION5", ARCGameAction.ACTION1)
                        fresh_frame = env.step(advance_act)
                        if fresh_frame and fresh_frame.frame:
                            frame_data = fresh_frame
                    except Exception as e:
                        logger.debug(f"Level transition frame advance: {e}")

            total_actions += lvl_actions
            total_baseline += baseline

            eff = (baseline / lvl_actions) if completed and lvl_actions > 0 else 0.0
            mean_conf = sum(lvl_confidences) / len(lvl_confidences) if lvl_confidences else 0.5
            target_out = 1.0 if completed else 0.0
            brier = (mean_conf - target_out) ** 2

            lvl_res = ARC3LevelResult(
                level_index=lvl_idx,
                completed=completed,
                actions_taken=lvl_actions,
                baseline_actions=baseline,
                efficiency_ratio=eff,
                brier_uncertainty=brier,
                time_seconds=time.time() - lvl_start,
                discovered_dynamics={
                    a: f"{m.delta_r:+d},{m.delta_c:+d}" for a, m in self.agent.action_models.items()
                },
            )
            level_results.append(lvl_res)

        duration = time.time() - start_time
        win_rate = levels_completed / total_levels if total_levels > 0 else 0.0
        mean_eff = (
            sum(r.efficiency_ratio for r in level_results) / len(level_results)
            if level_results
            else 0.0
        )
        mean_brier = (
            sum(r.brier_uncertainty for r in level_results) / len(level_results)
            if level_results
            else 0.0
        )

        return ARC3EnvironmentResult(
            game_id=game_id,
            total_levels=total_levels,
            levels_completed=levels_completed,
            win_rate=win_rate,
            total_actions=total_actions,
            total_baseline_actions=total_baseline,
            mean_efficiency=mean_eff,
            mean_brier=mean_brier,
            duration_seconds=duration,
            level_results=level_results,
        )

    def run_benchmark(
        self,
        game_ids: list[str] | None = None,
        max_levels_per_game: int = 2,
    ) -> ARC3BenchmarkReport:
        """Run full ARC-AGI-3 benchmark battery across multiple official environments."""
        if Arcade is None:
            raise RuntimeError(
                "ARC-AGI package is not installed. Please run `pip install arc-agi`."
            )

        arc = Arcade()
        env_infos = arc.get_environments() if hasattr(arc, "get_environments") else []
        all_ids = [e.game_id.split("-")[0] for e in env_infos] if env_infos else ["ls20", "su15"]
        if game_ids and "all" in game_ids:
            target_ids = all_ids
        else:
            target_ids = game_ids or all_ids[:3]  # Evaluate on first 3 games by default

        start_time = time.time()
        env_results: list[ARC3EnvironmentResult] = []

        for gid in target_ids:
            try:
                res = self.run_environment(arc, gid, max_levels=max_levels_per_game)
                env_results.append(res)
            except Exception as e:
                logger.error(f"Error executing ARC-AGI-3 environment '{gid}': {e}")

        total_envs = len(env_results)
        envs_done = sum(1 for e in env_results if e.levels_completed == e.total_levels)
        tot_levels = sum(e.total_levels for e in env_results)
        tot_levels_done = sum(e.levels_completed for e in env_results)
        tot_actions = sum(e.total_actions for e in env_results)

        comp_rate = tot_levels_done / tot_levels if tot_levels > 0 else 0.0
        mean_eff = (
            sum(e.mean_efficiency for e in env_results) / total_envs if total_envs > 0 else 0.0
        )
        mean_brier = sum(e.mean_brier for e in env_results) / total_envs if total_envs > 0 else 0.0

        # Fetch scorecard
        scorecard = arc.get_scorecard() if hasattr(arc, "get_scorecard") else {}
        if hasattr(scorecard, "model_dump"):
            scorecard_dict = scorecard.model_dump()
        elif hasattr(scorecard, "dict"):
            scorecard_dict = scorecard.dict()
        elif isinstance(scorecard, dict):
            scorecard_dict = scorecard
        else:
            scorecard_dict = {"raw": str(scorecard)}

        return ARC3BenchmarkReport(
            benchmark_title="ARC-AGI-3 Official Interactive Reasoning Challenge",
            total_environments=total_envs,
            environments_completed=envs_done,
            total_levels=tot_levels,
            levels_completed=tot_levels_done,
            overall_completion_rate=comp_rate,
            mean_action_efficiency=mean_eff,
            mean_brier_score=mean_brier,
            total_actions_taken=tot_actions,
            total_time_seconds=time.time() - start_time,
            environment_results=env_results,
            raw_scorecard=scorecard_dict,
        )
