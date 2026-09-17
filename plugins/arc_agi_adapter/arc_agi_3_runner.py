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

from hbllm.hcir.counterfactual_planner import CounterfactualPlanner
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
from hbllm.hcir.types import UncertaintyVector
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.predictors.physics import PhysicsPredictor
from hbllm.hcir.world_kernel import WorldKernel

from .arc_agi_runner import ARCGrid, GridTopologyExtractor

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
    ) -> list[tuple[int, int]]:
        """Breadth-first search for shortest path avoiding barrier cells via native HCIR PhysicsPredictor."""
        barrier_cells = set(zip(*np.where(barrier_mask)))
        return PhysicsPredictor.compute_geodesic_path(
            start=start,
            goal=goal,
            barrier_cells=barrier_cells,
            grid_shape=grid_shape,
            step_size=step_size,
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
        self.action_models: dict[int, ActionDynamicsModel] = {}
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

    def reset_episode(self, retain_dynamics: bool = False) -> None:
        """Reset internal agent hypothesis state for a new level/episode."""
        if not retain_dynamics:
            self.action_models.clear()
            self.state_mutations.clear()
            self.step_size = 1
        self.avatar_centroid = None
        self.avatar_color = None
        self.goal_centroid = None
        self.last_action_data = None
        self.blocked_actions.clear()
        self.available_actions.clear()
        self.stuck_counter = 0
        self.known_barriers = None
        self.target_zones.clear()
        self.pushable_colors.clear()
        self.decomposer = HierarchicalGoalDecomposer()
        self.workspace = HCIRWorkspaceState()
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
        self.walkable_colors.clear()
        self.primary_goal_node = None
        self.active_goal_node = None
        self.target_zone_bounds = None
        self.delivered_positions.clear()
        self.carried_offset = (0.0, 0.0)
        self.current_facing = (0, 0)
        self.click_active_controls.clear()
        self.click_inert_targets.clear()
        self.click_visited_states.clear()
        self.click_candidate_targets.clear()
        self.last_click_target = None
        self.click_step_counter = 0

    def active_probe_action(self, available_actions: list[int]) -> int:
        """Select exploratory action to maximize causal information gain on motor dynamics."""

        def probe_priority(a: int) -> tuple[int, float]:
            if a not in self.action_models:
                return (0, 0.0)
            m = self.action_models[a]
            return (m.probes_tested, m.confidence)

        return min(available_actions, key=probe_priority)

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
                    if new_c != 0 and new_c != self.avatar_color:
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

            if len(p_prev[0]) > 0 and len(p_curr[0]) > 0:
                old_r, old_c = float(np.mean(p_prev[0])), float(np.mean(p_prev[1]))
                new_r, new_c = float(np.mean(p_curr[0])), float(np.mean(p_curr[1]))
                dr = int(round(new_r - old_r))
                dc = int(round(new_c - old_c))
                self.avatar_centroid = (new_r, new_c)
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
                            # Mark the blocked destination cell as barrier with step footprint
                            dest_r = int(round(old_r + m.delta_r))
                            dest_c = int(round(old_c + m.delta_c))
                            half_w = max(0, (self.step_size - 1) // 2)
                            H, W = curr_grid.shape
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
                                        150, int(curr_grid.size * 0.05)
                                    ):
                                        self.known_barriers[curr_grid == obs_col] = True

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
    ) -> tuple[HCIRWorkspaceState, GoalNode, list[ActionNode]]:
        """Lift 2D visual sensory observation into native HCIR CognitiveGraph & Workspace."""
        ws = HCIRWorkspaceState()
        H, W = curr_grid.shape

        # 1. Controllable Avatar PhysicalEntityNode
        if self.avatar_centroid is not None:
            ar, ac = int(round(self.avatar_centroid[0])), int(round(self.avatar_centroid[1]))
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
        best_plan = await planner.evaluate_and_select(goal_node, valid_candidates, horizon=2)
        best_a = best_plan.action.properties.get("action_id", available_actions[0])
        return best_a, float(best_plan.utility_score)

    def _plan_pure_click_action(
        self,
        curr_grid: np.ndarray,
    ) -> tuple[int, float]:
        """Synthesize coordinate click action for pure-click environments using causal affordance search."""
        H, W = curr_grid.shape
        h = curr_grid.tobytes()
        is_revisit = h in self.click_visited_states
        self.click_visited_states.add(h)

        # 1. Extract foreground candidates if candidate queue is empty
        if not self.click_candidate_targets:
            bg_color = int(np.bincount(curr_grid.flatten()).argmax())
            border_pixels = np.concatenate(
                [curr_grid[0, :], curr_grid[-1, :], curr_grid[:, 0], curr_grid[:, -1]]
            )
            border_color = int(np.bincount(border_pixels).argmax())

            arc_grid = ARCGrid.from_list(curr_grid.tolist())
            objs = GridTopologyExtractor.extract_objects(arc_grid, background_color=bg_color)

            # Filter valid interactive candidates
            max_area = int(H * W * 0.25)
            valid_objs = [
                o
                for o in objs
                if o.color != border_color
                and 2 <= o.area <= max_area
                and not (o.min_r == 0 and o.max_r == 0)
                and not (o.min_r == H - 1 and o.max_r == H - 1)
                and not (o.min_c == 0 and o.max_c == 0)
                and not (o.min_c == W - 1 and o.max_c == W - 1)
            ]

            # Prioritize: larger area first, rarer colors first (distinct control buttons over repeated tiles)
            color_counts = {c: int(np.sum(curr_grid == c)) for c in np.unique(curr_grid)}
            sorted_objs = sorted(
                valid_objs, key=lambda o: (-o.area, color_counts.get(o.color, 9999))
            )

            for o in sorted_objs:
                cx = int(round(o.centroid[1]))
                cy = int(round(o.centroid[0]))
                cx = max(0, min(W - 1, cx))
                cy = max(0, min(H - 1, cy))
                if (cx, cy) not in self.click_inert_targets and (
                    cx,
                    cy,
                ) not in self.click_candidate_targets:
                    self.click_candidate_targets.append((cx, cy))

        # 2. Decision Logic
        target_coord: tuple[int, int] | None = None
        if self.click_active_controls:
            # If current state was revisited (loop detected), rotate active controls or probe next candidate
            if is_revisit:
                if self.click_candidate_targets:
                    target_coord = self.click_candidate_targets.pop(0)
                elif len(self.click_active_controls) > 1:
                    self.click_active_controls.append(self.click_active_controls.pop(0))
                    target_coord = self.click_active_controls[0]
            if target_coord is None:
                target_coord = self.click_active_controls[0]
        elif self.click_candidate_targets:
            target_coord = self.click_candidate_targets.pop(0)

        # Fallback if candidates exhausted
        if target_coord is None:
            target_coord = (W // 2, H // 2)

        self.last_click_target = target_coord
        self.last_action_data = {"x": target_coord[0], "y": target_coord[1]}
        self.click_step_counter += 1
        return 6, 0.85

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
        calibrated_models = [
            m
            for a, m in self.action_models.items()
            if a in available_actions and m.confidence >= 0.8 and (m.delta_r != 0 or m.delta_c != 0)
        ]
        directional_avail = [a for a in available_actions if a in [1, 2, 3, 4]]
        if len(calibrated_models) < min(4, len(directional_avail)):
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
                probe = self.active_probe_action(available_actions)
                self.last_action_data = None
                return probe, 0.40
        else:
            probe = self.active_probe_action(available_actions)
            self.last_action_data = None
            return probe, 0.40

        # 4. Extract Objects & Decompose Hierarchical Subgoals
        H, W = curr_grid.shape
        arc_grid = ARCGrid.from_list(curr_grid.tolist())
        objs = GridTopologyExtractor.extract_objects(arc_grid)

        candidate_goals = [
            o
            for o in objs
            if o.color != self.avatar_color
            and o.color != 0
            and o.color not in self.walkable_colors
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

        barrier_cells: set[tuple[int, int]] = set()
        if self.known_barriers is not None:
            barrier_cells = set(zip(*np.where(self.known_barriers)))

        # Identify primary destination: distinct target zone (color 2 / enclosed zone) or furthest target/exit
        target_zones = [
            o
            for o in candidate_goals
            if o.color == 2 or (getattr(o, "is_frame", False) and o.area > 15)
        ]
        if target_zones:
            if self.target_zone_bounds is None:
                tz_min_r = min(o.min_r for o in target_zones)
                tz_max_r = max(o.max_r for o in target_zones)
                tz_min_c = min(o.min_c for o in target_zones)
                tz_max_c = max(o.max_c for o in target_zones)
                self.target_zone_bounds = (tz_min_r, tz_max_r, tz_min_c, tz_max_c)

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

        tz_bounds = self.target_zone_bounds

        def is_in_zone(o) -> bool:
            if tz_bounds:
                return (tz_bounds[0] - 1) <= o.centroid[0] <= (tz_bounds[1] + 1) and (
                    tz_bounds[2] - 1
                ) <= o.centroid[1] <= (tz_bounds[3] + 1)
            return math.hypot(o.centroid[0] - target_pos[0], o.centroid[1] - target_pos[1]) <= max(
                2.0, self.step_size * 2.2
            )

        candidate_items = [o for o in candidate_goals if not is_in_zone(o)]
        affordance_type = "INTERACTION" if 5 in available_actions else "CONTACT"
        candidate_subgoals = [
            {
                "id": f"sub_{o.color}_{int(o.centroid[0])}_{int(o.centroid[1])}",
                "position": (int(round(o.centroid[0])), int(round(o.centroid[1]))),
                "color": int(o.color),
                "area": o.area,
                "affordance": affordance_type,
                "description": f"Prerequisite element color={o.color} at ({int(o.centroid[0])}, {int(o.centroid[1])})",
            }
            for o in candidate_items
        ]

        # Item holding / delivery state machine
        if self.holding_item:
            offset = getattr(self, "carried_offset", (0.0, 0.0))
            item_r = curr_r + offset[0]
            item_c = curr_c + offset[1]

            # Find open delivery slot in target zone bounds
            open_target = None
            if tz_bounds:
                tz_min_r, tz_max_r, tz_min_c, tz_max_c = tz_bounds
                slots = []
                for sr in range(tz_min_r, tz_max_r + 1, max(1, self.step_size)):
                    for sc in range(tz_min_c, tz_max_c + 1, max(1, self.step_size)):
                        if not any(
                            math.hypot(sr - dr, sc - dc) < self.step_size * 0.7
                            for dr, dc in self.delivered_positions
                        ):
                            slots.append((sr, sc))
                if slots:
                    open_target = min(slots, key=lambda s: math.hypot(s[0] - item_r, s[1] - item_c))
                else:
                    open_target = ((tz_min_r + tz_max_r) // 2, (tz_min_c + tz_max_c) // 2)
            else:
                open_target = target_pos

            in_delivery_zone = False
            if tz_bounds:
                tz_min_r, tz_max_r, tz_min_c, tz_max_c = tz_bounds
                in_bounds = (tz_min_r <= item_r <= tz_max_r + 1) and (
                    tz_min_c <= item_c <= tz_max_c + 1
                )
                not_overlapping = not any(
                    math.hypot(item_r - dr, item_c - dc) < self.step_size * 0.7
                    for dr, dc in self.delivered_positions
                )
                in_delivery_zone = in_bounds and not_overlapping
            else:
                deliv_r = int(round(open_target[0] - offset[0]))
                deliv_c = int(round(open_target[1] - offset[1]))
                in_delivery_zone = math.hypot(deliv_r - curr_r, deliv_c - curr_c) <= max(
                    1.5, self.step_size * 0.6
                )

            if in_delivery_zone and 5 in available_actions:
                self.holding_item = False
                self.delivered_positions.add((int(round(item_r)), int(round(item_c))))
                self.carried_offset = (0.0, 0.0)
                self.blocked_actions.clear()
                self.visited_positions.clear()
                self.last_action_data = None
                return 5, 0.99

            deliv_r = int(round(open_target[0] - offset[0]))
            deliv_c = int(round(open_target[1] - offset[1]))
            goal_r, goal_c = deliv_r, deliv_c
        else:
            unobserved_mask = None
            if self.avatar_centroid is not None and np.count_nonzero(curr_grid == 0) > (
                curr_grid.size * 0.4
            ):
                r_grid, c_grid = np.ogrid[:H, :W]
                dist_from_av = np.hypot(r_grid - curr_r, c_grid - curr_c)
                unobserved_mask = (curr_grid == 0) & (dist_from_av > max(5.0, self.step_size * 2.0))

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
            goal_r, goal_c = int(t_pos[0]), int(t_pos[1])

            # Check if orthogonally adjacent to prerequisite item with affordance
            dr_diff = abs(goal_r - curr_r)
            dc_diff = abs(goal_c - curr_c)
            is_ortho = (
                dr_diff <= max(1.5, self.step_size * 0.4)
                and dc_diff <= max(2.0, self.step_size + 1.5)
            ) or (
                dc_diff <= max(1.5, self.step_size * 0.4)
                and dr_diff <= max(2.0, self.step_size + 1.5)
            )
            if is_ortho and self.active_goal_node.id != self.primary_goal_node.id:
                dr_dir = int(np.sign(goal_r - curr_r))
                dc_dir = int(np.sign(goal_c - curr_c))
                facing = getattr(self, "current_facing", (0, 0))
                if (dr_dir != 0 or dc_dir != 0) and facing != (dr_dir, dc_dir):
                    for a in available_actions:
                        if a in self.action_models:
                            m = self.action_models[a]
                            if np.sign(m.delta_r) == dr_dir and np.sign(m.delta_c) == dc_dir:
                                self.current_facing = (dr_dir, dc_dir)
                                return a, 0.98

                if 5 in available_actions:
                    self.holding_item = True
                    self.carried_offset = (goal_r - curr_r, goal_c - curr_c)
                    self._on_subgoal_resolved(self.active_goal_node.id)
                    self.last_action_data = None
                    return 5, 0.99

        self.goal_centroid = (float(goal_r), float(goal_c))

        # 5. Native HCIR Scene Lifting & Topological Geodesic Search
        _ws, _goal_node, _candidates = self.lift_to_hcir(curr_grid, (goal_r, goal_c))

        shortest_path = PhysicsPredictor.compute_geodesic_path(
            start=(curr_r, curr_c),
            goal=(goal_r, goal_c),
            barrier_cells=barrier_cells,
            grid_shape=(H, W),
            step_size=self.step_size,
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

            barrier_penalty = 20000.0 if (dest_ir, dest_ic) in barrier_cells else 0.0

            score = float(alignment - penalty - deadlock_penalty - loop_penalty - barrier_penalty)

            if score > best_score:
                best_score = score
                best_action = a

        # If all actions are blocked or deadlocked, clear blocked set to allow detour
        if best_score < -5000.0:
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
