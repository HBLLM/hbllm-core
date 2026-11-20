"""Causal Theory Planner & Relational Subgoal Scheduling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


def get_sub_attr(target: Any, attr: str, default: Any = None) -> Any:
    return getattr(target, attr, default) if target is not None else default


@dataclass
class CausalPlanningContext:
    """Frozen planning context derived from the symbolic world theory.

    Computed once per planning cycle, consumed by the A* heuristic
    without re-querying the live causal theory mid-search.
    """

    goal_positions: set[tuple[int, int]]
    barrier_positions: set[tuple[int, int]]
    lethal_features: set[int]
    goal_features: set[int]
    barrier_features: set[int]
    walkable_features: set[int]
    action_displacements: dict[int, tuple[int, int]]  # action_id -> (dr, dc)
    grid_shape: tuple[int, int] = (0, 0)


@dataclass
class RelationalWaypoint:
    """A semantic milestone entity along a winning trajectory."""

    role: str  # "tool", "key", "barrier", "goal", "waypoint"
    feature: int
    target_pos: tuple[int, int] = (0, 0)
    relative_step: int = 0


@dataclass
class RelationalSubgoalMacro:
    """A coordinate-invariant relational macro composed of waypoint subgoals."""

    macro_id: str
    waypoints: list[RelationalWaypoint]
    action_space: frozenset[int]
    confidence: float = 1.0
    source_context_epoch: int = 0
    source_level: int | None = None
    replay_successes: int = 0
    replay_failures: int = 0

    def __post_init__(self) -> None:
        if self.source_level is None:
            self.source_level = self.source_context_epoch
        elif self.source_context_epoch == 0 and self.source_level != 0:
            self.source_context_epoch = self.source_level

    def matches_grid_features(self, grid: np.ndarray) -> bool:
        """Check if all required waypoint features are present in the target grid."""
        present = set(int(x) for x in grid.flat)
        return all(wp.feature in present for wp in self.waypoints)


class CausalTheoryPlanner:
    """Constructs dynamic A* heuristics from discovered CausalPredicates.

    Instead of hand-coding which features are goals/barriers/hazards, this
    planner reads the symbolic world theory and action dynamics to construct
    a grid-aware planning heuristic dynamically.
    """

    @staticmethod
    def extract_planning_context(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
    ) -> CausalPlanningContext:
        """Extract a frozen planning context from the engine's current causal theory."""
        theory = engine.symbolic_theory

        # Extract goal positions from grid using discovered goal features
        goal_positions: set[tuple[int, int]] = set(engine.learned_goal_positions)
        barrier_positions: set[tuple[int, int]] = set(engine.learned_barriers)

        # Scan grid for goal and barrier cells
        for r in range(curr_grid.shape[0]):
            for c in range(curr_grid.shape[1]):
                feat = int(curr_grid[r, c])
                if feat in theory.goal_features:
                    goal_positions.add((r, c))
                if feat in theory.barrier_features:
                    barrier_positions.add((r, c))

        lethal_features = set(engine.episodic_cortex.discovered_lethal_features) | set(
            engine.hazard_tracker.known_lethal_features
        )
        mob_feats = set(engine.mobile_threat_features) | (
            set(engine.hazard_tracker.mobile_entities)
            - set(theory.cargo_features)
            - set(engine.learned_cargo_features)
        )
        shroud_f = engine.working_memory.shroud_feature

        # Fallback to candidate goal features if no concrete goal cells located
        if not goal_positions and theory.candidate_goal_features:
            for r in range(curr_grid.shape[0]):
                for c in range(curr_grid.shape[1]):
                    feat = int(curr_grid[r, c])
                    if (
                        feat in theory.candidate_goal_features
                        and feat not in lethal_features
                        and feat not in mob_feats
                        and feat != shroud_f
                    ):
                        goal_positions.add((r, c))

        # Fallback to structural perception goals if still empty
        if not goal_positions:
            try:
                struct_goals = engine.detect_structural_goals(curr_grid)
                for sg in struct_goals:
                    sg_f = sg.get("feature")
                    if sg_f is not None and (
                        int(sg_f) in lethal_features
                        or int(sg_f) in mob_feats
                        or int(sg_f) == shroud_f
                    ):
                        continue
                    if sg.get("cells"):
                        for cp in sg["cells"]:
                            if cp != engine.avatar_pos:
                                goal_positions.add(cp)
                    elif sg.get("position") and sg["position"] != engine.avatar_pos:
                        goal_positions.add(sg["position"])
                    if sg_f is not None and int(np.count_nonzero(curr_grid == int(sg_f))) == 1:
                        theory.induce_candidate_goal(int(sg_f), evidence="structural_goal_detector")
            except Exception as exc:
                logger.debug("Structural goal induction at reset skipped: %s", exc)

        if goal_positions and not theory.goal_features:
            singleton_goals = {
                gp
                for gp in goal_positions
                if int(np.count_nonzero(curr_grid == int(curr_grid[gp[0], gp[1]]))) == 1
            }
            if singleton_goals:
                goal_positions = singleton_goals

        # Discard avatar position and exhausted goals
        if engine.avatar_pos is not None:
            goal_positions.discard(engine.avatar_pos)
        exhausted = get_sub_attr(engine.working_memory, "exhausted_candidate_goals", set())
        if exhausted and len(goal_positions - exhausted) > 0:
            goal_positions.difference_update(exhausted)

        # Extract action displacements from empirical action dynamics
        action_displacements: dict[int, tuple[int, int]] = {}
        for act_id, dyn in engine.action_dynamics.items():
            if dyn.is_displacement_action():
                action_displacements[act_id] = (dyn.delta_r, dyn.delta_c)

        return CausalPlanningContext(
            goal_positions=goal_positions,
            barrier_positions=barrier_positions,
            lethal_features=lethal_features,
            goal_features=set(theory.goal_features),
            barrier_features=set(theory.barrier_features),
            walkable_features=set(theory.walkable_features),
            action_displacements=action_displacements,
            grid_shape=(int(curr_grid.shape[0]), int(curr_grid.shape[1])),
        )

    @staticmethod
    def compute_causal_heuristic(
        pos: tuple[int, int],
        context: CausalPlanningContext,
    ) -> float:
        """Manhattan distance heuristic to nearest goal, with barrier penalty.

        Uses discovered causal knowledge rather than hard-coded features.
        """
        if not context.goal_positions:
            return 0.0
        min_dist = min(abs(pos[0] - g[0]) + abs(pos[1] - g[1]) for g in context.goal_positions)
        # Penalty if position is adjacent to lethal/barrier cells
        penalty = 0.0
        if pos in context.barrier_positions and pos not in context.goal_positions:
            penalty += 100.0
        return float(min_dist) + penalty

    @staticmethod
    def is_position_safe(
        pos: tuple[int, int],
        grid: np.ndarray,
        context: CausalPlanningContext,
    ) -> bool:
        """Check if a position is safe to traverse based on causal theory."""
        r, c = pos
        if r < 0 or c < 0 or r >= context.grid_shape[0] or c >= context.grid_shape[1]:
            return False
        feat = int(grid[r, c])
        if feat in context.lethal_features:
            return False
        if pos not in context.goal_positions:
            if pos in context.barrier_positions:
                return False
            if feat in context.barrier_features:
                return False
            if context.walkable_features and feat not in context.walkable_features:
                return False
        return True

    @staticmethod
    def plan_path(
        start: tuple[int, int],
        grid: np.ndarray,
        context: CausalPlanningContext,
        max_expansions: int = 500,
    ) -> list[int] | None:
        """A* pathfinding using causally-discovered action dynamics and hazards.

        Returns a list of action_ids forming a path from start to nearest goal,
        or None if no path found.
        """
        if not context.goal_positions or not context.action_displacements:
            return None

        import heapq

        # Priority queue: (f_score, g_score, position, action_path)
        start_h = CausalTheoryPlanner.compute_causal_heuristic(start, context)
        open_set: list[tuple[float, int, tuple[int, int], list[int]]] = [(start_h, 0, start, [])]
        visited: set[tuple[int, int]] = set()
        expansions = 0

        while open_set and expansions < max_expansions:
            f, g, pos, path = heapq.heappop(open_set)
            if pos in visited:
                continue
            visited.add(pos)
            expansions += 1

            # Check if we reached a goal
            if pos in context.goal_positions:
                logger.info(
                    "CausalTheoryPlanner: Found path of %d actions (%d nodes expanded)",
                    len(path),
                    expansions,
                )
                return path

            # Expand neighbors using discovered action dynamics
            for act_id, (dr, dc) in context.action_displacements.items():
                nr, nc = pos[0] + dr, pos[1] + dc
                next_pos = (nr, nc)
                if next_pos in visited:
                    continue
                if not CausalTheoryPlanner.is_position_safe(next_pos, grid, context):
                    continue
                new_g = g + 1
                new_h = CausalTheoryPlanner.compute_causal_heuristic(next_pos, context)
                new_f = new_g + new_h
                new_path = path + [act_id]
                heapq.heappush(open_set, (new_f, new_g, next_pos, new_path))

        return None

    @staticmethod
    def plan_chained_path(
        start: tuple[int, int],
        grid: np.ndarray,
        context: CausalPlanningContext,
        tool_barrier_affinities: dict[int, set[int]],
        max_expansions: int = 500,
    ) -> list[int] | None:
        """Synthesize a multi-stage prerequisite plan (e.g. acquire tool -> unlock barrier -> reach goal).

        Recursively backchains when direct path is blocked by barriers.
        """
        direct = CausalTheoryPlanner.plan_path(start, grid, context, max_expansions=max_expansions)
        if direct is not None:
            return direct

        if not tool_barrier_affinities or not context.goal_positions:
            return None

        H, W = context.grid_shape
        if H <= 0 or W <= 0:
            H, W = grid.shape[:2]

        for tool_feat, barrier_feats in tool_barrier_affinities.items():
            relevant_barriers = barrier_feats & context.barrier_features
            if not relevant_barriers and not any(
                0 <= r < H and 0 <= c < W and int(grid[r, c]) in barrier_feats
                for r, c in context.barrier_positions
            ):
                continue

            tool_positions: list[tuple[int, int]] = []
            for r in range(H):
                for c in range(W):
                    if int(grid[r, c]) == tool_feat and (r, c) != start:
                        tool_positions.append((r, c))

            if not tool_positions:
                continue

            # Stage 1: Plan path from start to tool
            stage1_context = CausalPlanningContext(
                goal_positions=set(tool_positions),
                barrier_positions=context.barrier_positions,
                lethal_features=context.lethal_features,
                goal_features={tool_feat},
                barrier_features=context.barrier_features,
                walkable_features=context.walkable_features,
                action_displacements=context.action_displacements,
                grid_shape=context.grid_shape,
            )
            path_to_tool = CausalTheoryPlanner.plan_path(
                start, grid, stage1_context, max_expansions=max_expansions
            )
            if not path_to_tool:
                continue

            curr_pos = start
            for act_id in path_to_tool:
                dr, dc = context.action_displacements.get(act_id, (0, 0))
                curr_pos = (curr_pos[0] + dr, curr_pos[1] + dc)

            # Stage 2: Barriers of this type become walkable
            unlocked_barriers = {
                pos
                for pos in context.barrier_positions
                if not (
                    0 <= pos[0] < H
                    and 0 <= pos[1] < W
                    and int(grid[pos[0], pos[1]]) in barrier_feats
                )
            }
            stage2_barrier_features = context.barrier_features - barrier_feats
            stage2_context = CausalPlanningContext(
                goal_positions=context.goal_positions,
                barrier_positions=unlocked_barriers,
                lethal_features=context.lethal_features,
                goal_features=context.goal_features,
                barrier_features=stage2_barrier_features,
                walkable_features=context.walkable_features | barrier_feats,
                action_displacements=context.action_displacements,
                grid_shape=context.grid_shape,
            )
            path_to_goal = CausalTheoryPlanner.plan_path(
                curr_pos, grid, stage2_context, max_expansions=max_expansions
            )
            if path_to_goal:
                logger.info(
                    "CausalTheoryPlanner: Synthesized multi-stage chained plan "
                    "(Tool %d: %d steps -> Goal: %d steps)",
                    tool_feat,
                    len(path_to_tool),
                    len(path_to_goal),
                )
                return path_to_tool + path_to_goal

        return None

    @staticmethod
    def plan_relational_macro(
        start: tuple[int, int],
        grid: np.ndarray,
        context: CausalPlanningContext,
        macro: RelationalSubgoalMacro,
        max_expansions_per_segment: int = 300,
    ) -> list[int] | None:
        """Plan composite path dynamically through a sequence of relational waypoints."""
        if not macro.waypoints:
            return None

        H, W = context.grid_shape
        if H <= 0 or W <= 0:
            H, W = grid.shape[:2]
        curr_start = start
        total_path: list[int] = []

        for wp in macro.waypoints:
            target_positions: list[tuple[int, int]] = []
            for r in range(H):
                for c in range(W):
                    if int(grid[r, c]) == wp.feature and (r, c) != curr_start:
                        target_positions.append((r, c))

            if not target_positions:
                return None

            segment_context = CausalPlanningContext(
                goal_positions=set(target_positions),
                barrier_positions=context.barrier_positions,
                lethal_features=context.lethal_features,
                goal_features={wp.feature},
                barrier_features=context.barrier_features,
                walkable_features=context.walkable_features,
                action_displacements=context.action_displacements,
                grid_shape=context.grid_shape,
            )
            segment_path = CausalTheoryPlanner.plan_path(
                curr_start, grid, segment_context, max_expansions=max_expansions_per_segment
            )
            if not segment_path:
                return None

            total_path.extend(segment_path)
            for act_id in segment_path:
                dr, dc = context.action_displacements.get(act_id, (0, 0))
                curr_start = (curr_start[0] + dr, curr_start[1] + dc)

        return total_path if total_path else None
