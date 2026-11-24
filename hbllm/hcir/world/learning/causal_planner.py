"""Causal Theory Planner & Relational Subgoal Scheduling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    failed_transitions: set[tuple[tuple[int, int], int]] = field(default_factory=set)
    inhibited_actions: set[Any] = field(default_factory=set)
    transition_kb: Any | None = None
    active_state_tokens: frozenset[str] = field(default_factory=frozenset)


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

        # Extract goal positions from grid using discovered TransitionKB / theory beliefs
        goal_positions: set[tuple[int, int]] = set(engine.learned_goal_positions)
        barrier_positions: set[tuple[int, int]] = set(engine.learned_barriers)
        has_loci = bool(theory.transition_kb._locus_beliefs)

        # Scan grid for goal and barrier cells
        for r in range(curr_grid.shape[0]):
            for c in range(curr_grid.shape[1]):
                pos = (r, c)
                feat = int(curr_grid[r, c])
                ev = theory.evaluate_transition(feat, locus=pos)
                if ev.goal_value > 0.5 or feat in theory.goal_features:
                    goal_positions.add(pos)
                if not ev.traversable or ev.risk_cost >= 40.0 or feat in theory.barrier_features:
                    barrier_positions.add(pos)
                elif (
                    has_loci
                    and ev.traversable
                    and ev.confidence > 0.0
                    and theory.transition_kb.get_belief(pos, scope="locus") is not None
                ):
                    barrier_positions.discard(pos)

        lethal_features = set(engine.episodic_cortex.discovered_lethal_features) | set(
            engine.hazard_tracker.known_lethal_features
        )
        mob_feats = set(engine.mobile_threat_features) | (
            set(engine.hazard_tracker.mobile_entities)
            - set(theory.cargo_features)
            - set(engine.learned_cargo_features)
        )
        shroud_f = engine.working_memory.shroud_feature

        wm_exog = engine.working_memory
        visited_cells = wm_exog.visited_cells
        exhausted = get_sub_attr(wm_exog, "exhausted_candidate_goals", set())
        bg_val = 0
        if curr_grid.size > 0:
            vals, cnts = np.unique(curr_grid, return_counts=True)
            bg_val = int(vals[int(np.argmax(cnts))])

        # Fallback to candidate goal features if no concrete goal cells located
        if not goal_positions and theory.candidate_goal_features:
            for r in range(curr_grid.shape[0]):
                for c in range(curr_grid.shape[1]):
                    cp = (r, c)
                    if cp in visited_cells or cp in exhausted or theory.is_locus_explored(cp):
                        continue
                    feat = int(curr_grid[r, c])
                    if (
                        feat in theory.candidate_goal_features
                        and feat not in lethal_features
                        and feat not in mob_feats
                        and feat != shroud_f
                    ):
                        goal_positions.add(cp)

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
                    added_nonexog = False
                    if sg.get("cells"):
                        for cp in sg["cells"]:
                            if (
                                cp != engine.avatar_pos
                                and cp not in visited_cells
                                and cp not in exhausted
                                and not theory.is_locus_explored(cp)
                                and not wm_exog.is_exogenous_locus(
                                    cp, lattice=curr_grid, bg=bg_val, include_outer_edge=False
                                )
                            ):
                                goal_positions.add(cp)
                                added_nonexog = True
                    elif (
                        sg.get("position")
                        and sg["position"] != engine.avatar_pos
                        and sg["position"] not in visited_cells
                        and sg["position"] not in exhausted
                        and not theory.is_locus_explored(sg["position"])
                        and not wm_exog.is_exogenous_locus(
                            sg["position"], lattice=curr_grid, bg=bg_val, include_outer_edge=False
                        )
                    ):
                        goal_positions.add(sg["position"])
                        added_nonexog = True
                    if (
                        added_nonexog
                        and sg_f is not None
                        and int(np.count_nonzero(curr_grid == int(sg_f))) == 1
                    ):
                        theory.induce_candidate_goal(int(sg_f), evidence="structural_goal_detector")
            except Exception as exc:
                logger.debug("Structural goal induction at reset skipped: %s", exc)

        goal_positions = {
            gp
            for gp in goal_positions
            if not wm_exog.is_exogenous_locus(
                gp, lattice=curr_grid, bg=bg_val, include_outer_edge=False
            )
        }

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

        # Extract action displacements from harmonized engine action_deltas
        action_displacements = dict(engine.action_deltas)
        inhibited_actions = {act_id for act_id, ttl in engine.inhibited_actions.items() if ttl > 0}
        failed_transitions = set(engine.failed_transitions)

        return CausalPlanningContext(
            goal_positions=goal_positions,
            barrier_positions=barrier_positions,
            lethal_features=lethal_features,
            goal_features=set(theory.goal_features),
            barrier_features=set(theory.barrier_features),
            walkable_features=set(theory.walkable_features),
            action_displacements=action_displacements,
            grid_shape=(int(curr_grid.shape[0]), int(curr_grid.shape[1])),
            failed_transitions=failed_transitions,
            inhibited_actions=inhibited_actions,
            transition_kb=theory.transition_kb,
            active_state_tokens=frozenset(),
        )

    @staticmethod
    def is_position_safe(
        pos: tuple[int, int],
        grid: np.ndarray,
        context: CausalPlanningContext,
    ) -> bool:
        """Check if a position is safe to traverse based on causal theory and transition KB."""
        r, c = pos
        if r < 0 or c < 0 or r >= context.grid_shape[0] or c >= context.grid_shape[1]:
            return False
        feat = int(grid[r, c])
        if feat in context.lethal_features:
            return False
        is_goal = pos in context.goal_positions
        if context.transition_kb is not None:
            ev = context.transition_kb.evaluate_transition(
                feat,
                locus=pos,
                active_state_tokens=context.active_state_tokens,
            )
            if ev.risk_cost >= 40.0 and ev.goal_value <= 0.0:
                return False
            if ev.confidence > 0.0:
                if not is_goal and not ev.traversable:
                    return False
                locus_belief = context.transition_kb.get_belief(pos, scope="locus")
                if (
                    not is_goal
                    and pos in context.barrier_positions
                    and not (locus_belief is not None and ev.traversable)
                ):
                    return False
                return True
        if not is_goal:
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
        """A* pathfinding via GridTopologyGraph.shortest_path_from_kb and TransitionKB."""
        if not context.goal_positions or not context.action_displacements:
            return None

        from hbllm.hcir.world.causal.transition_kb import TransitionEdgeEvaluation
        from hbllm.hcir.world.topological_graph_induction import GridTopologyGraph

        H, W = context.grid_shape

        def _eval_edge(pos: tuple[int, int], feat: int) -> TransitionEdgeEvaluation:
            r, c = pos
            if r < 0 or c < 0 or r >= H or c >= W or feat in context.lethal_features:
                return TransitionEdgeEvaluation(
                    traversable=False,
                    step_cost=1.0,
                    risk_cost=50.0,
                    goal_value=0.0,
                    info_gain=0.0,
                    mutates_state=False,
                    confidence=1.0,
                )
            safe = CausalTheoryPlanner.is_position_safe(pos, grid, context)
            is_goal = pos in context.goal_positions
            if context.transition_kb is not None:
                ev = context.transition_kb.evaluate_transition(
                    feat,
                    locus=pos,
                    active_state_tokens=context.active_state_tokens,
                )
                return TransitionEdgeEvaluation(
                    traversable=safe,
                    step_cost=float(ev.step_cost),
                    risk_cost=float(ev.risk_cost),
                    goal_value=(max(float(ev.goal_value), 1.0) if (is_goal and safe) else 0.0),
                    info_gain=float(ev.info_gain),
                    mutates_state=bool(ev.mutates_state),
                    confidence=float(ev.confidence),
                )
            return TransitionEdgeEvaluation(
                traversable=safe,
                step_cost=1.0,
                risk_cost=0.0,
                goal_value=(1.0 if (is_goal and safe) else 0.0),
                info_gain=0.0,
                mutates_state=False,
                confidence=1.0,
            )

        steps = GridTopologyGraph.shortest_path_from_kb(
            grid,
            start,
            context.goal_positions,
            context.action_displacements,
            _eval_edge,
            max_expansions=max_expansions,
            failed_transitions=context.failed_transitions,
            inhibited_actions=context.inhibited_actions,
            allow_goal_entry=False,
        )
        if steps is None:
            return None
        return [int(act_id) for act_id, _ in steps]

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

        if (
            context.transition_kb is not None
            and context.goal_positions
            and context.transition_kb.query_causal_rules()
        ):
            from hbllm.hcir.world.topological_graph_induction import GridTopologyGraph

            causal_steps = GridTopologyGraph.shortest_path_with_causal_ops(
                grid,
                start,
                context.goal_positions,
                context.action_displacements,
                context.transition_kb,
                initial_state_tokens=context.active_state_tokens,
                failed_transitions=context.failed_transitions,
                inhibited_actions=context.inhibited_actions,
                max_expansions=max_expansions * 2,
            )
            if causal_steps is not None:
                return [int(act_id) for act_id, _ in causal_steps]

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
                failed_transitions=context.failed_transitions,
                inhibited_actions=context.inhibited_actions,
                transition_kb=context.transition_kb,
                active_state_tokens=context.active_state_tokens,
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
                failed_transitions=context.failed_transitions,
                inhibited_actions=context.inhibited_actions,
                transition_kb=context.transition_kb,
                active_state_tokens=context.active_state_tokens,
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
        *,
        visited_cells: set[tuple[int, int]] | None = None,
        working_memory: Any = None,
        bg: int = 0,
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
            if (
                wp.feature in context.barrier_features
                or wp.feature in context.walkable_features
                or wp.feature in context.lethal_features
            ):
                return None

            target_positions: list[tuple[int, int]] = []
            already_visited = (
                0 <= curr_start[0] < H
                and 0 <= curr_start[1] < W
                and int(grid[curr_start[0], curr_start[1]]) == wp.feature
            )
            for r in range(H):
                for c in range(W):
                    if int(grid[r, c]) == wp.feature:
                        if working_memory is not None and working_memory.is_exogenous_locus(
                            (r, c), lattice=grid, bg=bg, include_outer_edge=True
                        ):
                            continue
                        if visited_cells and (r, c) in visited_cells:
                            already_visited = True
                        if (r, c) != curr_start:
                            target_positions.append((r, c))

            if already_visited:
                continue
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
                failed_transitions=context.failed_transitions,
                inhibited_actions=context.inhibited_actions,
                transition_kb=context.transition_kb,
                active_state_tokens=context.active_state_tokens,
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
