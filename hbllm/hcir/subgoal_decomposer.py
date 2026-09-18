"""Hierarchical Subgoal Decomposer — Recursive Obstacle Decomposition & Subgoal Dependency Trees.

Enables the cognitive architecture to:
1. Detect when a primary goal is obstructed by barriers, closed doors, or locked zones.
2. Inspect environment affordances and state mutations (switches, keys, transformer tiles).
3. Synthesize an HCIR dependency graph (HCIREdgeType.DEPENDS_ON) of prerequisite subgoals.
4. Dynamically advance through subgoals as prerequisite milestones are resolved.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from hbllm.hcir.graph import (
    GoalLifecycle,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
)
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.predictors.physics import PhysicsPredictor

logger = logging.getLogger(__name__)


class EpistemicFrontierDetector:
    """Detects and ranks epistemic exploration frontiers in partially observable spatial domains."""

    @staticmethod
    def detect_frontiers(
        avatar_pos: tuple[int, int],
        unobserved_mask: np.ndarray,
        barrier_cells: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int = 1,
        known_walls: set[tuple[int, int]] | None = None,
        deadlock_cells: set[tuple[int, int]] | None = None,
    ) -> list[tuple[tuple[int, int], float]]:
        """Identify reachable cells bordering unobserved space, ranked by barrier-aware information gain.

        Filters out unobserved cells that are known solid barriers and ensures candidate frontiers
        have accessible orthogonal exposure to unobserved regions without corner occlusion.

        Returns a list of ((r, c), info_score) tuples sorted descending by score.
        """
        H, W = grid_shape
        if not np.any(unobserved_mask):
            return []

        all_barriers = set(barrier_cells)
        if known_walls:
            all_barriers.update(known_walls)
        deadlocks = set(deadlock_cells) if deadlock_cells else set()

        candidates: list[tuple[tuple[int, int], float]] = []
        visited = {avatar_pos}
        queue = [avatar_pos]

        frontier_cells: list[tuple[int, int]] = []

        # BFS from avatar to find physically reachable cells
        while queue:
            curr_r, curr_c = queue.pop(0)

            # Check direct orthogonal adjacency to valid (non-barrier) unobserved cells
            cardinal_unobserved = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if unobserved_mask[nr, nc] and (nr, nc) not in all_barriers:
                        cardinal_unobserved += 1

            # Check diagonal adjacency only if not blocked by adjacent cardinal walls
            diagonal_unobserved = 0
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if unobserved_mask[nr, nc] and (nr, nc) not in all_barriers:
                        # Ensure both cardinal projections are not solid walls (corner pinch)
                        if (curr_r + dr, curr_c) not in all_barriers or (
                            curr_r,
                            curr_c + dc,
                        ) not in all_barriers:
                            diagonal_unobserved += 1

            total_open_unobserved = cardinal_unobserved + diagonal_unobserved
            if (
                total_open_unobserved > 0
                and cardinal_unobserved > 0
                and (curr_r, curr_c) != avatar_pos
                and (curr_r, curr_c) not in deadlocks
            ):
                frontier_cells.append((curr_r, curr_c))

            # Expand neighbors
            for dr, dc in [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if (
                        (nr, nc) not in visited
                        and (nr, nc) not in all_barriers
                        and not unobserved_mask[nr, nc]
                    ):
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        # Rank frontiers by info gain: proximity to avatar, high adjacent traversable unobserved cells
        for fr, fc in frontier_cells:
            dist = math.hypot(fr - avatar_pos[0], fc - avatar_pos[1])
            rad = max(2, step_size * 2)
            r_min, r_max = max(0, fr - rad), min(H, fr + rad + 1)
            c_min, c_max = max(0, fc - rad), min(W, fc + rad + 1)

            # Count unobserved cells in window that are NOT known solid barriers
            traversable_unobserved = 0
            for r in range(r_min, r_max):
                for c in range(c_min, c_max):
                    if unobserved_mask[r, c] and (r, c) not in all_barriers:
                        traversable_unobserved += 1

            # Deadlock proximity penalty
            deadlock_penalty = 1.0
            if any(math.hypot(fr - dr, fc - dc) <= 1.0 for dr, dc in deadlocks):
                deadlock_penalty = 0.2

            info_score = (float(traversable_unobserved) * deadlock_penalty) / (1.0 + dist * 0.1)
            candidates.append(((fr, fc), info_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates


class HierarchicalGoalDecomposer:
    """Decomposes complex or obstructed goals into an HCIR dependency graph of subgoals."""

    def __init__(self) -> None:
        self.completed_subgoals: set[str] = set()

    def decompose_goal(
        self,
        workspace: HCIRWorkspaceState,
        primary_goal: GoalNode,
        avatar_pos: tuple[int, int],
        barrier_cells: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        candidate_subgoals: list[dict[str, Any]] | None = None,
        step_size: int = 1,
        force_subgoals: bool = False,
        unobserved_mask: np.ndarray | None = None,
    ) -> GoalNode:
        """Analyze reachability of primary goal and return the active prerequisite GoalNode.

        If the primary goal path is clear, returns the primary goal.
        If the primary goal is obstructed (or force_subgoals is True), inspects candidate subgoals
        (switches, keys, tiles, delivery items), generates an HCIR GoalNode linked via DEPENDS_ON,
        and returns the active leaf subgoal.
        If no subgoals are reachable but unobserved_mask is provided, returns an epistemic exploration frontier.
        """
        # Ensure primary goal is in workspace
        if workspace.graph.get_node(primary_goal.id) is None:
            workspace.upsert_node(primary_goal)

        target_pos = primary_goal.properties.get("target_position")
        if target_pos is None:
            return primary_goal

        goal_r, goal_c = int(target_pos[0]), int(target_pos[1])

        # 1. Check if primary goal has an unobstructed geodesic path through known space
        effective_barriers = set(barrier_cells)
        if unobserved_mask is not None and np.any(unobserved_mask):
            unobserved_cells = set(zip(*np.where(unobserved_mask)))
            effective_barriers.update(unobserved_cells)

        primary_path = PhysicsPredictor.compute_geodesic_path(
            start=avatar_pos,
            goal=(goal_r, goal_c),
            barrier_cells=effective_barriers,
            grid_shape=grid_shape,
            step_size=step_size,
        )

        is_primary_reachable = bool(
            primary_path
            and math.hypot(goal_r - primary_path[-1][0], goal_c - primary_path[-1][1])
            <= step_size * 0.95
        )

        # Check existing registered subgoals for this primary goal
        active_prerequisites: list[GoalNode] = []
        for edge in workspace.graph.edges_from(primary_goal.id):
            if edge.edge_type == HCIREdgeType.DEPENDS_ON:
                for tid in edge.targets:
                    node = workspace.graph.get_node(tid)
                    if isinstance(node, GoalNode) and not getattr(node, "resolved", False):
                        active_prerequisites.append(node)

        # If primary goal is reachable and all prerequisites are resolved, focus on primary goal
        if is_primary_reachable and not active_prerequisites and not force_subgoals:
            return primary_goal

        # If there are already active prerequisite subgoals, return the closest unresolved one
        if active_prerequisites:
            active_prerequisites.sort(
                key=lambda g: math.hypot(
                    g.properties.get("target_position", (0, 0))[0] - avatar_pos[0],
                    g.properties.get("target_position", (0, 0))[1] - avatar_pos[1],
                )
            )
            return active_prerequisites[0]

        # 2. Obstructed: Synthesize new prerequisite subgoals from candidates
        if candidate_subgoals:
            reachable_candidates: list[tuple[int, dict[str, Any]]] = []
            for cand in candidate_subgoals:
                c_pos = cand.get("position")
                if c_pos is None:
                    continue
                cand_r, cand_c = int(c_pos[0]), int(c_pos[1])
                # Skip if already completed
                cand_id = str(cand.get("id", f"{cand_r}_{cand_c}"))
                if cand_id in self.completed_subgoals:
                    continue

                # Check reachability of candidate
                c_path = PhysicsPredictor.compute_geodesic_path(
                    start=avatar_pos,
                    goal=(cand_r, cand_c),
                    barrier_cells=barrier_cells,
                    grid_shape=grid_shape,
                    step_size=step_size,
                )
                if (
                    c_path
                    and math.hypot(cand_r - c_path[-1][0], cand_c - c_path[-1][1])
                    <= step_size * 1.5
                ):
                    reachable_candidates.append((len(c_path), cand))

            if reachable_candidates:
                reachable_candidates.sort(key=lambda item: item[0])
                best_cand = reachable_candidates[0][1]
                cand_pos = (int(best_cand["position"][0]), int(best_cand["position"][1]))
                cand_id = str(best_cand.get("id", f"{cand_pos[0]}_{cand_pos[1]}"))

                subgoal_id = f"subgoal_{cand_id}"
                subgoal = GoalNode(
                    id=subgoal_id,
                    description=best_cand.get(
                        "description", f"Activate prerequisite milestone at {cand_pos}"
                    ),
                    priority=min(1.0, max(0.0, float(primary_goal.priority))),
                    resolved=False,
                    properties={
                        "target_position": cand_pos,
                        "target_entity": cand_id,
                        "parent_goal_id": primary_goal.id,
                        "affordance": best_cand.get("affordance", "INTERACTION"),
                    },
                )
                workspace.upsert_node(subgoal)

                # Link: primary_goal DEPENDS_ON subgoal
                dep_edge = HCIREdge(
                    edge_type=HCIREdgeType.DEPENDS_ON,
                    sources=[primary_goal.id],
                    targets=[subgoal_id],
                    weight=1.0,
                )
                workspace.graph.add_edge(dep_edge)

                logger.info(
                    "HierarchicalGoalDecomposer: Obstructed goal '%s' decomposed into prerequisite '%s' at %s",
                    primary_goal.id,
                    subgoal_id,
                    cand_pos,
                )
                return subgoal

        # 3. If primary goal is obstructed, no subgoals are reachable, but unobserved_mask is provided:
        # Detect and return the best epistemic exploration frontier
        if unobserved_mask is not None and np.any(unobserved_mask):
            frontiers = EpistemicFrontierDetector.detect_frontiers(
                avatar_pos=avatar_pos,
                unobserved_mask=unobserved_mask,
                barrier_cells=barrier_cells,
                grid_shape=grid_shape,
                step_size=step_size,
            )
            if frontiers:
                best_frontier_pos, info_gain = frontiers[0]
                fr, fc = best_frontier_pos
                frontier_subgoal_id = f"epistemic_frontier_{fr}_{fc}"
                frontier_subgoal = GoalNode(
                    id=frontier_subgoal_id,
                    description=f"Explore unobserved spatial frontier at {(fr, fc)} (info_gain={info_gain:.2f})",
                    priority=min(1.0, max(0.0, float(primary_goal.priority))),
                    resolved=False,
                    properties={
                        "target_position": (fr, fc),
                        "is_epistemic": True,
                        "information_gain": info_gain,
                        "affordance": "CONTACT",
                    },
                )
                workspace.upsert_node(frontier_subgoal)
                dep_edge = HCIREdge(
                    edge_type=HCIREdgeType.DEPENDS_ON,
                    sources=[primary_goal.id],
                    targets=[frontier_subgoal_id],
                    weight=1.0,
                )
                workspace.graph.add_edge(dep_edge)
                logger.info(
                    "HierarchicalGoalDecomposer: Obstructed goal '%s' decomposed into epistemic frontier '%s' at (%d, %d)",
                    primary_goal.id,
                    frontier_subgoal_id,
                    fr,
                    fc,
                )
                return frontier_subgoal

        return primary_goal

    def decompose_compound_preconditions(
        self,
        workspace: HCIRWorkspaceState,
        primary_goal: GoalNode,
        avatar_pos: tuple[int, int],
        current_state: dict[str, Any],
        required_preconditions: dict[str, Any],
        available_modifiers: list[dict[str, Any]],
        barrier_cells: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        step_size: int = 1,
    ) -> GoalNode:
        """Decompose a goal requiring multiple prerequisite state conditions.

        Finds unsatisfied preconditions, pairs them with available modifying entities/tiles,
        constructs a dependency tree linked with DEPENDS_ON, and returns the next immediate subgoal.
        """
        if workspace.graph.get_node(primary_goal.id) is None:
            workspace.upsert_node(primary_goal)

        # Identify unsatisfied preconditions
        unsatisfied: list[tuple[str, Any]] = []
        for prop, req_val in required_preconditions.items():
            if current_state.get(prop) != req_val:
                unsatisfied.append((prop, req_val))

        if not unsatisfied:
            return primary_goal

        # Find available modifiers for unsatisfied preconditions
        parent_id = primary_goal.id
        immediate_subgoal: GoalNode | None = None

        for prop, req_val in unsatisfied:
            matching_modifiers = [
                m
                for m in available_modifiers
                if m.get("property") == prop and m.get("value") == req_val
            ]
            if not matching_modifiers:
                matching_modifiers = [m for m in available_modifiers if m.get("property") == prop]

            if matching_modifiers:
                matching_modifiers.sort(
                    key=lambda m: math.hypot(
                        m.get("position", (0, 0))[0] - avatar_pos[0],
                        m.get("position", (0, 0))[1] - avatar_pos[1],
                    )
                )
                best_mod = matching_modifiers[0]
                m_pos = (int(best_mod["position"][0]), int(best_mod["position"][1]))
                subgoal_id = (
                    f"subgoal_precond_{prop}_{best_mod.get('id', f'{m_pos[0]}_{m_pos[1]}')}"
                )

                existing = workspace.graph.get_node(subgoal_id)
                if isinstance(existing, GoalNode) and existing.resolved:
                    continue

                subgoal = GoalNode(
                    id=subgoal_id,
                    description=f"Satisfy precondition '{prop}={req_val}' at {m_pos}",
                    priority=min(1.0, max(0.0, float(primary_goal.priority))),
                    resolved=False,
                    properties={
                        "target_position": m_pos,
                        "target_entity": best_mod.get("id", subgoal_id),
                        "precondition_property": prop,
                        "target_value": req_val,
                        "affordance": best_mod.get("affordance", "CONTACT"),
                    },
                )
                workspace.upsert_node(subgoal)

                dep_edge = HCIREdge(
                    edge_type=HCIREdgeType.DEPENDS_ON,
                    sources=[parent_id],
                    targets=[subgoal_id],
                    weight=1.0,
                )
                workspace.graph.add_edge(dep_edge)

                if immediate_subgoal is None:
                    immediate_subgoal = subgoal
                parent_id = subgoal_id

        return immediate_subgoal or primary_goal

    def resolve_subgoal(self, workspace: HCIRWorkspaceState, subgoal_id: str) -> None:
        """Mark a completed prerequisite subgoal as resolved in HCIR."""
        node = workspace.graph.get_node(subgoal_id)
        if isinstance(node, GoalNode):
            node.resolved = True
            node.goal_lifecycle = GoalLifecycle.COMPLETED
            workspace.upsert_node(node)
            target_ent = node.properties.get("target_entity", subgoal_id)
            self.completed_subgoals.add(str(target_ent))
            self.completed_subgoals.add(subgoal_id)
            logger.info("HierarchicalGoalDecomposer: Resolved subgoal '%s'", subgoal_id)

    @staticmethod
    def check_subgoal_completion(
        subgoal: GoalNode, avatar_pos: tuple[int, int], tolerance: float = 0.5
    ) -> bool:
        """Check whether the avatar position satisfies the subgoal target."""
        # If the prerequisite requires a discrete physical action (e.g. INTERACTION),
        # spatial proximity alone does not resolve the subgoal.
        affordance = subgoal.properties.get("affordance")
        if affordance == "INTERACTION":
            return False
        t_pos = subgoal.properties.get("target_position")
        if t_pos:
            d = math.hypot(t_pos[0] - avatar_pos[0], t_pos[1] - avatar_pos[1])
            return d <= tolerance
        return False

    def decompose_pattern_alignment_into_subgoals(
        self,
        workspace: HCIRWorkspaceState,
        primary_goal: GoalNode,
        canvas_matrix: np.ndarray,
        target_template: np.ndarray,
        available_tools: list[dict[str, Any]],
        current_tool_id: Any | None = None,
        current_palette_color: int | None = None,
        palette_swatches: list[dict[str, Any]] | None = None,
    ) -> GoalNode | None:
        """Decompose a pattern alignment goal into tool selection, color selection, and application subgoals.

        Finds the tool and color that maximize visual Hamming distance reduction between the
        editable canvas and the reference template.
        """
        if canvas_matrix.shape != target_template.shape:
            return None

        diff_mask = canvas_matrix != target_template
        if not np.any(diff_mask):
            return None

        best_tool: dict[str, Any] | None = None
        best_target_color: int | None = None
        best_net_gain: int = -999999

        for tool in available_tools:
            mask = tool.get("mask")
            if mask is None or mask.shape != canvas_matrix.shape:
                continue

            # Check mismatched pixels within this tool's application sector
            sector_diff = mask & diff_mask
            if not np.any(sector_diff):
                continue

            # Candidate color: most frequent color in target template within the sector
            target_colors = target_template[sector_diff]
            if len(target_colors) == 0:
                continue
            cand_color = int(np.bincount(target_colors).argmax())

            # Evaluate net improvement if we stamp with cand_color
            corrects = int(
                np.sum((canvas_matrix != cand_color) & (target_template == cand_color) & mask)
            )
            corrupts = int(
                np.sum((canvas_matrix == cand_color) & (target_template != cand_color) & mask)
            )
            net_gain = corrects - corrupts

            if net_gain > best_net_gain:
                best_net_gain = net_gain
                best_tool = tool
                best_target_color = cand_color

        if best_tool is None or best_target_color is None or best_net_gain <= 0:
            return None

        tool_id = best_tool.get("tool_id")
        # Step 1: Precondition - Palette color match
        if current_palette_color is not None and current_palette_color != best_target_color:
            swatch_coord = None
            if palette_swatches:
                for sw in palette_swatches:
                    if sw.get("color") == best_target_color:
                        swatch_coord = sw.get("coord")
                        break

            subgoal_id = f"subgoal_select_color_{best_target_color}"
            color_subgoal = GoalNode(
                id=subgoal_id,
                description=f"Select palette color {best_target_color} for pattern alignment",
                priority=min(1.0, max(0.0, float(primary_goal.priority))),
                resolved=False,
                properties={
                    "affordance": "SELECT_COLOR",
                    "target_color": best_target_color,
                    "target_position": swatch_coord,
                    "parent_goal_id": primary_goal.id,
                },
            )
            workspace.upsert_node(color_subgoal)
            return color_subgoal

        # Step 2: Precondition - Tool alignment match
        if current_tool_id is not None and current_tool_id != tool_id:
            subgoal_id = f"subgoal_align_tool_{tool_id}"
            tool_subgoal = GoalNode(
                id=subgoal_id,
                description=f"Align tool selector to sector {tool_id}",
                priority=min(1.0, max(0.0, float(primary_goal.priority))),
                resolved=False,
                properties={
                    "affordance": "ALIGN_TOOL",
                    "target_tool": tool_id,
                    "nav_position": best_tool.get("nav_position"),
                    "parent_goal_id": primary_goal.id,
                },
            )
            workspace.upsert_node(tool_subgoal)
            return tool_subgoal

        # Step 3: Application - Apply stamp
        subgoal_id = f"subgoal_apply_stamp_{tool_id}"
        stamp_subgoal = GoalNode(
            id=subgoal_id,
            description=f"Apply tool stamp for sector {tool_id}",
            priority=min(1.0, max(0.0, float(primary_goal.priority))),
            resolved=False,
            properties={
                "affordance": "APPLY_STAMP",
                "target_tool": tool_id,
                "action_id": best_tool.get("application_action", 5),
                "parent_goal_id": primary_goal.id,
            },
        )
        workspace.upsert_node(stamp_subgoal)
        return stamp_subgoal
