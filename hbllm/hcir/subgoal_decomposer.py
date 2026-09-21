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
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.graph import (
    GoalLifecycle,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
    WorldVariableNode,
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
        resource_budget: int | None = None,
        recharge_stations: list[dict[str, Any]] | None = None,
    ) -> GoalNode:
        """Analyze reachability of primary goal and return the active prerequisite GoalNode.

        If the primary goal path is clear and within budget, returns the primary goal.
        If the primary goal is obstructed, exceeds resource budget, or force_subgoals is True,
        inspects candidate subgoals / recharge stations, generates an HCIR GoalNode linked via DEPENDS_ON,
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

        # Check resource constraint
        if is_primary_reachable and resource_budget is not None and primary_path:
            if len(primary_path) > resource_budget:
                is_primary_reachable = False

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

        # Resource budgeting: if primary goal exceeds budget, synthesize nearest reachable recharge subgoal
        if (
            not is_primary_reachable
            and resource_budget is not None
            and recharge_stations
            and not active_prerequisites
        ):
            best_station: dict[str, Any] | None = None
            best_s_path: list[tuple[int, int]] | None = None
            min_s_len = float("inf")
            for station in recharge_stations:
                s_pos = station.get("position")
                if not s_pos:
                    continue
                s_r, s_c = int(s_pos[0]), int(s_pos[1])
                s_path = PhysicsPredictor.compute_geodesic_path(
                    start=avatar_pos,
                    goal=(s_r, s_c),
                    barrier_cells=effective_barriers,
                    grid_shape=grid_shape,
                    step_size=step_size,
                )
                if s_path and len(s_path) <= resource_budget and len(s_path) < min_s_len:
                    min_s_len = len(s_path)
                    best_station = station
                    best_s_path = s_path

            if best_station and best_s_path:
                st_pos = best_station["position"]
                recharge_node = GoalNode(
                    id=f"subgoal_recharge_{int(st_pos[0])}_{int(st_pos[1])}",
                    description="Recharge resource budget at recharge station",
                    properties={
                        "target_position": (int(st_pos[0]), int(st_pos[1])),
                        "is_recharge": True,
                    },
                )
                workspace.upsert_node(recharge_node)
                workspace.add_edge(
                    HCIREdge(
                        sources=[primary_goal.id],
                        targets=[recharge_node.id],
                        edge_type=HCIREdgeType.DEPENDS_ON,
                    )
                )
                return recharge_node

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
                # Prioritize candidates matching learned target features from HCIR workspace
                target_feat_vars = workspace.graph.get_node(
                    "var_target_entity_features"
                ) or workspace.graph.get_node("var_learned_item_colors")
                learned_features = (
                    set(target_feat_vars.value)
                    if isinstance(target_feat_vars, WorldVariableNode)
                    and isinstance(target_feat_vars.value, list)
                    else set()
                )

                def candidate_priority(item_tuple: tuple[int, dict[str, Any]]) -> tuple[int, int]:
                    path_len, cand_dict = item_tuple
                    c_feat = (
                        cand_dict.get("feature_id")
                        or cand_dict.get("visual_id")
                        or cand_dict.get("color")
                    )
                    is_known = 0 if (c_feat is not None and c_feat in learned_features) else 1
                    return (is_known, path_len)

                reachable_candidates.sort(key=candidate_priority)
                best_cand = reachable_candidates[0][1]
                cand_pos = (int(best_cand["position"][0]), int(best_cand["position"][1]))
                cand_id = str(best_cand.get("id", f"{cand_pos[0]}_{cand_pos[1]}"))
                cand_feat = (
                    best_cand.get("feature_id")
                    or best_cand.get("visual_id")
                    or best_cand.get("color")
                )

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
                        "item_position": best_cand.get("item_position", cand_pos),
                        "target_feature": cand_feat,
                        "item_color": cand_feat,
                        "item_area": best_cand.get("area"),
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

    @staticmethod
    def decompose_with_skills(
        workspace: HCIRWorkspaceState,
        primary_goal: GoalNode,
        skills: list[HCIRSkill],
        current_state: dict[str, Any],
    ) -> list[GoalNode]:
        """Synthesize an ordered sequence of subgoals matching available skills against unmet preconditions.

        Builds a dependency graph of subgoals in the HCIR workspace where each subgoal represents
        an executable skill step, linked via HCIREdgeType.DEPENDS_ON edges.
        """
        subgoals: list[GoalNode] = []
        prev_subgoal: GoalNode | None = None

        for skill in skills:
            # Check if skill effect is already satisfied in current_state
            is_satisfied = True
            for k, expected_v in skill.expected_effect.items():
                if current_state.get(k) != expected_v:
                    is_satisfied = False
                    break

            if is_satisfied:
                continue

            subgoal_id = f"subgoal_skill_{skill.skill_id}"
            node = GoalNode(
                id=subgoal_id,
                description=f"Execute skill {skill.skill_id} to satisfy {skill.expected_effect}",
                priority=min(1.0, max(0.0, float(primary_goal.priority) * skill.confidence)),
                resolved=False,
                properties={
                    "skill_id": skill.skill_id,
                    "action_sequence": list(skill.action_sequence),
                    "action_data_sequence": list(skill.action_data_sequence),
                    "expected_effect": dict(skill.expected_effect),
                    "preconditions": dict(skill.preconditions),
                    "parent_goal_id": primary_goal.id,
                },
            )
            workspace.upsert_node(node)
            subgoals.append(node)

            if prev_subgoal is not None:
                # Link sequential dependency: node depends on prev_subgoal
                dep_edge = HCIREdge(
                    edge_type=HCIREdgeType.DEPENDS_ON,
                    sources=[node.id],
                    targets=[prev_subgoal.id],
                    weight=1.0,
                )
                workspace.graph.add_edge(dep_edge)

            prev_subgoal = node

        return subgoals

    @staticmethod
    def order_multibox_deliveries(
        boxes: list[tuple[int, int]],
        targets: list[tuple[int, int]],
        barrier_cells: set[tuple[int, int]] | None = None,
        avatar_pos: tuple[int, int] | None = None,
        bottleneck_cells: set[tuple[int, int]] | None = None,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Computes an optimal, deadlock-free matching and delivery order of boxes to targets.

        1. Uses Hungarian matching (scipy.optimize.linear_sum_assignment) on Manhattan distance
           between candidate boxes and targets to minimize total transportation displacement.
        2. Orders matched pairs topologically to prevent corridor / target-room deadlocks:
           - Boxes that sit in or obstruct bottlenecks/doorways are prioritized for first delivery.
           - Targets positioned deeper in enclosed target zones (furthest from entrances/bottlenecks)
             are filled first so subsequent deliveries are not obstructed.
           - If depths are equal, prioritizes boxes closest to avatar.
        """
        if not boxes or not targets:
            return []

        try:
            from scipy.optimize import linear_sum_assignment

            has_scipy = True
        except ImportError:
            has_scipy = False

        num_boxes = len(boxes)
        num_targets = len(targets)
        cost_matrix = np.zeros((num_boxes, num_targets), dtype=float)

        for i, (br, bc) in enumerate(boxes):
            for j, (tr, tc) in enumerate(targets):
                cost_matrix[i, j] = abs(br - tr) + abs(bc - tc)

        if has_scipy:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_pairs = [(boxes[r], targets[c]) for r, c in zip(row_ind, col_ind)]
        else:
            matched_pairs = []
            used_targets = set()
            for r, b in enumerate(boxes):
                best_c = -1
                best_dist = float("inf")
                for c, t in enumerate(targets):
                    if c not in used_targets and cost_matrix[r, c] < best_dist:
                        best_dist = cost_matrix[r, c]
                        best_c = c
                if best_c >= 0:
                    used_targets.add(best_c)
                    matched_pairs.append((b, targets[best_c]))

        def pair_priority(
            pair: tuple[tuple[int, int], tuple[int, int]],
        ) -> tuple[int, float, float]:
            box, target = pair
            is_bottleneck_box = 0 if (bottleneck_cells and box in bottleneck_cells) else 1

            if bottleneck_cells:
                min_bottleneck_dist = min(
                    math.hypot(target[0] - bnr, target[1] - bnc) for bnr, bnc in bottleneck_cells
                )
                target_depth_score = -min_bottleneck_dist
            else:
                target_depth_score = 0.0

            avatar_dist = (
                math.hypot(box[0] - avatar_pos[0], box[1] - avatar_pos[1])
                if avatar_pos is not None
                else 0.0
            )

            return (is_bottleneck_box, target_depth_score, avatar_dist)

        matched_pairs.sort(key=pair_priority)
        return matched_pairs


@dataclass
class HCIRSkill:
    """A learned procedural skill representing an executable action policy for achieving a sub-state."""

    skill_id: str
    preconditions: dict[str, Any] = field(default_factory=dict)
    action_sequence: list[int] = field(default_factory=list)
    action_data_sequence: list[dict[str, Any] | None] = field(default_factory=list)
    expected_effect: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    times_executed: int = 0
    times_succeeded: int = 0
