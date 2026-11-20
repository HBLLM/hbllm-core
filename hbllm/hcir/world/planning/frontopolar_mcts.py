"""BA10 Frontopolar Subgoal Monte Carlo Tree Search Planner."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.frontopolar_subgoal_stack import FrontopolarSubgoal, SubgoalType
from hbllm.hcir.world.prefrontal_working_memory import HeldItem

logger = logging.getLogger(__name__)


@dataclass
class FrontopolarSubgoalMCTSNode:
    """MCTS node representing an abstract subgoal state in Frontopolar BA10 tree search."""

    subgoal: FrontopolarSubgoal | None
    state_avatar_pos: tuple[int, int]
    state_held_items: frozenset[int]
    state_cleared_barriers: frozenset[tuple[int, int]]
    parent: FrontopolarSubgoalMCTSNode | None = None
    children: list[FrontopolarSubgoalMCTSNode] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    untried_subgoals: list[FrontopolarSubgoal] = field(default_factory=list)
    depth: int = 0

    @property
    def q_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb1(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        if not self.parent or self.parent.visits == 0:
            return self.q_value
        return self.q_value + c * float(np.sqrt(np.log(self.parent.visits) / self.visits))


@dataclass
class FrontopolarSubgoalMCTSResult:
    """Outcome of BA10 Frontopolar Subgoal Monte Carlo Tree Search."""

    best_subgoals: list[FrontopolarSubgoal]
    total_iterations: int
    tree_depth: int
    expected_value: float
    pruned_dead_ends: int

    def to_causal_predicate(self) -> CausalPredicate:
        """Export MCTS plan outcome as a declarative causal predicate."""
        return CausalPredicate(
            variable="frontopolar_subgoal_mcts",
            operator="==",
            value=len(self.best_subgoals),
            relation=f"val:{self.expected_value:.2f}:pruned:{self.pruned_dead_ends}",
            target_entity="cortex_planner",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions registering the planned subgoal sequence."""
        ops: list[TransactionOperation] = []
        for idx, sg in enumerate(self.best_subgoals):
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=f"mcts_subgoal_{idx}_{sg.subgoal_id}",
                    node_data={
                        "order": idx,
                        "subgoal_id": sg.subgoal_id,
                        "subgoal_type": str(sg.subgoal_type),
                        "target_pos": list(sg.target_entity_pos) if sg.target_entity_pos else None,
                        "priority": sg.priority,
                    },
                )
            )
        return ops


class FrontopolarSubgoalMCTSPlanner:
    """BA10 Frontopolar Subgoal Monte Carlo Tree Search Planner.

    Executes hierarchical tree search over abstract subgoals rather than primitive steps:
    1. Selection: UCB1 exploration-exploitation over abstract subgoal nodes.
    2. Expansion & Physical/Habenular Dead-End Pruning:
       - Prunes branches requiring movement across impermeable barriers without required keys (Spelke continuity).
       - Prunes paths entering grounded Habenular lethal zones or death loci.
    3. Rollout Simulation:
       - Evaluates path connectivity from current state through candidates to terminal goal.
    4. Backpropagation:
       - Backpropagates path efficiency and reward up the subgoal hierarchy.
    """

    @staticmethod
    def _is_reachable(
        start: tuple[int, int],
        goal: tuple[int, int],
        effective_barriers: set[tuple[int, int]],
        grid_bounds: tuple[int, int],
    ) -> bool:
        """Check 4-way BFS topological reachability from start to goal."""
        if start == goal:
            return True
        H, W = grid_bounds
        queue = deque([start])
        visited = {start}

        while queue:
            curr = queue.popleft()
            if curr == goal:
                return True
            cr, cc = curr
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                nxt = (nr, nc)
                if 0 <= nr < H and 0 <= nc < W and nxt not in visited:
                    if nxt == goal or nxt not in effective_barriers:
                        visited.add(nxt)
                        queue.append(nxt)
        return False

    @classmethod
    def plan_subgoal_mcts(
        cls,
        avatar_pos: tuple[int, int],
        candidate_subgoals: Sequence[FrontopolarSubgoal],
        goal_positions: set[tuple[int, int]],
        barriers: set[tuple[int, int]],
        grid_bounds: tuple[int, int],
        held_items: set[int] | set[HeldItem] | Sequence[HeldItem] | None = None,
        lethal_positions: set[tuple[int, int]] | None = None,
        max_iterations: int = 48,
        exploration_weight: float = 1.414,
        max_depth: int = 5,
        body_schema: Any | None = None,
        tool_barrier_affinities: dict[int, set[int]] | None = None,
        curr_grid: np.ndarray | None = None,
    ) -> FrontopolarSubgoalMCTSResult:
        """Run Frontopolar Subgoal MCTS with Spelke/Habenular pruning and Extended Body Schema dynamics."""
        if not candidate_subgoals:
            return FrontopolarSubgoalMCTSResult(
                best_subgoals=[],
                total_iterations=0,
                tree_depth=0,
                expected_value=0.0,
                pruned_dead_ends=0,
            )

        lethal = lethal_positions or set()
        normalized_held: set[int] = set()
        if held_items:
            for item in held_items:
                if isinstance(item, HeldItem):
                    normalized_held.add(item.feature_id)
                elif hasattr(item, "feature_id"):
                    normalized_held.add(int(item.feature_id))
                else:
                    normalized_held.add(int(item))

        root = FrontopolarSubgoalMCTSNode(
            subgoal=None,
            state_avatar_pos=avatar_pos,
            state_held_items=frozenset(normalized_held),
            state_cleared_barriers=frozenset(),
            untried_subgoals=list(candidate_subgoals),
            depth=0,
        )

        pruned_count = 0

        for _ in range(max_iterations):
            node = root

            # 1. Selection
            while node.children and not node.untried_subgoals:
                node = max(node.children, key=lambda c: c.ucb1(exploration_weight))

            # 2. Expansion
            if node.untried_subgoals and node.depth < max_depth:
                cand_sg = node.untried_subgoals.pop(0)
                cand_target = cand_sg.target_entity_pos or cand_sg.target_destination

                # Habenular Lethal Pruning
                if cand_target in lethal:
                    pruned_count += 1
                    continue

                # Spelke Continuity / Barrier Reachability Pruning with Extended Body Schema Permeability
                effective_barriers = set(barriers) - set(node.state_cleared_barriers)
                if curr_grid is not None and (tool_barrier_affinities or body_schema):
                    permeable_feats: set[int] = set()
                    for h_feat in node.state_held_items:
                        if tool_barrier_affinities and h_feat in tool_barrier_affinities:
                            permeable_feats.update(tool_barrier_affinities[h_feat])
                        if body_schema and hasattr(body_schema, "tool_barrier_affinities"):
                            permeable_feats.add(h_feat)
                            permeable_feats.update(
                                body_schema.tool_barrier_affinities.get(h_feat, set())
                            )
                    if permeable_feats:
                        H_b, W_b = grid_bounds
                        effective_barriers = {
                            b
                            for b in effective_barriers
                            if not (
                                0 <= b[0] < H_b
                                and 0 <= b[1] < W_b
                                and int(curr_grid[b[0], b[1]]) in permeable_feats
                            )
                        }

                # Prerequisite checking: clearing obstacles or unlocking mechanisms requires held key/tool
                if cand_sg.subgoal_type in [
                    SubgoalType.CLEAR_OBSTACLE,
                    SubgoalType.UNLOCK_REMOTE_MECHANISM,
                ]:
                    req_f = getattr(cand_sg, "required_feature", None)
                    if req_f is not None and req_f not in node.state_held_items:
                        pruned_count += 1
                        continue

                if cand_target is not None:
                    if not cls._is_reachable(
                        node.state_avatar_pos,
                        cand_target,
                        effective_barriers,
                        grid_bounds,
                    ):
                        pruned_count += 1
                        continue

                # Transition state
                next_pos = cand_target if cand_target is not None else node.state_avatar_pos
                next_held = set(node.state_held_items)
                next_cleared = set(node.state_cleared_barriers)

                if cand_sg.subgoal_type in [SubgoalType.COLLECT_ITEM, SubgoalType.ACQUIRE_TOOL]:
                    feat = getattr(
                        cand_sg, "required_feature", getattr(cand_sg, "target_feature", None)
                    )
                    if feat is not None:
                        next_held.add(feat)
                        # Predict resonant barrier clearing upon tool acquisition
                        if curr_grid is not None:
                            unlocked_f: set[int] = set()
                            if tool_barrier_affinities and feat in tool_barrier_affinities:
                                unlocked_f.update(tool_barrier_affinities[feat])
                            if body_schema and hasattr(body_schema, "tool_barrier_affinities"):
                                unlocked_f.add(feat)
                                unlocked_f.update(
                                    body_schema.tool_barrier_affinities.get(feat, set())
                                )
                            if unlocked_f:
                                H_b, W_b = grid_bounds
                                for br, bc in barriers:
                                    if (
                                        0 <= br < H_b
                                        and 0 <= bc < W_b
                                        and int(curr_grid[br, bc]) in unlocked_f
                                    ):
                                        next_cleared.add((br, bc))
                elif cand_sg.subgoal_type in [
                    SubgoalType.CLEAR_OBSTACLE,
                    SubgoalType.UNLOCK_REMOTE_MECHANISM,
                ]:
                    if cand_sg.target_destination:
                        next_cleared.add(cand_sg.target_destination)
                    if cand_sg.target_entity_pos:
                        next_cleared.add(cand_sg.target_entity_pos)

                remaining_cands = [
                    s for s in candidate_subgoals if s.subgoal_id != cand_sg.subgoal_id
                ]
                child = FrontopolarSubgoalMCTSNode(
                    subgoal=cand_sg,
                    state_avatar_pos=next_pos,
                    state_held_items=frozenset(next_held),
                    state_cleared_barriers=frozenset(next_cleared),
                    parent=node,
                    untried_subgoals=remaining_cands,
                    depth=node.depth + 1,
                )
                node.children.append(child)
                target_node = child
            else:
                target_node = node

            # 3. Simulation / Value Evaluation
            sim_pos = target_node.state_avatar_pos
            if goal_positions and sim_pos in goal_positions:
                rollout_val = 100.0 - target_node.depth * 5.0
            elif goal_positions:
                min_d = min(
                    abs(gr - sim_pos[0]) + abs(gc - sim_pos[1]) for gr, gc in goal_positions
                )
                rollout_val = 50.0 / (1.0 + min_d) - target_node.depth * 2.0
            else:
                rollout_val = 10.0 - target_node.depth

            # 4. Backpropagation
            curr = target_node
            while curr is not None:
                curr.visits += 1
                curr.total_value += rollout_val
                curr = curr.parent

        # Extract optimal trajectory from tree
        best_path: list[FrontopolarSubgoal] = []
        curr = root
        while curr.children:
            curr = max(curr.children, key=lambda c: c.visits)
            if curr.subgoal is not None:
                best_path.append(curr.subgoal)

        return FrontopolarSubgoalMCTSResult(
            best_subgoals=best_path,
            total_iterations=max_iterations,
            tree_depth=len(best_path),
            expected_value=root.q_value,
            pruned_dead_ends=pruned_count,
        )
