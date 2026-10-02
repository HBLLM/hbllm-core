"""
Sokoban Action Adapter.

Implements causal push planning and dead-end avoidance search to determine
optimal collision-free and deadlock-free action sequences.
"""

from __future__ import annotations

import heapq
import logging
from collections import deque
from typing import Any

from hbllm.brain.reasoning.operators.base import ProblemType, ReasoningProblem
from hbllm.brain.reasoning.operators.registry import create_default_operator_registry
from hbllm.brain.reasoning.unified_runtime import UnifiedReasoningRuntime
from hbllm.hcir.graph import ActionNode, CognitiveGraph
from hbllm.hcir.world.predictors.physics import PhysicsPredictor

from .perception import SokobanPerceptionAdapter
from .predicates import register_sokoban_predicates
from .types import SokobanAction, SokobanObservation, SokobanTile

logger = logging.getLogger(__name__)

DIRECTION_DELTAS = [
    (SokobanAction.UP, -1, 0),
    (SokobanAction.DOWN, 1, 0),
    (SokobanAction.LEFT, 0, -1),
    (SokobanAction.RIGHT, 0, 1),
]


class SokobanActionAdapter:
    """Causal planner for Sokoban that prunes deadlocks and solves box arrangements."""

    def __init__(self) -> None:
        register_sokoban_predicates()
        self.planned_actions: list[SokobanAction] = []
        self.perception = SokobanPerceptionAdapter()
        self.runtime = UnifiedReasoningRuntime(create_default_operator_registry())

    def reset(self) -> None:
        """Clear action buffer and perception state."""
        self.planned_actions.clear()
        self.perception.reset()

    def enumerate_affordances(
        self, obs: SokobanObservation, graph: CognitiveGraph
    ) -> list[ActionNode]:
        """Declare candidate push ActionNodes for boxes towards targets."""
        affordances: list[ActionNode] = []
        stale = [n.id for n in graph.all_nodes() if n.id.startswith("act_")]
        for sid in stale:
            graph.remove_node(sid)

        for br, bc in obs.boxes:
            box_id = f"box_{br}_{bc}"
            for tr, tc in obs.targets:
                target_id = f"target_{tr}_{tc}"
                affordances.append(
                    ActionNode(
                        id=f"act_push_{br}_{bc}_to_{tr}_{tc}",
                        intent=f"push_box_{box_id}_to_{target_id}",
                        requirements=[f"near({box_id})"],
                        produces=[f"box_on_target({box_id}, {target_id})"],
                    )
                )

        for act in affordances:
            graph.add_node(act)

        return affordances

    def select_action(
        self,
        obs: SokobanObservation,
        perception_data: dict[str, Any],
    ) -> SokobanAction:
        """Select next primitive action using HCIR causal reasoning and A* motor execution."""
        # 1. Ingest graph and declare affordances
        self.perception.ingest_observation(obs)
        goal_node = self.perception.ingest_goal(obs)
        self.enumerate_affordances(obs, self.perception.graph)

        # 2. Query UnifiedReasoningRuntime for high-level plan step
        problem = ReasoningProblem(
            problem_type=ProblemType.PLANNING,
            goal_node_ids=(goal_node.id,),
            description="Sokoban box pushing puzzle",
        )
        self.runtime.reason(graph=self.perception.graph, problem=problem)

        # 3. Motor dispatch via A* push path
        if not self.planned_actions:
            self._plan_solution(obs, perception_data)

        if self.planned_actions:
            return self.planned_actions.pop(0)

        # Fallback default action
        return SokobanAction.UP

    def _get_reachable(
        self,
        start: tuple[int, int],
        walls: set[tuple[int, int]],
        boxes: frozenset[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        """Compute flood-fill reachable cells for player."""
        visited = {start}
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            r, c = queue.popleft()
            for _, dr, dc in DIRECTION_DELTAS:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in walls and (nr, nc) not in boxes and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return visited

    def _find_walk_path(
        self,
        start: tuple[int, int],
        target: tuple[int, int],
        walls: set[tuple[int, int]],
        boxes: frozenset[tuple[int, int]],
    ) -> list[SokobanAction]:
        """Find shortest path of walking steps from start to target."""
        if start == target:
            return []
        visited: dict[tuple[int, int], tuple[tuple[int, int] | None, SokobanAction | None]] = {
            start: (None, None)
        }
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            curr = queue.popleft()
            if curr == target:
                break
            cr, cc = curr
            for act, dr, dc in DIRECTION_DELTAS:
                nxt = (cr + dr, cc + dc)
                if nxt not in walls and nxt not in boxes and nxt not in visited:
                    visited[nxt] = (curr, act)
                    queue.append(nxt)
        if target not in visited:
            return []
        path: list[SokobanAction] = []
        curr_trace: tuple[int, int] | None = target
        while curr_trace is not None and curr_trace != start:
            prev, act = visited[curr_trace]
            if act is not None:
                path.append(act)
            curr_trace = prev
        path.reverse()
        return path

    def _plan_solution(
        self,
        obs: SokobanObservation,
        perception_data: dict[str, Any],
    ) -> None:
        """Compute deadlock-free push path using reachability-based macro-push A* search."""
        player_pos = obs.player_pos
        boxes = frozenset(obs.boxes)
        targets = frozenset(obs.targets)
        taboo_cells = perception_data.get("deadlock_taboo_cells", set())

        height = len(obs.grid)
        width = len(obs.grid[0]) if height > 0 else 0

        walls = set()
        for r in range(height):
            for c in range(width):
                if obs.grid[r][c] == int(SokobanTile.WALL):
                    walls.add((r, c))

        def heuristic(
            curr_reachable: set[tuple[int, int]], curr_boxes: frozenset[tuple[int, int]]
        ) -> int:
            if not targets:
                return 0
            h = 0
            for br, bc in curr_boxes:
                h += min(abs(br - tr) + abs(bc - tc) for tr, tc in targets) * 3
            unsolved = [b for b in curr_boxes if b not in targets]
            if unsolved and curr_reachable:
                sample_pts = (
                    curr_reachable
                    if len(curr_reachable) <= 8
                    else [min(curr_reachable), max(curr_reachable)]
                )
                h += min(
                    abs(pr - br) + abs(pc - bc) for pr, pc in sample_pts for br, bc in unsolved
                )
            return h

        start_reachable = self._get_reachable(player_pos, walls, boxes)
        start_canonical = min(start_reachable) if start_reachable else player_pos
        start_state = (start_canonical, boxes)

        h0 = heuristic(start_reachable, boxes)
        counter = 0
        # (f, g, counter, canonical_player, boxes, macro_path)
        # macro_path entry: (push_from_cell, box_pos, act, (dr, dc))
        pq: list[
            tuple[
                int,
                int,
                int,
                tuple[tuple[int, int], frozenset[tuple[int, int]]],
                list[tuple[tuple[int, int], tuple[int, int], SokobanAction, tuple[int, int]]],
            ]
        ] = [(h0, 0, counter, start_state, [])]
        visited = {start_state}

        max_nodes = 2500
        nodes = 0
        best_macro_path: list[
            tuple[tuple[int, int], tuple[int, int], SokobanAction, tuple[int, int]]
        ] = []
        best_score = float("inf")

        def reconstruct_actions(
            macro_path: list[
                tuple[tuple[int, int], tuple[int, int], SokobanAction, tuple[int, int]]
            ],
        ) -> list[SokobanAction]:
            full_actions: list[SokobanAction] = []
            curr_p = player_pos
            active_b = set(obs.boxes)
            for push_from, box_pos, act, delta in macro_path:
                walk = self._find_walk_path(curr_p, push_from, walls, frozenset(active_b))
                full_actions.extend(walk)
                full_actions.append(act)
                active_b.remove(box_pos)
                active_b.add((box_pos[0] + delta[0], box_pos[1] + delta[1]))
                curr_p = box_pos
            return full_actions

        while pq and nodes < max_nodes:
            nodes += 1
            _f, g, _, (canon_p, curr_boxes), macro_path = heapq.heappop(pq)

            if curr_boxes == targets:
                self.planned_actions = reconstruct_actions(macro_path)
                return

            if targets:
                score = sum(
                    min(abs(br - tr) + abs(bc - tc) for tr, tc in targets) for br, bc in curr_boxes
                )
                if score < best_score and macro_path:
                    best_score = score
                    best_macro_path = macro_path

            # Reconstruct reachable cells from canonical player
            curr_reachable = self._get_reachable(canon_p, walls, curr_boxes)

            # Generate all valid macro-pushes from reachable area
            for br, bc in curr_boxes:
                for act, dr, dc in DIRECTION_DELTAS:
                    push_from = (br - dr, bc - dc)
                    push_to = (br + dr, bc + dc)

                    if push_from not in curr_reachable:
                        continue
                    if push_to in walls or push_to in curr_boxes:
                        continue

                    # Deadlock pruning
                    if push_to in taboo_cells and push_to not in targets:
                        continue

                    new_boxes = set(curr_boxes)
                    new_boxes.remove((br, bc))
                    new_boxes.add(push_to)

                    if self._is_2x2_deadlock(push_to, walls, new_boxes, targets):
                        continue

                    if PhysicsPredictor.is_line_deadlock(
                        box_pos=push_to,
                        barrier_cells=walls,
                        target_positions=targets,
                        grid_shape=(height, width),
                        step_size=1,
                    ):
                        continue

                    frozen_new_boxes = frozenset(new_boxes)
                    # New player position after push is (br, bc)
                    new_reachable = self._get_reachable((br, bc), walls, frozen_new_boxes)
                    new_canonical = min(new_reachable) if new_reachable else (br, bc)
                    next_state = (new_canonical, frozen_new_boxes)

                    if next_state not in visited:
                        visited.add(next_state)
                        counter += 1
                        next_g = g + 1
                        next_h = heuristic(new_reachable, frozen_new_boxes)
                        macro_step = (push_from, (br, bc), act, (dr, dc))
                        heapq.heappush(
                            pq,
                            (
                                next_g + next_h,
                                next_g,
                                counter,
                                next_state,
                                macro_path + [macro_step],
                            ),
                        )

        logger.debug("Sokoban planner reached search limit, executing best partial trajectory")
        if best_macro_path:
            self.planned_actions = reconstruct_actions(best_macro_path)
        else:
            self.planned_actions = [SokobanAction.UP] * 4

    def _is_2x2_deadlock(
        self,
        box: tuple[int, int],
        walls: set[tuple[int, int]],
        boxes: set[tuple[int, int]],
        targets: frozenset[tuple[int, int]],
    ) -> bool:
        """Check whether pushing a box creates an irreversible 2x2 box/wall deadlock."""
        r, c = box
        for dr, dc in [(-1, -1), (-1, 0), (0, -1), (0, 0)]:
            quad = [
                (r + dr, c + dc),
                (r + dr + 1, c + dc),
                (r + dr, c + dc + 1),
                (r + dr + 1, c + dc + 1),
            ]
            if all(pos in walls or pos in boxes for pos in quad):
                if any(pos in boxes and pos not in targets for pos in quad):
                    return True
        return False
