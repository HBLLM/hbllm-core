"""
NetHack Action Adapter and Causal Dungeon Crawler Planner.

Implements NetHackCausalPlanner:
1. Descend stairs when standing on '>'.
2. Door handling: open closed doors blocking corridors.
3. Tactical combat: engage monsters blocking critical paths.
4. Epistemic frontier exploration: navigate through fog of war to locate staircases.
"""

from __future__ import annotations

import logging
from collections import deque

from .types import (
    ACTION_VECTORS,
    NetHackAction,
    NetHackGlyph,
    NetHackGoal,
    NetHackObservation,
)

logger = logging.getLogger(__name__)


from hbllm.brain.reasoning.operators.base import ProblemType, ReasoningProblem
from hbllm.brain.reasoning.operators.registry import create_default_operator_registry
from hbllm.brain.reasoning.unified_runtime import UnifiedReasoningRuntime
from hbllm.hcir.graph import ActionNode, CognitiveGraph

from .perception import NetHackPerceptionAdapter
from .predicates import register_nethack_predicates


class NetHackActionAdapter:
    """
    Pure Device Driver for NetHack / MiniHack environment.

    Decoupled into:
    1. enumerate_affordances: Declares ActionNodes with causal preconditions and outcomes.
    2. execute_action: Low-level motor dispatch (corridor BFS, door kicking, combat attacks).
    3. plan_next_action: Routes causal planning directly through UnifiedReasoningRuntime.
    """

    def __init__(self) -> None:
        register_nethack_predicates()
        self.visited_tiles: set[tuple[int, int]] = set()
        self.traversed_doors: set[tuple[int, int]] = set()
        self.blocked_moves: set[tuple[tuple[int, int], NetHackAction]] = set()
        self.failed_door_open_attempts: int = 0
        self.stairs_pos: tuple[int, int] | None = None
        self.key_pos: tuple[int, int] | None = None
        self.last_pos: tuple[int, int] | None = None
        self.last_action: NetHackAction | None = None
        self.perception = NetHackPerceptionAdapter()
        self.runtime = UnifiedReasoningRuntime(create_default_operator_registry())

    def reset(self) -> None:
        """Reset internal exploration history and landmarks."""
        self.visited_tiles.clear()
        self.traversed_doors.clear()
        self.blocked_moves.clear()
        self.failed_door_open_attempts = 0
        self.stairs_pos = None
        self.key_pos = None
        self.last_pos = None
        self.last_action = None
        self.perception = NetHackPerceptionAdapter()

    def enumerate_affordances(
        self, obs: NetHackObservation, graph: CognitiveGraph
    ) -> list[ActionNode]:
        """Declare candidate ActionNodes with explicit causal requirements and outputs."""
        affordances: list[ActionNode] = [
            ActionNode(
                id="act_descend",
                intent="descend_stairs",
                requirements=["standing_on(stairs_down)"],
                produces=["descended"],
            ),
            ActionNode(
                id="act_approach_stairs",
                intent="approach_stairs",
                requirements=[],
                produces=["standing_on(stairs_down)"],
            ),
            ActionNode(
                id="act_pickup",
                intent="pickup",
                requirements=["standing_on(key)"],
                produces=["has(key)"],
            ),
            ActionNode(
                id="act_approach_key",
                intent="approach_key",
                requirements=[],
                produces=["standing_on(key)"],
            ),
            ActionNode(
                id="act_open_door",
                intent="open_door",
                requirements=["adjacent(door_closed)"],
                produces=["door_open"],
            ),
            ActionNode(
                id="act_kick_door",
                intent="kick_door",
                requirements=["adjacent(door_closed)"],
                produces=["door_open"],
            ),
            ActionNode(
                id="act_attack",
                intent="attack_monster",
                requirements=[],
                produces=["safe_from_monster"],
            ),
            ActionNode(
                id="act_explore",
                intent="explore",
                requirements=[],
                produces=["explore_done"],
            ),
        ]

        for act in affordances:
            if graph.has_node(act.id):
                graph.remove_node(act.id)
            graph.add_node(act)

        return affordances

    def _is_valid_move(
        self,
        obs: NetHackObservation,
        from_pos: tuple[int, int],
        to_pos: tuple[int, int],
        act: NetHackAction,
    ) -> bool:
        """Verify move is not blocked and respects NetHack doorway movement rules."""
        if (from_pos, act) in self.blocked_moves:
            return False
        fx, fy = from_pos
        tx, ty = to_pos
        if tx != fx and ty != fy:
            # NetHack rule: Cannot move diagonally into or out of doorways
            is_from_door = (from_pos in self.traversed_doors) or (
                obs.glyphs[fy][fx] in (NetHackGlyph.DOOR_OPEN, NetHackGlyph.DOOR_CLOSED)
            )
            is_to_door = (to_pos in self.traversed_doors) or (
                obs.glyphs[ty][tx] in (NetHackGlyph.DOOR_OPEN, NetHackGlyph.DOOR_CLOSED)
            )
            if is_from_door or is_to_door:
                return False
        return True

    def execute_action(self, intent: str, obs: NetHackObservation) -> NetHackAction:
        """Low-level actuator dispatch: translates declarative intent into discrete motor actions."""
        px, py = obs.player_pos

        if intent == "descend_stairs":
            return NetHackAction.DESCEND_STAIRS

        if intent == "pickup":
            self.key_pos = None
            return NetHackAction.PICKUP

        if intent in ("open_door", "kick_door"):
            if "locked" in obs.message.lower() or self.failed_door_open_attempts >= 2:
                return NetHackAction.KICK
            self.failed_door_open_attempts += 1
            return NetHackAction.OPEN_DOOR

        if intent == "attack_monster":
            width = len(obs.glyphs[0])
            height = len(obs.glyphs)
            for act, (dx, dy) in ACTION_VECTORS.items():
                nx, ny = px + dx, py + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if obs.glyphs[ny][nx] == NetHackGlyph.MONSTER:
                        return act
            return self.execute_action("explore", obs)

        if intent == "approach_stairs":
            target_stairs = self.stairs_pos or self._find_glyph_pos(obs, NetHackGlyph.STAIRS_DOWN)
            if target_stairs is not None:
                if (px, py) == target_stairs:
                    return NetHackAction.DESCEND_STAIRS
                step = self._bfs_path_step(obs, target_stairs, allow_monsters=True)
                if step is not None:
                    return step
            return self.execute_action("explore", obs)

        if intent == "approach_key":
            target_key = self.key_pos or self._find_glyph_pos(obs, NetHackGlyph.KEY)
            if target_key is not None:
                step = self._bfs_path_step(obs, target_key)
                if step is not None:
                    return step
            return self.execute_action("explore", obs)

        # Explore / Fallback
        width = len(obs.glyphs[0])
        height = len(obs.glyphs)

        # 1. Stairs priority: if stairs are known, route to them!
        target_stairs = self.stairs_pos or self._find_glyph_pos(obs, NetHackGlyph.STAIRS_DOWN)
        if target_stairs is not None:
            self.stairs_pos = target_stairs
            if (px, py) == target_stairs:
                return NetHackAction.DESCEND_STAIRS
            step = self._bfs_path_step(obs, target_stairs, allow_monsters=True)
            if step is not None and ((px, py), step) not in self.blocked_moves:
                return step

        # 2. Closed doors: find closed door and approach it
        door_pos = self._find_glyph_pos(obs, NetHackGlyph.DOOR_CLOSED)
        if door_pos is not None:
            step = self._bfs_path_step(obs, door_pos, allow_monsters=True, to_adjacent=True)
            if step is not None and ((px, py), step) not in self.blocked_moves:
                return step

        # 3. Open doors leading into new rooms:
        open_doors = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if obs.glyphs[y][x] == NetHackGlyph.DOOR_OPEN and (x, y) not in self.traversed_doors
        ]
        if open_doors:
            open_doors.sort(key=lambda p: abs(p[0] - px) + abs(p[1] - py))
            for od in open_doors:
                step = self._bfs_path_step(obs, od, allow_monsters=True)
                if step is not None and ((px, py), step) not in self.blocked_moves:
                    return step

        # 4. Immediate unvisited neighbors
        for act, (dx, dy) in ACTION_VECTORS.items():
            nx, ny = px + dx, py + dy
            if 0 <= nx < width and 0 <= ny < height:
                if (nx, ny) not in self.visited_tiles and self._is_valid_move(
                    obs, (px, py), (nx, ny), act
                ):
                    if obs.glyphs[ny][nx] in (
                        NetHackGlyph.CORRIDOR,
                        NetHackGlyph.FLOOR,
                        NetHackGlyph.DOOR_OPEN,
                    ):
                        return act

        # 5. BFS to nearest unvisited passable tile
        step_unvisited = self._bfs_to_nearest_unvisited(obs)
        if step_unvisited is not None and ((px, py), step_unvisited) not in self.blocked_moves:
            return step_unvisited

        # 6. BFS to epistemic frontier
        frontier_step = self._explore_frontier(obs)
        if frontier_step is not None and ((px, py), frontier_step) not in self.blocked_moves:
            return frontier_step

        # 7. Fallback: Any adjacent passable tile not blocked
        for act, (dx, dy) in ACTION_VECTORS.items():
            nx, ny = px + dx, py + dy
            if 0 <= nx < width and 0 <= ny < height:
                if obs.glyphs[ny][nx] in (
                    NetHackGlyph.CORRIDOR,
                    NetHackGlyph.FLOOR,
                    NetHackGlyph.DOOR_OPEN,
                ) and self._is_valid_move(obs, (px, py), (nx, ny), act):
                    return act

        # 8. Last resort: any move not stepping into a known wall
        for act, (dx, dy) in ACTION_VECTORS.items():
            nx, ny = px + dx, py + dy
            if 0 <= nx < width and 0 <= ny < height:
                if obs.glyphs[ny][nx] != NetHackGlyph.WALL and self._is_valid_move(
                    obs, (px, py), (nx, ny), act
                ):
                    return act

        return NetHackAction.EAST

    def plan_next_action(
        self, obs: NetHackObservation, goal: NetHackGoal | None = None
    ) -> NetHackAction:
        """Select next action using UnifiedReasoningRuntime with EmbodiedCausalOperator."""
        px, py = obs.player_pos
        self.visited_tiles.add((px, py))

        # Check if previous move failed (position unchanged or explicit bump)
        if self.last_pos is not None and self.last_action is not None:
            if (px, py) == self.last_pos and self.last_action != NetHackAction.OPEN_DOOR:
                self.blocked_moves.add((self.last_pos, self.last_action))

        # If standing on a door, mark it traversed
        if 0 <= py < len(obs.glyphs) and 0 <= px < len(obs.glyphs[0]):
            if obs.glyphs[py][px] in (NetHackGlyph.DOOR_OPEN, NetHackGlyph.DOOR_CLOSED):
                self.traversed_doors.add((px, py))

        # Scan for landmarks
        visible_stairs = self._find_glyph_pos(obs, NetHackGlyph.STAIRS_DOWN)
        if visible_stairs is not None:
            self.stairs_pos = visible_stairs

        visible_key = self._find_glyph_pos(obs, NetHackGlyph.KEY)
        if visible_key is not None:
            self.key_pos = visible_key

        # 1. Tactical Combat Fast-Path: If adjacent to monster, engage immediately
        width = len(obs.glyphs[0])
        height = len(obs.glyphs)
        for act, (dx, dy) in ACTION_VECTORS.items():
            nx, ny = px + dx, py + dy
            if 0 <= nx < width and 0 <= ny < height:
                if obs.glyphs[ny][nx] == NetHackGlyph.MONSTER:
                    self.last_pos = (px, py)
                    self.last_action = act
                    return act

        # 2. Door interaction: If orthogonally adjacent to closed door, open or kick it
        orthogonal_vectors = ((0, -1), (1, 0), (0, 1), (-1, 0))
        for dx, dy in orthogonal_vectors:
            nx, ny = px + dx, py + dy
            if 0 <= nx < width and 0 <= ny < height:
                if obs.glyphs[ny][nx] == NetHackGlyph.DOOR_CLOSED:
                    act = self.execute_action("open_door", obs)
                    self.last_pos = (px, py)
                    self.last_action = act
                    return act

        self.failed_door_open_attempts = 0

        # 3. Perception Ingestion
        self.perception.ingest_observation(obs)

        # 4. Goal Translation
        goal_node = self.perception.ingest_goal(goal, obs)

        # 5. Affordance Declaration
        self.enumerate_affordances(obs, self.perception.graph)

        # 6. Cognitive Causal Planning via UnifiedReasoningRuntime
        problem = ReasoningProblem(
            problem_type=ProblemType.PLANNING,
            goal_node_ids=(goal_node.id,),
            description="NetHack dungeon exploration and staircase descent",
        )
        trace = self.runtime.reason(graph=self.perception.graph, problem=problem)

        # 7. Actuator Motor Dispatch
        chosen_intent = ""
        if (
            trace
            and trace.final_result
            and trace.final_result.conclusions
            and "best_action" in trace.final_result.conclusions
        ):
            chosen_intent = trace.final_result.conclusions["best_action"]

        if chosen_intent and chosen_intent != "no_op":
            act = self.execute_action(chosen_intent, obs)
        else:
            act = self.execute_action("explore", obs)

        self.last_pos = (px, py)
        self.last_action = act
        return act

    def _bfs_to_nearest_unvisited(self, obs: NetHackObservation) -> NetHackAction | None:
        """Find the closest reachable unvisited passable tile."""
        start = obs.player_pos
        height = len(obs.glyphs)
        width = len(obs.glyphs[0])
        passable_glyphs = {
            NetHackGlyph.FLOOR,
            NetHackGlyph.CORRIDOR,
            NetHackGlyph.DOOR_OPEN,
            NetHackGlyph.STAIRS_DOWN,
            NetHackGlyph.STAIRS_UP,
            NetHackGlyph.KEY,
            NetHackGlyph.GOLD,
            NetHackGlyph.MONSTER,
        }
        queue = deque([(start[0], start[1], [])])
        visited = {start}
        while queue:
            cx, cy, path = queue.popleft()
            if (cx, cy) not in self.visited_tiles and obs.glyphs[cy][cx] in (
                NetHackGlyph.FLOOR,
                NetHackGlyph.CORRIDOR,
                NetHackGlyph.DOOR_OPEN,
            ):
                if path:
                    return path[0]
            if len(path) >= 100:
                continue
            for act, (dx, dy) in ACTION_VECTORS.items():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    if not self._is_valid_move(obs, (cx, cy), (nx, ny), act):
                        continue
                    if obs.glyphs[ny][nx] in passable_glyphs:
                        visited.add((nx, ny))
                        queue.append((nx, ny, path + [act]))
        return None

    def _find_glyph_pos(
        self, obs: NetHackObservation, target_glyph: NetHackGlyph
    ) -> tuple[int, int] | None:
        height = len(obs.glyphs)
        width = len(obs.glyphs[0])
        px, py = obs.player_pos
        candidates: list[tuple[int, int]] = []
        for y in range(height):
            for x in range(width):
                if obs.glyphs[y][x] == target_glyph:
                    candidates.append((x, y))
        if not candidates:
            return None
        # Sort by distance to player
        candidates.sort(key=lambda p: abs(p[0] - px) + abs(p[1] - py))
        return candidates[0]

    def _bfs_path_step(
        self,
        obs: NetHackObservation,
        target_pos: tuple[int, int],
        allow_monsters: bool = True,
        to_adjacent: bool = False,
    ) -> NetHackAction | None:
        """Compute one-step movement towards target_pos using BFS."""
        start = obs.player_pos
        tx, ty = target_pos
        height = len(obs.glyphs)
        width = len(obs.glyphs[0])

        passable_glyphs = {
            NetHackGlyph.FLOOR,
            NetHackGlyph.CORRIDOR,
            NetHackGlyph.DOOR_OPEN,
            NetHackGlyph.STAIRS_DOWN,
            NetHackGlyph.STAIRS_UP,
            NetHackGlyph.KEY,
            NetHackGlyph.GOLD,
        }
        if allow_monsters:
            passable_glyphs.add(NetHackGlyph.MONSTER)

        queue = deque([(start[0], start[1], [])])
        visited = {start}

        while queue:
            cx, cy, path = queue.popleft()

            if to_adjacent:
                if abs(cx - tx) + abs(cy - ty) == 1:
                    if path:
                        return path[0]
                    return None
            else:
                if (cx, cy) == (tx, ty):
                    if path:
                        return path[0]
                    return None

            if len(path) >= 150:  # Search depth cap for multi-room dungeons
                continue

            for act, (dx, dy) in ACTION_VECTORS.items():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    if not self._is_valid_move(obs, (cx, cy), (nx, ny), act):
                        continue
                    if (nx, ny) == (tx, ty) or obs.glyphs[ny][nx] in passable_glyphs:
                        visited.add((nx, ny))
                        queue.append((nx, ny, path + [act]))

        return None

    def _explore_frontier(self, obs: NetHackObservation) -> NetHackAction | None:
        """Find an unexplored cell adjacent to known passable floor/corridor."""
        px, py = obs.player_pos
        height = len(obs.glyphs)
        width = len(obs.glyphs[0])

        candidates: list[tuple[tuple[int, int], int]] = []
        for y in range(height):
            for x in range(width):
                if obs.glyphs[y][x] == NetHackGlyph.UNEXPLORED:
                    # Check if adjacent to a known passable tile
                    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                        ax, ay = x + dx, y + dy
                        if 0 <= ax < width and 0 <= ay < height:
                            if obs.glyphs[ay][ax] in (
                                NetHackGlyph.FLOOR,
                                NetHackGlyph.CORRIDOR,
                                NetHackGlyph.DOOR_OPEN,
                            ):
                                dist = abs(px - ax) + abs(py - ay)
                                if (ax, ay) in self.visited_tiles:
                                    dist += 15
                                candidates.append(((ax, ay), dist))

        if candidates:
            candidates.sort(key=lambda item: item[1])
            for (ax, ay), _ in candidates[:12]:
                if (ax, ay) == (px, py):
                    for act, (dx, dy) in ACTION_VECTORS.items():
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if (
                                obs.glyphs[ny][nx] != NetHackGlyph.WALL
                                and ((px, py), act) not in self.blocked_moves
                            ):
                                return act
                else:
                    step = self._bfs_path_step(obs, (ax, ay))
                    if step is not None and ((px, py), step) not in self.blocked_moves:
                        return step

        return None
