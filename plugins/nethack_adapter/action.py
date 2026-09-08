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


class NetHackActionAdapter:
    """
    HCIR Causal Dungeon Navigation and Tactical Combat Planner for NetHack.
    """

    def __init__(self) -> None:
        self.visited_tiles: set[tuple[int, int]] = set()

    def plan_next_action(
        self, obs: NetHackObservation, goal: NetHackGoal | None = None
    ) -> NetHackAction:
        """Select next optimal action to reach stairs and descend."""
        px, py = obs.player_pos
        self.visited_tiles.add((px, py))
        curr_glyph = obs.glyphs[py][px]

        # 1. If standing on stairs down, descend!
        if curr_glyph == NetHackGlyph.STAIRS_DOWN:
            return NetHackAction.DESCEND_STAIRS

        # 2. If standing on key or gold, pick up!
        if curr_glyph in (NetHackGlyph.KEY, NetHackGlyph.GOLD):
            return NetHackAction.PICKUP

        # 3. If adjacent to closed door, open it!
        for act, (dx, dy) in ACTION_VECTORS.items():
            nx, ny = px + dx, py + dy
            if 0 <= nx < len(obs.glyphs[0]) and 0 <= ny < len(obs.glyphs):
                if obs.glyphs[ny][nx] == NetHackGlyph.DOOR_CLOSED:
                    return NetHackAction.OPEN_DOOR

        # 4. Check if stairs down has been observed
        stairs_pos = self._find_glyph_pos(obs, NetHackGlyph.STAIRS_DOWN)
        if stairs_pos is not None:
            # BFS path directly to stairs
            step = self._bfs_path_step(obs, stairs_pos, allow_monsters=True)
            if step is not None:
                return step

        # 5. Check if closed door is visible to open access to next room
        door_pos = self._find_glyph_pos(obs, NetHackGlyph.DOOR_CLOSED)
        if door_pos is not None:
            step = self._bfs_path_step(obs, door_pos, allow_monsters=True)
            if step is not None:
                return step

        # 6. Otherwise explore unexplored frontier
        frontier_step = self._explore_frontier(obs)
        if frontier_step is not None:
            return frontier_step

        return NetHackAction.WAIT

    def _find_glyph_pos(
        self, obs: NetHackObservation, target_glyph: NetHackGlyph
    ) -> tuple[int, int] | None:
        height = len(obs.glyphs)
        width = len(obs.glyphs[0])
        for y in range(height):
            for x in range(width):
                if obs.glyphs[y][x] == target_glyph:
                    return (x, y)
        return None

    def _bfs_path_step(
        self,
        obs: NetHackObservation,
        target_pos: tuple[int, int],
        allow_monsters: bool = True,
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

        # If target itself is a closed door, allow moving to it to trigger OPEN_DOOR
        if obs.glyphs[ty][tx] == NetHackGlyph.DOOR_CLOSED:
            passable_glyphs.add(NetHackGlyph.DOOR_CLOSED)

        queue = deque([(start[0], start[1], [])])
        visited = {start}

        while queue:
            cx, cy, path = queue.popleft()

            if (cx, cy) == (tx, ty):
                if path:
                    return path[0]
                return None

            if len(path) >= 60:  # depth cap
                continue

            for act, (dx, dy) in ACTION_VECTORS.items():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    if (nx, ny) == (tx, ty) or obs.glyphs[ny][nx] in passable_glyphs:
                        visited.add((nx, ny))
                        queue.append((nx, ny, path + [act]))

        return None

    def _explore_frontier(self, obs: NetHackObservation) -> NetHackAction | None:
        """Find an unexplored cell adjacent to known passable floor/corridor."""
        px, py = obs.player_pos
        height = len(obs.glyphs)
        width = len(obs.glyphs[0])

        best_target = None
        best_dist = float("inf")

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
                                if dist < best_dist:
                                    best_dist = dist
                                    best_target = (ax, ay)

        if best_target is not None:
            return self._bfs_path_step(obs, best_target)

        return None
