"""Universal Curiosity & Exit AGI Agent (Domain-Agnostic 4-Pillar Cognitive Engine).

Operates on the universal first principles of Artificial General Intelligence:
1. Scene Perception & Object Segmentation (Avatar, Walls, Interactables, Exits, Hazards)
2. Exit & Goal Teleology (Detect victory criteria, test if exit is open and reachable)
3. Causal Curiosity & Epistemic Probing ("Play with unknown objects" to learn affordances)
4. Geodesic Spatial Navigation (Shortest collision-free path with hazard avoidance)

Zero hardcoded game IDs, zero bespoke puzzle archetypes.
"""

from __future__ import annotations

import collections
import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plugins.arc_agi_adapter.arc_solvers.visual_analysis import (
    DiffType,
    FrameDiffAnalyzer,
    VisualTopologyExtractor,
)

logger = logging.getLogger(__name__)


@dataclass
class UniversalObject:
    """A segmented physical or logical entity on the canvas."""

    entity_id: str
    color: int
    size: int
    bounding_box: tuple[int, int, int, int]  # (min_r, max_r, min_c, max_c)
    centroid: tuple[float, float]
    coords: list[tuple[int, int]]
    role: str = "unknown"  # "avatar", "wall", "interactable", "exit", "hazard", "receptacle"
    is_tested: bool = False
    affordance: str = "none"  # "pickup", "unlock", "push", "hazard", "toggle", "none"


@dataclass
class UniversalSceneGraph:
    """Topological decomposition of the current frame."""

    grid: np.ndarray
    bg_color: int
    avatar: UniversalObject | None = None
    avatar_pos: tuple[int, int] | None = None
    scale: int = 1
    objects: list[UniversalObject] = field(default_factory=list)
    walls: set[tuple[int, int]] = field(default_factory=set)
    hazards: set[tuple[int, int]] = field(default_factory=set)
    candidate_exits: list[UniversalObject] = field(default_factory=list)
    interactables: list[UniversalObject] = field(default_factory=list)


class UniversalCuriosityAgent:
    """Universal 4-Pillar Curiosity-and-Exit AGI Agent."""

    def __init__(self, **kwargs: Any) -> None:
        self.step_counter: int = 0
        self.current_level: int = 0
        self.prev_grid: np.ndarray | None = None
        self.last_action: int | None = None
        self.last_action_data: dict[str, int] | None = None
        self.current_actor_pos: tuple[int, int] | None = None

        # ── Cross-level / Cross-step Memory ───────────────────────────────
        self.known_avatar_color: int | None = None
        self.known_hazard_colors: set[int] = {8}  # 8 is standard hazard in ARC
        self.learned_affordances: dict[int, str] = {}  # color -> affordance
        self.tested_positions: set[tuple[int, int]] = set()
        self.tested_colors: set[int] = set()
        self.inventory: set[int] = set()

        # Navigation & Loop Avoidance
        self.position_visits: collections.Counter[tuple[int, int]] = collections.Counter()
        self.action_history: list[int] = []
        self.planned_action_queue: list[tuple[int, dict[str, int] | None]] = []
        self.stuck_counter: int = 0
        self.last_target: tuple[int, int] | None = None
        self.target_attempts: collections.Counter[tuple[int, int]] = collections.Counter()

    def reset_episode(self, retain_dynamics: bool = False, is_retry: bool = False) -> None:
        """Reset internal episodic state."""
        self.prev_grid = None
        self.last_action = None
        self.last_action_data = None
        self.step_counter = 0
        self.stuck_counter = 0
        self.planned_action_queue.clear()
        self.position_visits.clear()
        self.action_history.clear()
        self.target_attempts.clear()
        self.last_target = None

        if not retain_dynamics and not is_retry:
            self.current_level = 0
            self.known_avatar_color = None
            self.known_hazard_colors = {8}
            self.learned_affordances.clear()
            self.tested_positions.clear()
            self.tested_colors.clear()
            self.inventory.clear()
        elif retain_dynamics and not is_retry:
            self.current_level += 1
            # Retain hazard colors and learned affordances across levels

    # ═══════════════════════════════════════════════════════════════════════
    # Pillar 1: Scene Perception & Object Segmentation
    # ═══════════════════════════════════════════════════════════════════════

    def perceive_scene(
        self,
        grid: np.ndarray,
        available_actions: list[int],
    ) -> UniversalSceneGraph:
        """Segment canvas into discrete objects, walls, avatar, hazards, and exits."""
        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        counts = np.bincount(grid.flatten())
        bg_color = int(np.argmax(counts))

        raw_entities = VisualTopologyExtractor.extract_entities(grid, ignore_colors={bg_color, 0})

        objects: list[UniversalObject] = []
        for i, re in enumerate(raw_entities):
            uobj = UniversalObject(
                entity_id=f"obj_{i}_{re.color}",
                color=re.color,
                size=re.size,
                bounding_box=re.bounding_box,
                centroid=re.centroid,
                coords=[(int(r), int(c)) for r, c in getattr(re, "coords", [])]
                or [(int(round(re.centroid[0])), int(round(re.centroid[1])))],
            )
            objects.append(uobj)

        scene = UniversalSceneGraph(grid=grid, bg_color=bg_color, objects=objects)

        # 1. Identify Controllable Avatar
        avatar = self._identify_avatar(scene, available_actions)
        if avatar:
            avatar.role = "avatar"
            scene.avatar = avatar
            scene.avatar_pos = (
                int(round(avatar.centroid[0])),
                int(round(avatar.centroid[1])),
            )
            self.current_actor_pos = scene.avatar_pos
            self.known_avatar_color = avatar.color
            scene.scale = max(1, int(round(math.sqrt(avatar.size))))

        # 2. Categorize remaining objects
        for obj in objects:
            if obj == avatar:
                continue

            # Check if wall / static frame
            min_r, max_r, min_c, max_c = obj.bounding_box
            is_frame = (
                (max_r - min_r >= H - 4 and max_c - min_c >= W - 4)
                or (obj.size > H * W * 0.25)
                or (min_r <= 1 and max_r <= 1 and max_c - min_c > 10)
                or (min_r >= H - 2 and max_r >= H - 2 and max_c - min_c > 10)
            )

            has_movement = any(a in available_actions for a in [1, 2, 3, 4])
            if is_frame:
                obj.role = "wall"
                for r, c in obj.coords:
                    scene.walls.add((r, c))
            elif obj.color in self.known_hazard_colors:
                obj.role = "hazard"
                for r, c in obj.coords:
                    scene.hazards.add((r, c))
            elif has_movement and self._is_candidate_exit(obj, grid, bg_color):
                obj.role = "exit"
                scene.candidate_exits.append(obj)
            else:
                obj.role = "interactable"
                obj.is_tested = (
                    int(round(obj.centroid[0])),
                    int(round(obj.centroid[1])),
                ) in self.tested_positions
                scene.interactables.append(obj)

        return scene

    def _identify_avatar(
        self, scene: UniversalSceneGraph, available_actions: list[int]
    ) -> UniversalObject | None:
        """Find the controllable avatar entity."""
        # 1. Prioritize known avatar color from previous observations
        if self.known_avatar_color is not None:
            matches = [o for o in scene.objects if o.color == self.known_avatar_color]
            if matches:
                # Pick the one closest to last known position
                if self.current_actor_pos:
                    matches.sort(
                        key=lambda o: math.hypot(
                            o.centroid[0] - self.current_actor_pos[0],
                            o.centroid[1] - self.current_actor_pos[1],
                        )
                    )
                return matches[0]

        # 2. Movement games: look for compact, movable entities (area 1..36)
        has_movement = any(a in available_actions for a in [1, 2, 3, 4])
        if has_movement:
            cands = [
                o
                for o in scene.objects
                if 1 <= o.size <= 36
                and o.color not in self.known_hazard_colors
                and o.color != scene.bg_color
            ]
            if cands:
                # Prefer square-like entities
                cands.sort(
                    key=lambda o: (
                        abs(
                            (o.bounding_box[1] - o.bounding_box[0])
                            - (o.bounding_box[3] - o.bounding_box[2])
                        ),
                        -o.size,
                    )
                )
                return cands[0]

        return None

    def _is_candidate_exit(self, obj: UniversalObject, grid: np.ndarray, bg_color: int) -> bool:
        """Heuristically identify whether an entity has exit/goal properties."""
        # Receptacle/door/zone markers: hollow boxes, border openings, distinct colors
        min_r, max_r, min_c, max_c = obj.bounding_box
        H, W = grid.shape
        touches_border = min_r <= 2 or max_r >= H - 3 or min_c <= 2 or max_c >= W - 3
        if touches_border and 2 <= obj.size <= 40:
            return True
        # Check if entity is marked as receptacle in affordance memory
        if self.learned_affordances.get(obj.color) in ("exit", "receptacle", "goal"):
            return True
        return False

    # ═══════════════════════════════════════════════════════════════════════
    # Pillar 2: Exit & Goal Teleology
    # ═══════════════════════════════════════════════════════════════════════

    def evaluate_exit_reachability(self, scene: UniversalSceneGraph) -> UniversalObject | None:
        """Check if an exit is identified, open, and directly reachable."""
        if not scene.candidate_exits or not scene.avatar_pos:
            return None

        # Check each candidate exit
        for exit_obj in scene.candidate_exits:
            target_pos = (
                int(round(exit_obj.centroid[0])),
                int(round(exit_obj.centroid[1])),
            )
            # If we already reached this target position and it didn't win, don't repeat
            if target_pos in self.tested_positions:
                continue

            # Find collision-free path to exit
            path = self._find_shortest_path(
                scene.grid,
                scene.avatar_pos,
                target_pos,
                scene.walls,
                scene.hazards,
                scale=scene.scale,
            )
            if path is not None and len(path) > 0:
                # Path exists and is unblocked!
                has_uncollected_keys = any(
                    o.affordance == "key" and o.color not in self.inventory
                    for o in scene.interactables
                )
                if not has_uncollected_keys:
                    return exit_obj

        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Pillar 3: Causal Curiosity & Epistemic Probing ("Play with Unknowns")
    # ═══════════════════════════════════════════════════════════════════════

    def select_curiosity_target(self, scene: UniversalSceneGraph) -> UniversalObject | None:
        """Select the best untested object in the scene to play with."""
        if not scene.avatar_pos:
            return None

        # 1. Check if we have a known key/item that needs delivery to a receptacle
        if self.inventory:
            receptacles = [
                o
                for o in scene.objects
                if self.learned_affordances.get(o.color) == "receptacle" or o.role == "exit"
            ]
            if receptacles:
                return receptacles[0]

        # 2. Filter untested candidate interactables
        untested = [
            o
            for o in scene.interactables
            if not o.is_tested
            and o.color not in self.known_hazard_colors
            and self.target_attempts[(int(round(o.centroid[0])), int(round(o.centroid[1])))] < 3
        ]

        if not untested:
            # All tested once: re-visit closest untested positions or interactables with lowest visit count
            untested = [
                o
                for o in scene.interactables
                if o.color not in self.known_hazard_colors
                and self.target_attempts[(int(round(o.centroid[0])), int(round(o.centroid[1])))] < 5
            ]

        if not untested:
            return None

        # Rank by shortest path / distance from avatar
        def score_target(obj: UniversalObject) -> float:
            target_pos = (int(round(obj.centroid[0])), int(round(obj.centroid[1])))
            dist = math.hypot(
                target_pos[0] - scene.avatar_pos[0],
                target_pos[1] - scene.avatar_pos[1],
            )
            attempts = self.target_attempts[target_pos]
            return dist + attempts * 50.0

        untested.sort(key=score_target)
        return untested[0]

    def assimilate_causal_differential(
        self, prev_grid: np.ndarray, action: int, curr_grid: np.ndarray
    ) -> None:
        """Learn object properties from the visual difference ΔGrid."""
        diff = FrameDiffAnalyzer.analyze(prev_grid, action, curr_grid)

        if diff.diff_type == DiffType.NO_CHANGE:
            self.stuck_counter += 1
            if self.last_target:
                # If touching/approaching this target produced NO CHANGE, mark it tested
                self.tested_positions.add(self.last_target)
            return

        self.stuck_counter = 0

        # 1. Did the canvas undergo a global reset? -> Hazard!
        if diff.diff_type == DiffType.GLOBAL_TRANSITION:
            if self.last_target:
                target_r, target_c = self.last_target
                hazard_color = int(prev_grid[target_r, target_c])
                self.known_hazard_colors.add(hazard_color)
                self.learned_affordances[hazard_color] = "hazard"
                logger.info(
                    "Causal Discovery: Discovered HAZARD color %d at %s",
                    hazard_color,
                    self.last_target,
                )
            return

        # 2. Did an object disappear into avatar? -> Item / Key Pickup!
        if self.last_target:
            tr, tc = self.last_target
            old_col = int(prev_grid[tr, tc])
            new_col = int(curr_grid[tr, tc])
            if old_col != new_col and old_col not in (
                0,
                int(np.bincount(curr_grid.flatten()).argmax()),
            ):
                self.inventory.add(old_col)
                self.learned_affordances[old_col] = "pickup"
                self.tested_colors.add(old_col)
                self.tested_positions.add((tr, tc))
                logger.info("Causal Discovery: Picked up item color %d", old_col)

        # 3. Did a wall or barrier elsewhere disappear? -> Door Unlocked!
        diff_mask = prev_grid != curr_grid
        if np.sum(diff_mask) > 0 and self.last_target:
            tr, tc = self.last_target
            act_col = int(prev_grid[tr, tc])
            if self.learned_affordances.get(act_col) != "hazard":
                self.learned_affordances[act_col] = "unlock"
                self.tested_colors.add(act_col)

    # ═══════════════════════════════════════════════════════════════════════
    # Pillar 4: Geodesic Spatial Navigation ($A^*$)
    # ═══════════════════════════════════════════════════════════════════════

    def _find_shortest_path(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        walls: set[tuple[int, int]],
        hazards: set[tuple[int, int]],
        scale: int = 1,
    ) -> list[int] | None:
        """Find the shortest 4-directional path using A* avoiding walls and hazards."""
        H, W = grid.shape
        scale = max(1, scale)

        start_node = (start[0] // scale, start[1] // scale)
        goal_node = (goal[0] // scale, goal[1] // scale)

        if start_node == goal_node:
            return []

        # Quantized impassable cells
        blocked_nodes = {
            (r // scale, c // scale)
            for r, c in walls | hazards
            if (r // scale, c // scale) != goal_node
        }

        # Queue: (f_score, cost, current_node, action_path)
        heap = [(0, 0, start_node, [])]
        visited: dict[tuple[int, int], int] = {start_node: 0}

        actions = {
            1: (-1, 0),  # UP
            2: (1, 0),  # DOWN
            3: (0, -1),  # LEFT
            4: (0, 1),  # RIGHT
        }

        max_expansions = 2500
        expansions = 0

        while heap and expansions < max_expansions:
            expansions += 1
            f, cost, curr, path = heapq.heappop(heap)

            if curr == goal_node:
                return path

            if visited.get(curr, 99999) < cost:
                continue

            for act, (dr, dc) in actions.items():
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr * scale < H and 0 <= nc * scale < W:
                    if (nr, nc) in blocked_nodes:
                        continue

                    # Extra penalty for frequently visited tiles to avoid loops
                    tile_penalty = self.position_visits.get((nr * scale, nc * scale), 0)
                    new_cost = cost + 1 + tile_penalty * 2
                    heuristic = math.hypot(goal_node[0] - nr, goal_node[1] - nc)

                    if (nr, nc) not in visited or new_cost < visited[(nr, nc)]:
                        visited[(nr, nc)] = new_cost
                        heapq.heappush(
                            heap,
                            (new_cost + heuristic, new_cost, (nr, nc), path + [act]),
                        )

        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Decision Engine: Plan Next Action
    # ═══════════════════════════════════════════════════════════════════════

    def plan_next_action(
        self,
        curr_grid: np.ndarray,
        available_actions: list[int],
        level: int = 0,
    ) -> tuple[int, float]:
        """Compute the next action via the 4-pillar curiosity-and-exit loop."""
        self.step_counter += 1
        self.current_level = level

        # 1. Causal Learning from previous step
        if self.prev_grid is not None and self.last_action is not None:
            self.assimilate_causal_differential(self.prev_grid, self.last_action, curr_grid)

        # 2. Scene Perception & Object Segmentation
        scene = self.perceive_scene(curr_grid, available_actions)

        if scene.avatar_pos:
            self.position_visits[scene.avatar_pos] += 1

        # Check if avatar reached last target
        if self.last_target and scene.avatar_pos:
            dist = math.hypot(
                scene.avatar_pos[0] - self.last_target[0],
                scene.avatar_pos[1] - self.last_target[1],
            )
            if dist <= max(1.5, scene.scale * 1.5):
                self.tested_positions.add(self.last_target)
                self.last_target = None

        # 3. If action queue already has in-flight planned steps, execute next
        if self.planned_action_queue:
            act, act_data = self.planned_action_queue.pop(0)
            self.prev_grid = curr_grid.copy()
            self.last_action = act
            self.last_action_data = act_data
            self.action_history.append(act)
            return act, 0.9

        # 4. Handle Click-Affordance Only Games (No directional actions)
        has_movement = any(a in available_actions for a in [1, 2, 3, 4])
        if not has_movement and 6 in available_actions:
            return self._plan_click_step(scene, curr_grid, available_actions)

        # 5. Check Pillar 2: Is Exit Reachable & Active?
        exit_target = self.evaluate_exit_reachability(scene)
        if exit_target and scene.avatar_pos:
            target_pos = (
                int(round(exit_target.centroid[0])),
                int(round(exit_target.centroid[1])),
            )
            path = self._find_shortest_path(
                curr_grid,
                scene.avatar_pos,
                target_pos,
                scene.walls,
                scene.hazards,
                scale=scene.scale,
            )
            if path:
                logger.info(
                    "Pillar 2 Goal Convergence: Routing to EXIT at %s (%d steps)",
                    target_pos,
                    len(path),
                )
                first_act = path[0]
                for rem in path[1:]:
                    self.planned_action_queue.append((rem, None))
                self.last_target = target_pos
                self.prev_grid = curr_grid.copy()
                self.last_action = first_act
                self.last_action_data = None
                self.action_history.append(first_act)
                return first_act, 0.95

        # 6. Pillar 3: Epistemic Curiosity — Pick Nearest Untested Object
        curiosity_target = self.select_curiosity_target(scene)
        if curiosity_target and scene.avatar_pos:
            target_pos = (
                int(round(curiosity_target.centroid[0])),
                int(round(curiosity_target.centroid[1])),
            )
            self.target_attempts[target_pos] += 1
            self.last_target = target_pos

            path = self._find_shortest_path(
                curr_grid,
                scene.avatar_pos,
                target_pos,
                scene.walls,
                scene.hazards,
                scale=scene.scale,
            )
            if path:
                logger.info(
                    "Pillar 3 Curiosity: Probing object color %d at %s (%d steps)",
                    curiosity_target.color,
                    target_pos,
                    len(path),
                )
                first_act = path[0]
                for rem in path[1:]:
                    self.planned_action_queue.append((rem, None))
                self.prev_grid = curr_grid.copy()
                self.last_action = first_act
                self.last_action_data = None
                self.action_history.append(first_act)
                return first_act, 0.85

        # 7. Fallback: Loop Breaking & Exploratory Step
        return self._fallback_exploration(scene, available_actions, curr_grid)

    def _plan_click_step(
        self,
        scene: UniversalSceneGraph,
        curr_grid: np.ndarray,
        available_actions: list[int],
    ) -> tuple[int, float]:
        """Execute epistemic click exploration with causal momentum and quiescence learning."""
        if not hasattr(self, "_quiescent_click_targets"):
            self._quiescent_click_targets = set()
            self._completed_click_controls = set()
            self._consecutive_effective_clicks = 0
            self._visited_click_states = set()

        curr_bytes = curr_grid.tobytes()
        is_revisit = curr_bytes in self._visited_click_states
        self._visited_click_states.add(curr_bytes)

        # 1. Causal Momentum: If last click was effective on a control and didn't loop, continue
        if self.last_action == 6 and self.last_target is not None and self.prev_grid is not None:
            diff = FrameDiffAnalyzer.analyze(self.prev_grid, 6, curr_grid)
            if diff.diff_type != DiffType.NO_CHANGE:
                self._consecutive_effective_clicks += 1
                # If change is a localized in-place toggle (<= 25 px) or saturated, mark complete
                if diff.changed_pixel_count <= 25 or is_revisit:
                    self._completed_click_controls.add(self.last_target)
                    self._consecutive_effective_clicks = 0
                elif self._consecutive_effective_clicks < 8:
                    tr, tc = self.last_target
                    self.prev_grid = curr_grid.copy()
                    self.last_action = 6
                    self.last_action_data = {"x": tc, "y": tr}
                    return 6, 0.95
            else:
                self._quiescent_click_targets.add(self.last_target)
                self._consecutive_effective_clicks = 0

        # 2. Select next untested candidate interactable
        candidates = []
        for o in scene.interactables:
            if o.size > curr_grid.size * 0.25:
                continue
            cr, cc = int(round(o.centroid[0])), int(round(o.centroid[1]))
            H, W = curr_grid.shape
            if cr < 0 or cr >= H or cc < 0 or cc >= W:
                continue
            if (cr, cc) not in self._quiescent_click_targets and (
                cr,
                cc,
            ) not in self._completed_click_controls:
                candidates.append((cr, cc, o))

        if candidates:
            # Prefer 2D buttons (size 4..300)
            candidates.sort(key=lambda item: (0 if 4 <= item[2].size <= 300 else 1, item[2].size))
            best_r, best_c, best_obj = candidates[0]
            self.last_target = (best_r, best_c)
            self._consecutive_effective_clicks = 1
            self.prev_grid = curr_grid.copy()
            self.last_action = 6
            self.last_action_data = {"x": best_c, "y": best_r}
            return 6, 0.85

        # 3. Interleave Action 7 (SUBMIT/COMMIT) if available
        other_actions = [a for a in available_actions if a != 6]
        if other_actions and (
            self.step_counter % 6 == 0 or len(self._completed_click_controls) > 0
        ):
            chosen = other_actions[0]
            self.prev_grid = curr_grid.copy()
            self.last_action = chosen
            self.last_action_data = None
            return chosen, 0.75

        # If exhausted, reset completed controls to allow cycling
        self._completed_click_controls.clear()
        H, W = curr_grid.shape
        self.prev_grid = curr_grid.copy()
        self.last_action = 6
        self.last_action_data = {"x": W // 2, "y": H // 2}
        return 6, 0.5

    def _fallback_exploration(
        self,
        scene: UniversalSceneGraph,
        available_actions: list[int],
        curr_grid: np.ndarray,
    ) -> tuple[int, float]:
        """Perform collision-safe exploratory step when no specific path is active."""
        movement_actions = [a for a in available_actions if a in (1, 2, 3, 4)]
        if not movement_actions:
            chosen = available_actions[0]
            self.prev_grid = curr_grid.copy()
            self.last_action = chosen
            self.last_action_data = None
            return chosen, 0.4

        # Break 2-step oscillation (e.g. alternating 3 <-> 4 or 1 <-> 2)
        taboo_actions: set[int] = set()
        if len(self.action_history) >= 4:
            recent = self.action_history[-4:]
            if recent[0] == recent[2] and recent[1] == recent[3] and recent[0] != recent[1]:
                taboo_actions.update([recent[0], recent[1]])

        viable = [a for a in movement_actions if a not in taboo_actions] or movement_actions

        # Score actions by least visited resulting position
        action_deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        scored: list[tuple[int, int]] = []
        scale = scene.scale
        pos = scene.avatar_pos or (0, 0)

        for act in viable:
            dr, dc = action_deltas[act]
            npos = (pos[0] + dr * scale, pos[1] + dc * scale)
            if npos in scene.walls or npos in scene.hazards:
                visits = 9999
            else:
                visits = self.position_visits.get(npos, 0)
            scored.append((visits, act))

        scored.sort(key=lambda s: s[0])
        best_act = scored[0][1]

        self.prev_grid = curr_grid.copy()
        self.last_action = best_act
        self.last_action_data = None
        self.action_history.append(best_act)
        return best_act, 0.45

    def load_knowledge(self, path: Any, game_id: str | None = None) -> None:
        """Placeholder for cross-game offline memory loading."""
        pass

    def save_knowledge(self, path: Any, game_id: str | None = None) -> None:
        """Placeholder for cross-game offline memory persistence."""
        pass
