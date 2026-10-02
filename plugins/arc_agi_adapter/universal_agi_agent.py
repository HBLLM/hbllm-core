"""Universal AGI Agent (Domain-Agnostic 4-Pillar Cognitive Engine).

Operates strictly on the first principles of Artificial General Intelligence:
1. Identify Exits & Exit Conditions (Victory criteria, receptacles, doors, portals)
2. Identify Objects in the Scene (Avatar, walls, interactables, hazards, exits)
3. Identify Causal Impact & Affordances via Epistemic Curiosity ("Play with unknowns")
4. Visit Object-to-Object via Shortest Collision-Free Path with Hazard Avoidance

Architecture:
- Pure AGI: Zero game ID checks, zero hardcoded coordinates, zero bespoke puzzle archetypes.
- Completely self-contained: Standard library + numpy only.
- Dynamic Avatar Learning: Detects avatar via causal action-displacement invariance.
- Persistent Epistemic Memory: Retains causal affordances (pickups, switches, barriers, hazards) across levels.
"""

from __future__ import annotations

import collections
import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("universal_agi")


@dataclass
class SceneEntity:
    """A segmented physical or functional entity on the canvas."""

    entity_id: int
    color: int
    size: int
    bounding_box: tuple[int, int, int, int]  # (min_r, max_r, min_c, max_c)
    centroid: tuple[float, float]
    coords: list[tuple[int, int]]
    role: str = "unknown"  # "avatar", "wall", "interactable", "exit", "hazard"
    affordance: str = "unknown"  # "pickup", "push", "trigger", "barrier", "hazard", "exit"


@dataclass
class SceneGraph:
    """Topological decomposition of the current visual frame."""

    grid: np.ndarray
    bg_color: int
    avatar: SceneEntity | None = None
    avatar_pos: tuple[int, int] | None = None
    scale: int = 1
    entities: list[SceneEntity] = field(default_factory=list)
    walls: set[tuple[int, int]] = field(default_factory=set)
    hazards: set[tuple[int, int]] = field(default_factory=set)
    candidate_exits: list[SceneEntity] = field(default_factory=list)
    interactables: list[SceneEntity] = field(default_factory=list)


class UniversalAGIAgent:
    """Universal 4-Pillar Cognitive AGI Agent."""

    def __init__(self, **kwargs: Any) -> None:
        self.step_counter: int = 0
        self.current_level: int = 0
        self.prev_grid: np.ndarray | None = None
        self.last_action: int | None = None
        self.last_action_data: dict[str, int] | None = None
        self.current_avatar_pos: tuple[int, int] | None = None

        # ── Cross-Episode & Cross-Level Cognitive Memory ──────────────────
        self.known_avatar_color: int | None = None
        self.known_exit_colors: set[int] = set()
        self.known_hazard_colors: set[int] = {8}  # Standard ARC hazard default
        self.known_barriers: set[tuple[int, int]] = set()
        self.learned_affordances: dict[int, str] = {}  # color -> affordance
        self.tested_positions: set[tuple[int, int]] = set()
        self.inventory: set[int] = set()  # Collected item colors

        # ── Navigation, Dynamic Planning & Anti-Stagnation ────────────────
        self.position_visits: collections.Counter[tuple[int, int]] = collections.Counter()
        self.action_history: list[int] = []
        self.planned_action_queue: list[tuple[int, dict[str, int] | None]] = []
        self.stuck_counter: int = 0
        self.last_target_pos: tuple[int, int] | None = None
        self.target_failure_counts: collections.Counter[tuple[int, int]] = collections.Counter()

        # ── Click & Manipulation State ────────────────────────────────────
        self.quiescent_clicks: set[tuple[int, int]] = set()
        self.completed_click_controls: set[tuple[int, int]] = set()
        self.consecutive_effective_clicks: int = 0
        self.visited_grid_hashes: set[int] = set()

    def reset_episode(self, retain_dynamics: bool = False, is_retry: bool = False) -> None:
        """Reset episodic state while retaining cross-level dynamics and affordances."""
        self.prev_grid = None
        self.last_action = None
        self.last_action_data = None
        self.step_counter = 0
        self.stuck_counter = 0
        self.planned_action_queue.clear()
        self.position_visits.clear()
        self.action_history.clear()
        self.target_failure_counts.clear()
        self.last_target_pos = None
        self.consecutive_effective_clicks = 0
        self.visited_grid_hashes.clear()

        if not retain_dynamics and not is_retry:
            self.current_level = 0
            self.known_avatar_color = None
            self.known_exit_colors.clear()
            self.known_hazard_colors = {8}
            self.known_barriers.clear()
            self.learned_affordances.clear()
            self.tested_positions.clear()
            self.inventory.clear()
            self.quiescent_clicks.clear()
            self.completed_click_controls.clear()
        elif retain_dynamics and not is_retry:
            self.current_level += 1
            # Retain learned dynamics, affordances, barriers, and hazards across levels
            self.inventory.clear()

    # ═══════════════════════════════════════════════════════════════════════
    # Step 2: Object Segmentation & Scene Perception
    # ═══════════════════════════════════════════════════════════════════════

    def perceive_scene(
        self,
        grid: np.ndarray,
        available_actions: list[int],
    ) -> SceneGraph:
        """Segment canvas into discrete entities, avatar, walls, hazards, and exits."""
        if grid.ndim == 3:
            grid = grid[-1]

        H, W = grid.shape
        counts = np.bincount(grid.flatten())
        bg_color = int(np.argmax(counts))

        # Extract connected components (pure standard library + numpy)
        entities = self._segment_connected_components(grid, bg_color)
        scene = SceneGraph(grid=grid, bg_color=bg_color, entities=entities)

        # 1. Identify Controllable Avatar
        has_movement = any(a in available_actions for a in (1, 2, 3, 4))
        avatar = self._identify_avatar(scene, grid, has_movement)
        if avatar:
            avatar.role = "avatar"
            scene.avatar = avatar
            scene.avatar_pos = (
                int(round(avatar.centroid[0])),
                int(round(avatar.centroid[1])),
            )
            self.current_avatar_pos = scene.avatar_pos
            self.known_avatar_color = avatar.color
            scene.scale = max(1, int(round(math.sqrt(avatar.size))))

        # 2. Categorize remaining entities into walls, hazards, exits, interactables
        for entity in entities:
            if entity == avatar:
                continue

            min_r, max_r, min_c, max_c = entity.bounding_box
            is_frame_wall = (
                (max_r - min_r >= H - 4 and max_c - min_c >= W - 4)
                or (entity.size > H * W * 0.28)
                or (min_r <= 1 and max_r <= 1 and max_c - min_c > 8)
                or (min_r >= H - 2 and max_r >= H - 2 and max_c - min_c > 8)
            )

            pos = (int(round(entity.centroid[0])), int(round(entity.centroid[1])))
            is_barrier = is_frame_wall or pos in self.known_barriers

            if is_barrier:
                entity.role = "wall"
                for r, c in entity.coords:
                    scene.walls.add((r, c))
            elif entity.color in self.known_hazard_colors:
                entity.role = "hazard"
                for r, c in entity.coords:
                    scene.hazards.add((r, c))
            elif has_movement and self._is_candidate_exit(entity, grid, bg_color):
                entity.role = "exit"
                scene.candidate_exits.append(entity)
            else:
                entity.role = "interactable"
                scene.interactables.append(entity)

        # Add all known barrier coordinates to walls
        scene.walls.update(self.known_barriers)

        return scene

    def _segment_connected_components(self, grid: np.ndarray, bg_color: int) -> list[SceneEntity]:
        """Extract connected components using 8-connectivity flood fill."""
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        entities: list[SceneEntity] = []
        entity_id = 0

        for r in range(H):
            for c in range(W):
                val = int(grid[r, c])
                if val == bg_color or visited[r, c]:
                    continue

                component_coords: list[tuple[int, int]] = []
                queue = collections.deque([(r, c)])
                visited[r, c] = True

                while queue:
                    curr_r, curr_c = queue.popleft()
                    component_coords.append((curr_r, curr_c))

                    for dr, dc in (
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                        (-1, -1),
                        (-1, 1),
                        (1, -1),
                        (1, 1),
                    ):
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if not visited[nr, nc] and int(grid[nr, nc]) == val:
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                rs = [coord[0] for coord in component_coords]
                cs = [coord[1] for coord in component_coords]
                bbox = (min(rs), max(rs), min(cs), max(cs))
                centroid = (float(np.mean(rs)), float(np.mean(cs)))
                size = len(component_coords)

                entity = SceneEntity(
                    entity_id=entity_id,
                    color=val,
                    size=size,
                    bounding_box=bbox,
                    centroid=centroid,
                    coords=component_coords,
                )
                entities.append(entity)
                entity_id += 1

        return entities

    def _identify_avatar(
        self,
        scene: SceneGraph,
        grid: np.ndarray,
        has_movement: bool,
    ) -> SceneEntity | None:
        """Identify controllable avatar entity."""
        if not has_movement:
            return None

        # 1. Match known avatar color
        if self.known_avatar_color is not None:
            matches = [e for e in scene.entities if e.color == self.known_avatar_color]
            if matches:
                if self.current_avatar_pos:
                    matches.sort(
                        key=lambda e: math.hypot(
                            e.centroid[0] - self.current_avatar_pos[0],
                            e.centroid[1] - self.current_avatar_pos[1],
                        )
                    )
                return matches[0]

        # 2. Look for compact, movable entities
        candidates = [
            e
            for e in scene.entities
            if 1 <= e.size <= 64
            and e.color not in self.known_hazard_colors
            and e.color != scene.bg_color
        ]
        if candidates:
            # Sort by compact shape and reasonable size
            candidates.sort(
                key=lambda e: (
                    abs(
                        (e.bounding_box[1] - e.bounding_box[0])
                        - (e.bounding_box[3] - e.bounding_box[2])
                    ),
                    e.size,  # Smallest compact entity first
                )
            )
            return candidates[0]

        return None

    def _is_candidate_exit(
        self,
        entity: SceneEntity,
        grid: np.ndarray,
        bg_color: int,
    ) -> bool:
        """Identify candidate exits, receptacles, doors, or goal flags."""
        if entity.color in self.known_exit_colors:
            return True

        min_r, max_r, min_c, max_c = entity.bounding_box
        H, W = grid.shape
        touches_border = min_r <= 2 or max_r >= H - 3 or min_c <= 2 or max_c >= W - 3
        if touches_border and 2 <= entity.size <= 60:
            return True

        if self.learned_affordances.get(entity.color) in ("exit", "goal", "receptacle"):
            return True

        return False

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: Exit & Victory Criteria Teleology
    # ═══════════════════════════════════════════════════════════════════════

    def evaluate_exit_conditions(self, scene: SceneGraph) -> SceneEntity | None:
        """Check if exit conditions are met and a path to the exit is open."""
        if not scene.candidate_exits or not scene.avatar_pos:
            return None

        # Check if there are active uncollected collectibles (keys/items)
        has_uncollected_items = any(
            e.affordance == "pickup" and e.color not in self.inventory for e in scene.interactables
        )
        if has_uncollected_items:
            return None

        for exit_entity in scene.candidate_exits:
            target_pos = (
                int(round(exit_entity.centroid[0])),
                int(round(exit_entity.centroid[1])),
            )
            if target_pos in self.tested_positions and self.target_failure_counts[target_pos] >= 2:
                continue

            path = self._find_shortest_path(
                scene.grid,
                scene.avatar_pos,
                target_pos,
                scene.walls,
                scene.hazards,
                scale=scene.scale,
            )
            if path is not None and len(path) > 0:
                return exit_entity

        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3: Causal Impact & Affordances ("Play with Unknowns")
    # ═══════════════════════════════════════════════════════════════════════

    def select_curiosity_target(self, scene: SceneGraph) -> SceneEntity | None:
        """Select the most informative untested entity to probe via shortest path."""
        if not scene.avatar_pos:
            return None

        # 1. If holding items, prioritize receptacles or candidate exits
        if self.inventory:
            receptacles = [
                e
                for e in scene.entities
                if self.learned_affordances.get(e.color) in ("receptacle", "exit")
                or e.role == "exit"
            ]
            if receptacles:
                return receptacles[0]

        # 2. Select untested interactables, strictly filtering out known barriers
        candidates: list[SceneEntity] = []
        for e in scene.interactables:
            pos = (int(round(e.centroid[0])), int(round(e.centroid[1])))
            if pos in self.known_barriers:
                continue
            if e.color in self.known_hazard_colors:
                continue
            if self.target_failure_counts[pos] >= 2:
                continue
            candidates.append(e)

        if not candidates:
            candidates = [
                e
                for e in scene.interactables
                if (int(round(e.centroid[0])), int(round(e.centroid[1]))) not in self.known_barriers
                and e.color not in self.known_hazard_colors
                and self.target_failure_counts[
                    (int(round(e.centroid[0])), int(round(e.centroid[1])))
                ]
                < 3
            ]

        if not candidates:
            return None

        def target_score(entity: SceneEntity) -> float:
            tpos = (int(round(entity.centroid[0])), int(round(entity.centroid[1])))
            dist = math.hypot(
                tpos[0] - scene.avatar_pos[0],
                tpos[1] - scene.avatar_pos[1],
            )
            penalty = self.target_failure_counts[tpos] * 50.0
            return dist + penalty

        candidates.sort(key=target_score)
        return candidates[0]

    def assimilate_causal_differential(
        self, prev_grid: np.ndarray, action: int, curr_grid: np.ndarray
    ) -> None:
        """Learn physical properties, avatar identity, and affordances from ΔGrid."""
        diff_mask = prev_grid != curr_grid
        diff_count = int(np.sum(diff_mask))

        # Dynamic Avatar Learning: If unknown, detect entity displaced by action
        if self.known_avatar_color is None and action in (1, 2, 3, 4):
            deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
            dr, dc = deltas[action]
            H, W = prev_grid.shape
            for r in range(H):
                for c in range(W):
                    val = int(prev_grid[r, c])
                    if val != 0 and val != int(np.bincount(prev_grid.flatten()).argmax()):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if curr_grid[r, c] != val and curr_grid[nr, nc] == val:
                                self.known_avatar_color = val
                                logger.info(
                                    "Causal Discovery: Confirmed AVATAR color %d by movement invariance",
                                    val,
                                )
                                break
                if self.known_avatar_color is not None:
                    break

        # Case 0: No visual change occurred
        if diff_count == 0:
            self.stuck_counter += 1
            if self.last_target_pos:
                self.target_failure_counts[self.last_target_pos] += 1
                if self.target_failure_counts[self.last_target_pos] >= 2:
                    self.known_barriers.add(self.last_target_pos)
                    logger.info(
                        "Causal Discovery: Marked %s as STATIC BARRIER", self.last_target_pos
                    )
            return

        self.stuck_counter = 0

        # Case 1: Avatar interaction in movement game
        if self.last_target_pos and self.current_avatar_pos:
            tr, tc = self.last_target_pos
            old_val = int(prev_grid[tr, tc])
            new_val = int(curr_grid[tr, tc])

            # Item pickup: object disappeared into avatar or became background
            if old_val != new_val and old_val not in (
                0,
                int(np.bincount(curr_grid.flatten()).argmax()),
            ):
                self.inventory.add(old_val)
                self.learned_affordances[old_val] = "pickup"
                self.tested_positions.add((tr, tc))
                logger.info(
                    "Causal Discovery: Picked up item color %d at (%d, %d)", old_val, tr, tc
                )

            # Remote trigger: pixels changed elsewhere on screen
            if diff_count > 1:
                self.learned_affordances[old_val] = "trigger"
                self.tested_positions.add((tr, tc))
                logger.info(
                    "Causal Discovery: Entity color %d caused trigger effect (%d pixels changed)",
                    old_val,
                    diff_count,
                )

    # ═══════════════════════════════════════════════════════════════════════
    # Step 4: Geodesic Spatial Navigation ($A^*$)
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

        blocked_nodes = {
            (r // scale, c // scale)
            for r, c in (walls | hazards | self.known_barriers)
            if (r // scale, c // scale) != goal_node
        }

        heap = [(0, 0, start_node, [])]
        visited: dict[tuple[int, int], int] = {start_node: 0}

        actions = {
            1: (-1, 0),  # UP
            2: (1, 0),  # DOWN
            3: (0, -1),  # LEFT
            4: (0, 1),  # RIGHT
        }

        max_expansions = 3000
        expansions = 0

        while heap and expansions < max_expansions:
            expansions += 1
            f, cost, curr, path = heapq.heappop(heap)

            if curr == goal_node:
                return path

            if visited.get(curr, 999999) < cost:
                continue

            for act, (dr, dc) in actions.items():
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr * scale < H and 0 <= nc * scale < W:
                    if (nr, nc) in blocked_nodes:
                        continue

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
    # Master Decision Engine: Plan Next Action
    # ═══════════════════════════════════════════════════════════════════════

    def plan_next_action(
        self,
        curr_grid: np.ndarray,
        available_actions: list[int],
        level: int = 0,
    ) -> tuple[int, float]:
        """Compute the next action via the 4-pillar curiosity-and-exit cognitive loop."""
        self.step_counter += 1
        self.current_level = level

        # 1. Causal learning from previous action differential
        if self.prev_grid is not None and self.last_action is not None:
            self.assimilate_causal_differential(self.prev_grid, self.last_action, curr_grid)

        # 2. Scene Perception & Object Segmentation
        scene = self.perceive_scene(curr_grid, available_actions)

        if scene.avatar_pos:
            self.position_visits[scene.avatar_pos] += 1

        # Check if target reached
        if self.last_target_pos and scene.avatar_pos:
            dist = math.hypot(
                scene.avatar_pos[0] - self.last_target_pos[0],
                scene.avatar_pos[1] - self.last_target_pos[1],
            )
            if dist <= max(1.5, scene.scale * 1.5):
                self.tested_positions.add(self.last_target_pos)
                self.last_target_pos = None

        # 3. If action queue already has in-flight planned steps, execute next
        if self.planned_action_queue:
            act, act_data = self.planned_action_queue.pop(0)
            self.prev_grid = curr_grid.copy()
            self.last_action = act
            self.last_action_data = act_data
            self.action_history.append(act)
            return act, 0.9

        # 4. Handle Click & Manipulation Affordance Puzzles (Actions 5, 6, 7)
        has_movement = any(a in available_actions for a in (1, 2, 3, 4))
        if not has_movement and 6 in available_actions:
            return self._plan_click_manipulation(scene, curr_grid, available_actions)

        # 5. Step 1: Check if Exit Conditions are Met
        exit_entity = self.evaluate_exit_conditions(scene)
        if exit_entity and scene.avatar_pos:
            target_pos = (
                int(round(exit_entity.centroid[0])),
                int(round(exit_entity.centroid[1])),
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
                    "Step 1 Exit Convergence: Routing directly to EXIT at %s (%d steps)",
                    target_pos,
                    len(path),
                )
                first_act = path[0]
                for rem in path[1:]:
                    self.planned_action_queue.append((rem, None))
                self.last_target_pos = target_pos
                self.prev_grid = curr_grid.copy()
                self.last_action = first_act
                self.last_action_data = None
                self.action_history.append(first_act)
                return first_act, 0.95

        # 6. Step 3: Epistemic Curiosity — Play with Unknown Objects
        curiosity_target = self.select_curiosity_target(scene)
        if curiosity_target and scene.avatar_pos:
            target_pos = (
                int(round(curiosity_target.centroid[0])),
                int(round(curiosity_target.centroid[1])),
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
                    "Step 3 Curiosity Probing: Target object color %d at %s (%d steps)",
                    curiosity_target.color,
                    target_pos,
                    len(path),
                )
                self.last_target_pos = target_pos
                first_act = path[0]
                for rem in path[1:]:
                    self.planned_action_queue.append((rem, None))
                self.prev_grid = curr_grid.copy()
                self.last_action = first_act
                self.last_action_data = None
                self.action_history.append(first_act)
                return first_act, 0.85

        # 7. Fallback: Loop Breaking & Exploratory Stepping
        return self._fallback_exploration(scene, available_actions, curr_grid)

    def _plan_click_manipulation(
        self,
        scene: SceneGraph,
        curr_grid: np.ndarray,
        available_actions: list[int],
    ) -> tuple[int, float]:
        """Execute epistemic click exploration with causal momentum and submission interleaving."""
        curr_hash = hash(curr_grid.tobytes())
        is_revisit = curr_hash in self.visited_grid_hashes
        self.visited_grid_hashes.add(curr_hash)

        # 1. If an auxiliary submission/run action is available, interleave it every 4 steps
        # or after completing a control manipulation to test if exit condition is met
        auxiliary_actions = [a for a in available_actions if a != 6]
        if (
            auxiliary_actions
            and self.last_action == 6
            and (self.step_counter % 4 == 0 or len(self.completed_click_controls) > 0)
        ):
            chosen = auxiliary_actions[0]
            self.prev_grid = curr_grid.copy()
            self.last_action = chosen
            self.last_action_data = None
            self.completed_click_controls.clear()
            return chosen, 0.9

        # 2. Causal Momentum: If last click caused progress, continue
        if (
            self.last_action == 6
            and self.last_target_pos is not None
            and self.prev_grid is not None
        ):
            diff_count = int(np.sum(self.prev_grid != curr_grid))
            if diff_count > 0:
                self.consecutive_effective_clicks += 1
                if diff_count <= 25 or is_revisit:
                    self.completed_click_controls.add(self.last_target_pos)
                    self.consecutive_effective_clicks = 0
                elif self.consecutive_effective_clicks < 6:
                    tr, tc = self.last_target_pos
                    self.prev_grid = curr_grid.copy()
                    self.last_action = 6
                    self.last_action_data = {"x": tc, "y": tr}
                    return 6, 0.95
            else:
                self.quiescent_clicks.add(self.last_target_pos)
                self.consecutive_effective_clicks = 0

        # 3. Select next candidate interactable button/control
        candidates = []
        for e in scene.interactables:
            if e.size > curr_grid.size * 0.25:
                continue
            cr, cc = int(round(e.centroid[0])), int(round(e.centroid[1]))
            H, W = curr_grid.shape
            if cr < 0 or cr >= H or cc < 0 or cc >= W:
                continue
            if (cr, cc) not in self.quiescent_clicks and (
                cr,
                cc,
            ) not in self.completed_click_controls:
                candidates.append((cr, cc, e))

        if candidates:
            candidates.sort(key=lambda item: (0 if 4 <= item[2].size <= 300 else 1, item[2].size))
            best_r, best_c, best_ent = candidates[0]
            self.last_target_pos = (best_r, best_c)
            self.consecutive_effective_clicks = 1
            self.prev_grid = curr_grid.copy()
            self.last_action = 6
            self.last_action_data = {"x": best_c, "y": best_r}
            return 6, 0.85

        # Fallback if exhausted
        self.completed_click_controls.clear()
        self.quiescent_clicks.clear()
        H, W = curr_grid.shape
        self.prev_grid = curr_grid.copy()
        self.last_action = 6
        self.last_action_data = {"x": W // 2, "y": H // 2}
        return 6, 0.5

    def _fallback_exploration(
        self,
        scene: SceneGraph,
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

        taboo_actions: set[int] = set()
        if len(self.action_history) >= 4:
            recent = self.action_history[-4:]
            if recent[0] == recent[2] and recent[1] == recent[3] and recent[0] != recent[1]:
                taboo_actions.update([recent[0], recent[1]])

        viable = [a for a in movement_actions if a not in taboo_actions] or movement_actions

        action_deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        scored: list[tuple[int, int]] = []
        scale = scene.scale
        pos = scene.avatar_pos or (0, 0)

        for act in viable:
            dr, dc = action_deltas[act]
            npos = (pos[0] + dr * scale, pos[1] + dc * scale)
            if npos in scene.walls or npos in scene.hazards or npos in self.known_barriers:
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
