"""ARC-3 Spatial Cognitive Agent — Autonomous, General Spatial Intelligence via HCIR.

Delegates scene lifting, topological cut-set discovery, state-space sequence planning,
and constraint induction to HCIRSpatialEntityPlanner, PhysicsPredictor, and native HCIR memory.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from hbllm.hcir.spatial_planner import (
    EntityGraph,
    EntityRole,
    HCIRSpatialEntityPlanner,
    SequencePlanStep,
    SpatialEntity,
)
from hbllm.hcir.topological_cut_set import TopologicalCutSetAnalyzer
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.predictors.physics import PhysicsPredictor
from plugins.arc_agi_adapter.arc_agi_3_runner import (
    ActionDynamicsModel,
    ARCGrid,
    GridTopologyExtractor,
)

logger = logging.getLogger(__name__)


class ARCPerceptualLifter:
    """Perceptual front-end for ARC-AGI-3 environments.

    Lifts raw pixel grids and segmented objects into domain-agnostic SpatialEntity instances
    and barrier coordinate sets for HCIRSpatialEntityPlanner.
    """

    @staticmethod
    def lift(
        grid: np.ndarray,
        raw_objects: list[Any],
        avatar_color: int | None = None,
        avatar_centroid: tuple[float, float] | None = None,
        learned_item_colors: set[int] | None = None,
        learned_receptacle_colors: set[int] | None = None,
        learned_barrier_colors: set[int] | None = None,
        walkable_colors: set[int] | None = None,
        step_size: int = 1,
        target_zone_bounds: tuple[int, int, int, int] | None = None,
    ) -> tuple[list[SpatialEntity], set[tuple[int, int]]]:
        H, W = grid.shape
        step = step_size
        counts = np.bincount(grid.ravel())
        bg_color = int(np.argmax(counts))
        walkable = set(walkable_colors or set()) | {0, bg_color}
        for col, count in enumerate(counts):
            if count >= int(H * W * 0.20) and col != avatar_color:
                walkable.add(int(col))
        learned_b = (learned_barrier_colors or set()) - walkable
        learned_i = (learned_item_colors or set()) - walkable
        learned_r = (learned_receptacle_colors or set()) - walkable

        raw_barriers: set[tuple[int, int]] = set()
        for b_col in learned_b:
            if b_col not in walkable and b_col != avatar_color:
                raw_barriers.update(set(zip(*np.where(grid == b_col))))

        # Detect collinear segmented wall blocks
        from collections import defaultdict

        col_blocks = defaultdict(list)
        row_blocks = defaultdict(list)
        for o in raw_objects:
            if o.color in walkable or o.color == 0 or o.color == avatar_color:
                continue
            span_r = o.max_r - o.min_r
            span_c = o.max_c - o.min_c
            # Single continuous wall spanning >= 40% of grid
            if span_r >= int(H * 0.4) and span_c <= max(4, step * 2):
                for br in range(o.min_r, o.max_r + 1):
                    for bc in range(o.min_c, o.max_c + 1):
                        raw_barriers.add((br, bc))
            elif span_c >= int(W * 0.4) and span_r <= max(4, step * 2):
                for br in range(o.min_r, o.max_r + 1):
                    for bc in range(o.min_c, o.max_c + 1):
                        raw_barriers.add((br, bc))
            else:
                col_key = int(round(o.centroid[1] / step)) * step
                row_key = int(round(o.centroid[0] / step)) * step
                col_blocks[(o.color, col_key)].append(o)
                row_blocks[(o.color, row_key)].append(o)

        for (b_col, c_pos), b_list in col_blocks.items():
            if len(b_list) >= 3:
                min_r = min(o.min_r for o in b_list)
                max_r = max(o.max_r for o in b_list)
                if (max_r - min_r) >= int(H * 0.4):
                    for o in b_list:
                        for br in range(o.min_r, o.max_r + 1):
                            for bc in range(o.min_c, o.max_c + 1):
                                raw_barriers.add((br, bc))

        for (b_col, r_pos), b_list in row_blocks.items():
            if len(b_list) >= 3:
                min_c = min(o.min_c for o in b_list)
                max_c = max(o.max_c for o in b_list)
                if (max_c - min_c) >= int(W * 0.4):
                    for o in b_list:
                        for br in range(o.min_r, o.max_r + 1):
                            for bc in range(o.min_c, o.max_c + 1):
                                raw_barriers.add((br, bc))

        # Detect receptacle bounds to prevent interior cavity colors from being classified as items
        receptacle_bounds = []
        if target_zone_bounds:
            receptacle_bounds.append(target_zone_bounds)
        else:
            for o in raw_objects:
                if o.color in walkable or o.color == 0 or o.area >= int(H * W * 0.30):
                    continue
                if (o.color in learned_r and o.area >= 16) or (
                    o.area >= 24
                    and o.color not in (avatar_color, 0)
                    and (o.max_r - o.min_r >= 4)
                    and (o.max_c - o.min_c >= 4)
                ):
                    receptacle_bounds.append((o.min_r, o.max_r, o.min_c, o.max_c))

        entities: list[SpatialEntity] = []
        max_item_area = max(36, int(step * step * 2.5))

        if target_zone_bounds:
            tz_r = int(round((target_zone_bounds[0] + target_zone_bounds[1]) * 0.5 / step)) * step
            tz_c = int(round((target_zone_bounds[2] + target_zone_bounds[3]) * 0.5 / step)) * step
            receptacle_ent = SpatialEntity(
                id=f"receptacle_{target_zone_bounds[0]}_{target_zone_bounds[2]}",
                role=EntityRole.RECEPTACLE,
                centroid=(
                    (target_zone_bounds[0] + target_zone_bounds[1]) * 0.5,
                    (target_zone_bounds[2] + target_zone_bounds[3]) * 0.5,
                ),
                grid_pos=(tz_r, tz_c),
                area=(target_zone_bounds[1] - target_zone_bounds[0] + 1)
                * (target_zone_bounds[3] - target_zone_bounds[2] + 1),
                bounding_box=target_zone_bounds,
            )
            entities.append(receptacle_ent)

        for o in raw_objects:
            if o.color in walkable or o.color == 0 or o.area >= int(H * W * 0.30):
                continue

            # 1. Avatar check FIRST (avatar must never be swallowed or marked as a barrier)
            is_avatar = False
            if avatar_color is not None:
                if o.color == avatar_color:
                    is_avatar = True
            elif (
                avatar_centroid
                and math.hypot(
                    o.centroid[0] - avatar_centroid[0], o.centroid[1] - avatar_centroid[1]
                )
                < 2.0
            ):
                is_avatar = True

            r = int(round((o.centroid[0] - (step - 1) * 0.5) / step)) * step
            c = int(round((o.centroid[1] - (step - 1) * 0.5) / step)) * step
            e_id = f"ent_{o.color}_{r}_{c}"

            if is_avatar:
                ent = SpatialEntity(
                    id=e_id,
                    role=EntityRole.AGENT,
                    centroid=(float(o.centroid[0]), float(o.centroid[1])),
                    grid_pos=(r, c),
                    area=int(o.area),
                    bounding_box=(int(o.min_r), int(o.max_r), int(o.min_c), int(o.max_c)),
                    color=int(o.color),
                )
                entities.append(ent)
                continue

            # 2. If inside receptacle bounds, it is part of receptacle or a delivered item
            is_inside_receptacle = any(
                b[0] <= o.min_r and o.max_r <= b[1] and b[2] <= o.min_c and o.max_c <= b[3]
                for b in receptacle_bounds
            )
            if is_inside_receptacle and target_zone_bounds:
                # Any external object inside receptacle (delivered items, NPC bots) is a physical obstacle
                if o.color not in learned_r and o.color not in walkable and o.color != avatar_color:
                    if hasattr(o, "coords"):
                        raw_barriers.update((int(cr), int(cc)) for cr, cc in o.coords)
                    else:
                        for br in range(o.min_r, o.max_r + 1):
                            for bc in range(o.min_c, o.max_c + 1):
                                raw_barriers.add((br, bc))
                continue

            span_r = o.max_r - o.min_r
            span_c = o.max_c - o.min_c

            role = EntityRole.UNKNOWN
            if (
                (span_r >= int(H * 0.4) and span_c <= max(4, step * 2))
                or (span_c >= int(W * 0.4) and span_r <= max(4, step * 2))
                or o.color in learned_b
                or (o.centroid[0], o.centroid[1]) in raw_barriers
            ):
                role = EntityRole.OBSTACLE
                if hasattr(o, "coords"):
                    raw_barriers.update((int(cr), int(cc)) for cr, cc in o.coords)
                else:
                    for br in range(o.min_r, o.max_r + 1):
                        for bc in range(o.min_c, o.max_c + 1):
                            raw_barriers.add((br, bc))
            elif (
                (o.color in learned_r and o.area >= 16)
                or (
                    o.area >= 24
                    and o.color not in (avatar_color, 0)
                    and (o.max_r - o.min_r >= 4)
                    and (o.max_c - o.min_c >= 4)
                )
                or is_inside_receptacle
            ):
                role = EntityRole.RECEPTACLE
            elif (
                ((o.color in learned_i) or (not learned_i and o.color != avatar_color))
                and (max(3, int(step * step * 0.25)) <= o.area <= max_item_area)
                and (step <= 1 or (span_r >= max(1, step // 2) and span_c >= max(1, step // 2)))
            ):
                role = EntityRole.MANIPULABLE
            else:
                role = EntityRole.ACTUATOR

            ent = SpatialEntity(
                id=e_id,
                role=role,
                centroid=(float(o.centroid[0]), float(o.centroid[1])),
                grid_pos=(r, c),
                area=int(o.area),
                bounding_box=(int(o.min_r), int(o.max_r), int(o.min_c), int(o.max_c)),
                color=int(o.color),
            )
            entities.append(ent)

        return entities, raw_barriers


class ARC3SpatialCognitiveAgent:
    """Universal Spatial Cognitive Agent powered by HCIRSpatialEntityPlanner."""

    def __init__(self, step_size: int = 1) -> None:
        self.step_size: int = step_size
        self.avatar_color: int | None = None
        self.avatar_centroid: tuple[float, float] | None = None
        self.action_models: dict[int, ActionDynamicsModel] = {}
        self.available_actions: list[int] = []

        # Cognitive World & Planning Engine
        self.spatial_planner = HCIRSpatialEntityPlanner(step_size=step_size)
        self.workspace = HCIRWorkspaceState()

        # Physical Interaction State
        self.holding_item: bool = False
        self.carried_offset: tuple[float, float] = (0.0, 0.0)
        self.current_facing: tuple[int, int] = (0, 0)
        self.delivered_positions: set[tuple[int, int]] = set()

        # Cumulative Memory across Episodes/Levels
        self.learned_barrier_colors: set[int] = set()
        self.learned_walkable_colors: set[int] = set()
        self.learned_item_colors: set[int] = set()
        self.learned_receptacle_colors: set[int] = set()
        self.target_zone_bounds: tuple[int, int, int, int] | None = None
        self.is_cooperative_handoff: bool = False

        # Navigation & Probing State
        self.probe_step_counter: int = 0
        self.visited_positions: list[tuple[int, int]] = []
        self.blocked_actions: set[int] = set()
        self.stuck_counter: int = 0
        self.last_action: int | None = None
        self.current_plan: list[SequencePlanStep] = []
        self.known_barriers: np.ndarray | None = None

    @classmethod
    def is_cooperative_candidate(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Domain-agnostic check if environment possesses multi-entity spatial manipulation or cooperative structure."""
        if set(available_actions) != {1, 2, 3, 4, 5}:
            return False
        H, W = grid.shape
        if H < 32 or W < 32:
            return False
        counts = np.bincount(grid.ravel())
        bg_color = int(np.argmax(counts))
        active = [c for c, cnt in enumerate(counts) if cnt > 0 and c != bg_color and c != 0]
        has_small = any(4 <= counts[c] <= 36 for c in active)
        has_large = any(counts[c] >= 32 for c in active)
        return bool(has_small and has_large)

    def reset_episode(self, retain_dynamics: bool = False) -> None:
        """Reset internal agent state for a new level/episode."""
        if not retain_dynamics:
            self.action_models.clear()
            self.avatar_color = None
            self.step_size = 1
            self.spatial_planner = HCIRSpatialEntityPlanner(step_size=1)
        else:
            self.spatial_planner = HCIRSpatialEntityPlanner(step_size=self.step_size)

        self.learned_barrier_colors.clear()
        self.learned_walkable_colors.clear()
        self.learned_item_colors.clear()
        self.learned_receptacle_colors.clear()
        self.target_zone_bounds = None

        self.avatar_centroid = None
        self.holding_item = False
        self.carried_offset = (0.0, 0.0)
        self.current_facing = (0, 0)
        self.delivered_positions.clear()
        self.visited_positions.clear()
        self.blocked_actions.clear()
        self.stuck_counter = 0
        self.last_action = None
        self.current_plan.clear()
        self.probe_step_counter = 0
        self.is_cooperative_handoff = False
        self.known_barriers = None
        self.target_zone_bounds = None
        self.workspace = HCIRWorkspaceState()
        self.spatial_planner.reset()

    def _active_probe_action(self, available_actions: list[int]) -> int:
        """Systematically probe available actions to discover motor displacements."""
        uncalibrated = [
            a
            for a in available_actions
            if a in [1, 2, 3, 4]
            and (a not in self.action_models or self.action_models[a].confidence < 0.8)
        ]
        if uncalibrated:
            return uncalibrated[self.probe_step_counter % len(uncalibrated)]
        directional = [a for a in available_actions if a in [1, 2, 3, 4]]
        if directional:
            return directional[self.probe_step_counter % len(directional)]
        return available_actions[0]

    def _snap_coord(self, raw_val: float) -> int:
        """Snap float coordinate to discrete lattice coordinate using sprite offset."""
        if self.step_size > 1:
            offset = (self.step_size - 1) * 0.5
            return int(round((raw_val - offset) / self.step_size)) * self.step_size
        return int(round(raw_val))

    def _stand_clear_action(
        self,
        curr_r: float,
        curr_c: float,
        eg,
        available_actions: list[int],
    ) -> tuple[int, float]:
        """Stand clear of the active zone and wait so companion bots can complete delivery."""
        if self.is_cooperative_handoff:
            doorways = [
                e for e in eg.entities.values() if e.role in (EntityRole.PORTAL, EntityRole.DOORWAY)
            ]
            if doorways:
                appr_c = doorways[0].properties.get("approach_cell", (curr_r, 28))[1]
                if curr_c > appr_c - self.step_size * 4:
                    return 3, 0.95  # Move Left clear of doorway
            if curr_r > 16:
                return 1, 0.90  # Move UP clear of active zone
            if 5 in available_actions:
                return 5, 0.95  # Wait safely for companion bot
            return available_actions[0], 0.90
        elif self.target_zone_bounds:
            tz_min_r, tz_max_r, tz_min_c, tz_max_c = self.target_zone_bounds
            if curr_r >= tz_min_r - self.step_size * 3:
                return 1, 0.90  # Move UP away from receptacle
            if 5 in available_actions:
                return 5, 0.95  # Wait safely for companion bot
            return available_actions[0], 0.90
        if 5 in available_actions:
            return 5, 0.95
        safe_act = 1 if curr_r >= 20 else 2
        return safe_act, 0.90

    def plan_next_action(
        self,
        curr_grid: np.ndarray,
        available_actions: list[int],
        tags: list[str] | None = None,
    ) -> tuple[int, float]:
        """Synthesize next action by lifting scene to EntityGraph and planning state-space paths."""
        self.available_actions = list(available_actions)
        H, W = curr_grid.shape

        # Learn dominant floor / background colors dynamically
        counts = np.bincount(curr_grid.ravel())
        bg_color = int(np.argmax(counts))
        self.learned_walkable_colors.add(bg_color)
        self.learned_walkable_colors.add(0)
        for col, count in enumerate(counts):
            if count >= int(H * W * 0.20) and (
                self.avatar_color is None or col != self.avatar_color
            ):
                self.learned_walkable_colors.add(int(col))
        self.learned_barrier_colors.difference_update(self.learned_walkable_colors)
        self.learned_item_colors.difference_update(self.learned_walkable_colors)

        # ── 1. Motor Calibration Probing ──
        calibrated_models = [
            m
            for a, m in self.action_models.items()
            if a in available_actions and m.confidence >= 0.8 and (m.delta_r != 0 or m.delta_c != 0)
        ]
        directional_avail = [a for a in available_actions if a in [1, 2, 3, 4]]
        if len(calibrated_models) < min(4, len(directional_avail)):
            self.probe_step_counter += 1
            return self._active_probe_action(available_actions), 0.50

        # ── 2. Locate Avatar on Grid Lattice ──
        if self.avatar_color is not None:
            p_av = np.where(curr_grid == self.avatar_color)
            if len(p_av[0]) > 0:
                curr_r = self._snap_coord(float(np.mean(p_av[0])))
                curr_c = self._snap_coord(float(np.mean(p_av[1])))
                self.avatar_centroid = (float(curr_r), float(curr_c))
            else:
                self.probe_step_counter += 1
                return self._active_probe_action(available_actions), 0.40
        else:
            self.probe_step_counter += 1
            return self._active_probe_action(available_actions), 0.40

        # ── 3. Extract Objects & Lift Scene to EntityGraph ──
        arc_grid = ARCGrid.from_list(curr_grid.tolist())
        objs = GridTopologyExtractor.extract_objects(arc_grid)

        # Check if target_zone_bounds is still valid on current frame
        if self.target_zone_bounds is not None and self.learned_receptacle_colors:
            tz_r1, tz_r2, tz_c1, tz_c2 = self.target_zone_bounds
            tz_slice = curr_grid[tz_r1 : tz_r2 + 1, tz_c1 : tz_c2 + 1]
            if tz_slice.size == 0 or not any(c in tz_slice for c in self.learned_receptacle_colors):
                self.target_zone_bounds = None
                self.current_plan.clear()

        # Detect Receptacle / Target Zone (only when not yet established for current level)
        if self.target_zone_bounds is None:
            candidate_goals = [
                o
                for o in objs
                if o.color not in (0, self.avatar_color)
                and o.color not in self.learned_walkable_colors
                and np.count_nonzero(curr_grid == o.color) < (curr_grid.size * 0.30)
            ]
            target_zones = [
                o
                for o in candidate_goals
                if (o.color in self.learned_receptacle_colors and o.area >= 24)
                or (getattr(o, "is_frame", False) and o.area > 16)
                or (o.area >= 24 and (o.max_r - o.min_r >= 4) and (o.max_c - o.min_c >= 4))
            ]
            if target_zones:
                best_tz = max(
                    target_zones,
                    key=lambda o: (
                        3
                        if (
                            o.color in self.learned_receptacle_colors
                            and getattr(o, "is_frame", False)
                        )
                        else (
                            2
                            if o.color in self.learned_receptacle_colors
                            else (1 if getattr(o, "is_frame", False) else 0)
                        ),
                        o.area,
                    ),
                )
                self.target_zone_bounds = (
                    best_tz.min_r,
                    best_tz.max_r,
                    best_tz.min_c,
                    best_tz.max_c,
                )
                tz_patch = curr_grid[
                    best_tz.min_r : best_tz.max_r + 1, best_tz.min_c : best_tz.max_c + 1
                ]
                self.learned_receptacle_colors.update(
                    int(c)
                    for c in np.unique(tz_patch)
                    if int(c) not in self.learned_item_colors
                    and int(c) != 0
                    and int(c) != self.avatar_color
                )

        # Dynamically learn item colors from objects outside target zone
        from collections import Counter

        candidate_item_colors = Counter()
        max_item_area = max(36, int(self.step_size * self.step_size * 2.5))
        min_item_area = max(3, int(self.step_size * self.step_size * 0.25))
        for o in objs:
            if (
                o.color in (0, self.avatar_color)
                or o.color in self.learned_walkable_colors
                or o.color in self.learned_receptacle_colors
            ):
                continue
            if self.target_zone_bounds:
                tz = self.target_zone_bounds
                if tz[0] <= o.min_r and o.max_r <= tz[1] and tz[2] <= o.min_c and o.max_c <= tz[3]:
                    continue
            span_r = o.max_r - o.min_r
            span_c = o.max_c - o.min_c
            if self.step_size > 1 and (
                span_r < max(1, self.step_size // 2) or span_c < max(1, self.step_size // 2)
            ):
                continue
            if min_item_area <= o.area <= max_item_area:
                if span_r <= self.step_size * 2 and span_c <= self.step_size * 2:
                    candidate_item_colors[o.color] += 1

        for col, count in candidate_item_colors.items():
            if count >= 2 or (col in self.learned_item_colors and count >= 1):
                self.learned_item_colors.add(col)
        self.learned_receptacle_colors.difference_update(self.learned_item_colors)

        # Track delivered items inside receptacle (e.g. delivered by self or cooperative NPC)
        if self.target_zone_bounds:
            tz = self.target_zone_bounds
            for o in objs:
                if tz[0] <= o.min_r and o.max_r <= tz[1] and tz[2] <= o.min_c and o.max_c <= tz[3]:
                    if o.color in self.learned_item_colors:
                        r = self._snap_coord(float(o.centroid[0]))
                        c = self._snap_coord(float(o.centroid[1]))
                        self.delivered_positions.add((r, c))

            # Prune external handoff positions that no longer contain an item (picked up by NPC)
            occupied_slots = {
                (self._snap_coord(float(o.centroid[0])), self._snap_coord(float(o.centroid[1])))
                for o in objs
                if o.color in self.learned_item_colors
            }
            to_remove = {
                dp
                for dp in self.delivered_positions
                if not (tz[0] <= dp[0] <= tz[1] and tz[2] <= dp[1] <= tz[3])
                and dp not in occupied_slots
            }
            self.delivered_positions.difference_update(to_remove)

        known_b: set[tuple[int, int]] = set()
        if self.known_barriers is not None:
            known_b = set(zip(*np.where(self.known_barriers)))

        # Lift scene using ARCPerceptualLifter (adapter domain)
        entities, raw_barriers = ARCPerceptualLifter.lift(
            grid=curr_grid,
            raw_objects=objs,
            avatar_color=self.avatar_color,
            avatar_centroid=self.avatar_centroid,
            learned_item_colors=self.learned_item_colors,
            learned_receptacle_colors=self.learned_receptacle_colors,
            learned_barrier_colors=self.learned_barrier_colors,
            walkable_colors=self.learned_walkable_colors,
            step_size=self.step_size,
            target_zone_bounds=self.target_zone_bounds,
        )

        # Construct topological EntityGraph using HCIRSpatialEntityPlanner (HCIR domain-agnostic)
        eg: EntityGraph = self.spatial_planner.construct_entity_graph(
            entities=entities,
            barriers=raw_barriers,
            grid_shape=(H, W),
            step_size=self.step_size,
            known_barriers=known_b,
        )

        # ── 4. Determine Cooperative Partition State ──
        if self.target_zone_bounds:
            tz_min_r, tz_max_r, tz_min_c, tz_max_c = self.target_zone_bounds
            tz_center = (
                self._snap_coord((tz_min_r + tz_max_r) / 2.0),
                self._snap_coord((tz_min_c + tz_max_c) / 2.0),
            )
            if not self.is_cooperative_handoff:
                p_comp = TopologicalCutSetAnalyzer.get_reachable_component(
                    (curr_r, curr_c),
                    eg.barriers,
                    (H, W),
                    self.step_size,
                )
                self.is_cooperative_handoff = bool(tz_center not in p_comp)
        else:
            tz_center = (H // 2, W // 2)

        # ── 5. Plan High-Level Entity Subgoal Sequence ──
        if not self.current_plan:
            self.current_plan = self.spatial_planner.plan_sequence(
                eg=eg,
                workspace=self.workspace,
                delivered_positions=self.delivered_positions,
                is_carrying=self.holding_item,
                carried_offset=self.carried_offset,
            )
        # ── 6. State Machine: Execute Active Subgoal Step ──
        # If no plan steps left:
        if not self.current_plan:
            return self._stand_clear_action(curr_r, curr_c, eg, available_actions)
        else:
            active_step = self.current_plan[0]
            # Dynamic re-planning if drop slot was occupied or invalid
            if active_step.action_type == "DROP":
                drop_slot = (
                    int(round(active_step.target_pos[0] + self.carried_offset[0])),
                    int(round(active_step.target_pos[1] + self.carried_offset[1])),
                )
                is_portal_drop = any(
                    e.role in (EntityRole.PORTAL, EntityRole.DOORWAY)
                    and (e.grid_pos == drop_slot or e.id == active_step.target_entity_id)
                    for e in eg.entities.values()
                )
                is_blocked = (drop_slot in self.delivered_positions) or (
                    not is_portal_drop and drop_slot in eg.barriers
                )
                if not is_blocked and is_portal_drop:
                    # Portal drops only re-plan if another movable item is physically in the doorway
                    if any(
                        e.role in (EntityRole.MANIPULABLE, EntityRole.MOVABLE_ITEM)
                        and math.hypot(e.grid_pos[0] - drop_slot[0], e.grid_pos[1] - drop_slot[1])
                        < self.step_size * 0.75
                        for e in eg.entities.values()
                    ):
                        is_blocked = True

                if is_blocked:
                    self.current_plan.clear()
                    self.current_plan = self.spatial_planner.plan_sequence(
                        eg=eg,
                        workspace=self.workspace,
                        delivered_positions=self.delivered_positions,
                        is_carrying=self.holding_item,
                        carried_offset=self.carried_offset,
                    )
                    if not self.current_plan:
                        return self._stand_clear_action(curr_r, curr_c, eg, available_actions)
                    active_step = self.current_plan[0]

            step_target = active_step.target_pos
            target_facing = active_step.approach_facing
            action_type = active_step.action_type
            carried_footprint: list[tuple[int, int]] = (
                [
                    (0, 0),
                    (
                        int(round(active_step.carried_offset[0])),
                        int(round(active_step.carried_offset[1])),
                    ),
                ]
                if self.holding_item and (active_step.carried_offset != (0, 0))
                else [(0, 0)]
            )

        # Check if arrived at stand position
        dist_to_stand = math.hypot(step_target[0] - curr_r, step_target[1] - curr_c)
        is_at_stand = dist_to_stand <= max(1.5, self.step_size * 0.75)

        if is_at_stand:
            # Action: PICKUP
            if action_type == "PICKUP":
                # Orientation alignment: ensure avatar faces target_facing before pickup
                if target_facing and self.current_facing != target_facing:
                    for a in available_actions:
                        m = self.action_models.get(a)
                        if m and m.confidence >= 0.8:
                            if (
                                target_facing[0] != 0
                                and np.sign(m.delta_r) == np.sign(target_facing[0])
                            ) or (
                                target_facing[1] != 0
                                and np.sign(m.delta_c) == np.sign(target_facing[1])
                            ):
                                self.current_facing = target_facing
                                return a, 0.95

                item_adj = (
                    step_target[0] + (target_facing[0] if target_facing else 0) * self.step_size,
                    step_target[1] + (target_facing[1] if target_facing else 0) * self.step_size,
                )
                item_exists = any(
                    e.role in (EntityRole.MANIPULABLE, EntityRole.MOVABLE_ITEM, EntityRole.ACTUATOR)
                    and math.hypot(e.grid_pos[0] - item_adj[0], e.grid_pos[1] - item_adj[1])
                    < self.step_size * 0.75
                    and not e.is_delivered
                    for e in eg.entities.values()
                )
                if not item_exists:
                    self.current_plan.clear()
                    safe_act = 1 if curr_r >= 20 else 2
                    return safe_act, 0.90

                if 5 in available_actions:
                    self.holding_item = True
                    if target_facing:
                        self.carried_offset = (
                            float(target_facing[0] * self.step_size),
                            float(target_facing[1] * self.step_size),
                        )
                    self.current_plan.pop(0)
                    return 5, 0.99

            # Action: DROP
            elif action_type == "DROP":
                if 5 in available_actions:
                    self.holding_item = False
                    if target_facing:
                        g_r = step_target[0] + target_facing[0] * self.step_size
                        g_c = step_target[1] + target_facing[1] * self.step_size
                        self.delivered_positions.add((g_r, g_c))
                    else:
                        self.delivered_positions.add(step_target)
                    self.current_plan.clear()
                    return 5, 0.99

            # Action: MOVE (Reached destination)
            elif action_type == "MOVE":
                if self.current_plan:
                    self.current_plan.pop(0)
                safe_act = 1 if curr_r >= 20 else 2
                return safe_act, 0.90

        # ── 7. Pathfinding: Navigate Towards Stand Position ──
        # Obstacles include barriers, delivered items, and other uncarried boxes
        carried_pos = (
            (
                int(round(curr_r + self.carried_offset[0])),
                int(round(curr_c + self.carried_offset[1])),
            )
            if self.holding_item
            else None
        )

        item_coords = set()
        for e in eg.entities.values():
            if e.role in (EntityRole.MANIPULABLE, EntityRole.MOVABLE_ITEM) and not e.is_delivered:
                if (
                    carried_pos
                    and math.hypot(e.grid_pos[0] - carried_pos[0], e.grid_pos[1] - carried_pos[1])
                    < self.step_size * 0.75
                ):
                    continue
                item_coords.add(e.grid_pos)
        item_coords.discard(step_target)
        item_coords.discard(step_target)

        effective_barriers = set(eg.barriers)
        if self.target_zone_bounds:
            tz_min_r, tz_max_r, tz_min_c, tz_max_c = self.target_zone_bounds
            for tz_r in range(tz_min_r, tz_max_r + 1):
                for tz_c in range(tz_min_c, tz_max_c + 1):
                    effective_barriers.add((tz_r, tz_c))
        effective_barriers.discard(step_target)
        if self.holding_item:
            goal_payload = (
                int(round(step_target[0] + self.carried_offset[0])),
                int(round(step_target[1] + self.carried_offset[1])),
            )
            effective_barriers.discard(goal_payload)

        for dp in self.delivered_positions:
            if dp != step_target:
                effective_barriers.add(dp)
        effective_barriers.update(item_coords)
        for e in eg.entities.values():
            if e != eg.agent and e.role not in (
                EntityRole.RECEPTACLE,
                EntityRole.GOAL,
                EntityRole.EXIT,
                EntityRole.MANIPULABLE,
                EntityRole.MOVABLE_ITEM,
            ):
                if (
                    math.hypot(e.grid_pos[0] - curr_r, e.grid_pos[1] - curr_c)
                    < self.step_size * 0.75
                ):
                    continue
                if (
                    carried_pos
                    and math.hypot(e.grid_pos[0] - carried_pos[0], e.grid_pos[1] - carried_pos[1])
                    < self.step_size * 0.75
                ):
                    continue
                if e.grid_pos != step_target:
                    effective_barriers.add(e.grid_pos)

        effective_barriers.discard((curr_r, curr_c))
        if carried_pos:
            effective_barriers.discard(carried_pos)

        shortest_path = PhysicsPredictor.compute_geodesic_path(
            start=(curr_r, curr_c),
            goal=step_target,
            barrier_cells=effective_barriers,
            grid_shape=(H, W),
            step_size=self.step_size,
            footprint_offsets=carried_footprint,
        )

        if not shortest_path:
            shortest_path = PhysicsPredictor.compute_geodesic_path(
                start=(curr_r, curr_c),
                goal=step_target,
                barrier_cells=eg.barriers,
                grid_shape=(H, W),
                step_size=self.step_size,
                footprint_offsets=carried_footprint,
            )

        if shortest_path and len(shortest_path) > 1 and shortest_path[-1] == step_target:
            next_pt = shortest_path[1]
            dr_des = next_pt[0] - curr_r
            dc_des = next_pt[1] - curr_c
            for a in available_actions:
                if a in self.blocked_actions:
                    continue
                m = self.action_models.get(a)
                if m and m.confidence >= 0.8:
                    if (dr_des != 0 and np.sign(m.delta_r) == np.sign(dr_des)) or (
                        dc_des != 0 and np.sign(m.delta_c) == np.sign(dc_des)
                    ):
                        self.last_action = a
                        return a, 0.95

        dr_des = step_target[0] - curr_r
        dc_des = step_target[1] - curr_c

        # ── 8. Action Scoring with Collision & Oscillation Avoidance ──
        best_action = available_actions[0]
        best_score = -999999.0

        for a in available_actions:
            m = self.action_models.get(a)
            if not m or (m.delta_r == 0 and m.delta_c == 0):
                continue

            alignment = (m.delta_r * dr_des) + (m.delta_c * dc_des)
            penalty = 10000.0 if a in self.blocked_actions else 0.0

            dest_r = curr_r + m.delta_r
            dest_c = curr_c + m.delta_c
            dest_ir, dest_ic = int(round(dest_r)), int(round(dest_c))

            # Barrier collision check for avatar and carried footprint
            dest_in_barrier = (dest_ir, dest_ic) in effective_barriers or not (
                0 <= dest_ir < H and 0 <= dest_ic < W
            )
            if self.holding_item and self.carried_offset != (0.0, 0.0):
                c_ir = int(round(dest_r + self.carried_offset[0]))
                c_ic = int(round(dest_c + self.carried_offset[1]))
                if not (0 <= c_ir < H and 0 <= c_ic < W) or (c_ir, c_ic) in effective_barriers:
                    dest_in_barrier = True

            barrier_penalty = 20000.0 if dest_in_barrier else 0.0

            # Oscillation penalty
            recents = self.visited_positions[-8:]
            loop_penalty = sum(
                35.0
                for vr, vc in recents
                if math.hypot(dest_r - vr, dest_c - vc) < self.step_size * 0.9
            )

            score = float(alignment - penalty - barrier_penalty - loop_penalty)
            if score > best_score:
                best_score = score
                best_action = a

        # Detour fallback if all actions are blocked
        if best_score < -5000.0:
            self.blocked_actions.clear()
            best_detour_score = -999999.0
            for a in available_actions:
                m = self.action_models.get(a)
                if m and (m.delta_r != 0 or m.delta_c != 0):
                    dest_r = curr_r + m.delta_r
                    dest_c = curr_c + m.delta_c
                    dest_ir, dest_ic = int(round(dest_r)), int(round(dest_c))
                    is_b = (dest_ir, dest_ic) in effective_barriers or not (
                        0 <= dest_ir < H and 0 <= dest_ic < W
                    )
                    b_pen = 20000.0 if is_b else 0.0
                    align = (m.delta_r * dr_des) + (m.delta_c * dc_des)
                    recents = self.visited_positions[-8:]
                    l_pen = sum(
                        35.0
                        for vr, vc in recents
                        if math.hypot(dest_r - vr, dest_c - vc) < self.step_size * 0.9
                    )
                    d_score = float(align - b_pen - l_pen)
                    if d_score > best_detour_score:
                        best_detour_score = d_score
                        best_action = a
            best_score = best_detour_score

        self.last_action = best_action
        confidence = 0.95 if best_score > 0 else 0.60
        return best_action, confidence

    def update_causal_dynamics(
        self,
        action_id: int,
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
    ) -> None:
        """Infer avatar displacement, calibrate motor dynamics, and mark barrier collisions."""
        if prev_grid.shape != curr_grid.shape:
            return

        H, W = curr_grid.shape

        # 1. Action 5 (Interact/Pickup/Drop) has zero displacement
        if action_id == 5:
            if action_id not in self.action_models:
                self.action_models[action_id] = ActionDynamicsModel(
                    action_id=action_id, delta_r=0, delta_c=0, confidence=0.99, probes_tested=1
                )
            return

        # 2. Case: Avatar color already known
        if self.avatar_color is not None:
            p_prev = np.where(prev_grid == self.avatar_color)
            p_curr = np.where(curr_grid == self.avatar_color)

            if len(p_prev[0]) > 0 and len(p_curr[0]) > 0:
                old_r = float(np.mean(p_prev[0]))
                old_c = float(np.mean(p_prev[1]))
                new_r = float(np.mean(p_curr[0]))
                new_c = float(np.mean(p_curr[1]))
                dr = int(round(new_r - old_r))
                dc = int(round(new_c - old_c))

                if dr != 0 or dc != 0:
                    if max(abs(dr), abs(dc)) <= 8:
                        self.step_size = max(self.step_size, abs(dr), abs(dc))
                        self.spatial_planner.step_size = self.step_size
                        self.current_facing = (int(np.sign(dr)), int(np.sign(dc)))

                        norm_dr = int(np.sign(dr)) * self.step_size if dr != 0 else 0
                        norm_dc = int(np.sign(dc)) * self.step_size if dc != 0 else 0

                        # Calibrate motor model
                        if action_id not in self.action_models:
                            self.action_models[action_id] = ActionDynamicsModel(
                                action_id=action_id,
                                delta_r=norm_dr,
                                delta_c=norm_dc,
                                confidence=0.95,
                                probes_tested=1,
                            )
                        else:
                            m = self.action_models[action_id]
                            m.delta_r = norm_dr
                            m.delta_c = norm_dc
                            m.confidence = min(0.99, m.confidence + 0.1)
                            m.probes_tested += 1

                    self.visited_positions.append(
                        (self._snap_coord(new_r), self._snap_coord(new_c))
                    )
                    if len(self.visited_positions) > 30:
                        self.visited_positions.pop(0)

                    self.blocked_actions.clear()
                    self.stuck_counter = 0
                    return
                else:
                    # Blocked by obstacle or wall
                    if action_id in self.action_models:
                        m = self.action_models[action_id]
                        m.probes_tested += 1
                        if m.delta_r != 0 or m.delta_c != 0:
                            dest_r = int(round(old_r + m.delta_r))
                            dest_c = int(round(old_c + m.delta_c))
                            if self.known_barriers is None:
                                self.known_barriers = np.zeros(curr_grid.shape, dtype=bool)
                            half_w = max(0, (self.step_size - 1) // 2)
                            for b_dr in range(-half_w, half_w + 1):
                                for b_dc in range(-half_w, half_w + 1):
                                    br, bc = dest_r + b_dr, dest_c + b_dc
                                    if 0 <= br < H and 0 <= bc < W:
                                        self.known_barriers[br, bc] = True

                    self.blocked_actions.add(action_id)
                    self.stuck_counter += 1
                    return

        # 3. Case: Discover Avatar by finding rigid moving pixel cluster
        for col in np.unique(prev_grid):
            if col == 0:
                continue
            p_prev = np.where(prev_grid == col)
            p_curr = np.where(curr_grid == col)
            if (
                0 < len(p_prev[0]) <= 25
                and 0 < len(p_curr[0]) <= 25
                and len(p_prev[0]) == len(p_curr[0])
            ):
                dr = int(round(np.mean(p_curr[0]) - np.mean(p_prev[0])))
                dc = int(round(np.mean(p_curr[1]) - np.mean(p_prev[1])))
                if 0 < max(abs(dr), abs(dc)) <= 8:
                    self.avatar_color = int(col)
                    self.step_size = max(1, abs(dr), abs(dc))
                    self.spatial_planner.step_size = self.step_size
                    norm_dr = int(np.sign(dr)) * self.step_size if dr != 0 else 0
                    norm_dc = int(np.sign(dc)) * self.step_size if dc != 0 else 0
                    self.action_models[action_id] = ActionDynamicsModel(
                        action_id=action_id,
                        delta_r=norm_dr,
                        delta_c=norm_dc,
                        confidence=0.95,
                        probes_tested=1,
                    )
                    break
        else:
            # Nothing moved: probe was blocked
            self.probe_step_counter += 1
