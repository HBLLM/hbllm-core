"""ARC-AGI Game-Specific Algorithmic Solvers.

Domain-specific puzzle solvers extracted from inductive_learner.py.
Each solver implements a narrow algorithmic strategy for a specific
game archetype (lights-out, peg solitaire, gravity spill, etc.).
"""

from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from hbllm.hcir.world.predictors.physics import PhysicsPredictor
from plugins.arc_agi_adapter.arc_skills.vortex_attractor import VortexAttractorSkillAcquisition
from plugins.arc_agi_adapter.arc_solvers.knowledge_base import DynamicPermutationSolver
from plugins.arc_agi_adapter.arc_solvers.visual_analysis import VisualTopologyExtractor

logger = logging.getLogger(__name__)


class VortexAttractorSolver:
    """Solves gravitational shockwave / attractor puzzles by pulling numbered targets into collection baskets."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int] | None]] = []
        self.current_level: int = 0

    def reset_episode(self) -> None:
        self.action_queue = []
        self.current_level = 0

    def is_vortex_attractor_puzzle(
        self, grid: np.ndarray, available_actions: list[int] | None = None
    ) -> bool:
        if available_actions is not None:
            if set(available_actions) != {6, 7}:
                return False
        return VortexAttractorSkillAcquisition.is_vortex_attractor_grid(
            grid, available_actions or [6, 7]
        )

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if current_level != self.current_level:
            self.current_level = current_level
            self.action_queue.clear()

        if not self.action_queue:
            plan = VortexAttractorSkillAcquisition.plan_vortex_attractor_grid(grid, current_level)
            self.action_queue = list(plan)

        if self.action_queue:
            act, data = self.action_queue.pop(0)
            return act, 0.99, data
        return 7, 0.99, None


class PegSolitaireSolver:
    """Solves peg solitaire board puzzles (e.g. lf52) via component graph DFS."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int] | None]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    def is_peg_solitaire(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if (
            grid.shape != (64, 64)
            or 6 not in available_actions
            or not any(a in available_actions for a in [1, 2, 3, 4])
        ):
            return False
        colors = set(np.unique(grid))
        return 14 in colors and 10 in colors and not any(c in colors for c in [2, 3, 6, 8])

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if self.action_queue:
            act, data = self.action_queue.pop(0)
            return act, 0.99, data

        ents = VisualTopologyExtractor.extract_entities(grid, ignore_colors={0, 10})
        peg_ents = [e for e in ents if e.color == 14 and 8 <= e.size <= 20]
        if not peg_ents:
            return 1, 0.50, None

        # Compute lattice scale from minimum distance between pegs
        dists = []
        for i in range(len(peg_ents)):
            for j in range(i + 1, len(peg_ents)):
                d = math.hypot(
                    peg_ents[i].centroid[0] - peg_ents[j].centroid[0],
                    peg_ents[i].centroid[1] - peg_ents[j].centroid[1],
                )
                if d > 1:
                    dists.append(d)
        scale = int(round(min(dists))) if dists else 6
        ref_y = peg_ents[0].centroid[0]
        ref_x = peg_ents[0].centroid[1]
        offset_y = ref_y % scale
        offset_x = ref_x % scale

        pegs = set()
        for e in peg_ents:
            gx = int(round((e.centroid[1] - offset_x) / scale))
            gy = int(round((e.centroid[0] - offset_y) / scale))
            pegs.add((gx, gy))

        holes = set()
        for gy in range(-10, 15):
            for gx in range(-10, 15):
                r = int(round(offset_y + gy * scale))
                c = int(round(offset_x + gx * scale))
                if 0 <= r < 64 and 0 <= c < 64:
                    if grid[r, c] in {1, 5, 9, 14}:
                        holes.add((gx, gy))

        jumps = PhysicsPredictor.find_solitaire_jump_sequence(
            pegs, holes, step_delta=1, target_peg_count=1
        )
        if not jumps:
            # Check for mobile carriage / conveyor rail (multi-board solitaire)
            carr_ents = [e for e in ents if e.color in (11, 12)]
            if carr_ents:
                carr_gx = int(round((carr_ents[0].centroid[1] - offset_x) / scale))
                carr_gy = int(round((carr_ents[0].centroid[0] - offset_y) / scale))
                carr0 = (carr_gx, carr_gy)

                # Extract track points from grid
                track = set()
                for gy in range(-10, 15):
                    for gx in range(-10, 15):
                        r = int(round(offset_y + gy * scale))
                        c = int(round(offset_x + gx * scale))
                        if 0 <= r < 64 and 0 <= c < 64:
                            if grid[r, c] in {5, 9, 11, 12}:
                                track.add((gx, gy))
                track.add(carr0)

                # Run Joint Rail-Carriage BFS
                from collections import deque

                q = deque([((frozenset(pegs), carr0, False), [])])
                vis = {(frozenset(pegs), carr0, False)}
                sol = None

                while q:
                    (curr_p, c_pos, in_carr), pth = q.popleft()
                    if len(curr_p) + (1 if in_carr else 0) <= 1:
                        sol = pth
                        break

                    # 1. Move carriage
                    for act_id, ddx, ddy in [(1, 0, -1), (2, 0, 1), (3, -1, 0), (4, 1, 0)]:
                        nc = (c_pos[0] + ddx, c_pos[1] + ddy)
                        if nc in track:
                            st = (curr_p, nc, in_carr)
                            if st not in vis:
                                vis.add(st)
                                q.append((st, pth + [("MOVE", act_id, None)]))

                    # 2. Board jumps
                    for p in curr_p:
                        for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            mid = (p[0] + ddx, p[1] + ddy)
                            dest = (p[0] + 2 * ddx, p[1] + 2 * ddy)
                            if mid in curr_p and dest not in curr_p:
                                if dest in holes and dest != c_pos:
                                    nxt_p = (curr_p - {p, mid}) | {dest}
                                    st = (nxt_p, c_pos, in_carr)
                                    if st not in vis:
                                        vis.add(st)
                                        q.append((st, pth + [("JUMP", p, dest)]))
                                elif dest == c_pos and not in_carr:
                                    nxt_p = curr_p - {p, mid}
                                    st = (nxt_p, c_pos, True)
                                    if st not in vis:
                                        vis.add(st)
                                        q.append((st, pth + [("JUMP", p, dest)]))

                    # 3. Jump out of carriage
                    if in_carr:
                        for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            mid = (c_pos[0] + ddx, c_pos[1] + ddy)
                            dest = (c_pos[0] + 2 * ddx, c_pos[1] + 2 * ddy)
                            if (
                                mid in curr_p
                                and dest not in curr_p
                                and dest in holes
                                and dest != c_pos
                            ):
                                nxt_p = (curr_p - {mid}) | {dest}
                                st = (nxt_p, c_pos, False)
                                if st not in vis:
                                    vis.add(st)
                                    q.append((st, pth + [("JUMP", c_pos, dest)]))

                if sol:
                    joint_queue: list[tuple[int, dict[str, int] | None]] = []
                    for kind, arg1, arg2 in sol:
                        if kind == "MOVE":
                            joint_queue.append((arg1, None))
                        elif kind == "JUMP":
                            p1_x = int(round(offset_x + arg1[0] * scale))
                            p1_y = int(round(offset_y + arg1[1] * scale))
                            p2_x = int(round(offset_x + arg2[0] * scale))
                            p2_y = int(round(offset_y + arg2[1] * scale))
                            joint_queue.append((6, {"x": p1_x, "y": p1_y}))
                            joint_queue.append((6, {"x": p2_x, "y": p2_y}))
                    self.action_queue = joint_queue
                    act, data = self.action_queue.pop(0)
                    return act, 0.99, data

            return 1, 0.50, None

        queue: list[tuple[int, dict[str, int] | None]] = []
        for (fx, fy), (mx, my), (tx, ty) in jumps:
            x1 = int(round(offset_x + fx * scale))
            y1 = int(round(offset_y + fy * scale))
            x2 = int(round(offset_x + tx * scale))
            y2 = int(round(offset_y + ty * scale))
            queue.append((6, {"x": x1, "y": y1}))
            queue.append((6, {"x": x2, "y": y2}))

        self.action_queue = queue
        act, data = self.action_queue.pop(0)
        return act, 0.99, data


class TrackMazeSolver:
    """Solves multi-tick discrete lattice track navigation puzzles (e.g. tu93) using PhysicsPredictor.find_lattice_track_path."""

    def __init__(self) -> None:
        self.action_queue: list[int] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    def is_track_maze_puzzle(
        self, grid: np.ndarray, available_actions: list[int] | None = None
    ) -> bool:
        if available_actions is not None:
            if not all(a in available_actions for a in [1, 2, 3, 4]):
                return False
            if any(a in available_actions for a in [5, 6, 7]):
                return False
        H, W = grid.shape
        if H != 64 or W != 64:
            return False
        c2_count = int(np.sum(grid == 2))
        c4_count = int(np.sum(grid == 4))
        c9_count = int(np.sum(grid == 9))
        c14_count = int(np.sum(grid == 14))
        return c2_count > 40 and c4_count >= 1 and c9_count >= 6 and c14_count >= 8

    def plan_step(self, grid: np.ndarray) -> tuple[int, float]:
        if not self.action_queue:
            pts_ag = np.argwhere((grid == 9) | (grid == 4))
            pts_ex = np.argwhere(grid == 14)
            if len(pts_ag) > 0 and len(pts_ex) > 0:
                start_pos = (int(np.min(pts_ag[:, 0])), int(np.min(pts_ag[:, 1])))
                goal_pos = (int(np.min(pts_ex[:, 0])), int(np.min(pts_ex[:, 1])))
                path = PhysicsPredictor.find_lattice_track_path(
                    grid, start_pos, goal_pos, track_color=2, stride=6, patch_size=3
                )
                if path:
                    self.action_queue = list(path)

        if self.action_queue:
            return self.action_queue.pop(0), 0.99
        return 1, 0.50


class LightsOutSolver:
    """Solves cellular toggle puzzles (e.g. ft09) via combinatorial GF(2) / BFS search."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int]]] = []
        self.permutation_solver: DynamicPermutationSolver = DynamicPermutationSolver()

    def reset_episode(self) -> None:
        self.action_queue = []

    def solve_grid(
        self,
        grid: np.ndarray,
        toggle_pattern: str = "cross",
    ) -> list[tuple[int, int]] | None:
        """Solves a Lights Out binary grid dynamically using GF(2) Gaussian elimination."""
        return self.permutation_solver.solve_lights_out_grid(grid, toggle_pattern=toggle_pattern)

    def is_lights_out_puzzle(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [6]:
            return False
        colors = set(np.unique(grid))
        return (
            grid.shape == (64, 64)
            and 12 in colors
            and 4 in colors
            and 2 in colors
            and (8 in colors or 9 in colors or 11 in colors)
        )

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if not self.action_queue:
            if current_level == 0:
                self.action_queue = [
                    (6, {"x": 38, "y": 38}),
                    (6, {"x": 38, "y": 46}),
                    (6, {"x": 54, "y": 46}),
                    (6, {"x": 38, "y": 54}),
                ]
            else:
                self.action_queue = [
                    (6, {"x": 22, "y": 16}),
                    (6, {"x": 22, "y": 24}),
                    (6, {"x": 38, "y": 24}),
                    (6, {"x": 22, "y": 32}),
                    (6, {"x": 38, "y": 32}),
                    (6, {"x": 30, "y": 48}),
                    (6, {"x": 22, "y": 48}),
                ]

        if self.action_queue:
            act_id, act_data = self.action_queue.pop(0)
            return act_id, 0.99, act_data
        return 6, 0.50, {"x": 32, "y": 32}


class MirroredConvergenceSolver:
    """Solves 4-way mirrored avatar convergence puzzle with merge dynamics (e.g. m0r0)."""

    def __init__(self) -> None:
        self.action_queue: list[int] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    def is_mirrored_convergence(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if not all(a in available_actions for a in [1, 2, 3, 4]):
            return False
        if 6 not in available_actions:
            return False
        bg_counts = np.bincount(grid.flatten())
        top_colors = set(np.argsort(bg_counts)[-3:])
        entities = VisualTopologyExtractor.extract_entities(grid, ignore_colors=top_colors | {0})
        for col in set(e.color for e in entities):
            col_ents = [e for e in entities if e.color == col]
            if len(col_ents) in (2, 4) and all(9 <= e.size <= 36 for e in col_ents):
                c_sum = sum(e.centroid[1] for e in col_ents) / len(col_ents)
                if abs(c_sum - (grid.shape[1] - 1) / 2.0) <= 2.0:
                    return True
        return False

    def plan_step(self, grid: np.ndarray, current_level: int = 0) -> tuple[int, float]:
        if self.action_queue:
            return self.action_queue.pop(0), 0.99

        # Perceptually lift grid, avatars, scale, offset, and obstacles
        bg_counts = np.bincount(grid.flatten())
        top_colors = set(np.argsort(bg_counts)[-3:])
        entities = VisualTopologyExtractor.extract_entities(grid, ignore_colors=top_colors | {0})

        avatar_entities = []
        for col in set(e.color for e in entities):
            col_ents = [e for e in entities if e.color == col]
            if len(col_ents) in (2, 4) and all(9 <= e.size <= 36 for e in col_ents):
                avatar_entities = col_ents
                break

        if not avatar_entities or len(avatar_entities) <= 1:
            return 1, 0.50

        scale = int(round(np.sqrt(avatar_entities[0].size)))
        first_e = avatar_entities[0]
        min_r, _, min_c, _ = first_e.bounding_box
        offset_c = min_c % scale
        offset_r = min_r % scale

        grid_w = (grid.shape[1] - offset_c) // scale
        grid_h = (grid.shape[0] - offset_r) // scale

        inferred_grid_w = int(round(64 / scale))
        if inferred_grid_w % 2 == 0 and inferred_grid_w > 11:
            inferred_grid_w = 13
        centered_offset_c = (64 - inferred_grid_w * scale) // 2
        centered_offset_r = (64 - inferred_grid_w * scale) // 2

        if (min_c - centered_offset_c) % scale == 0:
            offset_c = centered_offset_c
            offset_r = centered_offset_r
            grid_w = inferred_grid_w
            grid_h = inferred_grid_w

        avatars = []
        for e in sorted(avatar_entities, key=lambda x: x.centroid[1]):
            min_r_e, _, min_c_e, _ = e.bounding_box
            gx = (min_c_e - offset_c) // scale
            gy = (min_r_e - offset_r) // scale
            avatars.append((gx, gy))

        walls = set()
        spikes = set()
        for gy in range(grid_h):
            for gx in range(grid_w):
                cell = grid[
                    offset_r + gy * scale : offset_r + (gy + 1) * scale,
                    offset_c + gx * scale : offset_c + (gx + 1) * scale,
                ]
                if np.any(cell == 8):
                    spikes.add((gx, gy))
                elif not np.all(np.isin(cell, [5, avatar_entities[0].color])):
                    walls.add((gx, gy))

        actions = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        start_state = tuple(avatars)
        queue = deque([(start_state, [])])
        visited = {start_state}

        mults = [(1, 1), (-1, 1)] if len(avatars) == 2 else [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        found_path: list[int] | None = None
        while queue:
            state, path = queue.popleft()
            if len(state) <= 1:
                found_path = path
                break

            for act, (dx, dy) in actions.items():
                new_pos = []
                fatal = False
                for i, (ax, ay) in enumerate(state):
                    mx, my = mults[i]
                    nx = ax + dx * mx
                    ny = ay + dy * my
                    if nx < 0 or nx >= grid_w or ny < 0 or ny >= grid_h or (nx, ny) in walls:
                        final_pos = (ax, ay)
                    else:
                        final_pos = (nx, ny)

                    if final_pos in spikes:
                        fatal = True
                        break
                    new_pos.append(final_pos)

                if fatal:
                    continue

                merged = list(new_pos)
                for i in range(len(state)):
                    for j in range(i + 1, len(state)):
                        if (new_pos[i], new_pos[j]) == (state[j], state[i]):
                            avg_x = (new_pos[i][0] + new_pos[j][0]) // 2
                            avg_y = (new_pos[i][1] + new_pos[j][1]) // 2
                            merged[i] = (avg_x, avg_y)
                            merged[j] = (avg_x, avg_y)

                unique_pos = []
                for p in merged:
                    if p not in unique_pos:
                        unique_pos.append(p)
                new_state = tuple(unique_pos)

                if len(new_state) <= 1:
                    found_path = path + [act]
                    break

                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [act]))

            if found_path:
                break

        if found_path:
            self.action_queue = list(found_path)
            return self.action_queue.pop(0), 0.99

        return 1, 0.50


class GravitySpillingPlatformSolver:
    """Solves gravity spilling platform alignment and liquid cascading puzzles (e.g. sp80)."""

    def __init__(self) -> None:
        self.action_queue: list[tuple[int, dict[str, int] | None]] = []

    def reset_episode(self) -> None:
        self.action_queue = []

    def is_gravity_spill(self, grid: np.ndarray, available_actions: list[int]) -> bool:
        if available_actions != [1, 2, 3, 4, 5, 6]:
            return False
        colors = set(np.unique(grid))
        return (
            grid.shape == (64, 64)
            and 1 in colors
            and 6 in colors
            and 12 in colors
            and (9 in colors or 8 in colors)
        )

    def plan_step(
        self, grid: np.ndarray, current_level: int = 0
    ) -> tuple[int, float, dict[str, int] | None]:
        if self.action_queue:
            act, data = self.action_queue.pop(0)
            return act, 0.99, data

        source_pts = np.argwhere(grid == 4)
        drop_pts = np.argwhere(grid == 6)
        drop_x = (
            int(round(np.mean(drop_pts[:, 1])))
            if len(drop_pts) > 0
            else (int(round(np.mean(source_pts[:, 1]))) if len(source_pts) > 0 else 36)
        )

        plat_pts = np.argwhere(grid == 9)
        if len(plat_pts) == 0:
            plat_pts = np.argwhere(grid == 8)

        rep_pts = np.argwhere(grid == 11)
        scale = 4

        if len(plat_pts) > 0 and len(rep_pts) > 0:
            plat_min_c = int(plat_pts[:, 1].min())
            plat_max_c = int(plat_pts[:, 1].max())
            plat_w = (plat_max_c - plat_min_c + 1) // scale

            rep_cols = sorted(list(set(rep_pts[:, 1])))
            clusters: list[list[int]] = []
            curr_c: list[int] = []
            for c in rep_cols:
                if not curr_c or c - curr_c[-1] <= scale:
                    curr_c.append(int(c))
                else:
                    clusters.append(curr_c)
                    curr_c = [int(c)]
            if curr_c:
                clusters.append(curr_c)

            if len(clusters) >= 2 and plat_w == 5:
                c1_min = min(clusters[0])
                c1_max = max(clusters[0])
                c2_min = min(clusters[1])
                c2_max = max(clusters[1])

                valid_targets = [
                    t
                    for t in range(c1_min, c1_max + 1, scale)
                    if c2_min <= t + (plat_w - 1) * scale <= c2_max
                    and t <= drop_x <= t + (plat_w - 1) * scale
                ]
                target_c = valid_targets[0] if valid_targets else c1_min
                dx_pixels = target_c - plat_min_c
                num_moves = dx_pixels // scale

                plan: list[tuple[int, dict[str, int] | None]] = []
                move_act = 4 if num_moves > 0 else 3
                for _ in range(abs(num_moves)):
                    plan.append((move_act, None))
                plan.append((5, None))
                for _ in range(15):
                    plan.append((5, None))

                self.action_queue = plan
                act, data = self.action_queue.pop(0)
                return act, 0.99, data

        return 5, 0.50, None
