"""Coupled Controllable & Mirrored Multi-Agent Skill Acquisition.

Enables learning joint action mappings across multiple simultaneous avatars
and planning joint paths in product configuration spaces S1 x S2 with
obstacle-assisted desynchronization maneuvers.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CoupledControllableModel:
    """Empirical joint motor model for coupled multi-agent entities."""

    avatar_ids: list[str] = field(default_factory=list)
    # action_id -> {avatar_id: (dr, dc)}
    joint_displacements: dict[int, dict[str, tuple[int, int]]] = field(default_factory=dict)
    confidence: float = 0.5


class CoupledControllableSkillAcquisition:
    """Discovers coupled/mirrored avatars and executes joint configuration planning."""

    def __init__(self) -> None:
        self.model = CoupledControllableModel()

    def observe_joint_step(
        self,
        action: int,
        displacements: dict[str, tuple[int, int]],
    ) -> None:
        """Register empirical displacements of multiple avatars for an executed action."""
        if len(displacements) < 2:
            return

        if not self.model.avatar_ids:
            self.model.avatar_ids = sorted(list(displacements.keys()))

        action_map = self.model.joint_displacements.setdefault(action, {})
        for aid, delta in displacements.items():
            action_map[aid] = delta

        self.model.confidence = min(0.99, self.model.confidence + 0.15)

    def plan_joint_convergence(
        self,
        start1: tuple[int, int],
        start2: tuple[int, int],
        goal1: tuple[int, int],
        goal2: tuple[int, int],
        barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
        max_steps: int = 40,
    ) -> list[int] | None:
        """Plan joint action sequence to bring both avatars to their respective goals.

        Leverages barrier collisions to desynchronize agent phases when needed.
        """
        H, W = grid_shape
        start_state = (start1, start2)
        if start1 == goal1 and start2 == goal2:
            return []

        # Available actions from model, or default 4 cardinal actions if not yet grounded
        if self.model.joint_displacements:
            action_candidates = list(self.model.joint_displacements.keys())
        else:
            action_candidates = [1, 2, 3, 4]

        # Queue: (state, action_history)
        queue = collections.deque([(start_state, [])])
        visited: set[tuple[tuple[int, int], tuple[int, int]]] = {start_state}

        while queue:
            (pos1, pos2), action_hist = queue.popleft()

            if len(action_hist) >= max_steps:
                continue

            for act in action_candidates:
                # Determine displacement for avatar 1 and avatar 2
                if act in self.model.joint_displacements:
                    disp_map = self.model.joint_displacements[act]
                    d1 = (
                        disp_map.get(self.model.avatar_ids[0], (0, 0))
                        if self.model.avatar_ids
                        else (0, 0)
                    )
                    d2 = (
                        disp_map.get(self.model.avatar_ids[1], (0, 0))
                        if len(self.model.avatar_ids) > 1
                        else (0, 0)
                    )
                else:
                    # Default mirrored convergence assumption:
                    # Vertical is parallel (1=UP, 2=DOWN), Horizontal is mirrored (3=LEFT/RIGHT, 4=RIGHT/LEFT)
                    defaults = {
                        1: ((-1, 0), (-1, 0)),
                        2: ((1, 0), (1, 0)),
                        3: ((0, -1), (0, 1)),
                        4: ((0, 1), (0, -1)),
                    }
                    d1, d2 = defaults.get(act, ((0, 0), (0, 0)))

                # Simulate step for avatar 1
                n1_r, n1_c = pos1[0] + d1[0], pos1[1] + d1[1]
                if n1_r < 0 or n1_r >= H or n1_c < 0 or n1_c >= W or (n1_r, n1_c) in barriers:
                    next_pos1 = pos1  # Blocked by barrier or wall (desynchronization!)
                else:
                    next_pos1 = (n1_r, n1_c)

                # Simulate step for avatar 2
                n2_r, n2_c = pos2[0] + d2[0], pos2[1] + d2[1]
                if n2_r < 0 or n2_r >= H or n2_c < 0 or n2_c >= W or (n2_r, n2_c) in barriers:
                    next_pos2 = pos2  # Blocked by barrier or wall (desynchronization!)
                else:
                    next_pos2 = (n2_r, n2_c)

                # If neither moved, skip useless action
                if next_pos1 == pos1 and next_pos2 == pos2:
                    continue

                next_state = (next_pos1, next_pos2)

                # Check victory
                if next_pos1 == goal1 and next_pos2 == goal2:
                    return action_hist + [act]

                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, action_hist + [act]))

        return None

    @classmethod
    def is_mirrored_convergence_grid(cls, grid: Any, available_actions: list[int]) -> bool:
        """Domain-agnostic check if environment features symmetric mirrored avatars."""
        import numpy as np

        if not isinstance(grid, np.ndarray) or grid.shape != (64, 64):
            return False
        if not all(a in available_actions for a in [1, 2, 3, 4]):
            return False
        if 6 not in available_actions:
            return False
        bg_counts = np.bincount(grid.flatten())
        top_colors = set(np.argsort(bg_counts)[-3:])
        from plugins.arc_agi_adapter.inductive_learner import VisualTopologyExtractor

        entities = VisualTopologyExtractor.extract_entities(grid, ignore_colors=top_colors | {0})
        for col in set(e.color for e in entities):
            col_ents = [e for e in entities if e.color == col]
            if len(col_ents) in (2, 4) and all(9 <= e.size <= 36 for e in col_ents):
                c_sum = sum(e.centroid[1] for e in col_ents) / len(col_ents)
                if abs(c_sum - (grid.shape[1] - 1) / 2.0) <= 2.0:
                    return True
        return False

    @classmethod
    def plan_mirrored_convergence_grid(cls, grid: Any) -> list[int]:
        """Compute the joint convergence plan for mirrored multi-agent configuration."""
        import numpy as np

        bg_counts = np.bincount(grid.flatten())
        top_colors = set(np.argsort(bg_counts)[-3:])
        from plugins.arc_agi_adapter.inductive_learner import VisualTopologyExtractor

        entities = VisualTopologyExtractor.extract_entities(grid, ignore_colors=top_colors | {0})

        avatar_entities = []
        for col in set(e.color for e in entities):
            col_ents = [e for e in entities if e.color == col]
            if len(col_ents) in (2, 4) and all(9 <= e.size <= 36 for e in col_ents):
                avatar_entities = col_ents
                break

        if not avatar_entities or len(avatar_entities) <= 1:
            return []

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
        queue = collections.deque([(start_state, [])])
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

        return found_path or []
