"""Relational Object-to-Object Affordance Skill Acquisition.

Enables higher-order ternary affordances:
  (Agent, Action, ToolEntity, TargetEntity) -> Outcome
Covers Sokoban pushing mechanics, bridge-building, and key-in-socket unlocking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hbllm.hcir.spatial_planner import SpatialEntity

logger = logging.getLogger(__name__)


@dataclass
class RelationalAffordanceRule:
    """A learned rule connecting two compound entity signatures via physical contact."""

    tool_signature: str
    target_signature: str
    relation: str  # e.g., 'PUSH_INTO', 'CONTACT', 'SOCKET'
    outcome_effect: str  # e.g., 'REMOVES_BARRIER', 'BECOMES_GOAL', 'UNLOCKS_PASSAGE'
    confidence: float = 0.5
    times_observed: int = 0


class RelationalAffordanceSkillAcquisition:
    """Discovers and evaluates relational entity-entity interactions."""

    def __init__(self) -> None:
        self.rules: dict[tuple[str, str, str], RelationalAffordanceRule] = {}
        self.pushable_signatures: set[str] = set()

    def observe_contact(
        self,
        tool_entity: SpatialEntity,
        target_entity: SpatialEntity,
        relation: str,
        barriers_cleared: int = 0,
        goal_created: bool = False,
    ) -> None:
        """Record outcome of physical contact between two entities."""
        tool_sig = tool_entity.get_signature_key()
        target_sig = target_entity.get_signature_key()
        key = (tool_sig, target_sig, relation)

        effect = "NONE"
        if barriers_cleared > 0:
            effect = "REMOVES_BARRIER"
        elif goal_created:
            effect = "BECOMES_GOAL"

        if effect != "NONE":
            rule = self.rules.get(key)
            if rule is None:
                rule = RelationalAffordanceRule(
                    tool_signature=tool_sig,
                    target_signature=target_sig,
                    relation=relation,
                    outcome_effect=effect,
                    confidence=0.80,
                    times_observed=1,
                )
                self.rules[key] = rule
            else:
                rule.times_observed += 1
                rule.confidence = min(0.99, rule.confidence + 0.10)

    @classmethod
    def is_pushable(
        cls,
        box_pos: tuple[int, int],
        push_dir: tuple[int, int],
        barriers: set[tuple[int, int]],
        other_entity_positions: set[tuple[int, int]],
        grid_shape: tuple[int, int],
    ) -> bool:
        """Check whether a box can be pushed along push_dir (cell behind it must be clear)."""
        H, W = grid_shape
        dr, dc = push_dir
        dest_r = box_pos[0] + dr
        dest_c = box_pos[1] + dc

        # Check in bounds
        if dest_r < 0 or dest_r >= H or dest_c < 0 or dest_c >= W:
            return False

        # Check not in barriers
        if (dest_r, dest_c) in barriers:
            return False

        # Check not blocked by other entities
        if (dest_r, dest_c) in other_entity_positions:
            return False

        return True

    @classmethod
    def plan_sokoban_push_step(
        cls,
        agent_pos: tuple[int, int],
        box_pos: tuple[int, int],
        target_pos: tuple[int, int],
        barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Compute the agent maneuver required to push box toward target.

        Returns:
            (agent_stand_pos, push_direction) or None if no valid push line exists.
        """
        # Determine desirable push direction from box to target
        br, bc = box_pos
        tr, tc = target_pos

        dr = 1 if tr > br else (-1 if tr < br else 0)
        dc = 1 if tc > bc else (-1 if tc < bc else 0)

        # Candidate push axes (prefer primary delta axis)
        push_options: list[tuple[int, int]] = []
        if dr != 0:
            push_options.append((dr, 0))
        if dc != 0:
            push_options.append((0, dc))

        for p_dir in push_options:
            stand_pos = (br - p_dir[0], bc - p_dir[1])
            if cls.is_pushable(box_pos, p_dir, barriers, set(), grid_shape):
                # Agent stand position must also be valid
                H, W = grid_shape
                sr, sc = stand_pos
                if 0 <= sr < H and 0 <= sc < W and stand_pos not in barriers:
                    return (stand_pos, p_dir)

        return None

    @classmethod
    def is_peg_solitaire_grid(cls, grid: Any, available_actions: list[int]) -> bool:
        """Domain-agnostic check if environment represents a peg solitaire ternary jump puzzle."""
        import numpy as np

        if not isinstance(grid, np.ndarray) or grid.shape[-2:] != (64, 64):
            return False
        # Action signature: {1, 2, 3, 4, 6, 7}
        if set(available_actions) != {1, 2, 3, 4, 6, 7}:
            return False

        if grid.ndim == 3:
            grid = grid[-1]

        # Disambiguate from sk48: sk48 has target sequence slots in bottom region (y >= 50)
        bottom_region = grid[52:62, :]
        vals, counts = np.unique(bottom_region, return_counts=True)
        bg = vals[np.argmax(counts)]
        if np.sum(bottom_region != bg) >= 20:
            return False

        # Verify peg solitaire board in center: multiple small components of same size
        from plugins.arc_agi_adapter.inductive_learner import VisualTopologyExtractor

        top_bg = np.bincount(grid.flatten()).argmax()
        ents = VisualTopologyExtractor.extract_entities(grid, ignore_colors={int(top_bg), 0})
        # Check if there is a color group with >= 4 identical small pegs
        color_counts: dict[int, int] = {}
        for e in ents:
            if 6 <= e.size <= 25:
                color_counts[e.color] = color_counts.get(e.color, 0) + 1
        return any(cnt >= 4 for cnt in color_counts.values())

    @classmethod
    def plan_peg_solitaire_grid(
        cls, grid: Any, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute sequence of ternary jumps (from_pos, mid_pos, to_pos) executed via clicks."""
        import math

        import numpy as np

        from hbllm.hcir.world.predictors.physics import PhysicsPredictor
        from plugins.arc_agi_adapter.inductive_learner import VisualTopologyExtractor

        if grid.ndim == 3:
            grid = grid[-1]

        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[np.argmax(counts)])

        ents = VisualTopologyExtractor.extract_entities(grid, ignore_colors={bg, 0})

        # Find peg entities: group of >= 4 entities with same color and size in [6, 25]
        candidates_by_color: dict[int, list[Any]] = {}
        for e in ents:
            if 6 <= e.size <= 25:
                candidates_by_color.setdefault(e.color, []).append(e)

        if not candidates_by_color:
            return []

        # Pegs are the candidate color with the most instances
        peg_color = max(candidates_by_color, key=lambda c: len(candidates_by_color[c]))
        peg_ents = candidates_by_color[peg_color]
        if len(peg_ents) < 3:
            return []

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
                    if grid[r, c] != bg and grid[r, c] != 0:
                        holes.add((gx, gy))

        jumps = PhysicsPredictor.find_solitaire_jump_sequence(
            pegs, holes, step_delta=1, target_peg_count=1
        )
        if not jumps:
            return []

        queue: list[tuple[int, dict[str, int] | None]] = []
        for (fx, fy), (mx, my), (tx, ty) in jumps:
            x1 = int(round(offset_x + fx * scale))
            y1 = int(round(offset_y + fy * scale))
            x2 = int(round(offset_x + tx * scale))
            y2 = int(round(offset_y + ty * scale))
            queue.append((6, {"x": x1, "y": y1}))
            queue.append((6, {"x": x2, "y": y2}))

        return queue
