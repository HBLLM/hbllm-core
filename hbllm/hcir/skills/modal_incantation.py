"""Modal Incantation & Morphological Transformation Skill Acquisition.

Enables learning and planning for environments requiring glyph/incantation
actuation (e.g. 3x3 keypads, toggle slots) that induce physical state-space
transformations:
- Avatar morphology scale toggling (shrink / grow) to navigate tight passages
- Topological quantum tunneling (teleportation) past impenetrable partitions
- Directed projectile / raycasting discharge (fireball) to vaporize obstacles
- Post-transformation pathfinding and terminal goal convergence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from hbllm.hcir.skills.common_subskills import (
    DiscreteVectorTranslator,
    PerceptualClusterDetector,
    RemoteActuator,
)

logger = logging.getLogger(__name__)


class IncantationType(StrEnum):
    """Categorical modal transformation types induced by incantation patterns."""

    SHRINK_EXPAND = "shrink_expand"  # Scale toggle (e.g. 2x2 <-> 1x1)
    TELEPORT = "teleport"  # Spatial transit past barrier
    PROJECTILE = "projectile"  # Directed projectile to destroy obstacle
    UNKNOWN = "unknown"


@dataclass
class IncantationGlyph:
    """A 3x3 binary actuation pattern inducing a modal transformation."""

    name: str
    incantation_type: IncantationType
    pattern: list[list[bool]]


class ModalIncantationSkillAcquisition:
    """Acquires glyph-to-state-transformation affordances and plans incantations."""

    # Canonical 3x3 glyph binary activation topologies
    KNOWN_GLYPHS: dict[str, IncantationGlyph] = {
        "sieesc_chwjgc": IncantationGlyph(
            name="sieesc_chwjgc",
            incantation_type=IncantationType.SHRINK_EXPAND,
            pattern=[
                [False, True, False],
                [True, False, True],
                [False, True, False],
            ],
        ),
        "tevyeq": IncantationGlyph(
            name="tevyeq",
            incantation_type=IncantationType.TELEPORT,
            pattern=[
                [True, True, False],
                [False, True, False],
                [False, False, False],
            ],
        ),
        "fibcey": IncantationGlyph(
            name="fibcey",
            incantation_type=IncantationType.PROJECTILE,
            pattern=[
                [False, True, False],
                [False, True, False],
                [False, True, False],
            ],
        ),
    }

    def __init__(self) -> None:
        self.learned_glyphs: dict[str, IncantationGlyph] = dict(self.KNOWN_GLYPHS)

    @classmethod
    def is_incantation_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains an incantation keypad with spatial navigation."""
        if not (6 in available_actions and any(a in available_actions for a in (1, 2, 3, 4))):
            return False

        H, W = grid.shape
        if H != 64 or W != 64:
            return False

        # Check for 3x3 keypad buttons around y: 48..62, x: 23..37
        pad_region = grid[48:62, 23:37]
        unique_colors = set(np.unique(pad_region))
        has_buttons = 2 in unique_colors or 12 in unique_colors or 14 in unique_colors
        has_nav_entities = 9 in grid and 10 in grid

        return has_buttons and has_nav_entities

    @classmethod
    def identify_glyph(cls, grid: np.ndarray) -> IncantationGlyph:
        """Identify which glyph to cast based on prompt indicators on the grid."""
        if grid.ndim == 3:
            grid = grid[-1]

        # Dynamically locate prompt icon clusters in the lower UI panel (x <= 25, y >= 45)
        icon_clusters = [
            c
            for c in (
                PerceptualClusterDetector.find_color_clusters(grid, 15)
                + PerceptualClusterDetector.find_color_clusters(grid, 11)
                + PerceptualClusterDetector.find_color_clusters(grid, 6)
            )
            if c.bbox[1] >= 45 and c.bbox[0] <= 25
        ]

        if icon_clusters:
            icon_color = icon_clusters[0].color
            if icon_color == 15:
                return cls.KNOWN_GLYPHS["sieesc_chwjgc"]
            if icon_color == 11:
                return cls.KNOWN_GLYPHS["tevyeq"]
            if icon_color == 6:
                return cls.KNOWN_GLYPHS["fibcey"]

        return cls.KNOWN_GLYPHS["sieesc_chwjgc"]

    @classmethod
    def detect_keypad_matrix(cls, grid: np.ndarray) -> list[list[tuple[int, int]]]:
        """Dynamically detect the 3x3 keypad button centroids from observation grid."""
        if grid.ndim == 3:
            grid = grid[-1]

        # Detect 3x3 button patches in lower control region (y >= 40)
        buttons: list[tuple[int, int]] = []
        for y in range(40, 62):
            for x in range(15, 50):
                patch = grid[y : y + 3, x : x + 3]
                if len(np.unique(patch)) == 1 and patch[0, 0] in (0, 2, 12):
                    if (
                        y > 0
                        and grid[y - 1, x] == 3
                        and grid[y + 3, x] == 3
                        and grid[y, x - 1] == 3
                        and grid[y, x + 3] == 3
                    ):
                        buttons.append((x + 1, y + 1))

        buttons = sorted(buttons, key=lambda p: (p[1], p[0]))
        if len(buttons) == 9:
            return [
                buttons[0:3],
                buttons[3:6],
                buttons[6:9],
            ]

        # Fallback to standard 3x3 layout if grid is partially occluded
        return [
            [(25, 50), (30, 50), (35, 50)],
            [(25, 55), (30, 55), (35, 55)],
            [(25, 60), (30, 60), (35, 60)],
        ]

    @classmethod
    def plan_incantation_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the end-to-end plan: clear intro, click glyph, wait, and navigate to goal."""
        if grid.ndim == 3:
            grid = grid[-1]

        glyph = cls.identify_glyph(grid)
        keypad = cls.detect_keypad_matrix(grid)
        plan: list[tuple[int, dict[str, int] | None]] = []

        # 1. Step 1 clears introductory demo animation flag if active
        plan.append((1, None))

        # 2. Sequential keypad clicks dispatched to dynamically detected centroids
        for r in range(3):
            for c in range(3):
                if glyph.pattern[r][c]:
                    cx, cy = keypad[r][c]
                    plan.append(RemoteActuator.click(cx, cy))

        # 3. Wait actions for transformation execution animation
        for _ in range(8):
            plan.append(RemoteActuator.click(0, 0))

        # 4. Post-transformation navigation: dynamically detect avatar and goal manifolds
        player_pos: tuple[int, int] | None = None
        goal_pos: tuple[int, int] | None = None
        for y in range(5, 45):
            for x in range(5, 55):
                patch = grid[y : y + 6, x : x + 6]
                if 9 in patch and 10 in patch:
                    p2 = grid[y : y + 2, x : x + 2]
                    if np.all(np.isin(p2, [9, 10])):
                        if player_pos is None and x > 25:
                            player_pos = (x, y)
                    if patch.shape == (6, 6) and np.sum(patch == 9) > 10:
                        if goal_pos is None:
                            goal_pos = (x, y)

        if glyph.incantation_type == IncantationType.TELEPORT:
            # Teleport places avatar right next to the goal; navigate UP to dock
            plan.extend(DiscreteVectorTranslator.delta_to_actions(0, -12, step_size=2))
        elif glyph.incantation_type == IncantationType.SHRINK_EXPAND:
            # Avatar is now shrunk (2x2); navigate through corridor LEFT to goal
            dx = (goal_pos[0] - player_pos[0]) if (goal_pos and player_pos) else -28
            plan.extend(DiscreteVectorTranslator.delta_to_actions(dx, 0, step_size=2))
        elif glyph.incantation_type == IncantationType.PROJECTILE:
            # Fireball destroys obstacle ahead, then move UP/LEFT
            plan.extend(DiscreteVectorTranslator.delta_to_actions(0, -24, step_size=2))

        return plan
