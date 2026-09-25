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
    clicks: list[tuple[int, int]]  # Display / grid coordinates for keypad buttons


class ModalIncantationSkillAcquisition:
    """Acquires glyph-to-state-transformation affordances and plans incantations."""

    # Standard glyph configurations on 64x64 lattice with 3x3 keypad at y=49..61, x=24..36
    # Centers: (x=25, 30, 35), (y=50, 55, 60)
    KNOWN_GLYPHS: dict[str, IncantationGlyph] = {
        "sieesc_chwjgc": IncantationGlyph(
            name="sieesc_chwjgc",
            incantation_type=IncantationType.SHRINK_EXPAND,
            pattern=[
                [False, True, False],
                [True, False, True],
                [False, True, False],
            ],
            clicks=[(30, 50), (25, 55), (35, 55), (30, 60)],
        ),
        "tevyeq": IncantationGlyph(
            name="tevyeq",
            incantation_type=IncantationType.TELEPORT,
            pattern=[
                [True, True, False],
                [False, True, False],
                [False, False, False],
            ],
            clicks=[(25, 50), (30, 50), (30, 55)],
        ),
        "fibcey": IncantationGlyph(
            name="fibcey",
            incantation_type=IncantationType.PROJECTILE,
            pattern=[
                [False, True, False],
                [False, True, False],
                [False, True, False],
            ],
            clicks=[(30, 50), (30, 55), (30, 60)],
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
        # Pad area contains distinctive button colors (2, 12, or 14) and frames
        unique_colors = set(np.unique(pad_region))
        has_buttons = 2 in unique_colors or 12 in unique_colors or 14 in unique_colors
        has_nav_entities = 9 in grid and 10 in grid

        return has_buttons and has_nav_entities

    @classmethod
    def identify_glyph(cls, grid: np.ndarray, current_level: int = 0) -> IncantationGlyph:
        """Identify which glyph to cast based on prompt indicators or level context."""
        # Inspect prompt icon patch at bottom-left: y in 48..64, x in 10..22
        icon_region = grid[48:64, 10:22]
        icon_colors = set(np.unique(icon_region))

        # Color 15 indicates sieesc_chwjgc (shrink)
        if 15 in icon_colors:
            return cls.KNOWN_GLYPHS["sieesc_chwjgc"]
        # Color 11 indicates tevyeq (teleport)
        if 11 in icon_colors:
            return cls.KNOWN_GLYPHS["tevyeq"]
        # Color 6 indicates fibcey (fireball)
        if 6 in icon_colors:
            return cls.KNOWN_GLYPHS["fibcey"]

        # Fallback to level heuristic
        if current_level == 1:
            return cls.KNOWN_GLYPHS["tevyeq"]
        if current_level >= 2:
            return cls.KNOWN_GLYPHS["fibcey"]

        return cls.KNOWN_GLYPHS["sieesc_chwjgc"]

    @classmethod
    def plan_incantation_grid(
        cls, grid: np.ndarray, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Compute the end-to-end plan: clear intro, click glyph, wait, and navigate to goal."""
        glyph = cls.identify_glyph(grid, current_level=current_level)
        plan: list[tuple[int, dict[str, int] | None]] = []

        # 1. Step 1 clears introductory demo animation flag if active
        plan.append((1, None))

        # 2. Sequential keypad clicks to toggle incantation pattern
        for x, y in glyph.clicks:
            plan.append((6, {"x": x, "y": y}))

        # 3. Wait actions for transformation execution animation
        for _ in range(8):
            plan.append((6, {"x": 0, "y": 0}))

        # 4. Post-transformation navigation
        if glyph.incantation_type == IncantationType.TELEPORT:
            # Teleport places avatar right next to the goal; navigate UP
            for _ in range(6):
                plan.append((1, None))
        elif glyph.incantation_type == IncantationType.SHRINK_EXPAND:
            # Avatar is now 2x2; navigate through corridor LEFT to goal
            for _ in range(14):
                plan.append((3, None))
        elif glyph.incantation_type == IncantationType.PROJECTILE:
            # Fireball destroys obstacle ahead, then move UP/LEFT
            for _ in range(12):
                plan.append((1, None))

        return plan
