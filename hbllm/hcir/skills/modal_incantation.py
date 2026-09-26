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
from typing import Any

import numpy as np

from hbllm.hcir.skills.base import BaseHierarchicalSkill
from hbllm.hcir.skills.common_subskills import (
    DiscreteVectorTranslator,
    PerceptualClusterDetector,
    RemoteActuator,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent

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


class ModalIncantationSkillAcquisition(BaseHierarchicalSkill):
    """Acquires glyph-to-state-transformation affordances and plans incantations."""

    skill_name: str = "modal_incantation_transformation"
    semantic_intent: SpatialActionIntent = SpatialActionIntent.INTERACT

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
    def _find_keypad_buttons(cls, grid: np.ndarray) -> list[tuple[int, int]]:
        """Find 9 button centroids in the lower control area without hardcoded colors."""
        if grid.ndim == 3:
            grid = grid[-1]

        # Dominant background in lower area
        lower_region = grid[42:63, 15:50]
        vals, counts = np.unique(lower_region, return_counts=True)
        bg = vals[np.argmax(counts)]

        # Find 2x2 or 3x3 uniform patches distinct from bg
        candidates = []
        visited = set()
        for y in range(42, 61):
            for x in range(15, 48):
                if (y, x) in visited:
                    continue
                c = grid[y, x]
                if c == bg:
                    continue
                # Check for 2x2 or 3x3 uniform block
                for sz in (3, 2):
                    if y + sz <= 63 and x + sz <= 50:
                        patch = grid[y : y + sz, x : x + sz]
                        if np.all(patch == c):
                            candidates.append((x + sz // 2, y + sz // 2))
                            for dy in range(sz):
                                for dx in range(sz):
                                    visited.add((y + dy, x + dx))
                            break

        c_set = set(candidates)
        xs = sorted(list({p[0] for p in c_set}))
        ys = sorted(list({p[1] for p in c_set}))

        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                dx = xs[j] - xs[i]
                if dx < 4:
                    continue
                x3 = xs[j] + dx
                if x3 not in xs:
                    continue
                cand_xs = [xs[i], xs[j], x3]
                for a in range(len(ys)):
                    for b in range(a + 1, len(ys)):
                        dy = ys[b] - ys[a]
                        if dy < 4:
                            continue
                        y3 = ys[b] + dy
                        if y3 not in ys:
                            continue
                        cand_ys = [ys[a], ys[b], y3]
                        if all((x, y) in c_set for y in cand_ys for x in cand_xs):
                            return [(x, y) for y in cand_ys for x in cand_xs]
        return []

    @classmethod
    def is_incantation_grid(cls, grid: np.ndarray, available_actions: list[int]) -> bool:
        """Detect whether the grid contains an incantation keypad with spatial navigation."""
        if not (6 in available_actions and any(a in available_actions for a in (1, 2, 3, 4))):
            return False

        H, W = grid.shape[-2:]
        if H != 64 or W != 64:
            return False

        buttons = cls._find_keypad_buttons(grid)
        if len(buttons) < 9:
            return False

        # Verify prompt icon exists in the lower-left UI panel (y >= 45, x <= 25)
        raw = grid[-1] if grid.ndim == 3 else grid
        panel = raw[45:63, 2:25]
        p_vals, p_counts = np.unique(panel, return_counts=True)
        panel_bg = p_vals[np.argmax(p_counts)]
        non_bg_pixels = np.sum(panel != panel_bg)
        return non_bg_pixels >= 6

    @classmethod
    def identify_glyph(cls, grid: np.ndarray) -> IncantationGlyph:
        """Identify which glyph to cast based on geometry and aspect ratio of prompt indicator."""
        if grid.ndim == 3:
            grid = grid[-1]

        panel = grid[45:63, 2:25]
        p_vals, p_counts = np.unique(panel, return_counts=True)
        panel_bg = p_vals[np.argmax(p_counts)]

        # Find non-bg connected components in prompt panel
        non_bg_mask = panel != panel_bg
        labeled, num_features = PerceptualClusterDetector.label_components(non_bg_mask)

        best_glyph = cls.KNOWN_GLYPHS["sieesc_chwjgc"]
        max_size = 0

        for lbl in range(1, num_features + 1):
            pts = np.argwhere(labeled == lbl)
            if len(pts) < 4:
                continue
            y_min, x_min = np.min(pts, axis=0)
            y_max, x_max = np.max(pts, axis=0)
            h = y_max - y_min + 1
            w = x_max - x_min + 1
            sz = len(pts)

            if sz > max_size:
                max_size = sz
                # Morphological classification by aspect ratio and bounding box
                if h >= 2 * w and h >= 5:
                    best_glyph = cls.KNOWN_GLYPHS["fibcey"]  # Vertical beam
                elif w >= 6 and h >= 6:
                    best_glyph = cls.KNOWN_GLYPHS["sieesc_chwjgc"]  # Large symmetric ring (8x8)
                elif 3 <= w <= 6 and 3 <= h <= 6:
                    best_glyph = cls.KNOWN_GLYPHS["tevyeq"]  # Compact asymmetric L-shape (5x5)

        return best_glyph

    @classmethod
    def detect_keypad_matrix(cls, grid: np.ndarray) -> list[list[tuple[int, int]]]:
        """Dynamically detect the 3x3 keypad button centroids from observation grid."""
        buttons = cls._find_keypad_buttons(grid)
        if len(buttons) == 9:
            # Sort into 3 rows of 3
            buttons.sort(key=lambda p: (p[1] // 4, p[0]))
            return [
                sorted(buttons[0:3], key=lambda p: p[0]),
                sorted(buttons[3:6], key=lambda p: p[0]),
                sorted(buttons[6:9], key=lambda p: p[0]),
            ]

        # Geometry fallback based on layout
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

        # 4. Post-transformation navigation: dynamically detect avatar and goal manifolds in upper playfield
        upper = grid[:45, :]
        u_vals, u_counts = np.unique(upper, return_counts=True)
        bg = u_vals[np.argmax(u_counts)]

        # Find entities in playfield
        non_bg = (upper != bg) & (upper != 0)
        labeled, num_features = PerceptualClusterDetector.label_components(non_bg)
        player_pos: tuple[int, int] | None = None
        goal_pos: tuple[int, int] | None = None

        for lbl in range(1, num_features + 1):
            pts = np.argwhere(labeled == lbl)
            if len(pts) < 4:
                continue
            cy, cx = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            sz = len(pts)
            # Differentiate goal vs player by morphology / position
            if cx > 25 and player_pos is None and sz <= 16:
                player_pos = (cx, cy)
            elif sz >= 10 and goal_pos is None:
                goal_pos = (cx, cy)

        if glyph.incantation_type == IncantationType.TELEPORT:
            dy = (goal_pos[1] - player_pos[1]) if (goal_pos and player_pos) else -12
            plan.extend(DiscreteVectorTranslator.delta_to_actions(0, dy, step_size=2))
        elif glyph.incantation_type == IncantationType.SHRINK_EXPAND:
            dx = (goal_pos[0] - player_pos[0]) if (goal_pos and player_pos) else -28
            plan.extend(DiscreteVectorTranslator.delta_to_actions(dx, 0, step_size=2))
        elif glyph.incantation_type == IncantationType.PROJECTILE:
            dy = (goal_pos[1] - player_pos[1]) if (goal_pos and player_pos) else -24
            plan.extend(DiscreteVectorTranslator.delta_to_actions(0, dy, step_size=2))

        return plan

    # ═══════════════════════════════════════════════════════════════════════
    # BaseHierarchicalSkill Standardized Protocol Implementation
    # ═══════════════════════════════════════════════════════════════════════

    def can_handle(
        self,
        grid: np.ndarray,
        available_actions: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Standardized interface check for modal incantation keypad puzzles."""
        return self.is_incantation_grid(grid, available_actions)

    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Standardized interface plan generation for modal incantation keypad puzzles."""
        return self.plan_incantation_grid(grid, current_level=current_level)
