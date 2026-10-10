"""Domain-General 2D Transformation Rule Induction Engine.

Implements candidate rule generation, Popperian counterexample refutation,
and Minimum Description Length (MDL) simplicity ranking:
1. Candidate Rule Generator:
   - Proposes parameterized geometric (affine), morphological, color mapping,
     crop/extraction, translation, and composite transformation hypotheses.
2. Popperian Refutation Gate:
   - Rejects candidate rules immediately upon the first demonstration counterexample.
   - Evaluates all training demonstration pairs strictly before acceptance.
3. Simplicity & Evidence Ranker:
   - Evaluates description complexity (Kolmogorov / MDL cost).
   - Selects the simplest non-falsified rule explaining all demonstrations.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.world.predictors.whole_grid import WholeGridPredictor

logger = logging.getLogger(__name__)


class RuleFamily(StrEnum):
    """Categorization of transformation rule families."""

    IDENTITY = "IDENTITY"
    AFFINE = "AFFINE"
    COLOR_MAP = "COLOR_MAP"
    CROP = "CROP"
    TRANSLATION = "TRANSLATION"
    GRAVITY = "GRAVITY"
    SCALING = "SCALING"
    KRONECKER = "KRONECKER"
    DUPLICATION = "DUPLICATION"
    TILING = "TILING"
    RELATIONAL = "RELATIONAL"
    DEFORMATION = "DEFORMATION"
    COMPOSITION = "COMPOSITION"
    CONDITIONAL = "CONDITIONAL"
    EXCEPTION = "EXCEPTION"


class AffineOp(StrEnum):
    """Supported 2D discrete affine transformations."""

    ROT_90 = "ROT_90"  # 90 degrees clockwise
    ROT_180 = "ROT_180"  # 180 degrees
    ROT_270 = "ROT_270"  # 270 degrees clockwise (90 counter-clockwise)
    FLIP_H = "FLIP_H"  # Flip across horizontal axis (up-down)
    FLIP_V = "FLIP_V"  # Flip across vertical axis (left-right)
    TRANSPOSE = "TRANSPOSE"  # Main diagonal transposition
    ANTI_TRANSPOSE = "ANTI_TRANSPOSE"  # Anti-diagonal transposition


class CropOp(StrEnum):
    """Supported crop and extraction operators."""

    BBOX_NON_BG = "BBOX_NON_BG"
    BBOX_COLOR = "BBOX_COLOR"
    QUADRANT_TL = "QUADRANT_TL"
    QUADRANT_TR = "QUADRANT_TR"
    QUADRANT_BL = "QUADRANT_BL"
    QUADRANT_BR = "QUADRANT_BR"


@dataclass
class TransformationRule:
    """Base class for parameterized, executable transformation rules."""

    rule_id: str
    family: RuleFamily
    params: dict[str, Any] = field(default_factory=dict)
    complexity: float = 1.0  # MDL description length in bits / nats

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        """Apply transformation to 2D numpy grid and return new 2D grid."""
        raise NotImplementedError

    def describe(self) -> str:
        """Human-readable description of rule and parameters."""
        return f"{self.family.value}:{self.rule_id}({self.params})"


@dataclass
class IdentityRule(TransformationRule):
    """Identity transformation (f(X) = X)."""

    def __init__(self) -> None:
        super().__init__(
            rule_id="identity",
            family=RuleFamily.IDENTITY,
            params={},
            complexity=0.1,
        )

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        return grid.copy()


@dataclass
class AffineRule(TransformationRule):
    """2D discrete affine geometric transformation."""

    op: AffineOp = AffineOp.ROT_90

    def __init__(self, op: AffineOp) -> None:
        super().__init__(
            rule_id=f"affine_{op.value.lower()}",
            family=RuleFamily.AFFINE,
            params={"op": op.value},
            complexity=1.0,
        )
        self.op = op

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        if self.op == AffineOp.ROT_90:
            return np.rot90(grid, -1).copy()
        elif self.op == AffineOp.ROT_180:
            return np.rot90(grid, 2).copy()
        elif self.op == AffineOp.ROT_270:
            return np.rot90(grid, 1).copy()
        elif self.op == AffineOp.FLIP_H:
            return np.flipud(grid).copy()
        elif self.op == AffineOp.FLIP_V:
            return np.fliplr(grid).copy()
        elif self.op == AffineOp.TRANSPOSE:
            return grid.T.copy()
        elif self.op == AffineOp.ANTI_TRANSPOSE:
            return np.rot90(grid.T, 2).copy()
        return grid.copy()


@dataclass
class ColorMappingRule(TransformationRule):
    """Discrete palette substitution mapping (c_src -> c_dst)."""

    mapping: dict[int, int] = field(default_factory=dict)

    def __init__(self, mapping: dict[int, int]) -> None:
        # Cost is proportional to number of active remappings
        active_changes = sum(1 for src, dst in mapping.items() if src != dst)
        cost = 1.0 + 0.25 * active_changes
        super().__init__(
            rule_id=f"color_map_{len(mapping)}",
            family=RuleFamily.COLOR_MAP,
            params={"mapping": mapping},
            complexity=cost,
        )
        self.mapping = mapping

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        out = grid.copy()
        # Fast lookup substitution table for small integer indices (0-9)
        if np.all((grid >= 0) & (grid <= 9)):
            lut = np.arange(10, dtype=int)
            for src, dst in self.mapping.items():
                if 0 <= src <= 9:
                    lut[src] = dst
            return lut[out]
        # General dictionary fallback
        for src, dst in self.mapping.items():
            if src != dst:
                out[grid == src] = dst
        return out


@dataclass
class CropRule(TransformationRule):
    """Crop and subgrid extraction operator."""

    crop_op: CropOp = CropOp.BBOX_NON_BG
    target_color: int | None = None

    def __init__(self, crop_op: CropOp, target_color: int | None = None) -> None:
        c_str = f"_{target_color}" if target_color is not None else ""
        super().__init__(
            rule_id=f"crop_{crop_op.value.lower()}{c_str}",
            family=RuleFamily.CROP,
            params={"crop_op": crop_op.value, "target_color": target_color},
            complexity=1.8,
        )
        self.crop_op = crop_op
        self.target_color = target_color

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        H, W = grid.shape
        if H == 0 or W == 0:
            return grid.copy()

        bg = context.get("bg_color") if context else None
        if bg is None:
            bg = WholeGridPredictor.estimate_background_color(grid)

        if self.crop_op == CropOp.BBOX_NON_BG:
            non_bg_coords = np.argwhere(grid != bg)
            if len(non_bg_coords) == 0:
                return grid.copy()
            r_min, c_min = non_bg_coords.min(axis=0)
            r_max, c_max = non_bg_coords.max(axis=0)
            return grid[r_min : r_max + 1, c_min : c_max + 1].copy()

        elif self.crop_op == CropOp.BBOX_COLOR and self.target_color is not None:
            c_coords = np.argwhere(grid == self.target_color)
            if len(c_coords) == 0:
                return grid.copy()
            r_min, c_min = c_coords.min(axis=0)
            r_max, c_max = c_coords.max(axis=0)
            return grid[r_min : r_max + 1, c_min : c_max + 1].copy()

        elif self.crop_op == CropOp.QUADRANT_TL:
            return grid[: H // 2, : W // 2].copy()
        elif self.crop_op == CropOp.QUADRANT_TR:
            return grid[: H // 2, W // 2 :].copy()
        elif self.crop_op == CropOp.QUADRANT_BL:
            return grid[H // 2 :, : W // 2].copy()
        elif self.crop_op == CropOp.QUADRANT_BR:
            return grid[H // 2 :, W // 2 :].copy()

        return grid.copy()


@dataclass
class TranslationRule(TransformationRule):
    """Rigid grid translation with background fill."""

    dr: int = 0
    dc: int = 0

    def __init__(self, dr: int, dc: int) -> None:
        super().__init__(
            rule_id=f"translate_{dr}_{dc}",
            family=RuleFamily.TRANSLATION,
            params={"dr": dr, "dc": dc},
            complexity=1.5 + 0.1 * (abs(dr) + abs(dc)),
        )
        self.dr = dr
        self.dc = dc

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        H, W = grid.shape
        bg = context.get("bg_color") if context else None
        if bg is None:
            bg = WholeGridPredictor.estimate_background_color(grid)

        out = np.full((H, W), bg, dtype=int)
        r_src_start = max(0, -self.dr)
        r_src_end = min(H, H - self.dr)
        c_src_start = max(0, -self.dc)
        c_src_end = min(W, W - self.dc)

        r_dst_start = max(0, self.dr)
        r_dst_end = min(H, H + self.dr)
        c_dst_start = max(0, self.dc)
        c_dst_end = min(W, W + self.dc)

        if r_src_end > r_src_start and c_src_end > c_src_start:
            out[r_dst_start:r_dst_end, c_dst_start:c_dst_end] = grid[
                r_src_start:r_src_end, c_src_start:c_src_end
            ]
        return out


@dataclass
class GravityRule(TransformationRule):
    """Directional falling/compaction of foreground pixels."""

    direction: str = "DOWN"  # "DOWN", "UP", "LEFT", "RIGHT"

    def __init__(self, direction: str = "DOWN") -> None:
        super().__init__(
            rule_id=f"gravity_{direction.lower()}",
            family=RuleFamily.GRAVITY,
            params={"direction": direction},
            complexity=2.0,
        )
        self.direction = direction

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        H, W = grid.shape
        bg = context.get("bg_color") if context else None
        if bg is None:
            bg = WholeGridPredictor.estimate_background_color(grid)

        out = np.full((H, W), bg, dtype=int)
        if self.direction == "DOWN":
            for c in range(W):
                fg = [grid[r, c] for r in range(H) if grid[r, c] != bg]
                for i, val in enumerate(fg):
                    out[H - len(fg) + i, c] = val
        elif self.direction == "UP":
            for c in range(W):
                fg = [grid[r, c] for r in range(H) if grid[r, c] != bg]
                for i, val in enumerate(fg):
                    out[i, c] = val
        elif self.direction == "RIGHT":
            for r in range(H):
                fg = [grid[r, c] for c in range(W) if grid[r, c] != bg]
                for i, val in enumerate(fg):
                    out[r, W - len(fg) + i] = val
        elif self.direction == "LEFT":
            for r in range(H):
                fg = [grid[r, c] for c in range(W) if grid[r, c] != bg]
                for i, val in enumerate(fg):
                    out[r, i] = val
        return out


@dataclass
class CompositeRule(TransformationRule):
    """Sequential composition of two transformation rules: T2(T1(X))."""

    rule_1: TransformationRule | None = None
    rule_2: TransformationRule | None = None

    def __init__(self, rule_1: TransformationRule, rule_2: TransformationRule) -> None:
        super().__init__(
            rule_id=f"{rule_1.rule_id}__then__{rule_2.rule_id}",
            family=RuleFamily.COMPOSITION,
            params={"stage_1": rule_1.params, "stage_2": rule_2.params},
            complexity=rule_1.complexity + rule_2.complexity + 0.5,
        )
        self.rule_1 = rule_1
        self.rule_2 = rule_2

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        intermediate = self.rule_1.execute(grid, context)
        return self.rule_2.execute(intermediate, context)


@dataclass
class ScalingRule(TransformationRule):
    """Integer scaling (zoom) or fractional downsampling."""

    scale_r: int = 1
    scale_c: int = 1
    downsample_factor: int = 1

    def __init__(self, scale_r: int = 1, scale_c: int = 1, downsample_factor: int = 1) -> None:
        cost = 1.0 + 0.1 * (scale_r + scale_c) + (0.3 if downsample_factor > 1 else 0.0)
        super().__init__(
            rule_id=f"scaling_{scale_r}x{scale_c}_ds{downsample_factor}",
            family=RuleFamily.SCALING,
            params={
                "scale_r": scale_r,
                "scale_c": scale_c,
                "downsample_factor": downsample_factor,
            },
            complexity=cost,
        )
        self.scale_r = scale_r
        self.scale_c = scale_c
        self.downsample_factor = downsample_factor

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        out = grid
        if self.downsample_factor > 1:
            out = out[:: self.downsample_factor, :: self.downsample_factor]
        if self.scale_r > 1 or self.scale_c > 1:
            out = np.repeat(np.repeat(out, self.scale_r, axis=0), self.scale_c, axis=1)
        return out.copy()


@dataclass
class KroneckerRule(TransformationRule):
    """Kronecker product pattern stamping: stamps pattern K for each foreground cell."""

    pattern: np.ndarray | None = None

    def __init__(self, pattern: np.ndarray) -> None:
        cost = 1.6 + 0.05 * pattern.size
        super().__init__(
            rule_id=f"kronecker_pat_{pattern.shape[0]}x{pattern.shape[1]}",
            family=RuleFamily.KRONECKER,
            params={"shape": pattern.shape},
            complexity=cost,
        )
        self.pattern = pattern

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        if self.pattern is None or self.pattern.size == 0:
            return grid.copy()
        bg = context.get("bg_color") if context else None
        if bg is None:
            bg = WholeGridPredictor.estimate_background_color(grid)

        H, W = grid.shape
        kh, kw = self.pattern.shape
        out = np.full((H * kh, W * kw), bg, dtype=int)

        for r in range(H):
            for c in range(W):
                val = grid[r, c]
                if val != bg:
                    patch = np.where(self.pattern != 0, val, bg)
                    out[r * kh : (r + 1) * kh, c * kw : (c + 1) * kw] = patch
        return out


@dataclass
class LatticeDuplicationRule(TransformationRule):
    """Lattice replication with optional alternating kaleidoscope reflections."""

    repeats_r: int = 1
    repeats_c: int = 1
    flip_alt_r: bool = False
    flip_alt_c: bool = False

    def __init__(
        self,
        repeats_r: int = 1,
        repeats_c: int = 1,
        flip_alt_r: bool = False,
        flip_alt_c: bool = False,
    ) -> None:
        cost = 1.5 + 0.15 * (repeats_r + repeats_c) + (0.3 if (flip_alt_r or flip_alt_c) else 0.0)
        super().__init__(
            rule_id=f"duplicate_{repeats_r}x{repeats_c}_flips_{int(flip_alt_r)}_{int(flip_alt_c)}",
            family=RuleFamily.DUPLICATION,
            params={
                "repeats_r": repeats_r,
                "repeats_c": repeats_c,
                "flip_alt_r": flip_alt_r,
                "flip_alt_c": flip_alt_c,
            },
            complexity=cost,
        )
        self.repeats_r = repeats_r
        self.repeats_c = repeats_c
        self.flip_alt_r = flip_alt_r
        self.flip_alt_c = flip_alt_c

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        H, W = grid.shape
        out = np.zeros((H * self.repeats_r, W * self.repeats_c), dtype=int)

        for i in range(self.repeats_r):
            for j in range(self.repeats_c):
                block = grid.copy()
                if self.flip_alt_r and (i % 2 == 1):
                    block = np.flipud(block)
                if self.flip_alt_c and (j % 2 == 1):
                    block = np.fliplr(block)
                out[i * H : (i + 1) * H, j * W : (j + 1) * W] = block
        return out


@dataclass
class TilingRule(TransformationRule):
    """2D wallpaper periodic tiling across canvas."""

    unit_cell: np.ndarray | None = None
    period_r: int = 1
    period_c: int = 1
    masked_fill: bool = False

    def __init__(
        self,
        unit_cell: np.ndarray,
        period_r: int,
        period_c: int,
        masked_fill: bool = False,
    ) -> None:
        cost = 1.4 + 0.04 * unit_cell.size + (0.2 if masked_fill else 0.0)
        super().__init__(
            rule_id=f"tiling_p{period_r}x{period_c}_mask{int(masked_fill)}",
            family=RuleFamily.TILING,
            params={"period_r": period_r, "period_c": period_c, "masked_fill": masked_fill},
            complexity=cost,
        )
        self.unit_cell = unit_cell
        self.period_r = period_r
        self.period_c = period_c
        self.masked_fill = masked_fill

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        if self.unit_cell is None or self.unit_cell.size == 0:
            return grid.copy()

        target_shape = context.get("target_shape") if context else None
        H, W = target_shape if target_shape else grid.shape

        pr, pc = self.period_r, self.period_c
        tiled = np.zeros((H, W), dtype=int)
        for r in range(H):
            for c in range(W):
                tiled[r, c] = self.unit_cell[r % pr, c % pc]

        if self.masked_fill:
            bg = context.get("bg_color") if context else None
            if bg is None:
                bg = WholeGridPredictor.estimate_background_color(grid)
            out = grid.copy()
            out[grid == bg] = tiled[grid == bg]
            return out

        return tiled


@dataclass
class DeformationRule(TransformationRule):
    """Topological non-rigid deformation (connecting lines, flood fill)."""

    deformation_type: str = "CONNECT_LINE"

    def __init__(self, deformation_type: str = "CONNECT_LINE") -> None:
        super().__init__(
            rule_id=f"deformation_{deformation_type.lower()}",
            family=RuleFamily.DEFORMATION,
            params={"type": deformation_type},
            complexity=2.2,
        )
        self.deformation_type = deformation_type

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        out = grid.copy()
        bg = context.get("bg_color") if context else None
        if bg is None:
            bg = WholeGridPredictor.estimate_background_color(grid)

        if self.deformation_type == "CONNECT_LINE":
            for color in np.unique(grid):
                if color == bg:
                    continue
                coords = np.argwhere(grid == color)
                if len(coords) >= 2:
                    for i in range(len(coords) - 1):
                        r1, c1 = coords[i]
                        r2, c2 = coords[i + 1]
                        c_min, c_max = min(c1, c2), max(c1, c2)
                        out[r1, c_min : c_max + 1] = color
                        r_min, r_max = min(r1, r2), max(r1, r2)
                        out[r_min : r_max + 1, c2] = color

        elif self.deformation_type == "FLOOD_FILL":
            H, W = grid.shape
            visited = np.zeros((H, W), dtype=bool)
            queue: deque[tuple[int, int]] = deque()
            for r in range(H):
                if grid[r, 0] == bg and not visited[r, 0]:
                    queue.append((r, 0))
                    visited[r, 0] = True
                if grid[r, W - 1] == bg and not visited[r, W - 1]:
                    queue.append((r, W - 1))
                    visited[r, W - 1] = True
            for c in range(W):
                if grid[0, c] == bg and not visited[0, c]:
                    queue.append((0, c))
                    visited[0, c] = True
                if grid[H - 1, c] == bg and not visited[H - 1, c]:
                    queue.append((H - 1, c))
                    visited[H - 1, c] = True

            while queue:
                r, c = queue.popleft()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if not visited[nr, nc] and grid[nr, nc] == bg:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

            fg_colors = [c for c in np.unique(grid) if c != bg]
            fill_color = fg_colors[0] if fg_colors else bg
            for r in range(H):
                for c in range(W):
                    if grid[r, c] == bg and not visited[r, c]:
                        out[r, c] = fill_color

        return out


class CandidateRuleGenerator:
    """Proposes parameterized transformation hypotheses from demonstration features."""

    @classmethod
    def generate_candidates(
        cls,
        train_pairs: list[tuple[np.ndarray, np.ndarray]],
        max_candidates: int = 100,
    ) -> list[TransformationRule]:
        """Synthesize candidate transformation rules based on demonstration invariants."""
        if not train_pairs:
            return []

        candidates: list[TransformationRule] = []
        x0, y0 = train_pairs[0]
        bg0 = WholeGridPredictor.estimate_background_color(x0)

        # 1. Identity baseline
        candidates.append(IdentityRule())

        # 2. Geometric / Affine candidates
        for op in AffineOp:
            candidates.append(AffineRule(op))

        # 3. Deterministic Palette Substitution Candidates
        if all(x.shape == y.shape for x, y in train_pairs):
            mapping: dict[int, int] = {}
            is_deterministic = True
            for r in range(x0.shape[0]):
                for c in range(x0.shape[1]):
                    src = int(x0[r, c])
                    dst = int(y0[r, c])
                    if src in mapping and mapping[src] != dst:
                        is_deterministic = False
                        break
                    mapping[src] = dst
                if not is_deterministic:
                    break

            if is_deterministic and mapping:
                candidates.append(ColorMappingRule(mapping))

            unique_x = np.unique(x0)
            unique_y = np.unique(y0)
            all_colors = sorted(set(unique_x).union(set(unique_y)))
            if len(all_colors) <= 6:
                for i in range(len(all_colors)):
                    for j in range(i + 1, len(all_colors)):
                        c1, c2 = int(all_colors[i]), int(all_colors[j])
                        swap_map = {c1: c2, c2: c1}
                        candidates.append(ColorMappingRule(swap_map))

        # 4. Crop & Subgrid Extraction Candidates
        if any(y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1] for x, y in train_pairs):
            candidates.append(CropRule(CropOp.BBOX_NON_BG))
            for crop_op in [
                CropOp.QUADRANT_TL,
                CropOp.QUADRANT_TR,
                CropOp.QUADRANT_BL,
                CropOp.QUADRANT_BR,
            ]:
                candidates.append(CropRule(crop_op))

            for c in np.unique(x0):
                if c != bg0:
                    candidates.append(CropRule(CropOp.BBOX_COLOR, target_color=int(c)))

        # 5. Translations & Gravity
        if all(x.shape == y.shape for x, y in train_pairs):
            for dr in [-2, -1, 0, 1, 2]:
                for dc in [-2, -1, 0, 1, 2]:
                    if dr != 0 or dc != 0:
                        candidates.append(TranslationRule(dr, dc))

            for g_dir in ["DOWN", "UP", "LEFT", "RIGHT"]:
                candidates.append(GravityRule(g_dir))

        # 6. Scaling & Duplication Candidates (W054 & W056)
        if all(
            y.shape[0] % x.shape[0] == 0 and y.shape[1] % x.shape[1] == 0 for x, y in train_pairs
        ):
            kr = y0.shape[0] // x0.shape[0]
            kc = y0.shape[1] // x0.shape[1]
            if kr >= 1 and kc >= 1 and (kr > 1 or kc > 1):
                candidates.append(ScalingRule(scale_r=kr, scale_c=kc))
                candidates.append(
                    LatticeDuplicationRule(kr, kc, flip_alt_r=False, flip_alt_c=False)
                )
                candidates.append(LatticeDuplicationRule(kr, kc, flip_alt_r=True, flip_alt_c=False))
                candidates.append(LatticeDuplicationRule(kr, kc, flip_alt_r=False, flip_alt_c=True))
                candidates.append(LatticeDuplicationRule(kr, kc, flip_alt_r=True, flip_alt_c=True))

        if all(
            x.shape[0] % y.shape[0] == 0 and x.shape[1] % y.shape[1] == 0 for x, y in train_pairs
        ):
            dr = x0.shape[0] // y0.shape[0]
            dc = x0.shape[1] // y0.shape[1]
            if dr == dc and dr > 1:
                candidates.append(ScalingRule(downsample_factor=dr))

        # Kronecker Pattern Stamping
        for p_size in [2, 3]:
            if y0.shape[0] == x0.shape[0] * p_size and y0.shape[1] == x0.shape[1] * p_size:
                found_pat = False
                for r in range(x0.shape[0]):
                    for c in range(x0.shape[1]):
                        if x0[r, c] != bg0:
                            block = y0[r * p_size : (r + 1) * p_size, c * p_size : (c + 1) * p_size]
                            pat = (block != bg0).astype(int)
                            candidates.append(KroneckerRule(pat))
                            found_pat = True
                            break
                    if found_pat:
                        break

        # 7. Motif Autocorrelation & Wallpaper Tiling Candidates (W036 & W106)
        H_y, W_y = y0.shape
        for pr in range(1, min(H_y // 2 + 1, 6)):
            for pc in range(1, min(W_y // 2 + 1, 6)):
                is_periodic = True
                for r in range(H_y - pr):
                    for c in range(W_y - pc):
                        if y0[r, c] != y0[r + pr, c] or y0[r, c] != y0[r, c + pc]:
                            is_periodic = False
                            break
                    if not is_periodic:
                        break
                if is_periodic:
                    unit = y0[:pr, :pc].copy()
                    candidates.append(TilingRule(unit, period_r=pr, period_c=pc, masked_fill=False))
                    candidates.append(TilingRule(unit, period_r=pr, period_c=pc, masked_fill=True))

        # 8. Topological Deformation Candidates (W059)
        candidates.append(DeformationRule("CONNECT_LINE"))
        candidates.append(DeformationRule("FLOOD_FILL"))

        # 9. Two-stage Compositions (Affine + ColorMap, Crop + Affine)
        crop_candidates = [c for c in candidates if isinstance(c, CropRule)]
        affine_candidates = [c for c in candidates if isinstance(c, AffineRule)]
        for cr in crop_candidates[:3]:
            for af in affine_candidates[:4]:
                candidates.append(CompositeRule(cr, af))

        color_candidates = [c for c in candidates if isinstance(c, ColorMappingRule)]
        for af in affine_candidates[:3]:
            for col in color_candidates[:2]:
                candidates.append(CompositeRule(af, col))

        # De-duplicate candidate rules by rule_id
        seen_ids: set[str] = set()
        unique_candidates: list[TransformationRule] = []
        for cand in candidates:
            if cand.rule_id not in seen_ids:
                seen_ids.add(cand.rule_id)
                unique_candidates.append(cand)

        return unique_candidates[:max_candidates]


@dataclass
class RefutationRecord:
    """Record of an inductive hypothesis falsification."""

    rule_id: str
    falsifying_demo_idx: int
    reason: str
    predicted_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    pixel_discrepancy: int


class PopperianRefutationGate:
    """Rigorous Popperian refutation gate for inductive transformation rules."""

    def __init__(self) -> None:
        self.falsifications: list[RefutationRecord] = []

    def evaluate_and_filter(
        self,
        candidates: list[TransformationRule],
        train_pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[TransformationRule]:
        """Test candidate rules against all demonstrations; refute on first discrepancy.

        A rule survives IF AND ONLY IF it produces exact pixel-level matches
        across 100% of the training demonstration outputs.
        """
        surviving_rules: list[TransformationRule] = []

        for rule in candidates:
            is_refuted = False
            for demo_idx, (x, y) in enumerate(train_pairs):
                bg = WholeGridPredictor.estimate_background_color(x)
                context = {"bg_color": bg, "demo_idx": demo_idx}

                try:
                    pred_y = rule.execute(x, context)
                except Exception as exc:
                    self.falsifications.append(
                        RefutationRecord(
                            rule_id=rule.rule_id,
                            falsifying_demo_idx=demo_idx,
                            reason=f"EXECUTION_EXCEPTION: {exc}",
                            predicted_shape=(0, 0),
                            target_shape=y.shape,
                            pixel_discrepancy=y.size,
                        )
                    )
                    is_refuted = True
                    break

                is_exact, _, mismatch_count = WholeGridPredictor.compute_grid_metrics(pred_y, y)
                if not is_exact:
                    self.falsifications.append(
                        RefutationRecord(
                            rule_id=rule.rule_id,
                            falsifying_demo_idx=demo_idx,
                            reason="PIXEL_OR_SHAPE_MISMATCH",
                            predicted_shape=pred_y.shape,
                            target_shape=y.shape,
                            pixel_discrepancy=mismatch_count,
                        )
                    )
                    is_refuted = True
                    break

            if not is_refuted:
                surviving_rules.append(rule)

        return surviving_rules


class SimplicityRanker:
    """Ranks surviving non-falsified rules by Minimum Description Length (MDL)."""

    @staticmethod
    def rank_survivors(survivors: list[TransformationRule]) -> list[TransformationRule]:
        """Sort surviving rules by ascending description complexity score."""
        return sorted(survivors, key=lambda r: (r.complexity, r.rule_id))


class ConditionalRule(TransformationRule):
    """W074: Conditional rule executing rule_true if condition predicate holds, else rule_false."""

    def __init__(
        self,
        condition_name: str,
        predicate: Callable[[np.ndarray], bool],
        rule_true: TransformationRule,
        rule_false: TransformationRule,
    ) -> None:
        super().__init__(
            rule_id=f"cond_{condition_name}_{rule_true.rule_id}_{rule_false.rule_id}",
            family=RuleFamily.CONDITIONAL,
            params={
                "condition": condition_name,
                "rule_true": rule_true.rule_id,
                "rule_false": rule_false.rule_id,
            },
            complexity=rule_true.complexity + rule_false.complexity + 0.5,
        )
        self.condition_name = condition_name
        self.predicate = predicate
        self.rule_true = rule_true
        self.rule_false = rule_false

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        if self.predicate(grid):
            return self.rule_true.execute(grid, context)
        return self.rule_false.execute(grid, context)


class ExceptionAwareRule(TransformationRule):
    """W075: Base rule with specialized exception patch handling."""

    def __init__(
        self,
        base_rule: TransformationRule,
        exception_mask_fn: Callable[[np.ndarray], np.ndarray],
        patch_color: int,
        exception_name: str = "outlier_patch",
    ) -> None:
        super().__init__(
            rule_id=f"exc_{base_rule.rule_id}_{exception_name}",
            family=RuleFamily.EXCEPTION,
            params={
                "base_rule": base_rule.rule_id,
                "patch_color": patch_color,
                "exception_name": exception_name,
            },
            complexity=base_rule.complexity + 1.2,
        )
        self.base_rule = base_rule
        self.exception_mask_fn = exception_mask_fn
        self.patch_color = patch_color
        self.exception_name = exception_name

    def execute(self, grid: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        out = self.base_rule.execute(grid, context)
        exc_mask = self.exception_mask_fn(grid)
        if exc_mask.shape == out.shape:
            out[exc_mask] = self.patch_color
        return out


class RulePrecedenceResolver:
    """W077: Resolves rule precedence and conflicts among competing applicable rules."""

    @classmethod
    def resolve(
        cls,
        candidates: list[TransformationRule],
        grid: np.ndarray,
    ) -> TransformationRule | None:
        """Select the highest-precedence rule: specificity > simplicity (MDL) > deterministic ID."""
        if not candidates:
            return None

        def priority_key(r: TransformationRule) -> tuple[int, float, str]:
            specificity = 0
            if isinstance(r, (ConditionalRule, ExceptionAwareRule)):
                specificity = 2
            elif isinstance(r, CompositeRule):
                specificity = 1
            return (-specificity, r.complexity, r.rule_id)

        return sorted(candidates, key=priority_key)[0]


class LatentVariableBifurcationInducer:
    """W078: Induces latent discrete conditioning variables that bifurcate demonstration datasets."""

    LATENT_PREDICATES: dict[str, tuple[Callable[[np.ndarray], bool], str]] = {
        "foreground_count_even": (
            lambda g: bool(np.count_nonzero(g != 0) % 2 == 0),
            "Foreground parity",
        ),
        "height_greater_width": (
            lambda g: bool(g.shape[0] > g.shape[1]),
            "Orientation aspect",
        ),
        "has_marker_color_2": (
            lambda g: bool(2 in np.unique(g)),
            "Contains red marker",
        ),
        "has_marker_color_8": (
            lambda g: bool(8 in np.unique(g)),
            "Contains teal marker",
        ),
        "unique_colors_gt_2": (
            lambda g: bool(len(np.unique(g)) > 2),
            "Multicolor palette",
        ),
    }

    @classmethod
    def induce_bifurcation(
        cls,
        train_pairs: list[tuple[np.ndarray, np.ndarray]],
        engine: RuleInductionEngine,
    ) -> ConditionalRule | None:
        """Searches latent variable space to bifurcate demonstrations and induce conditional rules."""
        if len(train_pairs) < 2:
            return None

        for pred_name, (pred_fn, _) in cls.LATENT_PREDICATES.items():
            group_true = [(x, y) for x, y in train_pairs if pred_fn(x)]
            group_false = [(x, y) for x, y in train_pairs if not pred_fn(x)]

            if not group_true or not group_false:
                continue

            rule_t, survivors_t, _ = engine.induce_rule(group_true, allow_recursion=False)
            rule_f, survivors_f, _ = engine.induce_rule(group_false, allow_recursion=False)

            if rule_t is not None and rule_f is not None:
                return ConditionalRule(
                    condition_name=pred_name,
                    predicate=pred_fn,
                    rule_true=rule_t,
                    rule_false=rule_f,
                )

        return None


@dataclass
class RulePosteriorResult:
    """W080: Calibrated Bayesian posterior and epistemic uncertainty over induced rules."""

    best_rule: TransformationRule | None
    probabilities: dict[str, float]
    entropy_bits: float
    confidence: float
    is_ambiguous: bool


class RulePosteriorCalibrator:
    """W080: Evaluates rule confidence and epistemic uncertainty over hypothesis distributions."""

    @classmethod
    def compute_posterior(
        cls,
        surviving_rules: Sequence[TransformationRule],
        temperature: float = 1.0,
        ambiguity_entropy_threshold: float = 1.5,
    ) -> RulePosteriorResult:
        """Computes Boltzmann posterior distribution over surviving MDL rules."""
        if not surviving_rules:
            return RulePosteriorResult(
                best_rule=None,
                probabilities={},
                entropy_bits=0.0,
                confidence=0.0,
                is_ambiguous=True,
            )

        complexities = [r.complexity for r in surviving_rules]
        min_c = min(complexities)
        unnorm = [math.exp(-(c - min_c) / temperature) for c in complexities]
        total = sum(unnorm)
        probs = [w / total for w in unnorm]

        prob_dict = {r.rule_id: float(p) for r, p in zip(surviving_rules, probs, strict=False)}

        entropy = 0.0
        for p in probs:
            if p > 1e-9:
                entropy -= p * math.log2(p)

        best_idx = 0
        best_rule = surviving_rules[best_idx]
        top_prob = probs[best_idx]

        confidence = top_prob * math.exp(-entropy)
        confidence = max(0.0, min(1.0, confidence))
        is_ambiguous = len(surviving_rules) > 1 and entropy > ambiguity_entropy_threshold

        return RulePosteriorResult(
            best_rule=best_rule,
            probabilities=prob_dict,
            entropy_bits=entropy,
            confidence=confidence,
            is_ambiguous=is_ambiguous,
        )


class RuleInductionEngine:
    """Unified engine coordinating generation, refutation, simplicity selection, and bifurcation."""

    def __init__(self, max_candidates: int = 60) -> None:
        self.max_candidates = max_candidates
        self.generator = CandidateRuleGenerator()
        self.refutation_gate = PopperianRefutationGate()
        self.bifurcation_inducer = LatentVariableBifurcationInducer()
        self.posterior_calibrator = RulePosteriorCalibrator()

    def extract_common_transformations(
        self,
        train_pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[TransformationRule]:
        """W073: Extracts transformation rules shared consistently across demonstration pairs."""
        candidates = self.generator.generate_candidates(
            train_pairs, max_candidates=self.max_candidates
        )
        return self.refutation_gate.evaluate_and_filter(candidates, train_pairs)

    def induce_rule(
        self,
        train_pairs: list[tuple[np.ndarray, np.ndarray]],
        allow_recursion: bool = True,
    ) -> tuple[TransformationRule | None, list[TransformationRule], dict[str, Any]]:
        """Induce the optimal transformation rule from demonstrations.

        Returns:
            (selected_rule, surviving_rules, metadata)
        """
        candidates = self.generator.generate_candidates(
            train_pairs, max_candidates=self.max_candidates
        )
        survivors = self.refutation_gate.evaluate_and_filter(candidates, train_pairs)
        ranked = SimplicityRanker.rank_survivors(survivors)

        selected_rule = ranked[0] if ranked else None

        # W078: If no single rule explains demonstrations, search latent variable space
        bifurcation_rule: ConditionalRule | None = None
        if selected_rule is None and allow_recursion:
            bifurcation_rule = self.bifurcation_inducer.induce_bifurcation(train_pairs, self)
            if bifurcation_rule is not None:
                selected_rule = bifurcation_rule
                ranked = [bifurcation_rule]

        # W080: Calibrated posterior confidence & uncertainty
        posterior = self.posterior_calibrator.compute_posterior(ranked)

        metadata = {
            "candidates_generated": len(candidates),
            "refuted_count": len(self.refutation_gate.falsifications),
            "survivors_count": len(ranked),
            "selected_rule_id": selected_rule.rule_id if selected_rule else None,
            "selected_complexity": selected_rule.complexity if selected_rule else None,
            "is_latent_bifurcated": bifurcation_rule is not None,
            "posterior_entropy": posterior.entropy_bits,
            "rule_confidence": posterior.confidence,
            "is_ambiguous": posterior.is_ambiguous,
        }

        return selected_rule, ranked, metadata
