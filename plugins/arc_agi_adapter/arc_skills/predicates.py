"""ARC-AGI grid-puzzle declarative predicates (WP3.5).

Puzzle-specific ``SymbolicPredicate`` implementations relocated from the
task-agnostic core (``hbllm.hcir.skills.declarative``) in accordance with
Invariant 1.7.
"""

from __future__ import annotations

from hbllm.hcir.skills.declarative import SkillEvaluationContext, SymbolicPredicate


class PanelConstraint(SymbolicPredicate):
    """Extracts a relative spatial window (panel) and evaluates sub-predicates inside it."""

    def __init__(
        self,
        min_row_ratio: float = 0.0,
        max_row_ratio: float = 1.0,
        min_col_ratio: float = 0.0,
        max_col_ratio: float = 1.0,
        contains_entities: SymbolicPredicate | None = None,
        min_distinct_colors: int = 1,
    ) -> None:
        self.min_row_ratio = min_row_ratio
        self.max_row_ratio = max_row_ratio
        self.min_col_ratio = min_col_ratio
        self.max_col_ratio = max_col_ratio
        self.contains_entities = contains_entities
        self.min_distinct_colors = min_distinct_colors

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        h, w = ctx.height, ctx.width
        r_min = int(round(self.min_row_ratio * h))
        r_max = int(round(self.max_row_ratio * h))
        c_min = int(round(self.min_col_ratio * w))
        c_max = int(round(self.max_col_ratio * w))

        if r_max <= r_min or c_max <= c_min:
            return False

        sub_ctx = ctx.create_subcontext(r_min, r_max, c_min, c_max)
        distinct = len(set(int(x) for x in sub_ctx.grid.flatten()))
        if distinct < self.min_distinct_colors:
            return False

        if self.contains_entities is not None:
            return self.contains_entities.evaluate(sub_ctx)

        return True

    def __repr__(self) -> str:
        return (
            f"PanelConstraint(r=[{self.min_row_ratio}, {self.max_row_ratio}], "
            f"c=[{self.min_col_ratio}, {self.max_col_ratio}], contains={self.contains_entities!r})"
        )


class LatticeConduitPredicate(SymbolicPredicate):
    """Evaluates whether the grid contains a high-density periodic lattice conduit network."""

    def __init__(self, stride: int = 6, min_conduits: int = 15, patch_size: int = 3) -> None:
        self.stride = stride
        self.min_conduits = min_conduits
        self.patch_size = patch_size

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        from hbllm.hcir.skills.common_subskills import LatticeQuantizer

        anchor = LatticeQuantizer.detect_lattice_anchor(
            ctx.grid, stride=self.stride, patch_size=self.patch_size
        )
        off_y, off_x = anchor
        h, w = ctx.height, ctx.width
        bg = ctx.background_color
        bridges = 0
        for y in range(off_y, h - self.stride, self.stride):
            for x in range(off_x, w - self.stride, self.stride):
                if x + self.stride <= w and ctx.grid[y + 1, x + 3] != bg:
                    bridges += 1
                if y + self.stride <= h and ctx.grid[y + 3, x + 1] != bg:
                    bridges += 1
        return bridges >= self.min_conduits

    def __repr__(self) -> str:
        return f"LatticeConduitPredicate(stride={self.stride}, min_conduits={self.min_conduits})"


class TileGridPredicate(SymbolicPredicate):
    """Evaluates whether the grid contains an array of regular interactive tiles."""

    def __init__(self, min_tiles: int = 4, min_tile_size: int = 3) -> None:
        self.min_tiles = min_tiles
        self.min_tile_size = min_tile_size

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        entities = ctx.get_entities()
        tiles = [
            e
            for e in entities
            if e.width >= self.min_tile_size
            and e.height >= self.min_tile_size
            and abs(e.width - e.height) <= 2
        ]
        return len(tiles) >= self.min_tiles

    def __repr__(self) -> str:
        return f"TileGridPredicate(min_tiles={self.min_tiles})"


class TileFramePredicate(SymbolicPredicate):
    """Evaluates whether a region contains periodic wireframe or square tile frames."""

    def __init__(
        self,
        frame_size: int = 7,
        min_frames: int = 2,
        y_stride: int = 1,
        y_start: int = 0,
        x_stride: int = 1,
        x_start: int = 0,
    ) -> None:
        self.frame_size = frame_size
        self.min_frames = min_frames
        self.y_stride = y_stride
        self.y_start = y_start
        self.x_stride = x_stride
        self.x_start = x_start

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        grid = ctx.grid
        bg = ctx.background_color
        H, W = grid.shape[-2:]
        sz = self.frame_size
        found = 0
        visited_origins: set[tuple[int, int]] = set()
        for y in range(self.y_start, H - sz + 1, self.y_stride):
            for x in range(self.x_start, W - sz + 1, self.x_stride):
                if any(abs(y - vy) < sz and abs(x - vx) < sz for vy, vx in visited_origins):
                    continue
                c = grid[y, x]
                if (
                    c != bg
                    and grid[y + sz - 1, x] == c
                    and grid[y, x + sz - 1] == c
                    and grid[y + sz - 1, x + sz - 1] == c
                ):
                    found += 1
                    visited_origins.add((y, x))
                    if found >= self.min_frames:
                        return True
        return found >= self.min_frames

    def __repr__(self) -> str:
        return (
            f"TileFramePredicate(size={self.frame_size}, min={self.min_frames}, "
            f"y_stride={self.y_stride}, y_start={self.y_start})"
        )
