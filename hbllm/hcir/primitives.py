"""Domain-Agnostic Spatial, Topological & Relational Primitives for HCIR.

Pure mathematical, cellular, and topological operations over discrete 2D integer lattices.
Zero domain-specific hardcoding. Part of the core HBLLM Cognitive OS ISA.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Grid = np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# 1. Connected Component & Topological Segmentation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EntityComponent:
    """Discrete spatial entity segmented from 2D discrete lattice."""

    color: int
    coords: frozenset[tuple[int, int]]
    min_r: int
    max_r: int
    min_c: int
    max_c: int
    area: int
    is_frame: bool
    enclosed_coords: frozenset[tuple[int, int]]

    @property
    def height(self) -> int:
        return self.max_r - self.min_r + 1

    @property
    def width(self) -> int:
        return self.max_c - self.min_c + 1

    @property
    def shape_signature(self) -> tuple[tuple[int, ...], ...]:
        sub = np.zeros((self.height, self.width), dtype=int)
        for r, c in self.coords:
            sub[r - self.min_r, c - self.min_c] = 1
        return tuple(tuple(int(x) for x in row) for row in sub)

    @property
    def centroid(self) -> tuple[float, float]:
        if not self.coords:
            return (0.0, 0.0)
        return (
            sum(r for r, _ in self.coords) / self.area,
            sum(c for _, c in self.coords) / self.area,
        )


def detect_background_color(grid: Grid) -> int:
    """Detect dominant border/background color (defaulting to 0 if present)."""
    h, w = grid.shape
    if 0 in grid:
        return 0
    border = list(grid[0, :]) + list(grid[h - 1, :]) + list(grid[:, 0]) + list(grid[:, w - 1])
    if border:
        return int(Counter(border).most_common(1)[0][0])
    return int(Counter(grid.flatten()).most_common(1)[0][0])


def segment_entities(
    grid: Grid, bg_color: int | None = None, connectivity: int = 4
) -> list[EntityComponent]:
    """Segment lattice into discrete connected entity components."""
    if bg_color is None:
        bg_color = detect_background_color(grid)

    h, w = grid.shape
    visited: set[tuple[int, int]] = set()
    components: list[EntityComponent] = []

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg_color or (r, c) in visited:
                continue

            coords_set: set[tuple[int, int]] = set()
            queue: deque[tuple[int, int]] = deque([(r, c)])
            visited.add((r, c))

            while queue:
                cr, cc = queue.popleft()
                coords_set.add((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and (nr, nc) not in visited
                        and grid[nr, nc] == color
                    ):
                        visited.add((nr, nc))
                        queue.append((nr, nc))

            min_r = min(cr for cr, _ in coords_set)
            max_r = max(cr for cr, _ in coords_set)
            min_c = min(cc for _, cc in coords_set)
            max_c = max(cc for _, cc in coords_set)
            area = len(coords_set)

            # Detect enclosed void pixels inside bounding box
            enclosed = _detect_enclosed_voids(coords_set, min_r, max_r, min_c, max_c)
            is_frame = len(enclosed) > 0

            components.append(
                EntityComponent(
                    color=color,
                    coords=frozenset(coords_set),
                    min_r=min_r,
                    max_r=max_r,
                    min_c=min_c,
                    max_c=max_c,
                    area=area,
                    is_frame=is_frame,
                    enclosed_coords=frozenset(enclosed),
                )
            )

    return components


def _detect_enclosed_voids(
    boundary: set[tuple[int, int]], min_r: int, max_r: int, min_c: int, max_c: int
) -> set[tuple[int, int]]:
    """Flood fill from perimeter using 4-connectivity to isolate enclosed cavities."""
    if (max_r - min_r < 2) or (max_c - min_c < 2):
        return set()

    bbox_pixels = {
        (r, c)
        for r in range(min_r, max_r + 1)
        for c in range(min_c, max_c + 1)
        if (r, c) not in boundary
    }
    if not bbox_pixels:
        return set()

    outside: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()

    # Perimeter seeding
    for r in range(min_r, max_r + 1):
        for c in (min_c, max_c):
            if (r, c) in bbox_pixels and (r, c) not in outside:
                outside.add((r, c))
                queue.append((r, c))
    for c in range(min_c, max_c + 1):
        for r in (min_r, max_r):
            if (r, c) in bbox_pixels and (r, c) not in outside:
                outside.add((r, c))
                queue.append((r, c))

    while queue:
        cr, cc = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if (
                min_r <= nr <= max_r
                and min_c <= nc <= max_c
                and (nr, nc) in bbox_pixels
                and (nr, nc) not in outside
            ):
                outside.add((nr, nc))
                queue.append((nr, nc))

    return bbox_pixels - outside


# ─────────────────────────────────────────────────────────────────────────────
# 2. D4 Symmetries & Affine Primitives
# ─────────────────────────────────────────────────────────────────────────────


def op_identity(grid: Grid) -> Grid:
    return grid.copy()


def op_rot90(grid: Grid) -> Grid:
    return np.rot90(grid, 1).copy()


def op_rot180(grid: Grid) -> Grid:
    return np.rot90(grid, 2).copy()


def op_rot270(grid: Grid) -> Grid:
    return np.rot90(grid, 3).copy()


def op_fliplr(grid: Grid) -> Grid:
    return np.fliplr(grid).copy()


def op_flipud(grid: Grid) -> Grid:
    return np.flipud(grid).copy()


def op_transpose(grid: Grid) -> Grid:
    return grid.T.copy()


def op_anti_transpose(grid: Grid) -> Grid:
    return np.rot90(np.fliplr(grid), 1).copy()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Spatial Bounding, Panel Extraction & Superposition
# ─────────────────────────────────────────────────────────────────────────────


def op_crop_active_content(grid: Grid) -> Grid:
    """Crop padding away from non-background pixels."""
    bg = detect_background_color(grid)
    pts = np.argwhere(grid != bg)
    if len(pts) == 0:
        return grid.copy()
    min_r, min_c = pts.min(axis=0)
    max_r, max_c = pts.max(axis=0)
    return grid[min_r : max_r + 1, min_c : max_c + 1].copy()


def op_crop_inside_frame(grid: Grid) -> Grid:
    """If an enclosed frame entity exists, crop the interior subgrid."""
    bg = detect_background_color(grid)
    for conn in [4, 8]:
        ents = segment_entities(grid, bg_color=bg, connectivity=conn)
        frames = [e for e in ents if e.is_frame]
        if frames:
            f = max(frames, key=lambda e: len(e.enclosed_coords))
            r_coords = [r for r, _ in f.enclosed_coords]
            c_coords = [c for _, c in f.enclosed_coords]
            min_r, max_r = min(r_coords), max(r_coords)
            min_c, max_c = min(c_coords), max(c_coords)
            return grid[min_r : max_r + 1, min_c : max_c + 1].copy()
    return grid.copy()


def op_crop_extreme_entity(grid: Grid, extreme: str = "largest") -> Grid:
    """Crop bounding box of the largest or smallest connected entity."""
    bg = detect_background_color(grid)
    ents = segment_entities(grid, bg_color=bg)
    if not ents:
        return grid.copy()
    target = (
        max(ents, key=lambda e: e.area) if extreme == "largest" else min(ents, key=lambda e: e.area)
    )
    return grid[target.min_r : target.max_r + 1, target.min_c : target.max_c + 1].copy()


def op_crop_least_frequent_fg(grid: Grid) -> Grid:
    """Crop bounding box of the least frequent non-background color."""
    bg = detect_background_color(grid)
    counts = Counter([int(c) for c in grid.flatten() if c != bg])
    if not counts:
        return grid.copy()
    rarest = counts.most_common()[-1][0]
    pts = np.argwhere(grid == rarest)
    if len(pts) == 0:
        return grid.copy()
    min_r, min_c = pts.min(axis=0)
    max_r, max_c = pts.max(axis=0)
    return grid[min_r : max_r + 1, min_c : max_c + 1].copy()


def op_crop_most_frequent_fg(grid: Grid) -> Grid:
    """Crop bounding box of the most frequent non-background color."""
    bg = detect_background_color(grid)
    counts = Counter([int(c) for c in grid.flatten() if c != bg])
    if not counts:
        return grid.copy()
    most = counts.most_common(1)[0][0]
    pts = np.argwhere(grid == most)
    if len(pts) == 0:
        return grid.copy()
    min_r, min_c = pts.min(axis=0)
    max_r, max_c = pts.max(axis=0)
    return grid[min_r : max_r + 1, min_c : max_c + 1].copy()


def find_grid_dividers(
    grid: Grid, divider_color: int | None = None
) -> tuple[list[int], list[int], int | None]:
    """Find horizontal and vertical divider lines of uniform color that partition the grid."""
    h, w = grid.shape
    if h < 3 or w < 3:
        return [], [], None

    if divider_color is not None:
        div_rows = [r for r in range(1, h - 1) if np.all(grid[r, :] == divider_color)]
        div_cols = [c for c in range(1, w - 1) if np.all(grid[:, c] == divider_color)]
        if div_rows or div_cols:
            return div_rows, div_cols, divider_color
        return [], [], None

    bg = detect_background_color(grid)
    candidate_colors = set()
    for r in range(1, h - 1):
        if len(set(grid[r, :])) == 1:
            candidate_colors.add(int(grid[r, 0]))
    for c in range(1, w - 1):
        if len(set(grid[:, c])) == 1:
            candidate_colors.add(int(grid[0, c]))

    # Prioritize non-background divider lines
    ordered_colors = [c for c in candidate_colors if c != bg] + (
        [bg] if bg in candidate_colors else []
    )

    for div_col in ordered_colors:
        div_rows = [r for r in range(1, h - 1) if np.all(grid[r, :] == div_col)]
        div_cols = [c for c in range(1, w - 1) if np.all(grid[:, c] == div_col)]
        if div_rows or div_cols:
            return div_rows, div_cols, div_col

    return [], [], None


def op_extract_subgrid_panels(grid: Grid, divider_color: int | None = None) -> list[Grid]:
    """Extract subgrid panels partitioned by uniform dividing lines."""
    div_rows, div_cols, _ = find_grid_dividers(grid, divider_color=divider_color)
    if not div_rows and not div_cols:
        return [grid.copy()]

    h, w = grid.shape
    row_splits = [-1] + sorted(div_rows) + [h]
    col_splits = [-1] + sorted(div_cols) + [w]

    panels: list[Grid] = []
    for i in range(len(row_splits) - 1):
        r1, r2 = row_splits[i] + 1, row_splits[i + 1]
        if r2 <= r1:
            continue
        for j in range(len(col_splits) - 1):
            c1, c2 = col_splits[j] + 1, col_splits[j + 1]
            if c2 <= c1:
                continue
            panels.append(grid[r1:r2, c1:c2].copy())

    return panels


def op_panels_overlay(panels: list[Grid], bg: int = 0) -> Grid:
    """Superpose panels: later non-background pixels take precedence."""
    if not panels:
        return np.zeros((1, 1), dtype=int)
    h, w = panels[0].shape
    if any(p.shape != (h, w) for p in panels):
        return panels[0].copy()

    res = np.full((h, w), bg, dtype=int)
    for p in panels:
        mask = p != bg
        res[mask] = p[mask]
    return res


def op_panels_or(panels: list[Grid], bg: int = 0) -> Grid:
    return op_panels_overlay(panels, bg=bg)


def op_panels_and(panels: list[Grid], bg: int = 0) -> Grid:
    """Pixel-wise intersection: pixels non-bg across all panels."""
    if not panels:
        return np.zeros((1, 1), dtype=int)
    h, w = panels[0].shape
    if any(p.shape != (h, w) for p in panels):
        return panels[0].copy()

    common_mask = np.ones((h, w), dtype=bool)
    for p in panels:
        common_mask &= p != bg

    res = np.full((h, w), bg, dtype=int)
    res[common_mask] = panels[0][common_mask]
    return res


def op_panels_xor(panels: list[Grid], bg: int = 0) -> Grid:
    """Pixel-wise symmetric difference: pixels present in odd number of panels."""
    if not panels:
        return np.zeros((1, 1), dtype=int)
    h, w = panels[0].shape
    if any(p.shape != (h, w) for p in panels):
        return panels[0].copy()

    counts = np.zeros((h, w), dtype=int)
    sample_val = np.full((h, w), bg, dtype=int)
    for p in panels:
        mask = p != bg
        counts[mask] += 1
        sample_val[mask] = p[mask]

    res = np.full((h, w), bg, dtype=int)
    xor_mask = counts % 2 == 1
    res[xor_mask] = sample_val[xor_mask]
    return res


def op_panel_intersection_recolor(panels: list[Grid], fill_color: int, bg: int = 0) -> Grid:
    """Compute Boolean intersection across panels and recolor matching pixels."""
    if not panels:
        return np.zeros((1, 1), dtype=int)
    h, w = panels[0].shape
    if any(p.shape != (h, w) for p in panels):
        return panels[0].copy()
    mask = np.ones((h, w), dtype=bool)
    for p in panels:
        mask &= p != bg
    res = np.full((h, w), bg, dtype=int)
    res[mask] = fill_color
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 4. Topological Infilling & Morphological Dynamics
# ─────────────────────────────────────────────────────────────────────────────


def op_fill_enclosed_voids(grid: Grid, fill_color: int, connectivity: int = 8) -> Grid:
    """Fill all enclosed frame voids with fill_color."""
    res = grid.copy()
    bg = detect_background_color(grid)
    ents = segment_entities(grid, bg_color=bg, connectivity=connectivity)
    for e in ents:
        if e.is_frame:
            for r, c in e.enclosed_coords:
                res[r, c] = fill_color
    return res


def op_connect_matching_dots(grid: Grid) -> Grid:
    """Connect pairs of identical color dots with straight orthogonal segments."""
    res = grid.copy()
    bg = detect_background_color(grid)
    cols = set(np.unique(grid)) - {bg}
    for c in cols:
        pts = np.argwhere(grid == c)
        if len(pts) == 2:
            r1, c1 = pts[0]
            r2, c2 = pts[1]
            if r1 == r2:
                for col_idx in range(min(c1, c2), max(c1, c2) + 1):
                    res[r1, col_idx] = c
            elif c1 == c2:
                for row_idx in range(min(r1, r2), max(r1, r2) + 1):
                    res[row_idx, c1] = c
    return res


def op_connect_matching_dots_diagonal(grid: Grid) -> Grid:
    """Connect pairs of identical color dots with diagonal segments if 45-degree aligned."""
    res = grid.copy()
    bg = detect_background_color(grid)
    cols = set(np.unique(grid)) - {bg}
    for c in cols:
        pts = np.argwhere(grid == c)
        if len(pts) == 2:
            r1, c1 = pts[0]
            r2, c2 = pts[1]
            dr = r2 - r1
            dc = c2 - c1
            if abs(dr) == abs(dc) and dr != 0:
                step_r = 1 if dr > 0 else -1
                step_c = 1 if dc > 0 else -1
                steps = abs(dr)
                for s in range(steps + 1):
                    res[r1 + s * step_r, c1 + s * step_c] = c
    return res


def op_gravity(grid: Grid, direction: str = "down") -> Grid:
    """Settle non-zero elements along a cardinal axis."""
    res = grid.copy()
    h, w = res.shape
    if direction == "down":
        for c in range(w):
            col = res[:, c]
            nz = [v for v in col if v != 0]
            res[:, c] = [0] * (h - len(nz)) + nz
    elif direction == "up":
        for c in range(w):
            col = res[:, c]
            nz = [v for v in col if v != 0]
            res[:, c] = nz + [0] * (h - len(nz))
    elif direction == "right":
        for r in range(h):
            row = res[r, :]
            nz = [v for v in row if v != 0]
            res[r, :] = [0] * (w - len(nz)) + nz
    elif direction == "left":
        for r in range(h):
            row = res[r, :]
            nz = [v for v in row if v != 0]
            res[r, :] = nz + [0] * (w - len(nz))
    return res


def op_filter_keep_extreme(grid: Grid, extreme: str = "largest") -> Grid:
    """Keep only the largest or smallest entity; erase others to background."""
    res = grid.copy()
    bg = detect_background_color(grid)
    ents = segment_entities(grid, bg_color=bg)
    if not ents:
        return res

    target = (
        max(ents, key=lambda e: e.area) if extreme == "largest" else min(ents, key=lambda e: e.area)
    )
    for e in ents:
        if e != target:
            for r, c in e.coords:
                res[r, c] = bg
    return res


def op_translate(grid: Grid, dr: int, dc: int, bg_color: int | None = None) -> Grid:
    """Rigidly translate grid by (dr, dc) with background padding."""
    if bg_color is None:
        bg_color = detect_background_color(grid)
    h, w = grid.shape
    res = np.full((h, w), bg_color, dtype=int)
    for r in range(h):
        for c in range(w):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                res[nr, nc] = grid[r, c]
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 5. Scaling, Periodic & Reflected Tiling
# ─────────────────────────────────────────────────────────────────────────────


def op_scale_kronecker(grid: Grid, scale_r: int, scale_c: int) -> Grid:
    """Kronecker block scaling: each pixel expands into a scale_r x scale_c block."""
    return np.kron(grid, np.ones((scale_r, scale_c), dtype=int)).copy()


def op_kronecker_mask(grid: Grid) -> Grid:
    """Fractal self-expansion: mask(grid != bg) ⊗ grid."""
    bg = detect_background_color(grid)
    mask = (grid != bg).astype(int)
    return np.kron(mask, grid).copy()


def op_tile_periodic(grid: Grid, reps_r: int, reps_c: int) -> Grid:
    """Tile grid periodically reps_r times vertically and reps_c horizontally."""
    return np.tile(grid, (reps_r, reps_c)).copy()


def op_tile_reflected(grid: Grid) -> Grid:
    """Reflected 2x2 tiling: top-left (orig), top-right (fliplr), bottom-left (flipud), bottom-right (rot180)."""
    top = np.hstack([grid, np.fliplr(grid)])
    bottom = np.hstack([np.flipud(grid), np.rot90(grid, 2)])
    return np.vstack([top, bottom]).copy()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Algebraic Color Mapping, Rank Coloring, Modular Tiling & Contact Slide
# ─────────────────────────────────────────────────────────────────────────────


def op_apply_color_map(grid: Grid, mapping: dict[int, int]) -> Grid:
    """Apply a 1-to-1 or many-to-one discrete color mapping."""
    res = grid.copy()
    for src, dst in mapping.items():
        res[grid == src] = dst
    return res


def op_recolor_by_rank(grid: Grid, palette: list[int], rank_by: str = "height") -> Grid:
    """Recolor entities/bars according to their rank order of height or area."""
    res = grid.copy()
    bg = detect_background_color(grid)
    ents = segment_entities(grid, bg_color=bg)
    if not ents:
        return res
    if rank_by == "height":
        ents.sort(key=lambda e: (e.height, e.area), reverse=True)
    elif rank_by == "width":
        ents.sort(key=lambda e: (e.width, e.area), reverse=True)
    else:
        ents.sort(key=lambda e: e.area, reverse=True)
    for rank, e in enumerate(ents):
        if rank < len(palette):
            target_c = palette[rank]
            for r, c in e.coords:
                res[r, c] = target_c
    return res


def op_tile_modular(grid: Grid, period: int = 3, mode: str = "diagonal") -> Grid:
    """Tessellate canvas by modular coordinate repetition from non-zero seed pattern."""
    h, w = grid.shape
    res = np.zeros((h, w), dtype=int)
    bg = detect_background_color(grid)
    pts = np.argwhere(grid != bg)
    if len(pts) == 0:
        return grid.copy()
    if mode == "diagonal":
        val_map: dict[int, int] = {}
        for r, c in pts:
            val_map[(r + c) % period] = int(grid[r, c])
        for r in range(h):
            for c in range(w):
                idx = (r + c) % period
                if idx in val_map:
                    res[r, c] = val_map[idx]
    else:
        val_map_cart: dict[tuple[int, int], int] = {}
        for r, c in pts:
            val_map_cart[(r % period, c % period)] = int(grid[r, c])
        for r in range(h):
            for c in range(w):
                k = (r % period, c % period)
                if k in val_map_cart:
                    res[r, c] = val_map_cart[k]
    return res


def op_slide_until_contact(grid: Grid, direction: str = "down") -> Grid:
    """Slide moving entities until they contact stationary obstacles or boundary."""
    res = grid.copy()
    h, w = res.shape
    bg = detect_background_color(grid)
    ents = segment_entities(grid, bg_color=bg)
    if len(ents) < 2:
        return op_gravity(grid, direction=direction)
    if direction == "down":
        ents.sort(key=lambda e: e.max_r)
        obstacle_coords = set(ents[-1].coords)
        for e in ents[:-1]:
            curr_coords = set(e.coords)
            dr = 0
            while True:
                next_coords = {(r + dr + 1, c) for r, c in curr_coords}
                if any(r >= h for r, _ in next_coords):
                    break
                if next_coords & obstacle_coords:
                    break
                dr += 1
            if dr > 0:
                for r, c in curr_coords:
                    res[r, c] = bg
                for r, c in curr_coords:
                    res[r + dr, c] = e.color
                obstacle_coords.update({(r + dr, c) for r, c in curr_coords})
    return res


def op_filter_keep_unique_color(grid: Grid) -> Grid:
    """Retain only entities whose color appears uniquely among all entities."""
    bg = detect_background_color(grid)
    ents = segment_entities(grid, bg_color=bg)
    if not ents:
        return grid.copy()
    color_counts: dict[int, int] = {}
    for e in ents:
        color_counts[e.color] = color_counts.get(e.color, 0) + 1
    unique_colors = {c for c, cnt in color_counts.items() if cnt == 1}
    res = np.full_like(grid, bg)
    for e in ents:
        if e.color in unique_colors:
            for r, c in e.coords:
                res[r, c] = e.color
    return res


def op_filter_keep_extreme_coord(grid: Grid, extreme: str = "top") -> Grid:
    """Retain only the entity located at the extreme coordinate."""
    bg = detect_background_color(grid)
    ents = segment_entities(grid, bg_color=bg)
    if not ents:
        return grid.copy()
    if extreme == "top":
        target = min(ents, key=lambda e: e.min_r)
    elif extreme == "bottom":
        target = max(ents, key=lambda e: e.max_r)
    elif extreme == "left":
        target = min(ents, key=lambda e: e.min_c)
    elif extreme == "right":
        target = max(ents, key=lambda e: e.max_c)
    else:
        target = ents[0]
    res = np.full_like(grid, bg)
    for r, c in target.coords:
        res[r, c] = target.color
    return res


def op_filter_keep_framed(grid: Grid) -> Grid:
    """Retain only entities that form hollow frames (contain internal voids)."""
    bg = detect_background_color(grid)
    ents = segment_entities(grid, bg_color=bg)
    res = np.full_like(grid, bg)
    for e in ents:
        # Check if entity bounding box contains background pixels that don't reach canvas edge
        rmin, rmax, cmin, cmax = e.min_r, e.max_r, e.min_c, e.max_c
        if rmax - rmin >= 2 and cmax - cmin >= 2:
            sub = grid[rmin : rmax + 1, cmin : cmax + 1]
            sub_holes = op_fill_enclosed_voids(sub, fill_color=e.color, connectivity=4)
            if not np.array_equal(sub, sub_holes):
                for r, c in e.coords:
                    res[r, c] = e.color
    return res


def op_filter_keep_solid(grid: Grid) -> Grid:
    """Retain only solid entities (no enclosed interior voids)."""
    bg = detect_background_color(grid)
    ents = segment_entities(grid, bg_color=bg)
    res = np.full_like(grid, bg)
    for e in ents:
        rmin, rmax, cmin, cmax = e.min_r, e.max_r, e.min_c, e.max_c
        if rmax - rmin < 2 or cmax - cmin < 2:
            for r, c in e.coords:
                res[r, c] = e.color
        else:
            sub = grid[rmin : rmax + 1, cmin : cmax + 1]
            sub_holes = op_fill_enclosed_voids(sub, fill_color=e.color, connectivity=4)
            if np.array_equal(sub, sub_holes):
                for r, c in e.coords:
                    res[r, c] = e.color
    return res


def op_raycast_cardinal(
    grid: Grid, directions: tuple[str, ...] = ("up", "down", "left", "right")
) -> Grid:
    """Emit colored rays from foreground pixels along cardinal directions."""
    res = grid.copy()
    h, w = res.shape
    bg = detect_background_color(grid)
    pts = np.argwhere(grid != bg)
    for r0, c0 in pts:
        color = int(grid[r0, c0])
        if "up" in directions:
            for r in range(r0 - 1, -1, -1):
                if res[r, c0] != bg:
                    break
                res[r, c0] = color
        if "down" in directions:
            for r in range(r0 + 1, h):
                if res[r, c0] != bg:
                    break
                res[r, c0] = color
        if "left" in directions:
            for c in range(c0 - 1, -1, -1):
                if res[r0, c] != bg:
                    break
                res[r0, c] = color
        if "right" in directions:
            for c in range(c0 + 1, w):
                if res[r0, c] != bg:
                    break
                res[r0, c] = color
    return res


def op_raycast_diagonal(grid: Grid) -> Grid:
    """Emit colored rays from foreground pixels along 45-degree diagonal directions."""
    res = grid.copy()
    h, w = res.shape
    bg = detect_background_color(grid)
    pts = np.argwhere(grid != bg)
    for r0, c0 in pts:
        color = int(grid[r0, c0])
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            r, c = r0 + dr, c0 + dc
            while 0 <= r < h and 0 <= c < w:
                if res[r, c] != bg:
                    break
                res[r, c] = color
                r += dr
                c += dc
    return res


def op_extend_lines(grid: Grid, axis: str = "both") -> Grid:
    """Extend straight horizontal or vertical lines across canvas."""
    res = grid.copy()
    h, w = res.shape
    bg = detect_background_color(grid)

    # Horizontal lines of >= 2 pixels
    if axis in ("both", "h"):
        for r in range(h):
            row = grid[r]
            colors = set(row[row != bg])
            for col in colors:
                cols = np.where(row == col)[0]
                if len(cols) >= 2 and np.max(np.diff(cols)) == 1:
                    res[r, :] = col

    # Vertical lines of >= 2 pixels
    if axis in ("both", "v"):
        for c in range(w):
            col_arr = grid[:, c]
            colors = set(col_arr[col_arr != bg])
            for col in colors:
                rows = np.where(col_arr == col)[0]
                if len(rows) >= 2 and np.max(np.diff(rows)) == 1:
                    res[:, c] = col

    return res


def op_extract_perimeter_only(grid: Grid) -> Grid:
    """Retain only the boundary perimeter of foreground entities, clearing interior."""
    bg = detect_background_color(grid)
    res = grid.copy()
    h, w = res.shape
    fg_mask = grid != bg
    for r in range(h):
        for c in range(w):
            if not fg_mask[r, c]:
                continue
            # Interior pixel has all 4 cardinal neighbors in foreground
            is_interior = (
                r > 0
                and r < h - 1
                and c > 0
                and c < w - 1
                and fg_mask[r - 1, c]
                and fg_mask[r + 1, c]
                and fg_mask[r, c - 1]
                and fg_mask[r, c + 1]
            )
            if is_interior:
                res[r, c] = bg
    return res


def op_recolor_enclosed_cavities(
    grid: Grid, fill_color: int, border_color: int | None = None
) -> Grid:
    """Recolor internal cavities enclosed by a border of specific or any color."""
    bg = detect_background_color(grid)
    filled = op_fill_enclosed_voids(grid, fill_color=fill_color, connectivity=4)
    # Target only the newly filled void pixels
    void_mask = (filled == fill_color) & (grid == bg)
    res = grid.copy()
    res[void_mask] = fill_color
    return res


def op_sort_and_pack_entities(
    grid: Grid, sort_key: str = "area", direction: str = "horizontal"
) -> Grid:
    """Extract individual entities, sort them, and tightly pack them into a contiguous canvas."""
    bg = detect_background_color(grid)
    ents = segment_entities(grid, bg_color=bg)
    if not ents:
        return grid.copy()
    if sort_key == "area":
        ents.sort(key=lambda e: e.area, reverse=True)
    elif sort_key == "color":
        ents.sort(key=lambda e: e.color)
    elif sort_key == "height":
        ents.sort(key=lambda e: e.height, reverse=True)
    elif sort_key == "width":
        ents.sort(key=lambda e: e.width, reverse=True)

    crops = [grid[e.min_r : e.max_r + 1, e.min_c : e.max_c + 1] for e in ents]
    if direction == "horizontal":
        max_h = max(c.shape[0] for c in crops)
        total_w = sum(c.shape[1] for c in crops)
        packed = np.full((max_h, total_w), bg, dtype=int)
        curr_c = 0
        for c_grid in crops:
            ch, cw = c_grid.shape
            packed[:ch, curr_c : curr_c + cw] = c_grid
            curr_c += cw
        return packed
    else:
        total_h = sum(c.shape[0] for c in crops)
        max_w = max(c.shape[1] for c in crops)
        packed = np.full((total_h, max_w), bg, dtype=int)
        curr_r = 0
        for c_grid in crops:
            ch, cw = c_grid.shape
            packed[curr_r : curr_r + ch, :cw] = c_grid
            curr_r += ch
        return packed


# ─────────────────────────────────────────────────────────────────────────────
# 11. Milestone 2: Multi-Layer Cut-Set Decomposition & Layer Transformations
# ─────────────────────────────────────────────────────────────────────────────


def op_extract_layers(grid: Grid) -> tuple[Grid, Grid, Grid]:
    """Decompose grid into 3 orthogonal layers: background, frame/barrier, and entity."""
    bg = detect_background_color(grid)
    h, w = grid.shape
    bg_layer = np.full((h, w), bg, dtype=int)
    frame_layer = np.full((h, w), bg, dtype=int)
    entity_layer = np.full((h, w), bg, dtype=int)

    ents = segment_entities(grid, bg_color=bg, connectivity=4)
    for ent in ents:
        # Enclosing frame or barrier spanning >= 60% of canvas dimension
        if ent.is_frame or ent.height >= max(3, int(h * 0.6)) or ent.width >= max(3, int(w * 0.6)):
            for r, c in ent.coords:
                frame_layer[r, c] = ent.color
        else:
            for r, c in ent.coords:
                entity_layer[r, c] = ent.color

    return bg_layer, frame_layer, entity_layer


def op_layer_compose(bg_layer: Grid, frame_layer: Grid, entity_layer: Grid) -> Grid:
    """Composite layers with priority: entity > frame > background."""
    bg = detect_background_color(bg_layer)
    result = bg_layer.copy()

    frame_mask = frame_layer != bg
    result[frame_mask] = frame_layer[frame_mask]

    entity_mask = entity_layer != bg
    result[entity_mask] = entity_layer[entity_mask]

    return result


def op_transform_layer(grid: Grid, target_layer: str, transform_fn: Callable[[Grid], Grid]) -> Grid:
    """Transform only the targeted layer while preserving static layers."""
    bg_layer, frame_layer, entity_layer = op_extract_layers(grid)
    if target_layer == "entity":
        transformed = transform_fn(entity_layer)
        return op_layer_compose(bg_layer, frame_layer, transformed)
    elif target_layer == "frame":
        transformed = transform_fn(frame_layer)
        return op_layer_compose(bg_layer, transformed, entity_layer)
    else:
        return transform_fn(grid)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Milestone 3: Cellular Automata, Spatial Grammar & Symmetry Completion
# ─────────────────────────────────────────────────────────────────────────────


def op_cellular_grow(
    grid: Grid,
    seed_color: int | None = None,
    target_color: int | None = None,
    steps: int = 1,
    diagonals: bool = False,
) -> Grid:
    """Frontier diffusion: expand seed colored cells outwards into background cells."""
    res = grid.copy()
    bg = detect_background_color(grid)
    h, w = res.shape
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonals:
        dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for _ in range(steps):
        next_grid = res.copy()
        for r in range(h):
            for c in range(w):
                val = int(res[r, c])
                if seed_color is not None and val != seed_color:
                    continue
                if val == bg:
                    continue
                fill_c = target_color if target_color is not None else val
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and res[nr, nc] == bg:
                        next_grid[nr, nc] = fill_c
        res = next_grid
    return res


def op_complete_symmetry(grid: Grid, axis: str = "both") -> Grid:
    """Auto-complete reflective symmetry across the specified axis into background cells."""
    res = grid.copy()
    bg = detect_background_color(grid)
    h, w = res.shape

    if axis in ("horizontal", "both"):
        for r in range(h):
            for c in range(w):
                mc = w - 1 - c
                if res[r, c] == bg and res[r, mc] != bg:
                    res[r, c] = res[r, mc]

    if axis in ("vertical", "both"):
        for r in range(h):
            for c in range(w):
                mr = h - 1 - r
                if res[r, c] == bg and res[mr, c] != bg:
                    res[r, c] = res[mr, c]

    if axis in ("main_diag", "both") and h == w:
        for r in range(h):
            for c in range(w):
                if res[r, c] == bg and res[c, r] != bg:
                    res[r, c] = res[c, r]

    return res


def op_tessellate_periodic(grid: Grid, r_factor: int = 2, c_factor: int = 2) -> Grid:
    """Tile the lattice periodically with repetition factors."""
    return np.tile(grid, (r_factor, c_factor))


def op_maze_route(
    grid: Grid,
    start_color: int,
    target_color: int,
    wall_color: int | None = None,
    path_color: int | None = None,
) -> Grid:
    """Find shortest path connecting start_color to target_color avoiding obstacles."""
    res = grid.copy()
    bg = detect_background_color(grid)
    h, w = res.shape
    starts = [tuple(p) for p in np.argwhere(grid == start_color)]
    targets = [tuple(p) for p in np.argwhere(grid == target_color)]
    if not starts or not targets:
        return res

    draw_c = path_color if path_color is not None else start_color
    target_set = set(targets)
    queue: deque[tuple[int, int]] = deque([starts[0]])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {starts[0]: None}
    visited: set[tuple[int, int]] = {starts[0]}
    found_target = None

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        curr = queue.popleft()
        if curr in target_set:
            found_target = curr
            break
        cr, cc = curr
        for dr, dc in dirs:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                val = int(grid[nr, nc])
                if wall_color is not None and val == wall_color:
                    continue
                if val == bg or (nr, nc) in target_set:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = curr
                    queue.append((nr, nc))

    if found_target:
        curr = parent[found_target]
        while curr and curr != starts[0]:
            res[curr[0], curr[1]] = draw_c
            curr = parent[curr]

    return res
