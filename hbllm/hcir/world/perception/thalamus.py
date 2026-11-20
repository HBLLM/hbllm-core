"""Sensory Thalamus Normalization & Figure-Ground Segregation."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _fast_unique_counts(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """O(N) bincount-based unique value & frequency extractor for bounded integer lattices."""
    if arr.size > 0 and np.issubdtype(arr.dtype, np.integer):
        amin, amax = int(arr.min()), int(arr.max())
        if 0 <= amin and amax < 256:
            bc = np.bincount(arr.ravel(), minlength=amax + 1)
            nz = np.flatnonzero(bc)
            return nz, bc[nz]
    return np.unique(arr, return_counts=True)


class SensoryThalamus:
    """Universal sensory normalization converting multimodal signals to 2D spatial lattices."""

    @staticmethod
    def normalize(raw: Any) -> np.ndarray:
        """Normalize arbitrary sensory observations (camera, depth, lidar, 2D grid) into a spatial 2D array.

        Supports:
        - 2D integer grids (e.g. ARC, Roguelike, Atari RAM/grid)
        - 2D float arrays (e.g. Depth maps, LiDAR range grids) -> quantized to perceptual levels
        - 3D arrays (e.g. RGB camera (H, W, 3), LiDAR multi-channel, or temporal stack (T, H, W))
        - 3D Point Cloud (N, 3) (x, y, z) -> 2D Bird's-Eye View (BEV) occupancy grid
        - 1D LiDAR range arrays (N,) -> (1, N) range profile
        - Objects with .frame, .grid, .image, or .raw_data
        """
        if raw is None:
            return np.zeros((1, 1), dtype=int)

        if hasattr(raw, "raw_data") and raw.raw_data is not None:
            raw = raw.raw_data
        elif hasattr(raw, "frame") and raw.frame is not None:
            raw = raw.frame
        elif hasattr(raw, "grid") and raw.grid is not None:
            raw = raw.grid
        elif hasattr(raw, "image") and raw.image is not None:
            raw = raw.image

        if isinstance(raw, (list, tuple)):
            if len(raw) == 0:
                return np.zeros((1, 1), dtype=int)
            raw = np.asarray(raw)

        if not isinstance(raw, np.ndarray):
            try:
                raw = np.asarray(raw)
            except Exception:
                return np.zeros((1, 1), dtype=int)

        if raw.ndim == 3 and raw.shape[0] < min(raw.shape[1], raw.shape[2]):
            raw = raw[-1]

        if raw.ndim == 3 and raw.shape[2] in (1, 3, 4):
            if raw.shape[2] == 1:
                raw = raw[:, :, 0]
            elif raw.shape[2] in (3, 4):
                r = raw[:, :, 0].astype(float)
                g = raw[:, :, 1].astype(float)
                b = raw[:, :, 2].astype(float)
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                raw = np.digitize(lum, np.linspace(0, 255, 16)).astype(int)

        if raw.ndim == 2 and np.issubdtype(raw.dtype, np.floating):
            min_v, max_v = float(np.nanmin(raw)), float(np.nanmax(raw))
            if max_v > min_v:
                raw = np.digitize(raw, np.linspace(min_v, max_v, 16)).astype(int)
            else:
                raw = np.zeros(raw.shape, dtype=int)

        # 1D LiDAR range arrays (N,) -> (1, N) range profile
        if raw.ndim == 1:
            if np.issubdtype(raw.dtype, np.floating):
                min_v, max_v = float(np.nanmin(raw)), float(np.nanmax(raw))
                if max_v > min_v:
                    raw = np.digitize(raw, np.linspace(min_v, max_v, 16)).astype(int)
                else:
                    raw = np.zeros(raw.shape, dtype=int)
            raw = raw.reshape((1, -1))

        # 3D Point Cloud (N, 3) (x, y, z) -> 2D Bird's-Eye View (BEV) occupancy grid
        elif raw.ndim == 2 and raw.shape[1] == 3 and raw.shape[0] > 3:
            xs, ys, _zs = raw[:, 0], raw[:, 1], raw[:, 2]
            grid_res = 32
            min_x, max_x = float(np.min(xs)), float(np.max(xs))
            min_y, max_y = float(np.min(ys)), float(np.max(ys))
            if max_x > min_x and max_y > min_y:
                xi = np.clip(
                    np.digitize(xs, np.linspace(min_x, max_x, grid_res)) - 1, 0, grid_res - 1
                )
                yi = np.clip(
                    np.digitize(ys, np.linspace(min_y, max_y, grid_res)) - 1, 0, grid_res - 1
                )
                bev = np.zeros((grid_res, grid_res), dtype=int)
                bev[yi, xi] = 1
                raw = bev
            else:
                raw = np.zeros((grid_res, grid_res), dtype=int)

        if raw.ndim != 2:
            if raw.ndim > 2:
                raw = raw.reshape((raw.shape[0], -1))
            else:
                raw = np.zeros((1, 1), dtype=int)

        return raw.astype(int)


class FigureGroundSegregator:
    """Domain-agnostic background color estimation based on border dominance and frequency."""

    _BG_CACHE: OrderedDict[tuple[Any, ...], int] = OrderedDict()
    _BG_CACHE_MAX = 64

    @staticmethod
    def estimate_background(
        grid: np.ndarray,
        barrier_features: set[int] | None = None,
        avatar_features: set[int] | None = None,
        bg_feature: int = 0,
        walkable_features: set[int] | None = None,
    ) -> int:
        grid = SensoryThalamus.normalize(grid)
        barriers = barrier_features or set()
        av_feats = avatar_features or set()
        walkables = walkable_features or set()
        cache_key = (
            grid.shape,
            str(grid.dtype),
            frozenset(barriers),
            frozenset(av_feats),
            frozenset(walkables),
            grid.tobytes(),
        )
        cached = FigureGroundSegregator._BG_CACHE.get(cache_key)
        if cached is not None:
            FigureGroundSegregator._BG_CACHE.move_to_end(cache_key)
            return cached
        res = FigureGroundSegregator._estimate_background_uncached(
            grid, barriers, av_feats, bg_feature, walkables
        )
        FigureGroundSegregator._BG_CACHE[cache_key] = res
        if len(FigureGroundSegregator._BG_CACHE) > FigureGroundSegregator._BG_CACHE_MAX:
            FigureGroundSegregator._BG_CACHE.popitem(last=False)
        return res

    @staticmethod
    def _estimate_background_uncached(
        grid: np.ndarray,
        barriers: set[int],
        av_feats: set[int],
        bg_feature: int,
        walkables: set[int],
    ) -> int:
        H, W = grid.shape
        border_pixels = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
        vals, counts = _fast_unique_counts(border_pixels)
        top_val = int(vals[np.argmax(counts)])

        g_vals, g_counts = _fast_unique_counts(grid)
        g_dominant = int(g_vals[np.argmax(g_counts)])

        # 1. Empirically Verified Walkable Ground:
        # If the organism has traversed a feature present in the grid, that navigable ground
        # takes priority as the spatial background canvas over non-walkables.
        if walkables:
            valid_walkables = [w for w in walkables if w not in av_feats and w not in barriers]
            if valid_walkables:
                walk_counts = {
                    w: int(np.sum(grid == w)) for w in valid_walkables if np.any(grid == w)
                }
                if walk_counts:
                    best_w, best_cnt = max(walk_counts.items(), key=lambda item: item[1])
                    if top_val not in valid_walkables and (
                        top_val in barriers
                        or top_val != g_dominant
                        or best_cnt >= max(3, int(0.15 * grid.size))
                    ):
                        return best_w

        # 2. Gestalt Perimeter Enclosure vs Canvas Ground:
        # If the outer border is dominated by top_val (an enclosing bounding wall/frame),
        # verify if the interior canvas is dominated by a non-barrier/non-avatar floor feature.
        if H > 2 and W > 2:
            interior = grid[1:-1, 1:-1]
            int_vals, int_counts = _fast_unique_counts(interior)
            int_dominant = int(int_vals[np.argmax(int_counts)])
            int_dom_frac = float(np.max(int_counts)) / max(1, interior.size)
            if (
                top_val != g_dominant
                and g_dominant not in av_feats
                and g_dominant not in barriers
                and int_dominant == g_dominant
                and int_dom_frac > 0.40
            ):
                return g_dominant
            top_border_frac = float(np.max(counts)) / max(1, border_pixels.size)
            top_interior_frac = float(np.sum(interior == top_val)) / max(1, interior.size)
            int_border_frac = float(np.sum(border_pixels == int_dominant)) / max(
                1, border_pixels.size
            )
            if (
                int_dominant != top_val
                and int_dominant not in av_feats
                and int_dominant not in barriers
                and top_border_frac >= 0.60
                and top_interior_frac < 0.30
                and int_dom_frac >= 0.65
                and int_border_frac < 0.10
                and int(np.sum(grid == int_dominant)) >= 0.50 * int(np.sum(grid == top_val))
            ):
                return int_dominant

        if top_val in barriers:
            if top_val == g_dominant and g_counts[np.argmax(g_counts)] > 0.35 * grid.size:
                pass
            else:
                non_barriers = [
                    i
                    for i, v in enumerate(vals)
                    if int(v) not in barriers and int(v) not in av_feats
                ]
                if non_barriers:
                    return int(vals[non_barriers[np.argmax(counts[non_barriers])]])
                g_valid = [
                    i
                    for i, v in enumerate(g_vals)
                    if int(v) not in barriers and int(v) not in av_feats
                ]
                if g_valid:
                    return int(g_vals[g_valid[np.argmax(g_counts[g_valid])]])

        if av_feats and len(vals) > 1:
            other_idx = [i for i, v in enumerate(vals) if v not in av_feats and v not in barriers]
            if other_idx:
                return int(vals[other_idx[np.argmax(counts[other_idx])]])
        return top_val
