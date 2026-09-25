"""Reusable Sub-Skills and Primitive Operators for HCIR Skills.

Consolidates common perceptual, geometric, pathfinding, and actuation routines
to eliminate duplication across game-specific skill induction modules.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ColorCluster:
    """Represents a connected or grouped cluster of identical color pixels."""

    color: int
    centroid: tuple[int, int]  # (cx, cy)
    bbox: tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    pixel_count: int
    points: list[tuple[int, int]]


class DiscreteVectorTranslator:
    """Translates continuous or lattice displacements into discrete ARC actions.

    ARC standard navigation actions:
    Action 1: UP    (dy < 0)
    Action 2: DOWN  (dy > 0)
    Action 3: LEFT  (dx < 0)
    Action 4: RIGHT (dx > 0)
    Action 5: NOOP / WAIT
    Action 6: CLICK / INTERACT
    """

    @staticmethod
    def delta_to_actions(
        dx: int, dy: int, step_size: int = 1, horizontal_first: bool = True
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Convert a (dx, dy) displacement into discrete movement action tuples."""
        actions: list[tuple[int, dict[str, int] | None]] = []
        num_y = abs(dy) // step_size if step_size > 0 else abs(dy)
        num_x = abs(dx) // step_size if step_size > 0 else abs(dx)

        x_moves = [(3, None)] * num_x if dx < 0 else ([(4, None)] * num_x if dx > 0 else [])
        y_moves = [(1, None)] * num_y if dy < 0 else ([(2, None)] * num_y if dy > 0 else [])

        if horizontal_first:
            actions.extend(x_moves)
            actions.extend(y_moves)
        else:
            actions.extend(y_moves)
            actions.extend(x_moves)

        return actions

    @classmethod
    def points_to_actions(
        cls,
        p1: tuple[int, int],
        p2: tuple[int, int],
        step_size: int = 1,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Convert displacement from point p1=(x1, y1) to p2=(x2, y2) into actions."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return cls.delta_to_actions(dx, dy, step_size=step_size)

    @classmethod
    def path_to_actions(
        cls,
        path: list[tuple[int, int]],
        step_size: int = 1,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Convert a sequence of waypoints into sequential navigation actions."""
        actions: list[tuple[int, dict[str, int] | None]] = []
        for i in range(len(path) - 1):
            actions.extend(cls.points_to_actions(path[i], path[i + 1], step_size=step_size))
        return actions


class PerceptualClusterDetector:
    """Detects visual clusters, entities, and regions of interest on ARC grids."""

    @staticmethod
    def normalize_grid(grid: Any) -> np.ndarray:
        """Ensure grid is a 2D numpy array."""
        arr = np.asarray(grid)
        while arr.ndim > 2:
            arr = arr[0]
        return arr

    @classmethod
    def find_color_clusters(
        cls,
        grid: np.ndarray,
        color: int,
        roi: tuple[int, int, int, int] | None = None,
    ) -> list[ColorCluster]:
        """Find clusters of a specific color, optionally restricted to an ROI (x_min, y_min, x_max, y_max)."""
        arr = cls.normalize_grid(grid)
        H, W = arr.shape

        if roi is not None:
            x0, y0, x1, y1 = roi
            x0, x1 = max(0, x0), min(W, x1)
            y0, y1 = max(0, y0), min(H, y1)
            sub = arr[y0:y1, x0:x1]
            ys, xs = np.where(sub == color)
            xs = xs + x0
            ys = ys + y0
        else:
            ys, xs = np.where(arr == color)

        if len(xs) == 0:
            return []

        # Find connected components via 4-connectivity
        pts = set(zip(xs.tolist(), ys.tolist(), strict=False))
        visited: set[tuple[int, int]] = set()
        clusters: list[ColorCluster] = []

        for pt in pts:
            if pt in visited:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([pt])
            visited.add(pt)
            while q:
                curr = q.popleft()
                comp.append(curr)
                cx, cy = curr
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nbr = (cx + dx, cy + dy)
                    if nbr in pts and nbr not in visited:
                        visited.add(nbr)
                        q.append(nbr)

            comp_xs = [p[0] for p in comp]
            comp_ys = [p[1] for p in comp]
            centroid = (int(np.mean(comp_xs)), int(np.mean(comp_ys)))
            bbox = (min(comp_xs), min(comp_ys), max(comp_xs), max(comp_ys))
            clusters.append(
                ColorCluster(
                    color=color,
                    centroid=centroid,
                    bbox=bbox,
                    pixel_count=len(comp),
                    points=comp,
                )
            )

        return clusters

    @classmethod
    def find_centroid(
        cls,
        grid: np.ndarray,
        color: int,
        roi: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, int] | None:
        """Find the overall centroid of all pixels of a given color."""
        arr = cls.normalize_grid(grid)
        H, W = arr.shape
        if roi is not None:
            x0, y0, x1, y1 = roi
            x0, x1 = max(0, x0), min(W, x1)
            y0, y1 = max(0, y0), min(H, y1)
            sub = arr[y0:y1, x0:x1]
            ys, xs = np.where(sub == color)
            xs = xs + x0
            ys = ys + y0
        else:
            ys, xs = np.where(arr == color)

        if len(xs) == 0:
            return None
        return (int(round(float(np.mean(xs)))), int(round(float(np.mean(ys)))))


class LatticeNavigator:
    """General 4-connected grid pathfinding over walkable graphs."""

    @staticmethod
    def bfs_path(
        start: tuple[int, int],
        goal: tuple[int, int],
        walkable: set[tuple[int, int]],
        step_size: int = 1,
    ) -> list[tuple[int, int]] | None:
        """Find the shortest path from start to goal on a discrete walkable lattice."""
        if start == goal:
            return [start]
        if goal not in walkable:
            return None

        queue: deque[list[tuple[int, int]]] = deque([[start]])
        visited: set[tuple[int, int]] = {start}
        moves = [
            (0, -step_size),
            (0, step_size),
            (-step_size, 0),
            (step_size, 0),
        ]

        while queue:
            path = queue.popleft()
            cx, cy = path[-1]

            for dx, dy in moves:
                nxt = (cx + dx, cy + dy)
                if nxt == goal:
                    return path + [nxt]
                if nxt in walkable and nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])

        return None


class RemoteActuator:
    """Generates standard Action 6 remote click / interaction tuples."""

    @staticmethod
    def click(x: int, y: int) -> tuple[int, dict[str, int]]:
        """Create an Action 6 click tuple for display coordinates (x, y)."""
        return (6, {"x": int(x), "y": int(y)})

    @classmethod
    def click_color(
        cls,
        grid: np.ndarray,
        color: int,
        roi: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, dict[str, int]] | None:
        """Find the centroid of a color in grid (optionally in ROI) and return a click action."""
        centroid = PerceptualClusterDetector.find_centroid(grid, color, roi=roi)
        if centroid is None:
            return None
        return cls.click(centroid[0], centroid[1])


class LatticeQuantizer:
    """Quantizes high-resolution pixel matrices into discrete macro-grid lattices."""

    @staticmethod
    def downsample(
        grid: np.ndarray,
        step: int | tuple[int, int],
        offset: int | tuple[int, int] = 0,
    ) -> np.ndarray:
        """Subsamples a grid at regular stride intervals."""
        arr = PerceptualClusterDetector.normalize_grid(grid)
        sy, sx = (step, step) if isinstance(step, int) else step
        oy, ox = (offset, offset) if isinstance(offset, int) else offset
        return arr[oy::sy, ox::sx]

    @staticmethod
    def block_majority(
        grid: np.ndarray,
        block_size: int | tuple[int, int],
        background_color: int = 0,
    ) -> np.ndarray:
        """Downsamples grid by taking the most prominent foreground color in each block."""
        arr = PerceptualClusterDetector.normalize_grid(grid)
        H, W = arr.shape
        bh, bw = (block_size, block_size) if isinstance(block_size, int) else block_size
        macro_h, macro_w = H // bh, W // bw
        result = np.full((macro_h, macro_w), background_color, dtype=arr.dtype)

        for my in range(macro_h):
            for mx in range(macro_w):
                cell = arr[my * bh : (my + 1) * bh, mx * bw : (mx + 1) * bw]
                fg = cell[cell != background_color]
                if len(fg) > 0:
                    vals, counts = np.unique(fg, return_counts=True)
                    result[my, mx] = vals[np.argmax(counts)]
                else:
                    result[my, mx] = background_color
        return result


class TemporalSyncPlanner:
    """Coordinates multi-actor sequences, ghost replays, and temporal delays."""

    @staticmethod
    def wait(steps: int, noop_action: int = 5) -> list[tuple[int, dict[str, int] | None]]:
        """Generate a series of NOOP / wait actions."""
        return [(noop_action, None)] * max(0, steps)

    @classmethod
    def pad_to_length(
        cls,
        actions: list[tuple[int, dict[str, int] | None]],
        target_length: int,
        noop_action: int = 5,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Pads an action sequence with NOOPs to match a target step count."""
        needed = target_length - len(actions)
        if needed > 0:
            return list(actions) + cls.wait(needed, noop_action=noop_action)
        return list(actions)

    @classmethod
    def synchronize_arrival(
        cls,
        travel_actions: list[tuple[int, dict[str, int] | None]],
        arrival_step: int,
        noop_action: int = 5,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Pads prefix with NOOPs so that the final action coincides with arrival_step."""
        pad = arrival_step - len(travel_actions)
        if pad > 0:
            return cls.wait(pad, noop_action=noop_action) + list(travel_actions)
        return list(travel_actions)


class Raycaster2D:
    """Traces 2D discrete orthogonal rays and reflections across grid lattices."""

    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @staticmethod
    def trace_ray(
        origin: tuple[int, int],
        direction: tuple[int, int],
        grid: np.ndarray,
        obstacle_colors: set[int] | None = None,
        max_steps: int = 100,
    ) -> tuple[tuple[int, int], int | None, list[tuple[int, int]]]:
        """Traces a ray from origin until hitting an obstacle or grid boundary.

        Returns (hit_pos, hit_color_or_none, traversed_path).
        """
        arr = PerceptualClusterDetector.normalize_grid(grid)
        H, W = arr.shape
        x, y = origin
        dx, dy = direction
        path: list[tuple[int, int]] = []
        obs = obstacle_colors if obstacle_colors is not None else set()

        for _ in range(max_steps):
            x += dx
            y += dy
            if not (0 <= x < W and 0 <= y < H):
                return ((x - dx, y - dy), None, path)
            val = int(arr[y, x])
            path.append((x, y))
            if val in obs:
                return ((x, y), val, path)

        return ((x, y), None, path)


class BlockPushPlanner:
    """Plans kinematic push alignments to push target blocks across grid lattices."""

    @staticmethod
    def get_push_stance(
        block_pos: tuple[int, int],
        push_dir: tuple[int, int],
        block_size: int = 1,
    ) -> tuple[int, int]:
        """Returns the coordinate the agent must occupy to push the block in push_dir."""
        bx, by = block_pos
        pdx, pdy = push_dir
        return (bx - pdx * block_size, by - pdy * block_size)

    @classmethod
    def plan_push_maneuver(
        cls,
        agent_pos: tuple[int, int],
        block_pos: tuple[int, int],
        push_dir: tuple[int, int],
        push_distance: int = 1,
        walkable: set[tuple[int, int]] | None = None,
        step_size: int = 1,
    ) -> list[tuple[int, dict[str, int] | None]]:
        """Plans navigation to the push stance, followed by pushing the block."""
        stance = cls.get_push_stance(block_pos, push_dir, block_size=step_size)
        actions: list[tuple[int, dict[str, int] | None]] = []

        if walkable is not None:
            path = LatticeNavigator.bfs_path(agent_pos, stance, walkable, step_size=step_size)
            if path is not None:
                actions.extend(DiscreteVectorTranslator.path_to_actions(path, step_size=step_size))
            else:
                actions.extend(
                    DiscreteVectorTranslator.points_to_actions(
                        agent_pos, stance, step_size=step_size
                    )
                )
        else:
            actions.extend(
                DiscreteVectorTranslator.points_to_actions(agent_pos, stance, step_size=step_size)
            )

        pdx, pdy = push_dir
        push_actions = DiscreteVectorTranslator.delta_to_actions(
            pdx * push_distance, pdy * push_distance, step_size=step_size
        )
        actions.extend(push_actions)
        return actions


class GF2LinearSolver:
    """Solves systems of linear congruences A * x = b (mod 2) using Gaussian elimination."""

    @classmethod
    def solve(cls, A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        """Solve A * x = b (mod 2).

        Args:
            A: Binary matrix of shape (M, N) with values in {0, 1}.
            b: Binary vector of length M with values in {0, 1}.

        Returns:
            Binary solution vector x of length N, or None if system is inconsistent.
        """
        M, N = A.shape
        if M > 256 or N > 256:
            return None

        aug = np.zeros((M, N + 1), dtype=np.uint8)
        aug[:, :N] = (A % 2).astype(np.uint8)
        aug[:, N] = (b % 2).astype(np.uint8)

        row = 0
        pivot_cols: list[int] = []

        for col in range(N):
            if row >= M:
                break

            col_vals = aug[row:M, col]
            rel_piv = int(np.argmax(col_vals))
            if col_vals[rel_piv] == 0:
                continue

            pivot_row = row + rel_piv
            if pivot_row != row:
                aug[[row, pivot_row]] = aug[[pivot_row, row]]

            mask = aug[:, col] == 1
            mask[row] = False
            aug[mask] ^= aug[row]

            pivot_cols.append(col)
            row += 1

        for r in range(row, M):
            if aug[r, N] == 1:
                return None

        x = np.zeros(N, dtype=np.uint8)
        for r, col in enumerate(pivot_cols):
            x[col] = aug[r, N]

        return x
