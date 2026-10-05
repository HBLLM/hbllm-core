"""Spatiotemporal Hazard Tracking & Phase Precession System.

Modeled on mammalian entorhinal grid cell phase precession and hippocampal spatiotemporal
projection:
1. Dynamic Sensory Differencing: Detects oscillating cells and moving entities across time steps.
2. Periodicity & Phase Estimation: Computes cyclic period T and phase offset for periodic hazards
   (blinking obstacles, oscillating lasers, patrolling entities).
3. Predictive Hazard Projection: Predicts future spatial hazards at (x, y, t mod T) to enable
   collision-free navigation and temporal waiting/hesitation impulses.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class DynamicCellPhase:
    """Estimated temporal phase model for a dynamic/oscillating spatial cell."""

    r: int
    c: int
    period: int
    cycle_values: list[int]
    hazardous_values: set[int]
    confidence: float = 1.0


class SpatiotemporalHazardTracker:
    """Biologically-inspired tracker for moving hazards and periodic spatial oscillations."""

    def __init__(self, history_len: int = 24, max_period: int = 8) -> None:
        self.history_len = history_len
        self.max_period = max_period
        self.grid_history: deque[np.ndarray] = deque(maxlen=history_len)
        self.step_history: deque[int] = deque(maxlen=history_len)
        self.periodic_cells: dict[tuple[int, int], DynamicCellPhase] = {}
        self.environmental_period: int = 1
        self.known_lethal_features: set[int] = set()

    def reset_episode(self) -> None:
        """Reset temporal observation history while preserving learned lethal feature identities."""
        self.grid_history.clear()
        self.step_history.clear()
        self.periodic_cells.clear()
        self.environmental_period = 1

    def register_lethal_feature(self, feature_id: int) -> None:
        """Register a feature value that resulted in avatar destruction/death upon contact."""
        self.known_lethal_features.add(int(feature_id))
        # Update existing periodic cells with new lethal feature
        for cell in self.periodic_cells.values():
            if int(feature_id) in cell.cycle_values:
                cell.hazardous_values.add(int(feature_id))

    def record_frame(
        self,
        step: int,
        grid: np.ndarray,
        background_feature: int = 0,
        avatar_features: set[int] | None = None,
    ) -> None:
        if not isinstance(grid, np.ndarray):
            if isinstance(grid, (list, tuple)) and len(grid) > 0:
                grid = grid[-1]
            grid = np.asarray(grid, dtype=int)
        if grid.ndim == 3 and len(grid) > 0:
            grid = grid[-1]
        self.grid_history.append(grid.copy())
        self.step_history.append(step)

        if len(self.grid_history) < 4:
            return

        H, W = grid.shape
        # Identify dynamic cells that fluctuated across history
        grids_arr = np.stack(list(self.grid_history), axis=0)  # Shape: (N, H, W)
        min_vals = np.min(grids_arr, axis=0)
        max_vals = np.max(grids_arr, axis=0)
        fluctuating_mask = min_vals != max_vals

        fluctuating_indices = np.argwhere(fluctuating_mask)
        if len(fluctuating_indices) == 0:
            self.periodic_cells.clear()
            self.environmental_period = 1
            return

        discovered_periods: list[int] = []
        av_set = avatar_features or set()

        for r, c in fluctuating_indices:
            r_idx, c_idx = int(r), int(c)
            series = [int(g[r_idx, c_idx]) for g in self.grid_history]

            best_period = self._estimate_period(series)
            if best_period is not None and best_period >= 2:
                # Cycle values over the period
                cycle = series[-best_period:]
                # Determine which values in this cycle are hazardous:
                # Known lethal features or periodic non-background environmental features (excluding avatar)
                haz_vals = {
                    v
                    for v in cycle
                    if v in self.known_lethal_features
                    or (
                        v != background_feature
                        and v not in av_set
                        and not self.known_lethal_features
                    )
                }
                if haz_vals:
                    self.periodic_cells[(r_idx, c_idx)] = DynamicCellPhase(
                        r=r_idx,
                        c=c_idx,
                        period=best_period,
                        cycle_values=cycle,
                        hazardous_values=haz_vals,
                    )
                    discovered_periods.append(best_period)
                else:
                    self.periodic_cells.pop((r_idx, c_idx), None)
            else:
                self.periodic_cells.pop((r_idx, c_idx), None)

        # Compute environmental LCM period
        if discovered_periods:
            self.environmental_period = self._compute_lcm(discovered_periods)
        else:
            self.environmental_period = 1

    def _estimate_period(self, series: list[int]) -> int | None:
        """Estimate repeating periodicity T using backward autocorrelation."""
        n = len(series)
        for T in range(2, min(self.max_period + 1, n // 2 + 1)):
            # Check if series[i] == series[i - T] for recent history
            matches = sum(1 for i in range(T, n) if series[i] == series[i - T])
            total = n - T
            if total > 0 and (matches / total) >= 0.85:
                return T
        return None

    @staticmethod
    def _compute_lcm(numbers: Sequence[int]) -> int:
        """Compute the least common multiple of a sequence of integers (capped at 24)."""
        lcm = 1
        for num in set(numbers):
            if num <= 0:
                continue
            lcm = (lcm * num) // math.gcd(lcm, num)
            if lcm > 24:
                return 24
        return max(1, lcm)

    def is_hazard_at(
        self,
        r: int,
        c: int,
        future_relative_step: int,
        background_feature: int = 0,
        avatar_features: set[int] | None = None,
    ) -> bool:
        """Predict whether coordinate (r, c) will be lethal or impassable at t_current + future_relative_step."""
        if (r, c) not in self.periodic_cells:
            return False

        cell_phase = self.periodic_cells[(r, c)]
        T = cell_phase.period
        # Predicted index in cycle_values
        idx = (future_relative_step - 1) % T
        predicted_val = cell_phase.cycle_values[idx]
        if avatar_features and predicted_val in avatar_features:
            return False
        return predicted_val in cell_phase.hazardous_values

    def get_hazard_schedule(
        self,
        horizon: int = 24,
        background_feature: int = 0,
    ) -> dict[int, set[tuple[int, int]]]:
        """Generate a schedule mapping relative future step -> set of hazardous coordinates."""
        schedule: dict[int, set[tuple[int, int]]] = {}
        for dt in range(horizon + 1):
            haz_set: set[tuple[int, int]] = set()
            for (r, c), cell_phase in self.periodic_cells.items():
                T = cell_phase.period
                idx = (dt - 1) % T if dt > 0 else -1
                val = cell_phase.cycle_values[idx]
                if val in cell_phase.hazardous_values:
                    haz_set.add((r, c))
            if haz_set:
                schedule[dt] = haz_set
        return schedule
