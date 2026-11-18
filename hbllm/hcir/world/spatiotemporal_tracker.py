"""Spatiotemporal Hazard Tracking & Phase Precession System.

Biologically modeled on mammalian entorhinal grid cell phase precession and
hippocampal spatiotemporal projection:
1. Dynamic Sensory Differencing: Detects oscillating cells and moving entities across time steps.
2. Periodicity & Phase Estimation: Computes cyclic period T and phase offset for periodic hazards
   (blinking obstacles, oscillating lasers, patrolling entities) via autocorrelation.
3. Predictive Hazard Projection: Predicts future spatial hazards at (x, y, t mod T) to enable
   collision-free navigation and temporal waiting/hesitation impulses.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

# ── Domain 1: Entorhinal Phase Precession & Dynamic Hazard Tracking ───────────


@dataclass
class DynamicCellPhase:
    """Estimated temporal phase model for a dynamic/oscillating spatial cell."""

    r: int
    c: int
    period: int
    cycle_values: list[int]
    hazardous_values: set[int]
    confidence: float = 1.0

    def is_hazard_at_step(
        self,
        future_relative_step: int,
        safe_features: set[int] | None = None,
    ) -> bool:
        """Predict whether cell is hazardous at t_current + future_relative_step."""
        if self.period <= 0 or not self.cycle_values:
            return False
        idx = (future_relative_step - 1) % self.period if future_relative_step > 0 else -1
        val = self.cycle_values[idx]
        if safe_features and val in safe_features:
            return False
        return val in self.hazardous_values

    def to_causal_predicate(self) -> CausalPredicate:
        """Export dynamic cell phase model as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"cell_phase:{self.r}_{self.c}",
            operator="==",
            value=self.period,
            relation=f"hazards:{sorted(self.hazardous_values)}",
            target_entity=f"({self.r},{self.c})",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording dynamic cell phase."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"phase_cell_{self.r}_{self.c}",
                node_data={
                    "r": self.r,
                    "c": self.c,
                    "period": self.period,
                    "cycle_values": self.cycle_values,
                    "hazardous_values": sorted(self.hazardous_values),
                    "confidence": self.confidence,
                },
            )
        ]


class PhasePrecessionEstimator:
    """Pure computational faculty for periodicity estimation and phase decomposition."""

    @staticmethod
    def estimate_period(
        series: list[int],
        max_period: int = 8,
        min_match_ratio: float = 0.85,
    ) -> tuple[int | None, float]:
        """Estimate repeating periodicity T using backward autocorrelation.

        Returns:
            (best_period, confidence) or (None, 0.0) if no periodicity detected.
        """
        n = len(series)
        best_t: int | None = None
        best_confidence: float = 0.0

        for t in range(2, min(max_period + 1, n // 2 + 1)):
            matches = sum(1 for i in range(t, n) if series[i] == series[i - t])
            total = n - t
            if total > 0:
                ratio = matches / total
                if ratio >= min_match_ratio and ratio > best_confidence:
                    best_t = t
                    best_confidence = ratio

        if best_t is not None:
            return best_t, best_confidence
        return None, 0.0

    @staticmethod
    def estimate_periods_batch(
        series_matrix: np.ndarray,
        max_period: int = 8,
        min_match_ratio: float = 0.85,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised :meth:`estimate_period` over many series at once.

        Args:
            series_matrix: array of shape ``(n_steps, n_series)``.

        Returns:
            ``(periods, confidences)`` arrays of shape ``(n_series,)``; period 0 means none.
        """
        n, m = series_matrix.shape
        best_t = np.zeros(m, dtype=np.int64)
        best_conf = np.zeros(m, dtype=np.float64)
        for t in range(2, min(max_period + 1, n // 2 + 1)):
            ratio = (series_matrix[t:] == series_matrix[:-t]).mean(axis=0)
            better = (ratio >= min_match_ratio) & (ratio > best_conf)
            best_t[better] = t
            best_conf[better] = ratio[better]
        return best_t, best_conf

    @staticmethod
    def compute_lcm(numbers: Sequence[int], cap: int = 24) -> int:
        """Compute least common multiple of a sequence of integers (capped to prevent explosion)."""
        lcm = 1
        for num in set(numbers):
            if num <= 0:
                continue
            lcm = (lcm * num) // math.gcd(lcm, num)
            if lcm > cap:
                return cap
        return max(1, lcm)

    @classmethod
    def to_causal_predicate(cls) -> CausalPredicate:
        """Export phase precession estimator capability as a declarative causal predicate."""
        return CausalPredicate(
            variable="phase_precession_estimator",
            operator="==",
            value="ACTIVE",
            relation="entorhinal_phase_precession",
            target_entity="spatiotemporal_tracker",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering phase precession estimator."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="phase_precession_estimator",
                node_data={
                    "faculty": "entorhinal_phase_precession",
                    "status": "ACTIVE",
                },
            )
        ]


class SpatiotemporalHazardTracker:
    """Biologically-inspired tracker for moving hazards and periodic spatial oscillations."""

    def __init__(self, history_len: int = 24, max_period: int = 8) -> None:
        # Lazy: RetentionPolicy lives in percept (import cycle guard).
        if not SpatiotemporalHazardTracker.FIELD_RETENTION:
            from hbllm.hcir.world.percept import RetentionPolicy

            SpatiotemporalHazardTracker.FIELD_RETENTION.update(
                {
                    "grid_history": RetentionPolicy.EPISODE_LOCAL,
                    "step_history": RetentionPolicy.EPISODE_LOCAL,
                    "periodic_cells": RetentionPolicy.EPISODE_LOCAL,
                    "mobile_entities": RetentionPolicy.EPISODE_LOCAL,
                    "environmental_period": RetentionPolicy.EPISODE_LOCAL,
                    "known_lethal_features": RetentionPolicy.CONTEXT_LOCAL,
                    "static_lethal_positions": RetentionPolicy.CONTEXT_LOCAL,
                }
            )
        self.history_len = history_len
        self.max_period = max_period
        self.grid_history: deque[np.ndarray] = deque(maxlen=history_len)
        self.step_history: deque[int] = deque(maxlen=history_len)
        self.periodic_cells: dict[tuple[int, int], DynamicCellPhase] = {}
        self.mobile_entities: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = {}
        self.environmental_period: int = 1
        self.known_lethal_features: set[int] = set()
        self.static_lethal_positions: set[tuple[int, int]] = set()

    # WP0.5/F1: per-field retention — temporal observation history is
    # episode-local; learned lethal features/positions are context-local theory
    # (static_lethal_positions is rebind-shared with engine.context_hazards).
    FIELD_RETENTION: dict[str, Any] = {}

    def on_boundary(self, boundary) -> None:
        """Episode lifecycle boundary handler — generic per-field retention."""
        from hbllm.hcir.world.percept import apply_boundary_retention

        apply_boundary_retention(self, boundary)

    @property
    def boundary_retention(self):
        """Declared retention policy: default episode-local semantics."""
        from hbllm.hcir.world.percept import RetentionPolicy

        return RetentionPolicy.EPISODE_LOCAL

    def reset_episode(self) -> None:
        """Reset temporal observation history while preserving learned lethal feature identities and static lethal positions."""
        self.grid_history.clear()
        self.step_history.clear()
        self.periodic_cells.clear()
        self.mobile_entities.clear()
        self.environmental_period = 1

    def register_lethal_position(self, pos: tuple[int, int]) -> None:
        """Register a coordinate that resulted in avatar destruction/death upon entry."""
        self.static_lethal_positions.add((int(pos[0]), int(pos[1])))

    def clear_lethal_positions(self) -> None:
        """Clear recorded static lethal positions."""
        self.static_lethal_positions.clear()

    def register_lethal_feature(self, feature_id: int) -> None:
        """Register a feature value that resulted in avatar destruction/death upon contact."""
        self.known_lethal_features.add(int(feature_id))
        for cell in self.periodic_cells.values():
            if int(feature_id) in cell.cycle_values:
                cell.hazardous_values.add(int(feature_id))

    def record_frame(
        self,
        step: int,
        grid: np.ndarray,
        background_feature: int = 0,
        avatar_features: set[int] | None = None,
        walkable_features: set[int] | None = None,
        cargo_features: set[int] | None = None,
    ) -> None:
        """Record an environmental observation frame and update periodic hazard models."""
        if not isinstance(grid, np.ndarray):
            if isinstance(grid, (list, tuple)) and len(grid) > 0:
                grid = grid[-1]
            grid = np.asarray(grid, dtype=int)
        if grid.ndim == 3 and len(grid) > 0:
            grid = grid[-1]
        if self.grid_history and self.grid_history[-1].shape != grid.shape:
            # Lattice extent changed (new level/episode) — history from the
            # previous lattice is incomparable; restart the window.
            self.grid_history.clear()
            self.step_history.clear()
        self.grid_history.append(grid.copy())
        self.step_history.append(step)

        if len(self.grid_history) >= 2:
            prev_g = self.grid_history[-2]
            curr_g = self.grid_history[-1]
            av_ignore = set(avatar_features or ()) | {int(background_feature)}
            if cargo_features:
                av_ignore.update(int(c) for c in cargo_features)
                for cf in cargo_features:
                    self.mobile_entities.pop(int(cf), None)
            for av in set(avatar_features or ()):
                self.mobile_entities.pop(int(av), None)
            u_vals, u_cnts = np.unique(curr_g, return_counts=True)
            for v_np, c_np in zip(u_vals.tolist(), u_cnts.tolist(), strict=False):
                f_id = int(v_np)
                if f_id in av_ignore or not (1 <= int(c_np) <= 4):
                    continue
                curr_mask = curr_g == f_id
                prev_mask = prev_g == f_id
                if np.array_equal(curr_mask, prev_mask) or not np.any(prev_mask):
                    continue
                c_pts = [(int(r), int(c)) for r, c in np.argwhere(curr_mask)]
                p_pts = [(int(r), int(c)) for r, c in np.argwhere(prev_mask)]
                tracked: list[tuple[tuple[int, int], tuple[int, int]]] = []
                for cr, cc in c_pts:
                    best_p = min(p_pts, key=lambda pp: abs(cr - pp[0]) + abs(cc - pp[1]))
                    dist_m = abs(cr - best_p[0]) + abs(cc - best_p[1])
                    if 1 <= dist_m <= 2:
                        tracked.append(((cr, cc), (cr - best_p[0], cc - best_p[1])))
                if tracked:
                    self.mobile_entities[f_id] = tracked

        if len(self.grid_history) < 4:
            return

        grids_arr = np.stack(list(self.grid_history), axis=0)
        min_vals = np.min(grids_arr, axis=0)
        max_vals = np.max(grids_arr, axis=0)
        fluctuating_mask = min_vals != max_vals

        fluctuating_indices = np.argwhere(fluctuating_mask)
        if len(fluctuating_indices) == 0:
            self.periodic_cells.clear()
            self.environmental_period = 1
            return

        discovered_periods: list[int] = []
        safe_features = {0, background_feature}
        if walkable_features:
            safe_features.update(walkable_features)
        if avatar_features:
            safe_features.update(avatar_features)

        # One vectorised autocorrelation pass over every fluctuating cell
        series_matrix = grids_arr[:, fluctuating_indices[:, 0], fluctuating_indices[:, 1]]
        periods, confidences = PhasePrecessionEstimator.estimate_periods_batch(
            series_matrix, max_period=self.max_period
        )

        for k, (r, c) in enumerate(fluctuating_indices):
            r_idx, c_idx = int(r), int(c)
            best_period = int(periods[k])
            confidence = float(confidences[k])
            if best_period >= 2:
                series = series_matrix[:, k].tolist()
                cycle = series[-best_period:]
                haz_vals = {
                    v
                    for v in cycle
                    if v in self.known_lethal_features
                    or (v not in safe_features and not self.known_lethal_features)
                }
                if haz_vals:
                    self.periodic_cells[(r_idx, c_idx)] = DynamicCellPhase(
                        r=r_idx,
                        c=c_idx,
                        period=best_period,
                        cycle_values=cycle,
                        hazardous_values=haz_vals,
                        confidence=confidence,
                    )
                    discovered_periods.append(best_period)
                else:
                    self.periodic_cells.pop((r_idx, c_idx), None)
            else:
                self.periodic_cells.pop((r_idx, c_idx), None)

        if discovered_periods:
            self.environmental_period = PhasePrecessionEstimator.compute_lcm(discovered_periods)
        else:
            self.environmental_period = 1

    def _estimate_period(self, series: list[int]) -> int | None:
        """Estimate repeating periodicity T using backward autocorrelation (legacy hook)."""
        period, _ = PhasePrecessionEstimator.estimate_period(series, max_period=self.max_period)
        return period

    @staticmethod
    def _compute_lcm(numbers: Sequence[int]) -> int:
        """Compute the least common multiple of a sequence of integers (capped at 24)."""
        return PhasePrecessionEstimator.compute_lcm(numbers, cap=24)

    def is_hazard_at(
        self,
        r: int,
        c: int,
        future_relative_step: int,
        background_feature: int = 0,
        avatar_features: set[int] | None = None,
        walkable_features: set[int] | None = None,
    ) -> bool:
        """Predict whether coordinate (r, c) will be lethal or impassable at t_current + future_relative_step."""
        if (r, c) in self.static_lethal_positions:
            return True
        if (r, c) not in self.periodic_cells:
            return False

        cell_phase = self.periodic_cells[(r, c)]
        t = cell_phase.period
        idx = (future_relative_step - 1) % t if future_relative_step > 0 else -1
        predicted_val = cell_phase.cycle_values[idx]
        if predicted_val == 0 or predicted_val == background_feature:
            return False
        if avatar_features and predicted_val in avatar_features:
            return False
        if walkable_features and predicted_val in walkable_features:
            return False
        return predicted_val in cell_phase.hazardous_values

    def get_mobile_avoidance_cells(
        self,
        grid: np.ndarray,
        barrier_features: set[int],
        background_feature: int = 0,
        cargo_features: set[int] | None = None,
        avatar_features: set[int] | None = None,
    ) -> set[tuple[int, int]]:
        """Return current and 1-step predicted cells occupied by autonomous non-self mobile entities."""
        avoid: set[tuple[int, int]] = set()
        if not self.mobile_entities:
            return avoid
        H, W = grid.shape
        cargo = {int(c) for c in (cargo_features or ())}
        av_set = {int(a) for a in (avatar_features or ())}
        for f_id, tracks in list(self.mobile_entities.items()):
            if int(f_id) in cargo or int(f_id) in av_set:
                self.mobile_entities.pop(f_id, None)
                continue
            live_pts = [(int(r), int(c)) for r, c in np.argwhere(grid == f_id)]
            if not live_pts:
                continue
            vel_map = {pos: vel for pos, vel in tracks}
            is_lethal_mob = f_id in self.known_lethal_features
            for cr, cc in live_pts:
                avoid.add((cr, cc))
                dr, dc = vel_map.get((cr, cc), tracks[0][1])
                nr, nc = cr + dr, cc + dc
                rr, rc = cr - dr, cc - dc
                if not (0 <= nr < H and 0 <= nc < W) or int(grid[nr, nc]) in barrier_features:
                    if 0 <= rr < H and 0 <= rc < W:
                        avoid.add((rr, rc))
                else:
                    avoid.add((nr, nc))
                if is_lethal_mob:
                    for dr4, dc4 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ar4, ac4 = cr + dr4, cc + dc4
                        if 0 <= ar4 < H and 0 <= ac4 < W:
                            avoid.add((ar4, ac4))
        return avoid

    def get_hazard_schedule(
        self,
        horizon: int = 24,
        background_feature: int = 0,
        walkable_features: set[int] | None = None,
    ) -> dict[int, set[tuple[int, int]]]:
        """Generate a schedule mapping relative future step -> set of hazardous coordinates."""
        schedule: dict[int, set[tuple[int, int]]] = {}
        safe_features = {0, background_feature}
        if walkable_features:
            safe_features.update(walkable_features)

        for dt in range(horizon + 1):
            haz_set: set[tuple[int, int]] = set(self.static_lethal_positions)
            for (r, c), cell_phase in self.periodic_cells.items():
                t = cell_phase.period
                idx = (dt - 1) % t if dt > 0 else -1
                val = cell_phase.cycle_values[idx]
                if val not in safe_features and val in cell_phase.hazardous_values:
                    haz_set.add((r, c))
            if haz_set:
                schedule[dt] = haz_set
        return schedule

    def to_causal_predicate(self) -> CausalPredicate:
        """Export spatiotemporal hazard tracker state as a declarative causal predicate."""
        return CausalPredicate(
            variable="spatiotemporal_hazard_tracker",
            operator="==",
            value=self.environmental_period,
            relation=f"periodic_cells:{len(self.periodic_cells)}",
            target_entity=f"lethal:{sorted(self.known_lethal_features)}",
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all periodic cell phase models as declarative causal predicates."""
        preds = [self.to_causal_predicate()]
        for cell in self.periodic_cells.values():
            preds.append(cell.to_causal_predicate())
        return preds

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting hazard tracker state."""
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="hazard_tracker_summary",
                node_data={
                    "environmental_period": self.environmental_period,
                    "periodic_cells_count": len(self.periodic_cells),
                    "known_lethal_features": sorted(self.known_lethal_features),
                    "static_lethal_positions": [list(p) for p in self.static_lethal_positions],
                },
            )
        ]
        for cell in self.periodic_cells.values():
            ops.extend(cell.to_transaction_operations())
        return ops
