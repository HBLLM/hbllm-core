"""Hippocampal CA1 Temporal Distance Cells & Relative Interval Timing.

Biological Foundations:
- MacDonald, Lepage, Eden, Eichenbaum (2011): "Hippocampal 'time cells' bridge the gap
  in memory for discontiguous events" (Cell 147: 238-249).
- Eichenbaum (2014): "Time cells in the hippocampus: a new dimension for mapping
  memories" (Nat Rev Neurosci 15: 732-744).
- Howard & Eichenbaum (2015): "The hippocampus, time, and memory across scales".

Key Cognitive Capabilities:
1. Receptive Tuning Fields: Individual CA1 pyramidal time cells fire at specific peak
   moments during an elapsed delay interval between an antecedent cue and consequence.
2. Scalar Timing & Weber-Fechner Law: Temporal firing fields widen proportionally to peak
   time (sigma_t(t_0) = sigma_0 * (1 + beta * t_0)), accurately capturing scalar variance.
3. Interval Rescaling: When action pacing or delay durations stretch or compress, cell
   tuning centers rescale proportionally, preserving relative relational timing.
4. Temporal Causal Bridge: Provides discount and proximity metrics to bridge causal credit
   assignment across discontiguous operational steps (Pearl's causal calculus).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from hbllm.hcir.world.causal_discovery import CausalPredicate


@dataclass(frozen=True)
class TemporalTimeCell:
    """Individual CA1 hippocampal time cell with Gaussian temporal receptive field."""

    cell_id: str
    peak_time: float
    tuning_width: float
    peak_rate: float = 1.0

    def firing_rate(self, current_time: float) -> float:
        """Compute instantaneous firing rate at given elapsed time t >= 0.0."""
        if self.tuning_width <= 1e-6:
            return self.peak_rate if abs(current_time - self.peak_time) < 1e-6 else 0.0
        diff = current_time - self.peak_time
        exponent = -(diff * diff) / (2.0 * self.tuning_width * self.tuning_width)
        return float(self.peak_rate * math.exp(exponent))


@dataclass(frozen=True)
class TimeIntervalState:
    """Instantaneous state of an active temporal interval representation."""

    interval_id: str
    elapsed_time: float
    duration: float
    progress: float
    active_cell_id: str
    temporal_confidence: float
    population_rates: tuple[float, ...]

    def to_causal_predicate(self) -> CausalPredicate:
        """Convert interval state into a formal HCIR CausalPredicate."""
        return CausalPredicate(
            variable="temporal_interval_progress",
            operator="==",
            value=f"id:{self.interval_id}:prog:{self.progress:.2f}:elapsed:{self.elapsed_time:.2f}:cell:{self.active_cell_id}",
            relation="hippocampal_time_cells",
            target_entity="ca1_time_cell_population",
        )


class HippocampalTimeCellEngine:
    """Engine orchestrating CA1 temporal distance cells and relative interval timing."""

    def __init__(
        self,
        default_sigma_0: float = 0.15,
        scalar_beta: float = 0.12,
        temporal_decay_tau: float = 4.0,
    ) -> None:
        """Initialize CA1 time cell engine.

        Args:
            default_sigma_0: Baseline temporal receptive field width at t = 0.
            scalar_beta: Weber-Fechner scalar widening factor per unit of time.
            temporal_decay_tau: Time constant for causal delay discounting.
        """
        self.default_sigma_0: float = max(0.01, float(default_sigma_0))
        self.scalar_beta: float = max(0.0, float(scalar_beta))
        self.temporal_decay_tau: float = max(0.1, float(temporal_decay_tau))

        self._intervals: dict[str, dict[str, Any]] = {}

    def register_interval(
        self,
        interval_id: str,
        duration: float,
        num_cells: int = 10,
        spacing: str = "weber",
    ) -> list[TemporalTimeCell]:
        """Register a new timed interval with a tuned population of CA1 time cells.

        Args:
            interval_id: Unique identifier for the timed task / delay phase.
            duration: Expected total interval duration in seconds / steps.
            num_cells: Number of pyramidal time cells in the ensemble.
            spacing: 'weber' for logarithmic/Weber-Fechner spacing, or 'linear'.

        Returns:
            List of initialized TemporalTimeCell instances.
        """
        dur = max(0.01, float(duration))
        n_cells = max(2, int(num_cells))

        cells: list[TemporalTimeCell] = []
        for i in range(n_cells):
            if spacing == "linear":
                peak_t = (i / (n_cells - 1)) * dur
            else:
                # Weber-Fechner logarithmic density: denser representation earlier
                fraction = math.log(1.0 + (math.e - 1.0) * (i / (n_cells - 1)))
                peak_t = fraction * dur

            # Scalar property: tuning width widens linearly with peak time
            width = self.default_sigma_0 * (1.0 + self.scalar_beta * peak_t)
            cid = f"{interval_id}_tc_{i:02d}"
            cells.append(
                TemporalTimeCell(
                    cell_id=cid,
                    peak_time=peak_t,
                    tuning_width=width,
                    peak_rate=1.0,
                )
            )

        self._intervals[interval_id] = {
            "duration": dur,
            "elapsed": 0.0,
            "cells": cells,
            "spacing": spacing,
        }
        return list(cells)

    def step(self, dt: float, interval_id: str | None = None) -> dict[str, TimeIntervalState]:
        """Advance time by dt for specified interval or all active intervals.

        Args:
            dt: Elapsed time delta to integrate.
            interval_id: Specific interval to update, or None for all.

        Returns:
            Dictionary mapping interval_id to current TimeIntervalState.
        """
        step_dt = max(0.0, float(dt))
        targets = (
            [interval_id]
            if interval_id is not None and interval_id in self._intervals
            else list(self._intervals.keys())
        )

        results: dict[str, TimeIntervalState] = {}
        for iid in targets:
            data = self._intervals[iid]
            data["elapsed"] += step_dt
            results[iid] = self._compute_state(iid)

        return results

    def reset_interval(self, interval_id: str) -> TimeIntervalState | None:
        """Reset elapsed time for an interval back to 0.0."""
        if interval_id not in self._intervals:
            return None
        self._intervals[interval_id]["elapsed"] = 0.0
        return self._compute_state(interval_id)

    def rescale_interval(
        self,
        interval_id: str,
        new_duration: float,
    ) -> list[TemporalTimeCell]:
        """Dynamically rescale cell tuning centers to fit a new duration."""
        if interval_id not in self._intervals:
            return []

        old_dur = self._intervals[interval_id]["duration"]
        new_dur = max(0.01, float(new_duration))
        ratio = new_dur / old_dur if old_dur > 1e-6 else 1.0

        old_cells: list[TemporalTimeCell] = self._intervals[interval_id]["cells"]
        new_cells: list[TemporalTimeCell] = []

        for cell in old_cells:
            new_peak = cell.peak_time * ratio
            new_width = self.default_sigma_0 * (1.0 + self.scalar_beta * new_peak)
            new_cells.append(
                TemporalTimeCell(
                    cell_id=cell.cell_id,
                    peak_time=new_peak,
                    tuning_width=new_width,
                    peak_rate=cell.peak_rate,
                )
            )

        self._intervals[interval_id]["duration"] = new_dur
        self._intervals[interval_id]["cells"] = new_cells
        return list(new_cells)

    def estimate_time_remaining(self, interval_id: str) -> tuple[float, float]:
        """Estimate remaining time and temporal certainty for an interval.

        Returns:
            Tuple of (estimated_remaining_time, confidence_score in [0.0, 1.0]).
        """
        if interval_id not in self._intervals:
            return 0.0, 0.0

        data = self._intervals[interval_id]
        dur = float(data["duration"])
        elapsed = float(data["elapsed"])
        remaining = max(0.0, dur - elapsed)

        # Confidence degrades with remaining delay according to scalar expectancy
        uncertainty = self.scalar_beta * remaining
        confidence = float(1.0 / (1.0 + uncertainty))
        return remaining, confidence

    def bridge_causal_credit(self, delta_t: float) -> float:
        """Compute exponential temporal discount factor to bridge delayed cause and effect."""
        dt = max(0.0, float(delta_t))
        return float(math.exp(-dt / self.temporal_decay_tau))

    def get_interval_state(self, interval_id: str) -> TimeIntervalState | None:
        """Retrieve current state of an interval without advancing time."""
        if interval_id not in self._intervals:
            return None
        return self._compute_state(interval_id)

    def _compute_state(self, interval_id: str) -> TimeIntervalState:
        """Internal helper to compute TimeIntervalState."""
        data = self._intervals[interval_id]
        dur = float(data["duration"])
        elapsed = float(data["elapsed"])
        cells: list[TemporalTimeCell] = data["cells"]

        rates: list[float] = [c.firing_rate(elapsed) for c in cells]
        max_idx = int(max(range(len(rates)), key=lambda idx: rates[idx])) if rates else 0
        active_cid = cells[max_idx].cell_id if cells else ""

        progress = min(1.0, elapsed / dur) if dur > 1e-6 else 1.0
        remaining = max(0.0, dur - elapsed)
        confidence = float(1.0 / (1.0 + self.scalar_beta * remaining))

        return TimeIntervalState(
            interval_id=interval_id,
            elapsed_time=elapsed,
            duration=dur,
            progress=progress,
            active_cell_id=active_cid,
            temporal_confidence=confidence,
            population_rates=tuple(rates),
        )
