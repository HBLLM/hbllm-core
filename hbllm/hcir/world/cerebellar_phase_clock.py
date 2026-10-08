"""Basal Ganglia & Cerebellar Predictive Phase Entrainment.

Modeled on mammalian cerebellar-basal ganglia interval timing and motor gating:
1. Macro Environmental Periodicity: Detects cyclic environmental clocks T in [2, 128]
   via frame signature recurrence and temporal autocorrelation.
2. Phase Tracking & Horizon Estimation: Maintains instantaneous phase phi(t) = t mod T
   and computes remaining epoch runway tau_remaining = T - (t mod T).
3. Motor Phase Gating: Gating mechanism for basal ganglia action release — when planning
   bottleneck crossings or hazardous transitions, computes required phase delays (hesitation / wait)
   to ensure arrival occurs during the verified safe phase window Phi_safe.
"""

from __future__ import annotations

import collections
import hashlib
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhaseGateDecision:
    """Motor gating recommendation from the cerebellar clock."""

    should_wait: bool = False
    wait_steps_recommended: int = 0
    safe_phase_offset: int = 0
    current_phase: int = 0
    environmental_period: int = 1
    reason: str = ""


class CerebellarPhaseClock:
    """Cerebellar interval timer and basal ganglia motor phase gating engine."""

    def __init__(self, history_len: int = 256, max_macro_period: int = 128) -> None:
        self.history_len = history_len
        self.max_macro_period = max_macro_period
        self.frame_signatures: deque[str] = deque(maxlen=history_len)
        self.step_history: deque[int] = deque(maxlen=history_len)
        self.detected_macro_period: int | None = None
        self.period_confidence: float = 0.0

        # Reset cycle tracking
        self.reset_intervals: list[int] = []
        self.last_reset_step: int = 0

    def record_step(self, step: int, grid: np.ndarray, is_reset: bool = False) -> None:
        """Record visual frame and update macro periodicity models.

        Args:
            step: Current environment step counter.
            grid: 2D numpy array of visual frame.
            is_reset: Whether an environmental reset/retry occurred at this step.
        """
        if is_reset:
            if self.last_reset_step > 0:
                interval = step - self.last_reset_step
                if 8 <= interval <= self.max_macro_period:
                    self.reset_intervals.append(interval)
                    self._recalibrate_from_resets()
            self.last_reset_step = step

        # Compute lightweight perceptual hash of grid state
        # Downsample or hash non-background features
        sig = hashlib.md5(grid.tobytes()).hexdigest()[:12]
        self.frame_signatures.append(sig)
        self.step_history.append(step)

        if len(self.frame_signatures) >= 32 and self.detected_macro_period is None:
            self._estimate_recurrence_period()

    def _recalibrate_from_resets(self) -> None:
        """Estimate periodic lifecycle clock from observed environmental reset intervals."""
        if len(self.reset_intervals) < 2:
            return

        # Check modal interval (e.g., 64 in BP35, 128 in TR87)
        counts = collections.Counter(self.reset_intervals)
        best_interval, num_matches = counts.most_common(1)[0]
        if num_matches >= 2:
            self.detected_macro_period = best_interval
            self.period_confidence = min(0.95, num_matches / len(self.reset_intervals))
            logger.info(
                "CerebellarPhaseClock: Discovered macro environmental clock T=%d (conf=%.2f) from reset intervals %s",
                best_interval,
                self.period_confidence,
                self.reset_intervals,
            )

    def _estimate_recurrence_period(self) -> None:
        """Autocorrelation over frame signatures to find repeating visual states."""
        sigs = list(self.frame_signatures)
        n = len(sigs)

        candidate_periods = [64, 128, 52, 48, 32, 24, 16, 8, 4]
        for T in candidate_periods:
            if n < T * 1.5:
                continue
            matches = sum(1 for i in range(T, n) if sigs[i] == sigs[i - T])
            total = n - T
            if total > 0 and (matches / total) >= 0.75:
                self.detected_macro_period = T
                self.period_confidence = matches / total
                logger.info(
                    "CerebellarPhaseClock: Discovered recurrence period T=%d from visual state recurrence",
                    T,
                )
                return

    def get_current_phase(self, step: int) -> int:
        """Return instantaneous phase phi(t) = step mod T."""
        T = self.detected_macro_period or 1
        return step % T

    def get_epoch_runway(self, step: int) -> int:
        """Return remaining steps before the next cyclic epoch boundary."""
        if not self.detected_macro_period or self.detected_macro_period <= 1:
            return 9999
        T = self.detected_macro_period
        return T - (step % T)

    def evaluate_phase_gate(
        self,
        current_step: int,
        travel_steps_to_hazard: int,
        hazard_period: int,
        hazard_cycle_values: list[int],
        safe_values: set[int],
    ) -> PhaseGateDecision:
        """Evaluate whether basal ganglia should withhold movement to time bottleneck entry safely.

        Args:
            current_step: Current global time step.
            travel_steps_to_hazard: Number of steps required to reach the bottleneck.
            hazard_period: Period T of the oscillating bottleneck cell.
            hazard_cycle_values: Values sequence over one cycle of the bottleneck.
            safe_values: Set of feature values that are non-lethal / passable.

        Returns:
            PhaseGateDecision indicating whether to wait, and for how many steps.
        """
        if hazard_period <= 1 or not hazard_cycle_values:
            return PhaseGateDecision(should_wait=False)

        # Arrival step if moving immediately
        arrival_step = current_step + travel_steps_to_hazard
        arrival_phase = (arrival_step - 1) % hazard_period
        arrival_val = hazard_cycle_values[arrival_phase]

        if arrival_val in safe_values:
            # Safe to enter without delay
            return PhaseGateDecision(
                should_wait=False,
                current_phase=arrival_phase,
                environmental_period=hazard_period,
            )

        # Immediate arrival is hazardous — calculate optimal delay k in [1, T-1]
        for wait_k in range(1, hazard_period):
            delayed_arrival = arrival_step + wait_k
            delayed_phase = (delayed_arrival - 1) % hazard_period
            if hazard_cycle_values[delayed_phase] in safe_values:
                return PhaseGateDecision(
                    should_wait=True,
                    wait_steps_recommended=wait_k,
                    safe_phase_offset=delayed_phase,
                    current_phase=arrival_phase,
                    environmental_period=hazard_period,
                    reason=f"Bottleneck unsafe at phase {arrival_phase}; wait {wait_k} step(s) for safe phase {delayed_phase}",
                )

        return PhaseGateDecision(should_wait=False)

    def evaluate_motion_hazard_gate(
        self,
        current_step: int,
        avatar_pos: tuple[int, int] | None,
        target_pos: tuple[int, int],
        hazard_tracker: Any,
    ) -> PhaseGateDecision:
        """Evaluate if basal ganglia should hesitate before entering target_pos due to cyclic hazard phase.

        If target_pos is periodic and entering next step is hazardous, but waiting k steps
        allows safe passage and avatar_pos is safe during the wait, recommends motor hesitation.
        """
        if not hasattr(hazard_tracker, "periodic_cells") or not avatar_pos:
            return PhaseGateDecision(should_wait=False)

        cell_phase = hazard_tracker.periodic_cells.get(target_pos)
        if cell_phase is None:
            return PhaseGateDecision(should_wait=False)

        period = cell_phase.period
        cycle_vals = cell_phase.cycle_values
        haz_vals = cell_phase.hazardous_values

        safe_vals = {v for v in cycle_vals if v not in haz_vals}
        if not safe_vals:
            return PhaseGateDecision(should_wait=False)

        decision = self.evaluate_phase_gate(
            current_step=current_step,
            travel_steps_to_hazard=1,
            hazard_period=period,
            hazard_cycle_values=cycle_vals,
            safe_values=safe_vals,
        )

        if decision.should_wait:
            # Verify avatar_pos is safe to wait on during the recommended delay
            for w in range(1, decision.wait_steps_recommended + 1):
                if hazard_tracker.is_hazard_at(
                    avatar_pos[0], avatar_pos[1], future_relative_step=w
                ):
                    return PhaseGateDecision(should_wait=False)

        return decision
