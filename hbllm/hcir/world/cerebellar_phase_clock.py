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
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

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

    def to_causal_predicate(self) -> CausalPredicate:
        """Export phase gate decision as a declarative causal predicate."""
        return CausalPredicate(
            variable="motor_phase_gate",
            operator="==",
            value="WAIT" if self.should_wait else "PROCEED",
            relation=f"wait_steps:{self.wait_steps_recommended}",
            target_entity=f"phase_{self.current_phase}_of_{self.environmental_period}",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording phase gate decision."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="phase_gate_decision",
                node_data={
                    "id": "phase_gate_decision",
                    "node_type": "event",
                    "category": "execution",
                    "should_wait": self.should_wait,
                    "wait_steps_recommended": self.wait_steps_recommended,
                    "safe_phase_offset": self.safe_phase_offset,
                    "current_phase": self.current_phase,
                    "environmental_period": self.environmental_period,
                    "reason": self.reason,
                },
            )
        ]


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
        self.entity_cycles: dict[int, tuple[int, int]] = {}  # feature_id -> (period, phase)

    def register_entity_cycle(self, feature_id: int, period: int, phase: int = 0) -> None:
        """Register a dynamic entity limit cycle discovered by SNN temporal tracking."""
        if period > 1:
            self.entity_cycles[feature_id] = (period, phase)
            logger.debug(
                "CerebellarPhaseClock: Registered entity cycle feat=%d, period=%d, phase=%d",
                feature_id,
                period,
                phase,
            )

    def evaluate_track_crossing_gate(
        self,
        current_step: int,
        travel_steps_to_hazard: int,
        entity_feature: int,
        collision_phases: set[int],
    ) -> PhaseGateDecision:
        """Evaluate motor gating for crossing a track occupied by a periodic dynamic entity."""
        cycle = self.entity_cycles.get(entity_feature)
        if cycle is None or cycle[0] <= 1 or not collision_phases:
            return PhaseGateDecision(should_wait=False)

        period, base_phase = cycle
        arrival_step = current_step + travel_steps_to_hazard
        arrival_phase = (arrival_step + base_phase) % period

        if arrival_phase not in collision_phases:
            return PhaseGateDecision(
                should_wait=False,
                current_phase=arrival_phase,
                environmental_period=period,
                reason="arrival_in_safe_phase",
            )

        # Collision imminent on immediate arrival — find minimal wait_k
        for wait_k in range(1, period):
            delayed_phase = (arrival_step + wait_k + base_phase) % period
            if delayed_phase not in collision_phases:
                return PhaseGateDecision(
                    should_wait=True,
                    wait_steps_recommended=wait_k,
                    safe_phase_offset=delayed_phase,
                    current_phase=arrival_phase,
                    environmental_period=period,
                    reason=f"delayed_by_{wait_k}_to_avoid_entity_{entity_feature}",
                )

        return PhaseGateDecision(
            should_wait=True,
            wait_steps_recommended=1,
            safe_phase_offset=0,
            current_phase=arrival_phase,
            environmental_period=period,
            reason="fallback_wait_step",
        )

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
        background_feature: int = 0,
        avatar_features: set[int] | None = None,
        walkable_features: set[int] | None = None,
    ) -> PhaseGateDecision:
        """Evaluate if basal ganglia should hesitate before entering target_pos due to cyclic hazard phase.

        If target_pos is periodic and entering next step is hazardous, but waiting k steps
        allows safe passage and avatar_pos is safe during the wait, recommends motor hesitation.
        """
        if not hazard_tracker.periodic_cells or not avatar_pos:
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
                    avatar_pos[0],
                    avatar_pos[1],
                    future_relative_step=w,
                    background_feature=background_feature,
                    avatar_features=avatar_features,
                    walkable_features=walkable_features,
                ):
                    return PhaseGateDecision(should_wait=False)

        return decision

    def evaluate_crossing_safety(
        self,
        agent_pos: tuple[int, int],
        target_pos: tuple[int, int],
        snn_predictor: Any,
        grid: np.ndarray | None = None,
        current_step: int = 0,
        horizon_steps: int = 16,
    ) -> PhaseGateDecision:
        """Evaluate crossing safety combining cerebellar timing with SNN limit cycle forecasting."""
        travel_steps = abs(agent_pos[0] - target_pos[0]) + abs(agent_pos[1] - target_pos[1])
        arrival_step = current_step + travel_steps

        # 1. If grid is available, forecast hazard density at arrival
        if grid is not None and hasattr(snn_predictor, "predict_hazard_density"):
            density_at_arrival = snn_predictor.predict_hazard_density(
                grid, horizon_steps=max(1, travel_steps)
            )
            H, W = grid.shape
            tr, tc = target_pos
            if 0 <= tr < H and 0 <= tc < W:
                hazard_prob = float(density_at_arrival[tr, tc])
                if hazard_prob > 0.4:
                    # Target will be occupied at arrival; find safe delay
                    for delay in range(1, min(horizon_steps, 8)):
                        future_density = snn_predictor.predict_hazard_density(
                            grid, horizon_steps=travel_steps + delay
                        )
                        if float(future_density[tr, tc]) <= 0.2:
                            return PhaseGateDecision(
                                should_wait=True,
                                wait_steps_recommended=delay,
                                safe_phase_offset=delay,
                                current_phase=arrival_step % (self.detected_macro_period or 4),
                                environmental_period=self.detected_macro_period or 4,
                                reason=f"SNN forecasted hazard at target (p={hazard_prob:.2f}), safe after {delay} steps",
                            )

        return PhaseGateDecision(
            should_wait=False,
            wait_steps_recommended=0,
            safe_phase_offset=0,
            current_phase=arrival_step % (self.detected_macro_period or 1),
            environmental_period=self.detected_macro_period or 1,
            reason="Corridor and target safe across planned transit window",
        )

    def to_causal_predicate(self) -> CausalPredicate:
        """Export macro periodicity state as a declarative causal predicate."""
        return CausalPredicate(
            variable="cerebellar_macro_period",
            operator="==",
            value=self.detected_macro_period if self.detected_macro_period is not None else 0,
            relation=f"confidence:{self.period_confidence:.2f}",
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all cerebellar phase state predicates."""
        return [self.to_causal_predicate()]

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting cerebellar clock state."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="cerebellar_phase_clock",
                node_data={
                    "id": "cerebellar_phase_clock",
                    "node_type": "event",
                    "category": "execution",
                    "detected_macro_period": self.detected_macro_period,
                    "period_confidence": self.period_confidence,
                    "last_reset_step": self.last_reset_step,
                    "reset_intervals": list(self.reset_intervals),
                },
            )
        ]


@dataclass
class EfferenceCopyRecord:
    """Dispatched motor action efference copy buffered in the cerebellar forward model."""

    step: int
    action_id: Any
    expected_delta: tuple[int, int]
    prior_pos: tuple[int, int]
    predicted_pos: tuple[int, int]
    timestamp: float = field(default_factory=time.time)


class CerebellarSmithPredictor:
    """Internal forward model (Smith Predictor) compensating sensory feedback latency."""

    def __init__(self, latency_steps: int = 1) -> None:
        self.latency_steps = latency_steps
        self.efference_queue: deque[EfferenceCopyRecord] = deque(maxlen=32)
        self.last_predicted_pos: tuple[int, int] | None = None
        self.prediction_errors: deque[float] = deque(maxlen=64)

    def on_boundary(self, boundary) -> None:
        """Episode lifecycle boundary handler (WP0.5). Delegates to the
        deprecated reset_episode for backward-compatible semantics."""
        self.reset_episode()

    @property
    def boundary_retention(self):
        """Declared retention policy: default level-local semantics."""
        from hbllm.hcir.world.percept import RetentionPolicy

        return RetentionPolicy.LEVEL_LOCAL

    def reset_episode(self) -> None:
        """Clear forward prediction buffer and error history."""
        self.efference_queue.clear()
        self.last_predicted_pos = None
        self.prediction_errors.clear()

    def record_efference_copy(
        self,
        step: int,
        action_id: Any,
        expected_delta: tuple[int, int],
        current_pos: tuple[int, int],
    ) -> tuple[int, int]:
        """Buffer efference copy and predict immediate forward position ahead of sensory afferents."""
        pred_pos = (
            int(current_pos[0] + expected_delta[0]),
            int(current_pos[1] + expected_delta[1]),
        )
        record = EfferenceCopyRecord(
            step=step,
            action_id=action_id,
            expected_delta=expected_delta,
            prior_pos=current_pos,
            predicted_pos=pred_pos,
        )
        self.efference_queue.append(record)
        self.last_predicted_pos = pred_pos
        return pred_pos

    def assimilate_sensory_feedback(
        self,
        step: int,
        observed_pos: tuple[int, int],
    ) -> float:
        """Compare observed afferent position against buffered prediction to compute sensory prediction error."""
        matched: EfferenceCopyRecord | None = None
        for rec in reversed(self.efference_queue):
            if rec.step == step or rec.step == step - 1:
                matched = rec
                break

        if matched is not None:
            err = math.hypot(
                observed_pos[0] - matched.predicted_pos[0],
                observed_pos[1] - matched.predicted_pos[1],
            )
        elif self.last_predicted_pos is not None:
            err = math.hypot(
                observed_pos[0] - self.last_predicted_pos[0],
                observed_pos[1] - self.last_predicted_pos[1],
            )
        else:
            err = 0.0

        self.prediction_errors.append(err)
        return float(err)

    def get_forward_compensated_position(self, observed_pos: tuple[int, int]) -> tuple[int, int]:
        """Return zero-latency forward compensated position if efference copy is unconfirmed."""
        if self.last_predicted_pos is not None and len(self.efference_queue) > 0:
            return self.last_predicted_pos
        return observed_pos

    def to_causal_predicate(self) -> CausalPredicate:
        """Export forward model status as a declarative causal predicate."""
        mean_err = float(np.mean(list(self.prediction_errors))) if self.prediction_errors else 0.0
        return CausalPredicate(
            variable="cerebellar_smith_predictor",
            operator="==",
            value=f"mean_err:{mean_err:.2f}",
            relation=f"buffered:{len(self.efference_queue)}",
            target_entity="cerebellum_forward_model",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting forward model state."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="cerebellar_smith_predictor",
                node_data={
                    "id": "cerebellar_smith_predictor",
                    "node_type": "event",
                    "category": "execution",
                    "buffered_count": len(self.efference_queue),
                    "last_predicted_pos": self.last_predicted_pos,
                },
            )
        ]
