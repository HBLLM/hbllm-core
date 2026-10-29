"""SNN Temporal Predictor — Spiking Neural Network Temporal Dynamics & Limit Cycle Predictor.

Modeled on biological cerebellar-cortical spiking circuits:
1. Leaky Integrate-and-Fire (LIF) Dynamics:
   - Integrates membrane potentials: V[t+1] = V[t] * (1 - dt / tau_m) + I_syn[t].
   - Emits all-or-none action potentials (spikes) when V >= V_th with refractory clamping.
2. Spatio-Temporal Phase Recurrence:
   - Tracks periodic limit cycles and motion trajectories of dynamic hazards and patrollers.
   - Computes expected collision cones and hazard densities at arbitrary future horizons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

try:
    import hbllm_compute  # type: ignore[import-not-found]

    _HAS_RUST_COMPUTE = True
except ImportError:
    hbllm_compute = None
    _HAS_RUST_COMPUTE = False

logger = logging.getLogger(__name__)


@dataclass
class DynamicEntityTrack:
    """Temporal trajectory track for a moving environmental entity."""

    feature_id: int
    positions: list[tuple[int, int]] = field(default_factory=list)
    velocity: tuple[int, int] = (0, 0)
    period: int = 1
    phase: int = 0


class SNNTemporalPredictor:
    """Spiking neural network temporal pattern and dynamic hazard predictor."""

    name: str = "snn"

    def __init__(
        self,
        v_rest: float = 0.0,
        v_thresh: float = 1.0,
        v_reset: float = 0.0,
        tau_m: float = 10.0,
        refractory_steps: int = 2,
        tau_adaptation: float = 20.0,
        adaptation_increment: float = 0.3,
        homeostatic_rate: float = 0.05,
    ) -> None:
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.tau_m = tau_m
        self.refractory_steps = refractory_steps
        self.tau_adaptation = tau_adaptation
        self.adaptation_increment = adaptation_increment
        self.homeostatic_rate = homeostatic_rate

        # Internal neuronal state
        self.membrane_potential: np.ndarray | None = None
        self.refractory_clock: np.ndarray | None = None
        self.adaptation_threshold: np.ndarray | None = None
        self.sensitivity_scale: float = 1.0
        self.entity_tracks: dict[int, DynamicEntityTrack] = {}
        self.last_grid_shape: tuple[int, int] = (0, 0)
        self.step_counter: int = 0

    def to_causal_predicate(self) -> CausalPredicate:
        """Export SNN temporal predictor status as a declarative causal predicate."""
        return CausalPredicate(
            variable="predictor:snn",
            operator="==",
            value=f"active:tracks:{len(self.entity_tracks)}",
            relation="spiking_temporal_pattern",
            target_entity="snn",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering SNN temporal predictor."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="predictor_snn",
                node_data={
                    "name": self.name,
                    "type": "spiking_temporal_pattern",
                    "active": True,
                    "tracked_entities": len(self.entity_tracks),
                },
            )
        ]

    def reset_state(self) -> None:
        """Reset neuronal membrane potentials and tracking state."""
        self.membrane_potential = None
        self.refractory_clock = None
        self.adaptation_threshold = None
        self.sensitivity_scale = 1.0
        self.entity_tracks.clear()
        self.step_counter = 0

    def step_lif(self, current_input: np.ndarray, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Execute one integration step of Leaky Integrate-and-Fire (LIF) dynamics with Spike-Frequency Adaptation.

        Returns:
            tuple[np.ndarray, np.ndarray]: (binary_spike_map, membrane_potential_map)
        """
        shape = current_input.shape
        if self.membrane_potential is None or self.membrane_potential.shape != shape:
            self.membrane_potential = np.full(shape, self.v_rest, dtype=np.float32)
            self.refractory_clock = np.zeros(shape, dtype=np.int32)
            self.adaptation_threshold = np.zeros(shape, dtype=np.float32)

        assert self.refractory_clock is not None
        if self.adaptation_threshold is None or self.adaptation_threshold.shape != shape:
            self.adaptation_threshold = np.zeros(shape, dtype=np.float32)

        decay = float(np.exp(-dt / self.tau_m))
        decay_theta = float(np.exp(-dt / self.tau_adaptation))

        # Biological SFA: Exponential decay of adaptation threshold
        self.adaptation_threshold *= decay_theta
        effective_thresh = self.v_thresh + self.adaptation_threshold

        # 1. Fast Rust SIMD acceleration path (when available, no SFA increment, and baseline sensitivity)
        if (
            _HAS_RUST_COMPUTE
            and hbllm_compute is not None
            and hasattr(hbllm_compute, "snn_step_lif_simd")
            and self.adaptation_increment == 0.0
            and abs(self.sensitivity_scale - 1.0) < 1e-3
        ):
            try:
                spikes, new_mem, new_ref = hbllm_compute.snn_step_lif_simd(
                    self.membrane_potential,
                    self.refractory_clock,
                    current_input.astype(np.float32),
                    float(self.v_thresh),
                    float(self.v_reset),
                    decay,
                    int(self.refractory_steps),
                )
                self.membrane_potential = new_mem
                self.refractory_clock = new_ref
                return spikes, self.membrane_potential.copy()
            except Exception as e:
                logger.debug("Rust snn_step_lif_simd fallback to numpy: %s", e)

        # 2. Pure NumPy path with biological SFA & homeostatic sensitivity scaling
        non_refractory = self.refractory_clock <= 0
        scaled_input = current_input.astype(np.float32) * self.sensitivity_scale

        self.membrane_potential[non_refractory] = (
            self.membrane_potential[non_refractory] * decay + scaled_input[non_refractory]
        )

        spikes = (self.membrane_potential >= effective_thresh) & non_refractory
        self.membrane_potential[spikes] = self.v_reset
        self.refractory_clock[spikes] = self.refractory_steps

        # SFA: Increase dynamic threshold upon spike
        self.adaptation_threshold[spikes] += self.adaptation_increment
        self.refractory_clock = np.maximum(0, self.refractory_clock - 1)

        # Homeostatic plasticity: adjust input sensitivity toward target firing rate
        mean_spike_rate = float(np.mean(spikes))
        self.sensitivity_scale = float(
            np.clip(
                self.sensitivity_scale + 0.01 * (self.homeostatic_rate - mean_spike_rate),
                0.2,
                3.0,
            )
        )

        return spikes.astype(np.int32), self.membrane_potential.copy()

    def update_tracks_from_grid(self, grid: np.ndarray) -> None:
        """Update kinematic entity trajectories and detect periodic limit cycles."""
        self.step_counter += 1
        H, W = grid.shape
        unique_features = np.unique(grid)

        for feat in unique_features:
            feat_int = int(feat)
            if feat_int == 0:
                continue

            coords = np.argwhere(grid == feat_int)
            if len(coords) == 1:
                pos = (int(coords[0][0]), int(coords[0][1]))
                if feat_int not in self.entity_tracks:
                    self.entity_tracks[feat_int] = DynamicEntityTrack(
                        feature_id=feat_int,
                        positions=[pos],
                    )
                else:
                    track = self.entity_tracks[feat_int]
                    if track.positions:
                        prev_pos = track.positions[-1]
                        dr = pos[0] - prev_pos[0]
                        dc = pos[1] - prev_pos[1]
                        if (dr != 0 or dc != 0) and abs(dr) <= 2 and abs(dc) <= 2:
                            track.velocity = (dr, dc)
                    track.positions.append(pos)
                    if len(track.positions) > 32:
                        track.positions.pop(0)

                    # Detect simple oscillation period (revisitation of same position)
                    if len(track.positions) >= 4:
                        for p in range(2, min(16, len(track.positions))):
                            if track.positions[-1] == track.positions[-p]:
                                track.period = p - 1
                                break

    def predict_hazard_density(
        self,
        grid: np.ndarray,
        horizon_steps: int = 4,
    ) -> np.ndarray:
        """Compute spatio-temporal hazard density map forecasted horizon_steps into the future."""
        H, W = grid.shape

        # 1. Fast Rust SIMD kinematic projection path
        if (
            _HAS_RUST_COMPUTE
            and hbllm_compute is not None
            and hasattr(hbllm_compute, "forecast_hazard_density_simd")
        ):
            tracks_data = [
                (
                    int(t.positions[-1][0]),
                    int(t.positions[-1][1]),
                    int(t.velocity[0]),
                    int(t.velocity[1]),
                )
                for t in self.entity_tracks.values()
                if t.positions and t.velocity != (0, 0)
            ]
            if tracks_data:
                try:
                    rust_density = hbllm_compute.forecast_hazard_density_simd(
                        H, W, tracks_data, horizon_steps
                    )
                    rust_density[grid != 0] += 0.1
                    max_d = float(np.max(rust_density))
                    if max_d > 0.0:
                        rust_density = rust_density / max_d
                    return rust_density
                except Exception as e:
                    logger.debug("Rust forecast_hazard_density_simd fallback to numpy: %s", e)

        # 2. Pure NumPy fallback path
        density = np.zeros((H, W), dtype=np.float32)

        # Base density from stationary hazards
        density[grid != 0] += 0.1

        # Project dynamic entities forward along their estimated limit cycles
        for feat_id, track in self.entity_tracks.items():
            if not track.positions or track.velocity == (0, 0):
                continue

            cur_r, cur_c = track.positions[-1]
            vr, vc = track.velocity

            # Multi-step projection with boundary bouncing
            r, c = cur_r, cur_c
            for step in range(1, horizon_steps + 1):
                r += vr
                c += vc
                # Reflect off borders
                if r < 0 or r >= H:
                    vr = -vr
                    r = max(0, min(H - 1, r))
                if c < 0 or c >= W:
                    vc = -vc
                    c = max(0, min(W - 1, c))

                weight = 1.0 / (1.0 + 0.1 * step)
                density[r, c] += weight

        # Normalize density between 0.0 and 1.0
        max_d = float(np.max(density))
        if max_d > 0.0:
            density = density / max_d

        return density

    def extract_limit_cycles(self) -> list[tuple[int, int, int]]:
        """Extract detected periodic limit cycles (feature_id, period, phase) across entity tracks."""
        cycles: list[tuple[int, int, int]] = []
        for feat_id, track in self.entity_tracks.items():
            if track.period > 1 and track.positions:
                phase = self.step_counter % track.period
                cycles.append((feat_id, track.period, phase))
        return cycles

    def forecast_entity_position(
        self,
        feature_id: int,
        steps_ahead: int,
        grid_shape: tuple[int, int],
    ) -> tuple[int, int] | None:
        """Forecast the coordinate of a dynamic entity steps_ahead into the future with boundary reflection."""
        track = self.entity_tracks.get(feature_id)
        if track is None or not track.positions or track.velocity == (0, 0):
            return None

        H, W = grid_shape
        r, c = track.positions[-1]
        vr, vc = track.velocity

        for _ in range(steps_ahead):
            r += vr
            c += vc
            if r < 0 or r >= H:
                vr = -vr
                r = max(0, min(H - 1, r))
            if c < 0 or c >= W:
                vc = -vc
                c = max(0, min(W - 1, c))

        return (r, c)

    def evaluate_hazard_at_trajectory(
        self,
        planned_positions: list[tuple[int, int]],
        grid_shape: tuple[int, int],
    ) -> float:
        """Compute cumulative hazard exposure along a planned trajectory across future horizons."""
        if not planned_positions or not self.entity_tracks:
            return 0.0

        total_hazard: float = 0.0
        for step_idx, pos in enumerate(planned_positions, start=1):
            for feat_id, track in self.entity_tracks.items():
                if not track.positions or track.velocity == (0, 0):
                    continue
                future_pos = self.forecast_entity_position(feat_id, step_idx, grid_shape)
                if future_pos == pos:
                    total_hazard += 1.0
                elif future_pos is not None:
                    # Proximity penalty
                    dist = abs(future_pos[0] - pos[0]) + abs(future_pos[1] - pos[1])
                    if dist == 1:
                        total_hazard += 0.5
                    elif dist == 2:
                        total_hazard += 0.2

        return min(1.0, total_hazard)

    def evaluate_phase_gate_for_track(
        self,
        feature_id: int,
        current_step: int,
        target_pos: tuple[int, int],
        grid_shape: tuple[int, int],
        travel_steps: int = 1,
        max_wait: int = 8,
    ) -> dict[str, Any]:
        """Evaluate if motor hesitation/wait steps are required to safely traverse target_pos.

        Returns dict with keys: should_wait, wait_steps_recommended, reason, period.
        """
        track = self.entity_tracks.get(feature_id)
        if track is None or track.period <= 1:
            return {
                "should_wait": False,
                "wait_steps_recommended": 0,
                "reason": "no_cycle",
                "period": 1,
            }

        # Check immediate arrival
        imm_pos = self.forecast_entity_position(feature_id, travel_steps, grid_shape)
        if imm_pos != target_pos:
            return {
                "should_wait": False,
                "wait_steps_recommended": 0,
                "reason": "immediate_arrival_safe",
                "period": track.period,
            }

        # Immediate arrival causes collision — find minimal wait_k in [1, min(max_wait, period)]
        for wait_k in range(1, min(max_wait, track.period) + 1):
            delayed_pos = self.forecast_entity_position(
                feature_id, travel_steps + wait_k, grid_shape
            )
            if delayed_pos != target_pos:
                return {
                    "should_wait": True,
                    "wait_steps_recommended": wait_k,
                    "reason": f"collision_avoided_with_entity_{feature_id}",
                    "period": track.period,
                }

        return {
            "should_wait": True,
            "wait_steps_recommended": 1,
            "reason": "fallback_single_step_wait",
            "period": track.period,
        }

    def predict_state(
        self,
        snapshot: WorldStateSnapshot,
        action_intent: str,
        horizon_ms: int = 60000,
    ) -> tuple[dict[str, Any], float]:
        """Compute SNN temporal spike and state prediction."""
        predicted = dict(snapshot.variables)

        # If a grid snapshot exists, run LIF spiking and update kinematic tracks
        grid = snapshot.variables.get("grid")
        if isinstance(grid, np.ndarray) and grid.ndim == 2:
            self.update_tracks_from_grid(grid)
            spikes, v_mem = self.step_lif(grid.astype(np.float32) * 0.2)
            predicted["snn_spike_count"] = int(np.sum(spikes))
            predicted["snn_hazard_density"] = self.predict_hazard_density(grid, horizon_steps=2)

        for var_name, var_value in snapshot.variables.items():
            if isinstance(var_value, (int, float)) and "vibration" in var_name.lower():
                predicted[var_name] = var_value * 0.9  # Dampening spike pattern

        confidence = 0.90 if len(self.entity_tracks) > 0 else 0.88
        return predicted, confidence
