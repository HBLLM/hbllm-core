"""Cortical Oscillatory Phase-Precession & Fast Action Chunking.

Biologically grounded in hippocampal place cell theta-gamma phase precession
(O'Keefe & Recce, 1993; Lisman & Idiart, 1995; Skaggs et al., 1996):
1. Oscillatory Phase Coding:
   - A global theta rhythm (6-10 Hz, default 8 Hz) coordinates cognitive processing.
   - Subordinate gamma cycles (~40 Hz, 5-7 cycles nested per theta period) represent
     discrete sub-states or candidate action steps.
2. Phase Precession Dynamics:
   - As an agent traverses a receptive place field of length L, place firing phases
     precess linearly relative to the theta cycle:
     phi(x) = phi_0 - 2*pi * (x mod L) / L.
3. Fast Temporal Sequence Compression:
   - Behavioral spatial sequences spanning seconds are compressed into millisecond-level
     spike patterns within a single theta wave (~125 ms).
   - Facilitates one-shot Spike-Timing-Dependent Plasticity (STDP) for procedural action
     macro chunking into atomic skill primitives.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class ThetaPhase:
    """Current state of the global hippocampal theta oscillation."""

    cycle_index: int
    phase_rad: float  # [0, 2*pi)
    frequency_hz: float = 8.0

    @property
    def phase_degrees(self) -> float:
        """Phase in degrees [0, 360)."""
        return math.degrees(self.phase_rad)


@dataclass
class PhasePrecessionBurst:
    """Compressed temporal burst representing an action sequence within a theta cycle."""

    chunk_id: str
    ordered_actions: list[Any]
    relative_phases: list[float]  # Firing phase for each step in radians
    gamma_subcycles: list[int]  # Assigned gamma cycle index (0..K-1)
    compression_ratio: float  # Behavioral time / compressed oscillatory time
    timestamp: float

    def to_causal_predicate(self) -> CausalPredicate:
        """Export phase-compressed burst as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="phase_precession_burst",
            operator="==",
            value=f"actions:{len(self.ordered_actions)}:ratio:{self.compression_ratio:.1f}",
            relation="theta_gamma_chunking",
            target_entity="hippocampal_precession",
        )


class HippocampalPhasePrecessionEngine:
    """Oscillatory phase precession engine for temporal sequence compression and macro chunking."""

    def __init__(
        self,
        theta_frequency_hz: float = 8.0,
        gamma_cycles_per_theta: int = 7,
        place_field_length: float = 8.0,
    ) -> None:
        """Initialize hippocampal phase precession engine.

        Args:
            theta_frequency_hz: Frequency of the base theta oscillation (Hz).
            gamma_cycles_per_theta: Number of nested gamma sub-cycles per theta wave.
            place_field_length: Spatial extent of a typical place field unit.
        """
        self.theta_frequency = max(1.0, float(theta_frequency_hz))
        self.gamma_cycles = max(3, int(gamma_cycles_per_theta))
        self.place_field_length = max(1.0, float(place_field_length))

        self._current_phase: float = 0.0  # [0, 2*pi)
        self._cycle_counter: int = 0
        self._elapsed_time: float = 0.0

    @property
    def current_theta_phase(self) -> ThetaPhase:
        """Get the current global theta phase."""
        return ThetaPhase(
            cycle_index=self._cycle_counter,
            phase_rad=self._current_phase,
            frequency_hz=self.theta_frequency,
        )

    def step_theta_phase(self, dt: float) -> ThetaPhase:
        """Advance the continuous theta oscillation by dt seconds."""
        self._elapsed_time += max(0.0, dt)
        phase_increment = 2.0 * math.pi * self.theta_frequency * dt
        new_phase = self._current_phase + phase_increment

        # Handle cycle wrap-around
        cycles_elapsed = int(new_phase // (2.0 * math.pi))
        self._cycle_counter += cycles_elapsed
        self._current_phase = new_phase % (2.0 * math.pi)

        return self.current_theta_phase

    def compute_place_field_phase(
        self,
        position_along_field: float,
        entry_phase: float = math.pi * 1.8,
    ) -> float:
        """Calculate firing phase relative to theta cycle as agent progresses through place field.

        At entry (x = 0), fires late near entry_phase (~320 deg).
        At exit (x = L), fires early near (entry_phase - 2*pi) (~0 deg).
        """
        norm_pos = max(0.0, min(1.0, position_along_field / self.place_field_length))
        raw_phase = entry_phase - (2.0 * math.pi * norm_pos)
        return float(raw_phase % (2.0 * math.pi))

    def encode_trajectory_burst(
        self,
        positions: list[tuple[int, int]],
        actions: list[Any],
        behavioral_duration_s: float = 2.0,
    ) -> PhasePrecessionBurst:
        """Compress a behavioral trajectory into a nested theta-gamma burst sequence.

        Args:
            positions: Sequence of (r, c) coordinates traversed.
            actions: Sequence of action commands executed.
            behavioral_duration_s: Real-world execution duration in seconds.

        Returns:
            PhasePrecessionBurst containing ordered actions and gamma phase assignments.
        """
        n_steps = len(actions)
        if n_steps == 0:
            return PhasePrecessionBurst(
                chunk_id=f"burst_{uuid.uuid4().hex[:8]}",
                ordered_actions=[],
                relative_phases=[],
                gamma_subcycles=[],
                compression_ratio=1.0,
                timestamp=self._elapsed_time,
            )

        # Calculate cumulative distance along path
        distances: list[float] = [0.0]
        for i in range(1, len(positions)):
            dr = positions[i][0] - positions[i - 1][0]
            dc = positions[i][1] - positions[i - 1][1]
            dist = math.sqrt(dr * dr + dc * dc)
            distances.append(distances[-1] + dist)

        total_dist = max(1.0, distances[-1])

        relative_phases: list[float] = []
        gamma_subcycles: list[int] = []

        gamma_cycle_width = (2.0 * math.pi) / self.gamma_cycles

        for i in range(n_steps):
            d = distances[min(i, len(distances) - 1)]
            # Phase precesses across trajectory
            phase = self.compute_place_field_phase((d / total_dist) * self.place_field_length)
            relative_phases.append(phase)

            # Assign gamma subcycle slot (0..gamma_cycles-1)
            gamma_idx = int(phase // gamma_cycle_width) % self.gamma_cycles
            gamma_subcycles.append(gamma_idx)

        # Oscillatory duration = 1 theta cycle = 1 / theta_frequency (~125 ms)
        oscillatory_time_s = 1.0 / self.theta_frequency
        compression_ratio = (
            behavioral_duration_s / oscillatory_time_s if oscillatory_time_s > 0 else 1.0
        )

        burst = PhasePrecessionBurst(
            chunk_id=f"burst_{uuid.uuid4().hex[:8]}",
            ordered_actions=actions,
            relative_phases=relative_phases,
            gamma_subcycles=gamma_subcycles,
            compression_ratio=float(compression_ratio),
            timestamp=self._elapsed_time,
        )
        return burst

    def chunk_into_macro_primitive(
        self,
        burst: PhasePrecessionBurst,
        min_steps: int = 2,
    ) -> dict[str, Any] | None:
        """Transform a phase-compressed burst into a procedural macro candidate.

        Returns None if sequence is too short to warrant chunking.
        """
        if len(burst.ordered_actions) < min_steps:
            return None

        return {
            "macro_id": f"macro_{burst.chunk_id}",
            "action_sequence": list(burst.ordered_actions),
            "step_count": len(burst.ordered_actions),
            "compression_ratio": burst.compression_ratio,
            "gamma_pattern": list(burst.gamma_subcycles),
            "created_at": burst.timestamp,
        }

    def reset(self) -> None:
        """Reset internal phase counter and clock."""
        self._current_phase = 0.0
        self._cycle_counter = 0
        self._elapsed_time = 0.0
