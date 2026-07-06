"""
Neural Oscillations — Rhythmic gating for cognitive subsystem coordination.

Provides natural synchronization of cognitive subsystems through
oscillatory phase gating, replacing manual coordination logic.

Oscillation bands and their cognitive roles:
    - **Gamma (30–100 Hz)**: Attention binding, active processing
    - **Beta (13–30 Hz)**: Planning, motor control, maintaining status quo
    - **Theta (4–8 Hz)**: Memory encoding/retrieval, navigation
    - **Delta (0.5–4 Hz)**: Sleep, deep consolidation

Phase-amplitude coupling enables cross-frequency coordination:
    Theta phase gates gamma bursts → memory-guided attention.
    Delta phase gates theta → sleep-guided consolidation.

Architecture::

    Instead of manually synchronizing:
        "when attention fires, trigger memory retrieval"

    Oscillations naturally coordinate:
        theta_peak → memory gate opens → retrieval permitted
        gamma_peak → attention gate opens → processing active
        delta_peak → consolidation gate opens → sleep activity

Usage::

    from hbllm.brain.snn.oscillations import OscillationManager, OscillationBand

    osc = OscillationManager()

    # Check if memory gate is open
    gate = osc.get_gate(OscillationBand.THETA, timestamp=t)
    if gate > 0.5:
        # Memory retrieval is permitted at this phase

    # Neuromodulation can modulate oscillation frequency
    osc.modulate_frequency(OscillationBand.GAMMA, factor=1.2)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Oscillation Bands
# ═══════════════════════════════════════════════════════════════════════════


class OscillationBand(StrEnum):
    """Standard neural oscillation frequency bands."""

    GAMMA = "gamma"  # 30–100 Hz — attention binding, active processing
    BETA = "beta"  # 13–30 Hz — planning, motor control, status quo
    THETA = "theta"  # 4–8 Hz — memory encoding/retrieval
    DELTA = "delta"  # 0.5–4 Hz — sleep, deep consolidation


# ═══════════════════════════════════════════════════════════════════════════
# BrainTick — rich heartbeat event
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BrainTick:
    """Rich heartbeat event published on each oscillation cycle.

    Every cognitive subsystem subscribes to BrainTick rather than
    polling the OscillationManager directly.

    Attributes:
        phase: Current phase for each oscillation band.
        gate: Current gate value for each band [0.0, 1.0].
        cycle: Monotonic cycle counter.
        timestamp: When this tick was generated (epoch seconds).
        cognitive_load: System-wide cognitive load [0.0, 1.0].
        attention_level: Current attention level [0.0, 1.0].
        fatigue: System fatigue level [0.0, 1.0].
        dominant_band: The band with highest gate value.
    """

    phase: dict[str, float] = field(default_factory=dict)
    gate: dict[str, float] = field(default_factory=dict)
    cycle: int = 0
    timestamp: float = 0.0
    cognitive_load: float = 0.0
    attention_level: float = 1.0
    fatigue: float = 0.0
    dominant_band: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": {k: round(v, 4) for k, v in self.phase.items()},
            "gate": {k: round(v, 4) for k, v in self.gate.items()},
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "cognitive_load": round(self.cognitive_load, 3),
            "attention_level": round(self.attention_level, 3),
            "fatigue": round(self.fatigue, 3),
            "dominant_band": self.dominant_band,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Band Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BandConfig:
    """Configuration for a single oscillation band.

    Attributes:
        frequency: Center frequency in Hz.
        amplitude: Oscillation amplitude [0.0, 1.0].
        phase_offset: Initial phase offset in radians.
        gate_threshold: Phase value above which the gate is "open".
    """

    frequency: float
    amplitude: float = 1.0
    phase_offset: float = 0.0
    gate_threshold: float = 0.5


# Default biological-inspired configurations
DEFAULT_BAND_CONFIGS: dict[OscillationBand, BandConfig] = {
    OscillationBand.GAMMA: BandConfig(frequency=40.0, amplitude=1.0),
    OscillationBand.BETA: BandConfig(frequency=20.0, amplitude=0.8),
    OscillationBand.THETA: BandConfig(frequency=6.0, amplitude=0.9),
    OscillationBand.DELTA: BandConfig(frequency=2.0, amplitude=1.0),
}


# ═══════════════════════════════════════════════════════════════════════════
# OscillationManager
# ═══════════════════════════════════════════════════════════════════════════


class OscillationManager:
    """Coordinates cognitive subsystems through rhythmic phase gating.

    Each oscillation band runs at its configured frequency. Subsystems
    can query the current phase or gate value to determine whether
    they should be active at a given moment.

    Phase-amplitude coupling enables cross-frequency coordination:
    the phase of a slow oscillation modulates the amplitude of a
    faster oscillation.

    Args:
        configs: Per-band configuration overrides.
        reference_time: Reference timestamp for phase computation.
    """

    def __init__(
        self,
        configs: dict[OscillationBand, BandConfig] | None = None,
        reference_time: float = 0.0,
    ) -> None:
        self._configs: dict[OscillationBand, BandConfig] = {}
        for band in OscillationBand:
            self._configs[band] = (configs or DEFAULT_BAND_CONFIGS).get(
                band, DEFAULT_BAND_CONFIGS[band]
            )

        self._reference_time = reference_time
        self._cycle_count: int = 0
        self._frequency_modifiers: dict[OscillationBand, float] = {
            band: 1.0 for band in OscillationBand
        }

    def get_phase(self, band: OscillationBand, timestamp: float) -> float:
        """Get the current phase of an oscillation band.

        Args:
            band: The oscillation band to query.
            timestamp: Current time in seconds.

        Returns:
            Phase in radians [0, 2π).
        """
        config = self._configs[band]
        effective_freq = config.frequency * self._frequency_modifiers[band]
        dt = timestamp - self._reference_time
        phase = (2.0 * math.pi * effective_freq * dt + config.phase_offset) % (2.0 * math.pi)
        return phase

    def get_gate(self, band: OscillationBand, timestamp: float) -> float:
        """Get the gating value for a band.

        The gate value determines whether a subsystem should be active.
        Based on a raised-cosine window centered at the peak phase.

        Args:
            band: The oscillation band to query.
            timestamp: Current time in seconds.

        Returns:
            Gate value [0.0, 1.0]. High = gate open, subsystem active.
        """
        config = self._configs[band]
        phase = self.get_phase(band, timestamp)
        # Raised cosine: peaks at phase=0 (or 2π), trough at phase=π
        gate = config.amplitude * (1.0 + math.cos(phase)) / 2.0
        return gate

    def get_amplitude(
        self,
        band: OscillationBand,
        timestamp: float,
        modulating_band: OscillationBand | None = None,
    ) -> float:
        """Get the amplitude of a band, optionally modulated by another.

        Implements phase-amplitude coupling: the phase of the
        modulating (slower) band modulates the amplitude of the
        target (faster) band.

        Example: theta phase modulates gamma amplitude →
        memory-guided attention bursts.

        Args:
            band: The target oscillation band.
            timestamp: Current time in seconds.
            modulating_band: Optional slower band for cross-frequency coupling.

        Returns:
            Effective amplitude [0.0, 1.0].
        """
        config = self._configs[band]
        base_amplitude = config.amplitude

        if modulating_band is not None:
            # Phase-amplitude coupling
            modulating_gate = self.get_gate(modulating_band, timestamp)
            return base_amplitude * modulating_gate

        return base_amplitude

    def modulate_frequency(self, band: OscillationBand, factor: float) -> None:
        """Modulate the frequency of an oscillation band.

        Used by neuromodulation to speed up or slow down rhythms.
        For example, increased norepinephrine speeds up gamma
        for heightened attention.

        Args:
            band: The band to modulate.
            factor: Multiplicative factor (>1 = faster, <1 = slower).
                Clamped to [0.1, 10.0] for stability.
        """
        factor = max(0.1, min(10.0, factor))
        self._frequency_modifiers[band] = factor
        logger.debug(
            "Oscillation %s frequency modulated by %.2f",
            band.value,
            factor,
        )

    def reset_modulation(self) -> None:
        """Reset all frequency modifiers to 1.0."""
        for band in OscillationBand:
            self._frequency_modifiers[band] = 1.0

    def get_dominant_band(self, timestamp: float) -> OscillationBand:
        """Return the band with the highest gate value at this moment.

        Useful for determining which cognitive mode should be
        dominant (attention vs memory vs sleep).

        Args:
            timestamp: Current time in seconds.

        Returns:
            The OscillationBand with the highest gate value.
        """
        best_band = OscillationBand.GAMMA
        best_gate = -1.0
        for band in OscillationBand:
            gate = self.get_gate(band, timestamp)
            if gate > best_gate:
                best_gate = gate
                best_band = band
        return best_band

    def snapshot(self, timestamp: float) -> dict[str, dict[str, float]]:
        """Capture the current state of all oscillation bands.

        Args:
            timestamp: Current time in seconds.

        Returns:
            Dict mapping band names to their phase, gate, and amplitude.
        """
        result: dict[str, dict[str, float]] = {}
        for band in OscillationBand:
            result[band.value] = {
                "phase": round(self.get_phase(band, timestamp), 4),
                "gate": round(self.get_gate(band, timestamp), 4),
                "amplitude": round(self.get_amplitude(band, timestamp), 4),
                "frequency": round(
                    self._configs[band].frequency * self._frequency_modifiers[band],
                    2,
                ),
            }
        return result

    def stats(self) -> dict[str, Any]:
        """Manager statistics."""
        return {
            "bands": len(self._configs),
            "cycle_count": self._cycle_count,
            "frequency_modifiers": {
                b.value: round(m, 2) for b, m in self._frequency_modifiers.items()
            },
        }

    def generate_tick(
        self,
        timestamp: float,
        cognitive_load: float = 0.0,
        attention_level: float = 1.0,
        fatigue: float = 0.0,
    ) -> BrainTick:
        """Generate a BrainTick heartbeat event.

        Captures the current oscillation state and system metrics
        into a single tick that subsystems can subscribe to.

        Args:
            timestamp: Current time in seconds.
            cognitive_load: System cognitive load [0, 1].
            attention_level: Current attention level [0, 1].
            fatigue: System fatigue [0, 1].

        Returns:
            A BrainTick event.
        """
        self._cycle_count += 1

        phase_dict: dict[str, float] = {}
        gate_dict: dict[str, float] = {}
        for band in OscillationBand:
            phase_dict[band.value] = self.get_phase(band, timestamp)
            gate_dict[band.value] = self.get_gate(band, timestamp)

        dominant = self.get_dominant_band(timestamp)

        return BrainTick(
            phase=phase_dict,
            gate=gate_dict,
            cycle=self._cycle_count,
            timestamp=timestamp,
            cognitive_load=cognitive_load,
            attention_level=attention_level,
            fatigue=fatigue,
            dominant_band=dominant.value,
        )
