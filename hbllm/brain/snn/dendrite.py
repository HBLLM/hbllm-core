"""
Dendritic Neuron — Predictive coding via basal/apical compartments.

Implements a two-compartment neuron model for predictive coding:

    - **Basal dendrites**: Receive bottom-up sensory/evidence input
    - **Apical dendrites**: Receive top-down prediction/context input
    - **Soma**: Integrates both streams with match/mismatch detection

When basal and apical inputs match (prediction confirmed), the neuron
is *suppressed* — expected signals are not worth propagating. When they
mismatch, a *prediction error spike* fires with strength proportional
to the mismatch magnitude.

This implements the core mechanism of predictive processing:
    - Expected events → minimal response (efficient processing)
    - Unexpected events → strong spike (surprise signal)

Architecture::

    Top-down predictions (apical)
            ↓
    ┌───────────────────┐
    │  DendriticNeuron  │
    │                   │
    │  basal ──┐        │
    │          ├→ soma  │──→ SpikeEvent
    │  apical ─┘        │
    └───────────────────┘
            ↑
    Bottom-up evidence (basal)

Usage::

    from hbllm.brain.snn.dendrite import DendriticNeuron, DendriticConfig

    neuron = DendriticNeuron(DendriticConfig(), "pred_0")

    # Prediction matches evidence → suppressed
    spike = neuron.step_dual(basal=5.0, apical=5.0, timestamp=t)
    assert not spike.fired  # Expected — no error

    # Prediction mismatches evidence → prediction error
    spike = neuron.step_dual(basal=8.0, apical=2.0, timestamp=t)
    assert spike.fired  # Surprise — prediction error spike
"""

from __future__ import annotations

from dataclasses import dataclass

from hbllm.brain.snn.neurons import BaseNeuron, SpikeEvent

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DendriticConfig:
    """Configuration for dendritic predictive coding neurons.

    Attributes:
        threshold: Soma firing threshold.
        match_suppression: Suppression factor when prediction matches
            evidence. Higher values = stronger suppression of expected signals.
        mismatch_gain: Amplification of prediction error signals.
        basal_weight: Weight for bottom-up (evidence) input.
        apical_weight: Weight for top-down (prediction) input.
        leak: Membrane potential leak rate per timestep.
        refractory_period: Minimum time between spikes (seconds).
    """

    threshold: float = 1.0
    match_suppression: float = 0.8
    mismatch_gain: float = 2.0
    basal_weight: float = 1.0
    apical_weight: float = 1.0
    leak: float = 0.95
    refractory_period: float = 0.002


# ═══════════════════════════════════════════════════════════════════════════
# DendriticNeuron
# ═══════════════════════════════════════════════════════════════════════════


class DendriticNeuron(BaseNeuron):
    """Two-compartment neuron for predictive coding.

    Basal dendrites receive bottom-up evidence; apical dendrites
    receive top-down predictions. The soma computes a match/mismatch
    signal and fires prediction error spikes on surprise.

    Match detection:
        match_score = 1 - |basal - apical| / max(|basal|, |apical|, ε)

    When match_score is high (prediction ≈ evidence):
        → Effective input is suppressed → rarely fires

    When match_score is low (prediction ≠ evidence):
        → Effective input is amplified → prediction error spike

    This neuron is backward-compatible with ``BaseNeuron.step()``
    by treating all input as basal (no top-down context).
    """

    def __init__(self, config: DendriticConfig | None = None, neuron_id: str = "") -> None:
        self.neuron_id = neuron_id
        self.v = 0.0  # Membrane potential (managed internally as _potential)
        self.config = config or DendriticConfig()

        # Internal state
        self._potential: float = 0.0
        self._last_spike_time: float = 0.0
        self._spike_count: int = 0

        # Diagnostic accumulators
        self._total_match_score: float = 0.0
        self._step_count: int = 0

    # ── BaseNeuron interface ─────────────────────────────────────────

    def step(self, current: float, timestamp: float) -> SpikeEvent:
        """Single-input step (backward-compatible).

        All input is treated as basal (bottom-up evidence) with no
        top-down prediction context (apical=0).

        Args:
            current: Input current (treated as basal).
            timestamp: Current time in seconds.

        Returns:
            SpikeEvent indicating whether a prediction error fired.
        """
        return self.step_dual(basal=current, apical=0.0, timestamp=timestamp)

    def reset_state(self) -> None:
        """Reset neuron state to initial conditions."""
        self._potential = 0.0
        self._last_spike_time = 0.0
        self._spike_count = 0
        self._total_match_score = 0.0
        self._step_count = 0

    def reset(self) -> None:
        """Alias for reset_state (backward compatibility)."""
        self.reset_state()

    def get_type(self) -> str:
        """Return the neuron model type identifier."""
        return "dendritic"

    def get_state(self) -> dict:
        """Serialize neuron state."""
        return {
            "neuron_id": self.neuron_id,
            "potential": self._potential,
            "spike_count": self._spike_count,
            "avg_match_score": self.average_match_score,
        }

    # ── Dual-compartment step ────────────────────────────────────────

    def step_dual(
        self,
        basal: float,
        apical: float,
        timestamp: float,
    ) -> SpikeEvent:
        """Process dual-compartment input and generate prediction error.

        Args:
            basal: Bottom-up evidence input (sensory, data-driven).
            apical: Top-down prediction input (context, expectation).
            timestamp: Current time in seconds.

        Returns:
            SpikeEvent with strength proportional to prediction error.
            Fires when prediction significantly mismatches evidence.
        """
        cfg = self.config
        self._step_count += 1

        # Refractory period check
        if self._last_spike_time > 0:
            dt = timestamp - self._last_spike_time
            if dt < cfg.refractory_period:
                return SpikeEvent(fired=False, strength=0.0, timestamp=timestamp)

        # Compute match score: how well does prediction match evidence?
        abs_basal = abs(basal * cfg.basal_weight)
        abs_apical = abs(apical * cfg.apical_weight)
        max_magnitude = max(abs_basal, abs_apical, 1e-10)

        prediction_error = abs(basal * cfg.basal_weight - apical * cfg.apical_weight)
        match_score = 1.0 - min(1.0, prediction_error / max_magnitude)

        self._total_match_score += match_score

        # Compute effective input based on match/mismatch
        if match_score > 0.7:
            # Prediction matches evidence — suppress
            effective = basal * cfg.basal_weight * (1.0 - cfg.match_suppression * match_score)
        else:
            # Prediction error — amplify mismatch signal
            effective = prediction_error * cfg.mismatch_gain

        # Integrate into membrane potential (leaky integration)
        self._potential = self._potential * cfg.leak + effective

        # Fire if above threshold
        if self._potential >= cfg.threshold:
            self._potential = 0.0  # Reset
            self._spike_count += 1
            self._last_spike_time = timestamp

            # Spike strength encodes prediction error magnitude
            strength = min(1.0, prediction_error / max_magnitude)
            return SpikeEvent(fired=True, strength=strength, timestamp=timestamp)

        return SpikeEvent(fired=False, strength=0.0, timestamp=timestamp)

    # ── Diagnostics ──────────────────────────────────────────────────

    @property
    def average_match_score(self) -> float:
        """Average match score across all steps."""
        if self._step_count == 0:
            return 0.0
        return self._total_match_score / self._step_count

    @property
    def spike_count(self) -> int:
        """Total spikes fired."""
        return self._spike_count
