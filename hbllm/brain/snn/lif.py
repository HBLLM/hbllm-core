"""
Leaky Integrate-and-Fire (LIF) Neuron Model and Accumulator.

Implements a time-dependent spiking neural model where decay is computed
relative to real elapsed time (dt in seconds) rather than discrete cycles,
making it suitable for asynchronous event-driven pipelines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from hbllm.brain.snn.neurons import BaseNeuron, SpikeEvent

# Re-export SpikeEvent so existing imports from lif.py still work:
#   from hbllm.brain.snn.lif import SpikeEvent
__all__ = ["LIFConfig", "LIFNeuron", "SpikeEvent", "SpikingAccumulator"]


@dataclass
class LIFConfig:
    """Configuration parameter set for a LIF neuron."""

    threshold: float = 1.0
    decay_half_life: float = 1.0  # seconds; time for potential to decay by 50%
    reset_potential: float = 0.0
    refractory_period: float = 0.0  # seconds; cooldown period after firing
    is_inhibitory: bool = False  # If True, this neuron's output current is inverted

    # Homeostatic plasticity: auto-adjust threshold to maintain target firing rate
    target_firing_rate: float = 0.0  # 0.0 = disabled; e.g., 0.1 = fire 10% of steps
    adaptation_rate: float = 0.01  # How fast threshold adapts per step


class LIFNeuron(BaseNeuron):
    """
    Leaky Integrate-and-Fire (LIF) Neuron.

    Accumulates input current over time, decays exponentially, and fires a spike
    once the membrane potential crosses a configured threshold.

    Extends ``BaseNeuron`` (from ``neurons.py``) for polymorphic use in
    ``NeuronLayer``, ``LayerProjection``, and ``SpikingNetwork``.
    """

    def __init__(self, config: LIFConfig, neuron_id: str = "unknown") -> None:
        self.config = config
        self.neuron_id = neuron_id
        self.v = 0.0  # Membrane potential
        self.last_reported_v = 0.0
        self.last_update_time: float | None = None
        self.refractory_time_remaining: float = 0.0

        # Homeostatic state
        self._firing_history: list[bool] = []  # Recent firing history (ring buffer)
        self._homeostatic_window: int = 50  # Steps to average over
        self._effective_threshold: float = config.threshold  # May drift via homeostasis

    def get_type(self) -> str:
        """Return ``"lif"`` — identifies this neuron model for serialization."""
        return "lif"

    def step(self, current: float, timestamp: float) -> SpikeEvent:
        """
        Advance the state of the neuron by applying decay and input current.

        Args:
            current: Input value/charge to accumulate on this step.
            timestamp: The epoch timestamp (in seconds) of this step.

        Returns:
            A SpikeEvent detailing if a spike occurred and its strength.
        """
        if self.last_update_time is None:
            self.last_update_time = timestamp

        dt = max(0.0, timestamp - self.last_update_time)
        self.last_update_time = timestamp

        # 1. Update refractory timer
        if self.refractory_time_remaining > 0:
            self.refractory_time_remaining = max(0.0, self.refractory_time_remaining - dt)

        # 2. Decay membrane potential (only if we have non-zero elapsed time)
        if dt > 0.0:
            if self.config.decay_half_life <= 0:
                decay_factor = 0.0
            else:
                # v(t) = v(t0) * 2^(-dt / t_half)
                decay_factor = 2.0 ** (-dt / self.config.decay_half_life)
            self.v *= decay_factor

        # 3. Accumulate current if not in refractory period
        if self.refractory_time_remaining <= 0:
            self.v += current

        # 4. Check if threshold is crossed (use effective threshold for homeostasis)
        fired = False
        strength = 0.0
        if self.v >= self._effective_threshold:
            fired = True
            # Strength is proportional to the overshoot ratio
            strength = self.v / self._effective_threshold
            # Reset membrane potential
            self.v = self.config.reset_potential
            # Activate refractory period
            self.refractory_time_remaining = self.config.refractory_period

        # 5. Homeostatic plasticity: adapt threshold to maintain target firing rate
        if self.config.target_firing_rate > 0.0:
            self._firing_history.append(fired)
            if len(self._firing_history) > self._homeostatic_window:
                self._firing_history = self._firing_history[-self._homeostatic_window :]
            if len(self._firing_history) >= self._homeostatic_window:
                actual_rate = sum(self._firing_history) / len(self._firing_history)
                error = actual_rate - self.config.target_firing_rate
                # If firing too much, raise threshold; too little, lower it
                self._effective_threshold += error * self.config.adaptation_rate
                # Clamp to reasonable range
                self._effective_threshold = max(
                    self.config.threshold * 0.5,
                    min(self.config.threshold * 2.0, self._effective_threshold),
                )

        # 5. Record SNN telemetry via MetricsCollector (Tier 2 delta-filtered)
        if (
            abs(self.v - self.last_reported_v) >= 0.05
            or fired
            or (self.v == 0.0 and self.last_reported_v != 0.0)
        ):
            from hbllm.network.metrics import MetricsCollector

            MetricsCollector.get_instance().record_snn_potential(self.neuron_id, self.v)
            self.last_reported_v = self.v

        if fired:
            from hbllm.network.metrics import MetricsCollector

            MetricsCollector.get_instance().record_snn_spike(self.neuron_id, strength)

        return SpikeEvent(fired=fired, strength=strength, timestamp=timestamp)

    def reset_state(self) -> None:
        """Reset the dynamic states of the neuron."""
        self.v = 0.0
        self.last_reported_v = 0.0
        self.last_update_time = None
        self.refractory_time_remaining = 0.0
        self._firing_history = []
        self._effective_threshold = self.config.threshold


class SpikingAccumulator:
    """
    A generic accumulator primitive wrapping a LIFNeuron.

    Can be used by any cognitive node to track quantities like attention fatigue,
    error counts, resource load, or threat level over time.
    """

    def __init__(self, config: LIFConfig, neuron_id: str = "unknown") -> None:
        self.neuron = LIFNeuron(config, neuron_id=neuron_id)

    def stimulate(self, value: float, timestamp: float | None = None) -> SpikeEvent:
        """
        Inject charge/stimulus into the accumulator.

        Args:
            value: Magnitude of the stimulus.
            timestamp: Optional custom timestamp. Defaults to time.time().
        """
        if timestamp is None:
            timestamp = time.time()
        return self.neuron.step(value, timestamp)

    def get_potential(self, timestamp: float | None = None) -> float:
        """
        Fetch the current decayed membrane potential without adding any stimulus.

        Args:
            timestamp: Optional custom timestamp. Defaults to time.time().
        """
        if timestamp is None:
            timestamp = time.time()
        self.neuron.step(0.0, timestamp)
        return self.neuron.v

    def reset(self) -> None:
        """Reset the accumulator state."""
        self.neuron.reset_state()
