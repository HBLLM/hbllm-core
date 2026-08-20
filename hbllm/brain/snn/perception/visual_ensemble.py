"""Visual Perception Ensemble — SNN-gated visual stream.

A 5-channel LIF neuron ensemble specialized for visual perception.
Each channel detects a different visual event type.

Channels:
    scene     — detects scene-level changes (room, environment)
    entity    — detects new/moved objects
    motion    — detects significant motion
    novelty   — detects unfamiliar visual patterns
    stability — detects scene stabilization after changes

Architecture:
    VisualSignals → PerceptionEnsemble → SpikeEvent[] → PerceptionGateDecision

The ensemble maps cheap visual signals to SNN spikes.
PerceptionGateDecision then determines the processing level
(NONE, LOW, STANDARD, HIGH, URGENT) for each frame.
"""

from __future__ import annotations

import logging
import time

from hbllm.brain.snn.lif import LIFConfig, LIFNeuron
from hbllm.brain.snn.neurons import SpikeEvent
from hbllm.brain.snn.perception.gate import PerceptionGateDecision
from hbllm.brain.snn.perception.visual_signals import VisualSignals

logger = logging.getLogger(__name__)


# ── Channel configurations ────────────────────────────────────────────
# LIFConfig: threshold, decay_half_life, reset_potential, refractory_period

VISUAL_CHANNEL_CONFIGS: dict[str, LIFConfig] = {
    "scene": LIFConfig(
        threshold=0.8,
        decay_half_life=5.0,  # Slow — accumulates over many frames
        reset_potential=0.0,
        refractory_period=2.0,  # 2s cooldown
    ),
    "entity": LIFConfig(
        threshold=0.6,
        decay_half_life=2.0,  # Medium — responds to individual objects
        reset_potential=0.0,
        refractory_period=1.0,
    ),
    "motion": LIFConfig(
        threshold=0.4,
        decay_half_life=0.5,  # Fast — immediate motion response
        reset_potential=0.0,
        refractory_period=0.3,
    ),
    "novelty": LIFConfig(
        threshold=0.7,
        decay_half_life=3.0,  # Medium-slow — needs consistent novelty
        reset_potential=0.0,
        refractory_period=1.5,
    ),
    "stability": LIFConfig(
        threshold=0.9,
        decay_half_life=10.0,  # Very slow — detects extended calm
        reset_potential=0.0,
        refractory_period=5.0,
    ),
}


# ── Signal-to-channel weight matrix ──────────────────────────────────

# Each row = channel, columns = [motion, intensity, edge, color, texture]
SIGNAL_WEIGHTS: dict[str, list[float]] = {
    "scene": [0.3, 0.4, 0.2, 0.3, 0.1],  # Intensity + color dominant
    "entity": [0.2, 0.1, 0.4, 0.2, 0.3],  # Edge + texture dominant
    "motion": [0.8, 0.1, 0.1, 0.0, 0.0],  # Pure motion
    "novelty": [0.1, 0.2, 0.2, 0.3, 0.3],  # Color + texture dominant
    "stability": [-0.5, -0.2, -0.1, -0.1, -0.1],  # Negative: calm = no change
}


class PerceptionEnsemble:
    """5-channel SNN ensemble for visual perception gating.

    Converts cheap visual signals into spike events that drive
    the PerceptionGateDecision for each frame.

    Usage::

        ensemble = PerceptionEnsemble()
        signals = extractor.extract(frame)
        decision = ensemble.step(signals, frame_index=42)

        if decision.should_process:
            # Run expensive perception
            assessment = await runtime.perceive(frame)
    """

    def __init__(self, configs: dict[str, LIFConfig] | None = None) -> None:
        cfgs = configs or VISUAL_CHANNEL_CONFIGS
        self.neurons: dict[str, LIFNeuron] = {
            name: LIFNeuron(config=cfg, neuron_id=f"vis_{name}") for name, cfg in cfgs.items()
        }
        self._frame_count = 0
        self._start_time = time.time()

    def step(
        self,
        signals: VisualSignals,
        frame_index: int | None = None,
    ) -> PerceptionGateDecision:
        """Process one frame's signals through the ensemble.

        Args:
            signals: Cheap visual features from VisualSignalExtractor.
            frame_index: Optional frame index for decision tracking.

        Returns:
            PerceptionGateDecision with processing level.

        """
        self._frame_count += 1
        idx = frame_index if frame_index is not None else self._frame_count
        timestamp = self._start_time + self._frame_count * 0.033  # ~30fps simulated

        signal_vec = [
            signals.motion,
            signals.intensity,
            signals.edge,
            signals.color,
            signals.texture,
        ]

        fired: list[tuple[str, SpikeEvent]] = []

        for channel_name, neuron in self.neurons.items():
            weights = SIGNAL_WEIGHTS[channel_name]
            # Weighted sum of signals → current input
            current = sum(w * s for w, s in zip(weights, signal_vec))

            spike = neuron.step(current, timestamp)
            if spike.fired:
                fired.append((channel_name, spike))
                logger.debug(
                    "Visual spike: channel=%s, strength=%.3f, frame=%d",
                    channel_name,
                    spike.strength,
                    idx,
                )

        return PerceptionGateDecision.from_spikes(fired, frame_index=idx)

    def reset(self) -> None:
        """Reset all neurons (new video stream)."""
        for neuron in self.neurons.values():
            neuron.reset_state()
        self._frame_count = 0
        self._start_time = time.time()

    @property
    def state(self) -> dict[str, float]:
        """Current membrane potentials for debugging."""
        return {name: neuron.v for name, neuron in self.neurons.items()}
