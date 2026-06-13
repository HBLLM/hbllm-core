"""
Role-Specialized Neuron Ensemble for input comprehension.

Implements a 5-channel LIF neuron ensemble where each channel has a
specialized semantic role rather than simply different timescales.

Channels:
    entity     — fast, detects entity/topic shifts
    clause     — medium, detects clause-level coherence boundaries (PRIMARY)
    discourse  — slow, detects discourse-level intent shifts
    surprise   — detects anomalies/contradictions
    constraint — detects qualifiers ("only", "except", "but")

The clause channel is the primary concept boundary trigger.
Other channels provide metadata about what's inside each concept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hbllm.brain.snn.lif import LIFConfig, LIFNeuron, SpikeEvent

if TYPE_CHECKING:
    from hbllm.brain.snn.plasticity import PlasticWeightMatrix


@dataclass
class ConceptSpike:
    """A spike from the comprehension ensemble."""

    channel: str  # Which neuron fired
    strength: float  # Overshoot ratio
    concept_text: str  # Accumulated buffer at spike time
    timestamp: float


# ── Per-domain parameter sets ─────────────────────────────────────────────

DOMAIN_PARAMS: dict[str, dict[str, float]] = {
    "general": {
        "entity_threshold": 0.6,
        "clause_threshold": 0.8,
        "discourse_threshold": 1.2,
    },
    "code": {
        # Code has dense, structured tokens — need higher thresholds
        # to avoid spiking on every identifier
        "entity_threshold": 0.8,
        "clause_threshold": 1.0,
        "discourse_threshold": 1.5,
    },
    "math": {
        # Math has strict boundaries — lower thresholds for
        # precise segmentation
        "entity_threshold": 0.5,
        "clause_threshold": 0.7,
        "discourse_threshold": 1.0,
    },
    "dialogue": {
        # Short bursts — fast, low thresholds
        "entity_threshold": 0.5,
        "clause_threshold": 0.6,
        "discourse_threshold": 0.9,
    },
}


class ComprehensionEnsemble:
    """5-channel neuron ensemble for input comprehension.

    Each channel has a specialized role and responds to different
    signal combinations.  All channels share the same word buffer
    but maintain independent membrane potentials.
    """

    def __init__(
        self,
        domain: str = "general",
        plastic_weights: PlasticWeightMatrix | None = None,
    ) -> None:
        self.domain = domain
        self.plastic_weights = plastic_weights
        params = DOMAIN_PARAMS.get(domain, DOMAIN_PARAMS["general"])

        self.channels: dict[str, LIFNeuron] = {
            # Fast: detects entity/topic shifts (low threshold, fast decay)
            "entity": LIFNeuron(
                config=LIFConfig(
                    threshold=params["entity_threshold"],
                    decay_half_life=0.3,
                    reset_potential=0.0,
                    refractory_period=0.02,
                ),
                neuron_id=f"comprehension.entity.{domain}",
            ),
            # Medium: detects clause-level coherence boundaries
            "clause": LIFNeuron(
                config=LIFConfig(
                    threshold=params["clause_threshold"],
                    decay_half_life=0.8,
                    reset_potential=0.1,
                    refractory_period=0.05,
                ),
                neuron_id=f"comprehension.clause.{domain}",
            ),
            # Slow: detects discourse-level intent shifts
            "discourse": LIFNeuron(
                config=LIFConfig(
                    threshold=params["discourse_threshold"],
                    decay_half_life=2.0,
                    reset_potential=0.2,
                    refractory_period=0.1,
                ),
                neuron_id=f"comprehension.discourse.{domain}",
            ),
            # Surprise: detects anomalies/contradictions
            "surprise": LIFNeuron(
                config=LIFConfig(
                    threshold=0.9,
                    decay_half_life=0.5,
                    reset_potential=0.0,
                    refractory_period=0.1,
                ),
                neuron_id=f"comprehension.surprise.{domain}",
            ),
            # Constraint: detects qualifiers ("only", "except", "but")
            "constraint": LIFNeuron(
                config=LIFConfig(
                    threshold=0.6,
                    decay_half_life=0.4,
                    reset_potential=0.0,
                    refractory_period=0.05,
                ),
                neuron_id=f"comprehension.constraint.{domain}",
            ),
        }

        # Signal routing: which signals feed which neurons
        self._signal_weights: dict[str, dict[str, float]] = {
            "entity": {
                "semantic_weight": 0.5,
                "topic_shift": 0.3,
                "novelty": 0.2,
            },
            "clause": {
                "punctuation": 0.3,
                "buffer_pressure": 0.3,
                "topic_shift": 0.2,
                "semantic_weight": 0.2,
            },
            "discourse": {
                "topic_shift": 0.4,
                "inter_novelty": 0.3,
                "buffer_pressure": 0.2,
                "novelty": 0.1,
            },
            "surprise": {
                "inter_novelty": 0.5,
                "constraint": 0.3,
                "novelty": 0.2,
            },
            "constraint": {
                "constraint": 0.7,
                "semantic_weight": 0.2,
                "punctuation": 0.1,
            },
        }

    def step(
        self,
        signals: dict[str, float],
        timestamp: float,
    ) -> list[tuple[str, SpikeEvent]]:
        """Feed signals to all channels.

        If a ``PlasticWeightMatrix`` is attached, uses learned weights
        instead of static ones and records spike timing for STDP updates.

        Returns list of (channel_name, spike_event) for any that fired.
        """
        # Record pre-synaptic activity for STDP
        if self.plastic_weights is not None:
            self.plastic_weights.record_signals(signals, timestamp)

        fired: list[tuple[str, SpikeEvent]] = []

        for channel_name, neuron in self.channels.items():
            # Use learned weights if available, otherwise static
            if self.plastic_weights is not None:
                weights = self.plastic_weights.get_weights(channel_name)
            else:
                weights = self._signal_weights[channel_name]

            current = sum(
                signals.get(sig, 0.0) * weight for sig, weight in weights.items()
            )

            spike = neuron.step(current, timestamp)
            if spike.fired:
                fired.append((channel_name, spike))

        # Record post-synaptic spikes for STDP
        if self.plastic_weights is not None and fired:
            fired_channels = [ch for ch, _ in fired]
            self.plastic_weights.record_spikes(fired_channels, timestamp)

        return fired

    def reset(self) -> None:
        """Reset all neuron states."""
        for neuron in self.channels.values():
            neuron.reset_state()

    def update_params(self, params: dict[str, float]) -> None:
        """Update neuron thresholds from a parameter dict.

        Used by SNNCalibrator to apply tuned parameters.
        """
        if "entity_threshold" in params:
            self.channels["entity"].config.threshold = params["entity_threshold"]
        if "clause_threshold" in params:
            self.channels["clause"].config.threshold = params["clause_threshold"]
        if "discourse_threshold" in params:
            self.channels["discourse"].config.threshold = params["discourse_threshold"]
