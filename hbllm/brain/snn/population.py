"""
Population Coding — Distributed value representation via neuron ensembles.

Encodes scalar values as spike patterns across a population of neurons,
and decodes population activity back to scalar values with uncertainty.

This is how biological neural systems represent continuous values:
instead of a single neuron encoding "0.7", an ensemble of neurons
with overlapping tuning curves collectively represent it.

Features:
    - **Encoding**: Gaussian tuning curves distribute a value across N neurons
    - **Decoding**: Population vector decoding recovers the original value
    - **Uncertainty**: Width of decoded distribution indicates confidence
    - **Noise tolerance**: Distributed representation is robust to single-neuron failure

``CognitiveStateEncoder`` extends this to encode/decode all fields of
``CognitiveState`` as population spike patterns for SNN processing.

Architecture::

    Scalar value (e.g., urgency = 0.7)
        ↓ encode
    Population: [0.1, 0.3, 0.8, ■0.95■, 0.7, 0.2, 0.05]
                                  ↑ peak near preferred value
        ↓ process through SNN
    Population: [0.05, 0.2, 0.7, ■0.9■, 0.6, 0.15, 0.02]
        ↓ decode
    Scalar value ≈ 0.72 ± 0.03
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from hbllm.brain.snn.neurons import SpikeEvent

# ═══════════════════════════════════════════════════════════════════════════
# PopulationEncoder — Gaussian tuning curve encoding/decoding
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PopulationConfig:
    """Configuration for population encoding.

    Attributes:
        num_neurons: Number of neurons in the population.
        min_value: Minimum encoded value.
        max_value: Maximum encoded value.
        tuning_width: Width (σ) of Gaussian tuning curves.
            Narrower → sharper encoding, less overlap.
    """

    num_neurons: int = 16
    min_value: float = 0.0
    max_value: float = 1.0
    tuning_width: float = 0.1


class PopulationEncoder:
    """Encodes scalar values as population spike patterns.

    Each neuron has a "preferred value" (center of its tuning curve).
    When a value is presented, each neuron's response is proportional
    to the Gaussian of the distance from its preferred value.

    Args:
        config: Population encoding configuration.
    """

    def __init__(self, config: PopulationConfig | None = None) -> None:
        self.config = config or PopulationConfig()
        cfg = self.config

        # Compute preferred values (evenly spaced)
        self._preferred: list[float] = []
        for i in range(cfg.num_neurons):
            pref = cfg.min_value + (cfg.max_value - cfg.min_value) * i / max(1, cfg.num_neurons - 1)
            self._preferred.append(pref)

    def encode(self, value: float) -> list[float]:
        """Encode a scalar value as population activity.

        Args:
            value: Scalar value to encode (clamped to [min, max]).

        Returns:
            List of activation levels [0.0, 1.0] for each neuron.
        """
        cfg = self.config
        value = max(cfg.min_value, min(cfg.max_value, value))
        sigma = cfg.tuning_width

        activations: list[float] = []
        for pref in self._preferred:
            dist = (value - pref) / max(sigma, 1e-10)
            activation = math.exp(-0.5 * dist * dist)
            activations.append(activation)
        return activations

    def decode(self, activations: list[float]) -> float:
        """Decode population activity back to a scalar value.

        Uses population vector decoding (weighted average of
        preferred values by activation levels).

        Args:
            activations: Activation levels for each neuron.

        Returns:
            Decoded scalar value.
        """
        total_weight = sum(activations)
        if total_weight < 1e-10:
            return (self.config.min_value + self.config.max_value) / 2.0

        weighted_sum = sum(a * p for a, p in zip(activations, self._preferred))
        return weighted_sum / total_weight

    def decode_from_spikes(self, spikes: list[SpikeEvent]) -> float:
        """Decode from SpikeEvent list (using strength as activation).

        Args:
            spikes: SpikeEvent list from a neuron layer.

        Returns:
            Decoded scalar value.
        """
        activations = [s.strength if s.fired else 0.0 for s in spikes]
        # Pad or truncate to match population size
        while len(activations) < len(self._preferred):
            activations.append(0.0)
        activations = activations[: len(self._preferred)]
        return self.decode(activations)

    def decode_with_uncertainty(self, activations: list[float]) -> tuple[float, float]:
        """Decode value with uncertainty estimate.

        Uncertainty is computed as the weighted standard deviation
        of the population activity. Narrower distribution = lower
        uncertainty = higher confidence.

        Args:
            activations: Activation levels for each neuron.

        Returns:
            Tuple of (decoded_value, uncertainty_stddev).
        """
        mean = self.decode(activations)
        total_weight = sum(activations)
        if total_weight < 1e-10:
            return mean, float("inf")

        # Weighted variance
        weighted_var = sum(a * (p - mean) ** 2 for a, p in zip(activations, self._preferred))
        variance = weighted_var / total_weight
        return mean, math.sqrt(max(0.0, variance))


# ═══════════════════════════════════════════════════════════════════════════
# CognitiveStateEncoder — encode CognitiveState as spike patterns
# ═══════════════════════════════════════════════════════════════════════════


class CognitiveStateEncoder:
    """Encodes CognitiveState fields as population spike patterns.

    Each field of ``CognitiveState`` (urgency, curiosity, confidence,
    emotional_valence, etc.) gets its own ``PopulationEncoder``.

    This bridges the gap between the symbolic ``CognitiveState``
    representation and the SNN's spike-based processing.

    Args:
        neurons_per_field: Number of neurons per encoded field.
        tuning_width: Width of Gaussian tuning curves.
    """

    # Standard cognitive state fields and their value ranges
    FIELD_RANGES: dict[str, tuple[float, float]] = {
        "urgency": (0.0, 1.0),
        "curiosity": (0.0, 1.0),
        "confidence": (0.0, 1.0),
        "emotional_valence": (-1.0, 1.0),
        "cognitive_load": (0.0, 1.0),
        "fatigue": (0.0, 1.0),
        "engagement": (0.0, 1.0),
        "active_goal_count": (0.0, 20.0),
    }

    def __init__(
        self,
        neurons_per_field: int = 16,
        tuning_width: float = 0.1,
    ) -> None:
        self._neurons_per_field = neurons_per_field
        self._encoders: dict[str, PopulationEncoder] = {}

        for field_name, (min_val, max_val) in self.FIELD_RANGES.items():
            config = PopulationConfig(
                num_neurons=neurons_per_field,
                min_value=min_val,
                max_value=max_val,
                tuning_width=tuning_width * (max_val - min_val),
            )
            self._encoders[field_name] = PopulationEncoder(config)

    def encode_state(self, state: Any) -> dict[str, list[float]]:
        """Encode a CognitiveState into population spike patterns.

        Args:
            state: CognitiveState (or dict with matching field names).

        Returns:
            Dict mapping field names to activation patterns.
        """
        result: dict[str, list[float]] = {}
        state_dict = state if isinstance(state, dict) else vars(state)

        for field_name, encoder in self._encoders.items():
            value = state_dict.get(field_name, 0.0)
            if isinstance(value, (int, float)):
                result[field_name] = encoder.encode(float(value))

        return result

    def decode_state(self, patterns: dict[str, list[float]]) -> dict[str, float]:
        """Decode population patterns back to scalar values.

        Args:
            patterns: Dict mapping field names to activation patterns.

        Returns:
            Dict of decoded field values.
        """
        result: dict[str, float] = {}
        for field_name, activations in patterns.items():
            encoder = self._encoders.get(field_name)
            if encoder:
                result[field_name] = encoder.decode(activations)
        return result

    def decode_with_uncertainty(
        self, patterns: dict[str, list[float]]
    ) -> dict[str, tuple[float, float]]:
        """Decode with uncertainty estimates per field.

        Args:
            patterns: Dict mapping field names to activation patterns.

        Returns:
            Dict of (value, uncertainty) tuples per field.
        """
        result: dict[str, tuple[float, float]] = {}
        for field_name, activations in patterns.items():
            encoder = self._encoders.get(field_name)
            if encoder:
                result[field_name] = encoder.decode_with_uncertainty(activations)
        return result

    @property
    def total_neurons(self) -> int:
        """Total number of neurons across all field encoders."""
        return self._neurons_per_field * len(self._encoders)
