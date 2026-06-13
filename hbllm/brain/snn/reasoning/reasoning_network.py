"""
Reasoning Network — SNN for evaluating causal chain quality.

A 3-layer SpikingNetwork that scores causal reasoning chains based
on their structural properties (probability, length, recency, diversity).

Layers:
    evidence (4 neurons):
        Encodes chain features as input currents.
        - chain_probability: product of link probabilities
        - chain_length: normalized depth (shorter = stronger)
        - recency: how recent the causal links are
        - diversity: variety of unique nodes in the chain

    evaluation (6 neurons):
        Pattern detectors that respond to specific evidence combinations.
        - neurons 0-1: strong evidence (high prob + short chain)
        - neurons 2-3: temporal evidence (recent + high prob)
        - neurons 4-5: diverse evidence (many unique nodes)

    confidence (2 neurons):
        Final output layer.
        - neuron 0: high confidence (fires on strong evidence)
        - neuron 1: low confidence (fires on weak/long chains)

The confidence score is computed from the relative firing strengths
of the two output neurons.

Usage::

    from hbllm.brain.snn.reasoning.reasoning_network import ReasoningNetwork

    network = ReasoningNetwork()
    confidence = network.evaluate({
        "chain_probability": 0.85,
        "chain_length": 0.67,  # 1 hop out of 3 max
        "recency": 0.9,
        "diversity": 0.5,
    })
    # confidence ∈ [0.0, 1.0]
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.brain.snn.lif import LIFConfig
from hbllm.brain.snn.network import NeuronLayer, SpikingNetwork

logger = logging.getLogger(__name__)


class ReasoningNetwork:
    """SNN for evaluating causal reasoning chains.

    Uses a 3-layer ``SpikingNetwork`` to convert structural chain
    features into a confidence score.  STDP plasticity (if provided)
    allows the network to learn which chain structures correlate with
    good downstream outcomes.

    Args:
        stdp_rule: Optional STDP rule for learnable projections.
        settle_steps: Number of simulation steps per evaluation
            (more steps = more accurate but slower). Default 3.
    """

    def __init__(
        self,
        stdp_rule: Any | None = None,
        settle_steps: int = 3,
    ) -> None:
        self._settle_steps = settle_steps

        self._network = SpikingNetwork(name="reasoning")

        # Evidence layer: encodes chain features
        self._network.add_layer(
            NeuronLayer(
                name="evidence",
                neuron_count=4,
                config=LIFConfig(
                    threshold=0.3,
                    decay_half_life=0.3,
                    reset_potential=0.0,
                    refractory_period=0.01,
                ),
            )
        )

        # Evaluation layer: pattern detectors
        self._network.add_layer(
            NeuronLayer(
                name="evaluation",
                neuron_count=6,
                config=LIFConfig(
                    threshold=0.4,
                    decay_half_life=0.5,
                    reset_potential=0.0,
                    refractory_period=0.02,
                ),
            )
        )

        # Confidence layer: final output
        self._network.add_layer(
            NeuronLayer(
                name="confidence",
                neuron_count=2,
                config=LIFConfig(
                    threshold=0.35,
                    decay_half_life=0.4,
                    reset_potential=0.0,
                    refractory_period=0.01,
                ),
            )
        )

        # Evidence → Evaluation projection
        # Layout:
        #   evidence[0] = chain_probability
        #   evidence[1] = chain_length (inverted: shorter = higher)
        #   evidence[2] = recency
        #   evidence[3] = diversity
        #
        #   eval[0-1] = strong (high prob + short)
        #   eval[2-3] = temporal (recent + high prob)
        #   eval[4-5] = diverse (many sources)
        evidence_to_eval = [
            # chain_probability →
            [0.6, 0.5, 0.4, 0.3, 0.1, 0.1],
            # chain_length (inverted) →
            [0.5, 0.4, 0.1, 0.1, 0.2, 0.1],
            # recency →
            [0.2, 0.1, 0.6, 0.5, 0.1, 0.2],
            # diversity →
            [0.1, 0.2, 0.1, 0.2, 0.5, 0.6],
        ]

        self._network.connect(
            "evidence", "evaluation",
            initial_weights=evidence_to_eval,
            stdp_rule=stdp_rule,
        )

        # Evaluation → Confidence projection
        # eval[0-1] (strong) → high confidence
        # eval[2-3] (temporal) → high confidence (weaker)
        # eval[4-5] (diverse) → moderate confidence
        eval_to_conf = [
            # strong → high confidence
            [0.6, 0.1],
            [0.5, 0.1],
            # temporal → high confidence (moderate)
            [0.4, 0.1],
            [0.3, 0.2],
            # diverse → moderate confidence
            [0.3, 0.2],
            [0.2, 0.3],
        ]

        self._network.connect(
            "evaluation", "confidence",
            initial_weights=eval_to_conf,
            stdp_rule=stdp_rule,
        )

    def evaluate(self, chain_features: dict[str, float]) -> float:
        """Evaluate a causal chain and return confidence score.

        Args:
            chain_features: Dict with keys:
                - ``chain_probability``: product of link probabilities [0,1]
                - ``chain_length``: normalized depth (1/depth) [0,1]
                - ``recency``: how recent the links are [0,1]
                - ``diversity``: variety of unique nodes [0,1]

        Returns:
            Confidence score in [0.0, 1.0].
            Higher = more confident in the causal chain.
        """
        self._network.reset()

        input_currents = [
            chain_features.get("chain_probability", 0.0),
            chain_features.get("chain_length", 0.0),
            chain_features.get("recency", 0.0),
            chain_features.get("diversity", 0.0),
        ]

        t = time.time()
        last_result: dict[str, list] = {}

        for step in range(self._settle_steps):
            last_result = self._network.step(
                {"evidence": input_currents},
                t + step * 0.001,
                learn=True,
            )

        # Extract confidence from output layer
        if "confidence" not in last_result:
            return 0.0

        conf_spikes = last_result["confidence"]
        high_conf = conf_spikes[0].strength if conf_spikes[0].fired else 0.0
        low_conf = conf_spikes[1].strength if conf_spikes[1].fired else 0.0

        # If neither fired, use membrane potentials as a softer signal
        if high_conf == 0.0 and low_conf == 0.0:
            potentials = self._network.get_layer("confidence").get_potential_vector()
            threshold = self._network.get_layer("confidence").config.threshold
            high_conf = min(1.0, potentials[0] / max(0.01, threshold))
            low_conf = min(1.0, potentials[1] / max(0.01, threshold))

        # Normalize to [0, 1]
        total = high_conf + low_conf
        if total > 0:
            return high_conf / total
        return 0.5  # uncertain

    @property
    def network(self) -> SpikingNetwork:
        """Access the underlying SpikingNetwork."""
        return self._network

    def reset(self) -> None:
        """Reset network state."""
        self._network.reset()
