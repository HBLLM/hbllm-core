"""
Association Layer — discovers relationships between comprehension concepts.

Uses a 2-layer SpikingNetwork to detect concept-pair relationships:

    Input layer (4 neurons):
        Encodes features of each concept pair:
        - embedding_similarity (cosine similarity between embeddings)
        - domain_overlap (fraction of shared domain activations)
        - channel_contrast (difference in constraint/surprise metadata)
        - temporal_proximity (closeness in sequence)

    Association layer (8 neurons):
        Specialized detectors:
        - neurons 0-1: similarity detection (high embedding sim + domain overlap)
        - neurons 2-3: contrast detection (high channel contrast)
        - neurons 4-5: causal detection (temporal proximity + domain continuity)
        - neurons 6-7: temporal detection (sequential ordering patterns)

The association layer uses STDP plasticity so that repeated concept
patterns strengthen the appropriate association neurons over time.

Cost model:
    For N concepts: N×(N-1)/2 pairs × ~0.02ms per pair
    10 concepts: 45 pairs × 0.02ms = ~0.9ms total
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.brain.snn.lif import LIFConfig
from hbllm.brain.snn.network import NeuronLayer, SpikingNetwork

logger = logging.getLogger(__name__)


@dataclass
class ConceptAssociation:
    """A discovered relationship between two comprehension concepts.

    Attributes:
        source_idx: Index of the first concept in the concepts list.
        target_idx: Index of the second concept.
        source_text: Text of the first concept.
        target_text: Text of the second concept.
        association_type: Type of relationship detected
            (``"similar"``, ``"contrast"``, ``"causal"``, ``"temporal"``).
        strength: Spike strength of the association neuron that fired.
        pair_features: The input features that were encoded for this pair.
    """

    source_idx: int = 0
    target_idx: int = 0
    source_text: str = ""
    target_text: str = ""
    association_type: str = "similar"
    strength: float = 0.0
    pair_features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for inclusion in UnderstandingState."""
        return {
            "source_idx": self.source_idx,
            "target_idx": self.target_idx,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "association_type": self.association_type,
            "strength": self.strength,
            "pair_features": self.pair_features,
        }


# Association type mapping: neuron index → type
_ASSOCIATION_TYPES = {
    0: "similar",
    1: "similar",
    2: "contrast",
    3: "contrast",
    4: "causal",
    5: "causal",
    6: "temporal",
    7: "temporal",
}


class AssociationLayer:
    """Discovers relationships between comprehension concepts.

    Analyzes all concept pairs from an ``UnderstandingState`` and
    returns discovered associations based on which neurons fire in
    the association layer.

    Uses a ``SpikingNetwork`` with:
    - Input layer (4 neurons): encodes concept-pair features
    - Association layer (8 neurons): specialized pattern detectors
    - STDP plasticity on the input→association projection

    Args:
        stdp_rule: Optional STDP rule for learnable connections.
            If None, connections are fixed.
        similarity_threshold: Minimum cosine similarity to register
            as a feature (default 0.3).
        min_strength: Minimum spike strength to report an association
            (default 0.0, report all).
    """

    def __init__(
        self,
        stdp_rule: Any | None = None,
        similarity_threshold: float = 0.3,
        min_strength: float = 0.0,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.min_strength = min_strength

        # Build the 2-layer network
        self._network = SpikingNetwork(name="association")

        self._network.add_layer(
            NeuronLayer(
                name="input",
                neuron_count=4,
                config=LIFConfig(
                    threshold=0.3,
                    decay_half_life=0.2,
                    reset_potential=0.0,
                    refractory_period=0.01,
                ),
            )
        )

        self._network.add_layer(
            NeuronLayer(
                name="association",
                neuron_count=8,
                config=LIFConfig(
                    threshold=0.5,
                    decay_half_life=0.4,
                    reset_potential=0.0,
                    refractory_period=0.02,
                ),
            )
        )

        # Projection: input → association with structured initial weights
        # Each pair of association neurons responds to specific input features
        #
        # Layout:
        #   input[0] = embedding_similarity
        #   input[1] = domain_overlap
        #   input[2] = channel_contrast
        #   input[3] = temporal_proximity
        #
        #   assoc[0-1] = similar  (high sim + domain overlap)
        #   assoc[2-3] = contrast (high channel contrast)
        #   assoc[4-5] = causal   (temporal + domain continuity)
        #   assoc[6-7] = temporal (temporal proximity)
        initial_weights = [
            # From embedding_similarity →
            [0.6, 0.5, 0.1, 0.1, 0.3, 0.2, 0.1, 0.1],
            # From domain_overlap →
            [0.4, 0.5, 0.1, 0.1, 0.4, 0.3, 0.1, 0.2],
            # From channel_contrast →
            [0.1, 0.1, 0.6, 0.5, 0.1, 0.2, 0.2, 0.1],
            # From temporal_proximity →
            [0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ]

        self._network.connect(
            "input",
            "association",
            initial_weights=initial_weights,
            stdp_rule=stdp_rule,
        )

    def find_associations(
        self,
        concepts: list[Any],
    ) -> list[ConceptAssociation]:
        """Analyze all concept pairs and return discovered associations.

        Args:
            concepts: List of ``ComprehensionUnit`` objects.

        Returns:
            List of discovered ConceptAssociation objects, sorted by
            strength descending.
        """
        if len(concepts) < 2:
            return []

        associations: list[ConceptAssociation] = []
        import time

        t = time.time()

        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                pair_features = self._encode_pair(concepts[i], concepts[j])
                t += 0.001  # small time increment per pair

                # Reset network for clean evaluation per pair
                self._network.reset()

                # Feed features through the network
                input_currents = [
                    pair_features.get("embedding_similarity", 0.0),
                    pair_features.get("domain_overlap", 0.0),
                    pair_features.get("channel_contrast", 0.0),
                    pair_features.get("temporal_proximity", 0.0),
                ]

                # Step multiple times to let the network settle
                spikes_result: dict[str, list[Any]] = {}
                for step in range(3):
                    spikes_result = self._network.step(
                        {"input": input_currents},
                        t + step * 0.001,
                        learn=True,
                    )

                # Check which association neurons fired
                if "association" in spikes_result:
                    for neuron_idx, spike in enumerate(spikes_result["association"]):
                        if spike.fired and spike.strength > self.min_strength:
                            assoc_type = _ASSOCIATION_TYPES.get(neuron_idx, "similar")
                            associations.append(
                                ConceptAssociation(
                                    source_idx=i,
                                    target_idx=j,
                                    source_text=getattr(concepts[i], "text", ""),
                                    target_text=getattr(concepts[j], "text", ""),
                                    association_type=assoc_type,
                                    strength=spike.strength,
                                    pair_features=pair_features,
                                )
                            )

        # Sort by strength descending and deduplicate types per pair
        associations.sort(key=lambda a: a.strength, reverse=True)

        # Keep only the strongest association per pair
        seen_pairs: set[tuple[int, int]] = set()
        unique: list[ConceptAssociation] = []
        for assoc in associations:
            key = (assoc.source_idx, assoc.target_idx)
            if key not in seen_pairs:
                seen_pairs.add(key)
                unique.append(assoc)

        return unique

    def _encode_pair(
        self,
        a: Any,
        b: Any,
    ) -> dict[str, float]:
        """Encode a concept pair as 4 input features.

        All features are normalized to [0.0, 1.0].

        Args:
            a: First ComprehensionUnit.
            b: Second ComprehensionUnit.

        Returns:
            Dict with keys: embedding_similarity, domain_overlap,
            channel_contrast, temporal_proximity.
        """
        features: dict[str, float] = {}

        # 1. Embedding similarity (cosine)
        emb_a = getattr(a, "embedding", None)
        emb_b = getattr(b, "embedding", None)
        if emb_a is not None and emb_b is not None:
            try:
                emb_a = np.asarray(emb_a, dtype=float)
                emb_b = np.asarray(emb_b, dtype=float)
                norm_a = float(np.linalg.norm(emb_a))
                norm_b = float(np.linalg.norm(emb_b))
                if norm_a > 0 and norm_b > 0:
                    cos_sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
                    features["embedding_similarity"] = max(0.0, cos_sim)
                else:
                    features["embedding_similarity"] = 0.0
            except (ValueError, TypeError):
                features["embedding_similarity"] = 0.0
        else:
            # Fallback: lexical overlap
            text_a = set(getattr(a, "text", "").lower().split())
            text_b = set(getattr(b, "text", "").lower().split())
            if text_a and text_b:
                overlap = len(text_a & text_b)
                features["embedding_similarity"] = min(
                    1.0, overlap / max(1, min(len(text_a), len(text_b)))
                )
            else:
                features["embedding_similarity"] = 0.0

        # 2. Domain overlap
        domains_a = getattr(a, "domain_activation", {})
        domains_b = getattr(b, "domain_activation", {})
        if domains_a and domains_b:
            shared = set(domains_a.keys()) & set(domains_b.keys())
            total = set(domains_a.keys()) | set(domains_b.keys())
            features["domain_overlap"] = len(shared) / max(1, len(total))
        else:
            features["domain_overlap"] = 0.0

        # 3. Channel contrast (constraint/surprise difference)
        meta_a = getattr(a, "channel_metadata", {})
        meta_b = getattr(b, "channel_metadata", {})
        contrast_sum = 0.0
        for ch in ("constraint", "surprise"):
            va = meta_a.get(ch, 0.0)
            vb = meta_b.get(ch, 0.0)
            contrast_sum += abs(va - vb)
        features["channel_contrast"] = min(1.0, contrast_sum)

        # 4. Temporal proximity
        ts_a = getattr(a, "timestamp", 0.0)
        ts_b = getattr(b, "timestamp", 0.0)
        dt = abs(ts_b - ts_a)
        # Normalize: concepts within 1s are "close"
        features["temporal_proximity"] = max(0.0, 1.0 - dt)

        return features

    @property
    def network(self) -> SpikingNetwork:
        """Access the underlying SpikingNetwork."""
        return self._network

    def reset(self) -> None:
        """Reset the network state."""
        self._network.reset()
