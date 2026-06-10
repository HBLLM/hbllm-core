"""
Synaptic Priming / Working Memory Primer Layer.

Maintains category-specific Leaky Integrate-and-Fire (LIF) accumulators.
Activations are stimulated by incoming text keywords or direct category domain triggers,
biasing retrieval towards active cognitive contexts.
"""

from __future__ import annotations

import time

import numpy as np

from hbllm.brain.snn import LIFConfig, SpikingAccumulator


class WorkingMemoryPrimer:
    """
    Working Memory Primer that maintains category-specific spiking accumulators.

    Keywords matching active categories or direct domain intent signals stimulate the SNN.
    The membrane potentials of these category neurons act as contextual relevance boosts
    applied during semantic document search.
    """

    def __init__(self, config: LIFConfig | None = None) -> None:
        # Default config: slow decay half-life of 60.0s (representing short-term cognitive focus)
        self.config = config or LIFConfig(
            threshold=99.0,
            decay_half_life=60.0,
            reset_potential=0.0,
            refractory_period=0.0,
        )

        self.categories = {
            "physics": SpikingAccumulator(self.config, neuron_id="priming_physics"),
            "math": SpikingAccumulator(self.config, neuron_id="priming_math"),
            "coding": SpikingAccumulator(self.config, neuron_id="priming_coding"),
            "finance": SpikingAccumulator(self.config, neuron_id="priming_finance"),
            "personal": SpikingAccumulator(self.config, neuron_id="priming_personal"),
            "general": SpikingAccumulator(self.config, neuron_id="priming_general"),
        }

        # Semantic keywords fallback mappings
        self._keyword_map = {
            "physics": [
                "quantum",
                "physics",
                "gravity",
                "energy",
                "mechanics",
                "atom",
                "molecule",
                "relativity",
            ],
            "math": [
                "math",
                "equation",
                "sum",
                "multiply",
                "divide",
                "perfect number",
                "prime",
                "proof",
                "divisor",
                "theorem",
            ],
            "coding": [
                "code",
                "python",
                "javascript",
                "rust",
                "function",
                "bug",
                "compiler",
                "github",
                "git",
                "api",
                "json",
                "syntax",
            ],
            "finance": [
                "money",
                "finance",
                "bank",
                "stock",
                "tax",
                "crypto",
                "bitcoin",
                "budget",
                "invoice",
                "payment",
                "cost",
            ],
            "personal": [
                "my name",
                "my favorite",
                "my dog",
                "personal",
                "prefer",
                "habit",
                "address",
                "phone",
            ],
        }

    def stimulate_by_text(self, text: str, charge: float = 0.35) -> None:
        """Scan input text for keywords and stimulate the matched category accumulators."""
        if not text:
            return

        text_lower = text.lower()
        now = time.time()

        for category, keywords in self._keyword_map.items():
            accumulator = self.categories.get(category)
            if not accumulator:
                continue

            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits > 0:
                # Accumulate current proportional to keywords found (capped at 1.0 per category per text scan)
                stimulus = min(1.0, hits * charge)
                accumulator.stimulate(stimulus, timestamp=now)

    def stimulate_by_vector(
        self, query_vector: np.ndarray, centroids: dict[int, np.ndarray]
    ) -> None:
        """Project query vector onto centroids and stimulate corresponding cluster accumulators."""
        if not centroids:
            return
        now = time.time()
        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            return
        for cluster_id, centroid in centroids.items():
            c_norm = np.linalg.norm(centroid)
            if c_norm == 0:
                continue
            similarity = float(np.dot(query_vector, centroid) / (q_norm * c_norm + 1e-9))
            if similarity >= 0.35:
                # Stimulus = max(0.0, similarity - 0.35)^2 * 3.0
                stimulus = ((similarity - 0.35) ** 2) * 3.0
                cluster_key = f"cluster_{cluster_id}"
                if cluster_key not in self.categories:
                    self.categories[cluster_key] = SpikingAccumulator(
                        self.config, neuron_id=f"priming_{cluster_key}"
                    )
                self.categories[cluster_key].stimulate(stimulus, timestamp=now)

    def stimulate_category(self, category: str, charge: float = 0.5) -> None:
        """Directly stimulate a category accumulator (e.g. from RouterNode intent)."""
        accumulator = self.categories.get(category)
        if not accumulator and category.startswith("cluster_"):
            accumulator = SpikingAccumulator(self.config, neuron_id=f"priming_{category}")
            self.categories[category] = accumulator

        if accumulator:
            accumulator.stimulate(charge, timestamp=time.time())

    def get_boosts(self) -> dict[str, float]:
        """Get the current membrane potentials of all category accumulators to use as search boosts."""
        now = time.time()
        return {cat: acc.get_potential(now) for cat, acc in self.categories.items()}

    def reset(self) -> None:
        """Reset all category accumulators."""
        for acc in self.categories.values():
            acc.reset()
