"""Concept Hypothesis Generator — proposes candidate concepts from feature similarity.

Generates candidate concepts using weighted multi-dimensional coherence scoring.

**Critical distinction:** The hypothesis generator only proposes candidates
based on coherence. Predictive utility is evaluated separately by the
ConceptConsolidator — that is the decisive criterion.

Weighted scoring::

    concept_score = (
        w_appearance × appearance_similarity
      + w_behavior   × behavioral_similarity
      + w_relation   × relational_similarity
      + w_temporal   × temporal_similarity
      + w_prediction × predictive_similarity
    )

Uses simple deterministic clustering — no LLM, no neural embeddings.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.concepts.feature_accumulator import (
    EntityFeatureVector,
    FeatureAccumulator,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Concept Hypothesis
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ConceptHypothesis:
    """A candidate concept proposed by feature similarity.

    Not yet validated — requires predictive utility test
    by ConceptConsolidator before becoming a GroundedConcept.
    """

    hypothesis_id: str = field(
        default_factory=lambda: f"hyp_{uuid.uuid4().hex[:8]}",
    )
    member_ids: list[str] = field(default_factory=list)
    feature_prototype: dict[str, Any] = field(default_factory=dict)
    coherence_scores: dict[str, float] = field(default_factory=dict)
    overall_coherence: float = 0.0
    domain: str = ""
    formation_source: str = ""  # "feature", "behavioral", "predictive"
    behavioral_regularities: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Concept Hypothesis Generator
# ═══════════════════════════════════════════════════════════════════════════


class ConceptHypothesisGenerator:
    """Proposes candidate concepts from entity feature similarity.

    Uses weighted multi-dimensional coherence scoring.
    Does NOT evaluate predictive utility (that's ConceptConsolidator).

    Usage::

        generator = ConceptHypothesisGenerator()
        hypotheses = generator.generate(features)
    """

    def __init__(
        self,
        # Dimension weights
        w_appearance: float = 0.20,
        w_behavior: float = 0.30,
        w_relation: float = 0.15,
        w_temporal: float = 0.10,
        w_prediction: float = 0.25,
        # Thresholds
        similarity_threshold: float = 0.6,  # Minimum weighted similarity
        min_cluster_size: int = 2,
    ) -> None:
        self._weights = {
            "appearance": w_appearance,
            "behavior": w_behavior,
            "relational": w_relation,
            "temporal": w_temporal,
            "epistemic": w_prediction,
        }
        self._similarity_threshold = similarity_threshold
        self._min_cluster_size = min_cluster_size

    def generate(
        self,
        features: dict[str, EntityFeatureVector],
    ) -> list[ConceptHypothesis]:
        """Generate concept hypotheses from entity feature vectors.

        Algorithm:
        1. Compute pairwise similarity (weighted multi-dimensional)
        2. Greedy clustering: entities above threshold form candidates
        3. Return hypotheses with coherence scores

        This is deliberately simple — no LLM, no neural embeddings.
        Sophistication comes from the feature dimensions, not the
        clustering algorithm.
        """
        entity_ids = list(features.keys())
        if len(entity_ids) < self._min_cluster_size:
            return []

        # Compute pairwise weighted similarity
        similarity_matrix: dict[tuple[str, str], float] = {}
        dimension_distances: dict[tuple[str, str], dict[str, float]] = {}

        for i, id_a in enumerate(entity_ids):
            for id_b in entity_ids[i + 1:]:
                distances = FeatureAccumulator.feature_distance(
                    features[id_a], features[id_b],
                )
                dimension_distances[(id_a, id_b)] = distances

                # Weighted similarity = 1 - weighted distance
                weighted_dist = sum(
                    self._weights.get(dim, 0) * dist
                    for dim, dist in distances.items()
                )
                similarity = 1.0 - weighted_dist
                similarity_matrix[(id_a, id_b)] = similarity
                similarity_matrix[(id_b, id_a)] = similarity

        # Greedy clustering: find groups above threshold
        used: set[str] = set()
        hypotheses: list[ConceptHypothesis] = []

        for id_a in entity_ids:
            if id_a in used:
                continue

            cluster = [id_a]
            for id_b in entity_ids:
                if id_b == id_a or id_b in used:
                    continue

                sim = similarity_matrix.get(
                    (id_a, id_b),
                    similarity_matrix.get((id_b, id_a), 0),
                )
                if sim >= self._similarity_threshold:
                    # Check similarity with all existing cluster members
                    all_similar = all(
                        similarity_matrix.get(
                            (id_b, member),
                            similarity_matrix.get((member, id_b), 0),
                        ) >= self._similarity_threshold
                        for member in cluster
                    )
                    if all_similar:
                        cluster.append(id_b)

            if len(cluster) >= self._min_cluster_size:
                hypothesis = self._build_hypothesis(cluster, features, dimension_distances)
                hypotheses.append(hypothesis)
                used.update(cluster)

        return hypotheses

    def _build_hypothesis(
        self,
        member_ids: list[str],
        features: dict[str, EntityFeatureVector],
        dimension_distances: dict[tuple[str, str], dict[str, float]],
    ) -> ConceptHypothesis:
        """Build a ConceptHypothesis from a cluster of entities."""
        # Compute prototype as centroid of features
        prototype: dict[str, Any] = {}
        all_props: dict[str, list] = {}

        for mid in member_ids:
            fv = features[mid]
            for key, val in fv.appearance.properties.items():
                all_props.setdefault(key, []).append(val)

        # Prototype: most common value for each property
        for key, vals in all_props.items():
            from collections import Counter
            counts = Counter(vals)
            prototype[key] = counts.most_common(1)[0][0]

        # Per-dimension coherence scores (1 - avg pairwise distance)
        coherence: dict[str, float] = {dim: 1.0 for dim in self._weights}
        pair_count = 0

        for i, id_a in enumerate(member_ids):
            for id_b in member_ids[i + 1:]:
                pair_key: tuple[str, str] = (min(id_a, id_b), max(id_a, id_b))
                distances: dict[str, float] = dimension_distances.get(pair_key, {})
                for dim in coherence:
                    coherence[dim] -= distances.get(dim, 0.5) / max(
                        len(member_ids) * (len(member_ids) - 1) / 2, 1,
                    )
                pair_count += 1

        # Overall coherence = weighted
        overall = sum(
            self._weights.get(dim, 0) * score
            for dim, score in coherence.items()
        )

        # Behavioral regularities: shared event types
        event_sets = []
        for mid in member_ids:
            fv = features[mid]
            event_sets.append(set(fv.behavior.event_type_distribution.keys()))
        shared_events = (
            list(set.intersection(*event_sets)) if event_sets else []
        )

        # Determine formation source
        max_dim = max(coherence, key=coherence.get)  # type: ignore[arg-type]
        source_map = {
            "appearance": "feature",
            "behavior": "behavioral",
            "epistemic": "predictive",
            "relational": "feature",
            "temporal": "feature",
        }

        return ConceptHypothesis(
            member_ids=member_ids,
            feature_prototype=prototype,
            coherence_scores=coherence,
            overall_coherence=overall,
            domain=features[member_ids[0]].entity_type if member_ids else "",
            formation_source=source_map.get(max_dim, "feature"),
            behavioral_regularities=shared_events,
        )
