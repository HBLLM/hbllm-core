from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.grounding.concepts.model import GroundedConcept
from hbllm.hcir.grounding.concepts.store import WorkingConceptStore
from hbllm.hcir.grounding.contracts.enums import ConceptLifecycle
from hbllm.hcir.grounding.contracts.quality import GroundingQuality
from hbllm.hcir.grounding.contracts.references import GroundedConceptRef
from hbllm.hcir.grounding.features.feature_extractor import PerceptualFeatureVector


def _gen_concept_id() -> str:
    return f"C-{uuid.uuid4().hex[:6].upper()}"


@dataclass(frozen=True)
class ConceptAssignment:
    """The result of assigning a feature vector to a concept cluster."""

    concept_ref: GroundedConceptRef
    distance: float  # Mahalanobis-like distance from concept centroid
    is_new_concept: bool = False
    split_from: str | None = None  # If this concept was just split from another


@dataclass(frozen=True)
class ConceptSplitResult:
    """Record of a concept being split into two or more child concepts."""

    parent_concept_id: str
    child_concept_ids: tuple[str, ...]
    rationale: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ConceptMergeResult:
    """Record of two or more concepts being merged."""

    source_concept_ids: tuple[str, ...]
    target_concept_id: str
    rationale: str
    timestamp: float = field(default_factory=time.time)


class ConceptLearner:
    """Unsupervised incremental concept learning engine (SGE Pipeline Stage 5).

    Implements online streaming clustering over incoming ``PerceptualFeatureVector``
    observations.  Unlike batch k-means or DBSCAN, this learner:

    1. **Assigns** each observation to the nearest existing concept using
       normalised Mahalanobis-like distance, or spawns a new cluster.
    2. **Detects multimodality** (concept splitting) when a concept's
       prototype spread exceeds a dispersion threshold, indicating that
       the cluster has absorbed heterogeneous instances.
    3. **Detects convergence** (concept merging) when two concepts'
       centroids drift close enough that they represent the same
       underlying category.
    4. Updates 7D ``GroundingQuality`` and triggers lifecycle transitions
       in the ``WorkingConceptStore``.

    All operations are deterministic and LLM-free.
    """

    def __init__(
        self,
        concept_store: WorkingConceptStore,
        assignment_threshold: float = 2.5,
        split_dispersion_threshold: float = 4.0,
        merge_distance_threshold: float = 0.8,
        min_prototypes_for_split: int = 4,
        min_split_gap: float = 2.0,
    ) -> None:
        self._store = concept_store
        self._assignment_threshold = assignment_threshold
        self._split_dispersion_threshold = split_dispersion_threshold
        self._merge_distance_threshold = merge_distance_threshold
        self._min_prototypes_for_split = min_prototypes_for_split
        self._min_split_gap = min_split_gap

        # Audit trails
        self._split_log: list[ConceptSplitResult] = []
        self._merge_log: list[ConceptMergeResult] = []
        self._total_assignments: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn(
        self,
        feature: PerceptualFeatureVector,
        exemplar_entity_id: str | None = None,
    ) -> ConceptAssignment:
        """Assigns a feature vector to the best-matching concept or spawns a new one.

        This is the primary entry point — called once per extracted feature
        vector in the pipeline.
        """
        features_dict = self._embedding_to_dict(feature)

        # 1. Find nearest concept
        best_concept, best_dist = self._find_nearest(features_dict)

        if best_concept is not None and best_dist <= self._assignment_threshold:
            # Assign to existing concept
            best_concept.add_prototype(features_dict)
            if exemplar_entity_id:
                best_concept.add_exemplar(exemplar_entity_id)
            self._update_quality_from_assignment(best_concept, best_dist)
            self._total_assignments += 1

            return ConceptAssignment(
                concept_ref=best_concept.concept_ref,
                distance=best_dist,
                is_new_concept=False,
            )
        else:
            # Spawn new concept
            concept_ref = GroundedConceptRef(id=_gen_concept_id())
            new_concept = GroundedConcept(
                concept_ref=concept_ref,
                lifecycle_state=ConceptLifecycle.EMERGING,
                grounding_quality=GroundingQuality.with_untested(
                    perceptual_stability=0.3,
                    temporal_persistence=0.1,
                    relational_consistency=0.0,
                    cross_modal_support=0.0,
                    predictive_utility=0.0,
                    interaction_support=0.0,
                    boundary_stability=0.3,
                    untested_dimensions={
                        "cross_modal_support",
                        "interaction_support",
                        "predictive_utility",
                    },
                ),
            )
            new_concept.add_prototype(features_dict)
            if exemplar_entity_id:
                new_concept.add_exemplar(exemplar_entity_id)
            self._store.register_concept(new_concept)
            self._total_assignments += 1

            return ConceptAssignment(
                concept_ref=concept_ref,
                distance=float("inf"),
                is_new_concept=True,
            )

    def learn_batch(
        self,
        features: list[PerceptualFeatureVector],
        exemplar_entity_ids: list[str | None] | None = None,
    ) -> list[ConceptAssignment]:
        """Assigns a batch of feature vectors, then runs maintenance."""
        entity_ids = exemplar_entity_ids or [None] * len(features)
        assignments = [
            self.learn(feat, eid) for feat, eid in zip(features, entity_ids, strict=False)
        ]
        self.maintain()
        return assignments

    def maintain(self) -> tuple[list[ConceptSplitResult], list[ConceptMergeResult]]:
        """Runs cluster maintenance: splits multimodal concepts and merges convergent ones.

        Should be called periodically (e.g., after each observation cycle) or
        after ``learn_batch``.
        """
        splits = self._detect_and_split()
        merges = self._detect_and_merge()
        return splits, merges

    def bind_linguistic_alias(
        self,
        concept_ref_or_id: str | GroundedConceptRef,
        alias: str,
        confidence: float = 1.0,
    ) -> GroundedConcept:
        """Binds a linguistic alias directly to a concept in the working store."""
        return self._store.bind_linguistic_alias(
            concept_ref_or_id,
            alias=alias,
            confidence=confidence,
        )

    def bind_text_feature(
        self,
        text_feature: PerceptualFeatureVector,
        target_concept_id_or_ref: str | GroundedConceptRef | None = None,
        target_entity_id: str | None = None,
        confidence: float = 1.0,
    ) -> GroundedConcept | None:
        """Extracts text or label from a text feature vector and binds it to a target concept,
        either by concept ID or by matching an exemplar entity ID.
        """
        raw_alias = text_feature.feature_metadata.get("text")
        if not raw_alias and text_feature.semantic_labels:
            raw_alias = next(iter(text_feature.semantic_labels.keys()))
        if not raw_alias:
            return None

        alias_str = str(raw_alias).strip()
        if not alias_str:
            return None

        # 1. Direct concept targeting
        if target_concept_id_or_ref is not None:
            return self.bind_linguistic_alias(
                target_concept_id_or_ref,
                alias=alias_str,
                confidence=confidence,
            )

        # 2. Match via exemplar entity ID
        if target_entity_id is not None:
            for concept in self._store.get_all_concepts():
                if concept.lifecycle_state == ConceptLifecycle.MERGED_SPLIT_DEPRECATED:
                    continue
                if target_entity_id in concept.exemplar_entity_ids:
                    return self.bind_linguistic_alias(
                        concept.concept_ref,
                        alias=alias_str,
                        confidence=confidence,
                    )

        # 3. Fallback: match via metadata entity_id
        meta_eid = text_feature.feature_metadata.get("entity_id")
        if meta_eid:
            for concept in self._store.get_all_concepts():
                if concept.lifecycle_state == ConceptLifecycle.MERGED_SPLIT_DEPRECATED:
                    continue
                if meta_eid in concept.exemplar_entity_ids:
                    return self.bind_linguistic_alias(
                        concept.concept_ref,
                        alias=alias_str,
                        confidence=confidence,
                    )

        return None

    def get_split_log(self) -> list[ConceptSplitResult]:
        return list(self._split_log)

    def get_merge_log(self) -> list[ConceptMergeResult]:
        return list(self._merge_log)

    @property
    def total_assignments(self) -> int:
        return self._total_assignments

    # ------------------------------------------------------------------
    # Nearest-concept assignment
    # ------------------------------------------------------------------

    def _find_nearest(
        self,
        features: dict[str, Any],
    ) -> tuple[GroundedConcept | None, float]:
        """Finds the closest concept using normalised Mahalanobis-like distance."""
        best: GroundedConcept | None = None
        best_dist = float("inf")

        for concept in self._store.get_all_concepts():
            if concept.lifecycle_state == ConceptLifecycle.MERGED_SPLIT_DEPRECATED:
                continue
            dist = self._mahalanobis_distance(features, concept)
            if dist < best_dist:
                best_dist = dist
                best = concept

        return best, best_dist

    @staticmethod
    def _mahalanobis_distance(
        features: dict[str, Any],
        concept: GroundedConcept,
    ) -> float:
        """Computes normalised Mahalanobis-like distance from a feature dict to a concept.

        Uses per-feature (mean, variance) from the concept's running
        Welford statistics.  Falls back to Euclidean when variance is
        unavailable.
        """
        if not concept.feature_distributions:
            return float("inf")

        shared = set(features.keys()) & set(concept.feature_distributions.keys())
        if not shared:
            return float("inf")

        sq_sum = 0.0
        for k in shared:
            val = features[k]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                mean, var = concept.feature_distributions[k]
                # Use a minimum std floor of 1.0 so that early low-sample
                # variance doesn't produce explosive z-scores.  This makes
                # Mahalanobis degrade gracefully to Euclidean distance when
                # the Welford variance estimate is unreliable (few samples).
                std = max(math.sqrt(var), 1.0) if var > 1e-12 else 1.0
                z = (float(val) - mean) / std
                sq_sum += z * z

        return math.sqrt(sq_sum / len(shared))

    # ------------------------------------------------------------------
    # Split detection (multimodality)
    # ------------------------------------------------------------------

    def _detect_and_split(self) -> list[ConceptSplitResult]:
        """Scans all concepts for multimodal feature distributions and splits them.

        Multimodality heuristic: compute the maximum pairwise distance between
        prototypes.  If the spread exceeds ``split_dispersion_threshold``, the
        concept is split into two clusters via a 2-means initialisation on
        the stored prototypes.
        """
        results: list[ConceptSplitResult] = []
        concepts = self._store.get_all_concepts()

        for concept in concepts:
            if concept.lifecycle_state == ConceptLifecycle.MERGED_SPLIT_DEPRECATED:
                continue
            if len(concept.prototypes) < self._min_prototypes_for_split:
                continue

            dispersion = self._max_prototype_dispersion(concept)
            if dispersion < self._split_dispersion_threshold:
                continue

            # 2-means initialisation over prototypes
            child_a_protos, child_b_protos = self._two_means_partition(concept.prototypes)

            if not child_a_protos or not child_b_protos:
                continue

            # Verify the split gap is meaningful
            gap = self._inter_cluster_distance(child_a_protos, child_b_protos)
            if gap < self._min_split_gap:
                continue

            # Create child concepts
            child_a_ref = GroundedConceptRef(id=_gen_concept_id())
            child_b_ref = GroundedConceptRef(id=_gen_concept_id())

            child_a = GroundedConcept(
                concept_ref=child_a_ref,
                lifecycle_state=ConceptLifecycle.EMERGING,
                grounding_quality=GroundingQuality.balanced(0.3),
                affordances=set(concept.affordances),
                relational_patterns=list(concept.relational_patterns),
                linguistic_aliases=dict(concept.linguistic_aliases),
            )
            child_b = GroundedConcept(
                concept_ref=child_b_ref,
                lifecycle_state=ConceptLifecycle.EMERGING,
                grounding_quality=GroundingQuality.balanced(0.3),
                affordances=set(concept.affordances),
                relational_patterns=list(concept.relational_patterns),
                linguistic_aliases=dict(concept.linguistic_aliases),
            )

            for proto in child_a_protos:
                child_a.add_prototype(proto)
            for proto in child_b_protos:
                child_b.add_prototype(proto)

            # Register children and deprecate parent
            self._store.register_concept(child_a)
            self._store.register_concept(child_b)
            concept.transition_lifecycle(
                ConceptLifecycle.MERGED_SPLIT_DEPRECATED,
                reason=f"Split into {child_a_ref.id} and {child_b_ref.id} "
                f"(dispersion={dispersion:.2f} > {self._split_dispersion_threshold})",
            )

            result = ConceptSplitResult(
                parent_concept_id=concept.concept_ref.id,
                child_concept_ids=(child_a_ref.id, child_b_ref.id),
                rationale=f"Dispersion {dispersion:.2f} exceeded threshold "
                f"{self._split_dispersion_threshold}; gap={gap:.2f}",
            )
            self._split_log.append(result)
            results.append(result)

        return results

    def _max_prototype_dispersion(self, concept: GroundedConcept) -> float:
        """Maximum pairwise distance between stored prototypes."""
        protos = concept.prototypes
        if len(protos) < 2:
            return 0.0

        max_dist = 0.0
        for i in range(len(protos)):
            for j in range(i + 1, len(protos)):
                d = self._prototype_distance(protos[i], protos[j])
                if d > max_dist:
                    max_dist = d
        return max_dist

    def _two_means_partition(
        self,
        prototypes: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Simple 2-means partitioning of prototypes.

        Initialises with the two most distant prototypes, then does up to
        10 Lloyd iterations to refine the split.
        """
        if len(prototypes) < 2:
            return prototypes, []

        # Find the two most distant prototypes as seeds
        best_i, best_j = 0, 1
        best_d = -1.0
        for i in range(len(prototypes)):
            for j in range(i + 1, len(prototypes)):
                d = self._prototype_distance(prototypes[i], prototypes[j])
                if d > best_d:
                    best_d = d
                    best_i, best_j = i, j

        centroid_a = dict(prototypes[best_i])
        centroid_b = dict(prototypes[best_j])

        cluster_a: list[dict[str, Any]] = []
        cluster_b: list[dict[str, Any]] = []

        for _ in range(10):
            cluster_a.clear()
            cluster_b.clear()

            for proto in prototypes:
                da = self._point_centroid_distance(proto, centroid_a)
                db = self._point_centroid_distance(proto, centroid_b)
                if da <= db:
                    cluster_a.append(proto)
                else:
                    cluster_b.append(proto)

            if not cluster_a or not cluster_b:
                break

            centroid_a = self._compute_centroid(cluster_a)
            centroid_b = self._compute_centroid(cluster_b)

        return cluster_a, cluster_b

    # ------------------------------------------------------------------
    # Merge detection (convergence)
    # ------------------------------------------------------------------

    def _detect_and_merge(self) -> list[ConceptMergeResult]:
        """Scans all active concept pairs and merges those whose centroids have converged."""
        results: list[ConceptMergeResult] = []
        concepts = [
            c
            for c in self._store.get_all_concepts()
            if c.lifecycle_state != ConceptLifecycle.MERGED_SPLIT_DEPRECATED
            and c.feature_distributions
        ]

        merged_ids: set[str] = set()

        for i in range(len(concepts)):
            if concepts[i].concept_ref.id in merged_ids:
                continue
            for j in range(i + 1, len(concepts)):
                if concepts[j].concept_ref.id in merged_ids:
                    continue

                dist = self._centroid_distance(concepts[i], concepts[j])
                if dist <= self._merge_distance_threshold:
                    merged = self._merge_concepts(concepts[i], concepts[j])
                    merged_ids.add(concepts[i].concept_ref.id)
                    merged_ids.add(concepts[j].concept_ref.id)

                    result = ConceptMergeResult(
                        source_concept_ids=(
                            concepts[i].concept_ref.id,
                            concepts[j].concept_ref.id,
                        ),
                        target_concept_id=merged.concept_ref.id,
                        rationale=f"Centroid distance {dist:.3f} <= "
                        f"{self._merge_distance_threshold}",
                    )
                    self._merge_log.append(result)
                    results.append(result)
                    break  # Concept i consumed; move to next

        return results

    def _merge_concepts(
        self,
        concept_a: GroundedConcept,
        concept_b: GroundedConcept,
    ) -> GroundedConcept:
        """Merges two concepts into one, preserving all exemplars and prototypes."""
        merged_ref = GroundedConceptRef(id=_gen_concept_id())
        merged = GroundedConcept(
            concept_ref=merged_ref,
            lifecycle_state=ConceptLifecycle.EMERGING,
            grounding_quality=GroundingQuality.balanced(0.4),
            affordances=concept_a.affordances | concept_b.affordances,
            relational_patterns=list(
                set(concept_a.relational_patterns) | set(concept_b.relational_patterns)
            ),
            exemplar_entity_ids=concept_a.exemplar_entity_ids | concept_b.exemplar_entity_ids,
            counterexamples=concept_a.counterexamples | concept_b.counterexamples,
            linguistic_aliases={**concept_a.linguistic_aliases, **concept_b.linguistic_aliases},
        )

        for proto in concept_a.prototypes + concept_b.prototypes:
            merged.add_prototype(proto)

        self._store.register_concept(merged)

        # Deprecate sources
        reason = f"Merged into {merged_ref.id}"
        concept_a.transition_lifecycle(ConceptLifecycle.MERGED_SPLIT_DEPRECATED, reason=reason)
        concept_b.transition_lifecycle(ConceptLifecycle.MERGED_SPLIT_DEPRECATED, reason=reason)

        return merged

    # ------------------------------------------------------------------
    # Quality updates
    # ------------------------------------------------------------------

    def _update_quality_from_assignment(
        self,
        concept: GroundedConcept,
        distance: float,
    ) -> None:
        """Updates perceptual stability and boundary stability based on assignment distance."""
        q = concept.grounding_quality

        # Closer assignments improve perceptual stability
        stability_delta = max(0.0, 1.0 - distance / self._assignment_threshold) * 0.05
        new_stability = min(1.0, q.perceptual_stability + stability_delta)

        # Temporal persistence improves with each observation
        obs_count = max(1, len(concept.prototypes))
        new_persistence = min(1.0, 0.1 + math.log1p(obs_count) * 0.2)

        # Boundary stability: inversely related to distance variance
        new_boundary = (
            min(1.0, q.boundary_stability + 0.02) if distance < 1.5 else q.boundary_stability
        )

        new_quality = GroundingQuality(
            perceptual_stability=new_stability,
            temporal_persistence=new_persistence,
            relational_consistency=q.relational_consistency,
            cross_modal_support=q.cross_modal_support,
            predictive_utility=q.predictive_utility,
            interaction_support=q.interaction_support,
            boundary_stability=new_boundary,
        )
        concept.update_grounding_quality(new_quality)

    # ------------------------------------------------------------------
    # Distance utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _prototype_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
        """Euclidean distance between two prototype feature dicts on shared numeric keys."""
        shared = set(a.keys()) & set(b.keys())
        numeric_keys = [
            k
            for k in shared
            if isinstance(a[k], (int, float))
            and not isinstance(a[k], bool)
            and isinstance(b[k], (int, float))
            and not isinstance(b[k], bool)
        ]
        if not numeric_keys:
            return float("inf")

        sq_sum = sum((float(a[k]) - float(b[k])) ** 2 for k in numeric_keys)
        return math.sqrt(sq_sum / len(numeric_keys))

    @staticmethod
    def _point_centroid_distance(point: dict[str, Any], centroid: dict[str, Any]) -> float:
        shared = set(point.keys()) & set(centroid.keys())
        numeric_keys = [
            k
            for k in shared
            if isinstance(point[k], (int, float))
            and not isinstance(point[k], bool)
            and isinstance(centroid[k], (int, float))
            and not isinstance(centroid[k], bool)
        ]
        if not numeric_keys:
            return float("inf")

        return math.sqrt(
            sum((float(point[k]) - float(centroid[k])) ** 2 for k in numeric_keys)
            / len(numeric_keys)
        )

    @staticmethod
    def _compute_centroid(prototypes: list[dict[str, Any]]) -> dict[str, Any]:
        """Computes the mean of numeric features across prototypes."""
        if not prototypes:
            return {}

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for proto in prototypes:
            for k, v in proto.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1

        return {k: sums[k] / counts[k] for k in sums}

    @staticmethod
    def _centroid_distance(
        concept_a: GroundedConcept,
        concept_b: GroundedConcept,
    ) -> float:
        """Distance between two concepts' feature distribution centroids."""
        shared = set(concept_a.feature_distributions.keys()) & set(
            concept_b.feature_distributions.keys()
        )
        if not shared:
            return float("inf")

        sq_sum = 0.0
        for k in shared:
            mean_a, _ = concept_a.feature_distributions[k]
            mean_b, _ = concept_b.feature_distributions[k]
            sq_sum += (mean_a - mean_b) ** 2

        return math.sqrt(sq_sum / len(shared))

    def _inter_cluster_distance(
        self,
        cluster_a: list[dict[str, Any]],
        cluster_b: list[dict[str, Any]],
    ) -> float:
        """Minimum distance between the centroids of two prototype clusters."""
        centroid_a = self._compute_centroid(cluster_a)
        centroid_b = self._compute_centroid(cluster_b)
        return self._point_centroid_distance(centroid_a, centroid_b)

    @staticmethod
    def _embedding_to_dict(feature: PerceptualFeatureVector) -> dict[str, Any]:
        """Converts a PerceptualFeatureVector's embedding into a keyed dict for prototype storage."""
        d: dict[str, Any] = {f"dim_{i}": v for i, v in enumerate(feature.embedding)}
        # Propagate semantic labels as features
        for label, conf in feature.semantic_labels.items():
            d[f"label_{label}"] = conf
        return d
