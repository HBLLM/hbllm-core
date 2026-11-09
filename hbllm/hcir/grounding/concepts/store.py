from __future__ import annotations

import math
import threading
import uuid
from typing import Any

from hbllm.hcir.grounding.concepts.lifecycle import (
    ConceptLifecycleManager,
    ConceptPromotionProposal,
)
from hbllm.hcir.grounding.concepts.model import GroundedConcept
from hbllm.hcir.grounding.contracts.enums import ConceptLifecycle
from hbllm.hcir.grounding.contracts.quality import GroundingQuality, QualityMeasurement
from hbllm.hcir.grounding.contracts.references import GroundedConceptRef


def _gen_concept_id() -> str:
    return f"C-{uuid.uuid4().hex[:6].upper()}"


class WorkingConceptStore:
    """SGE Working Concept Store: mutable working memory for dynamic ontology evolution.

    Upholds the two-store invariant:
    Emerging, provisional, and consolidated concepts evolve here during active perception
    and reasoning, completely isolated from direct mutation of the Canonical CognitiveGraph.
    Only validated ConceptPromotionProposals are emitted for transactional commit.
    """

    def __init__(
        self,
        lifecycle_manager: ConceptLifecycleManager | None = None,
        similarity_threshold: float = 1.5,
    ) -> None:
        self._lifecycle_manager = lifecycle_manager or ConceptLifecycleManager()
        self._similarity_threshold = similarity_threshold
        self._concepts: dict[str, GroundedConcept] = {}
        self._promotion_proposals: dict[str, ConceptPromotionProposal] = {}
        self._lock = threading.RLock()

    @property
    def lifecycle_manager(self) -> ConceptLifecycleManager:
        return self._lifecycle_manager

    def get_concept(self, concept_id: str) -> GroundedConcept | None:
        with self._lock:
            return self._concepts.get(concept_id)

    def register_concept(self, concept: GroundedConcept) -> None:
        with self._lock:
            self._concepts[concept.concept_ref.id] = concept

    def get_all_concepts(self) -> list[GroundedConcept]:
        with self._lock:
            return list(self._concepts.values())

    def get_all(self) -> list[GroundedConcept]:
        """Alias for get_all_concepts."""
        return self.get_all_concepts()

    def get_concepts_by_state(self, state: ConceptLifecycle) -> list[GroundedConcept]:
        with self._lock:
            return [c for c in self._concepts.values() if c.lifecycle_state == state]

    def ingest_feature_prototype(
        self,
        features: dict[str, Any],
        exemplar_entity_id: str | None = None,
    ) -> GroundedConcept:
        """Finds closest matching concept to update, or spawns a new EMERGING concept."""
        with self._lock:
            best_match: GroundedConcept | None = None
            min_dist = float("inf")

            for concept in self._concepts.values():
                dist = self._compute_feature_distance(features, concept)
                if dist < min_dist and dist <= self._similarity_threshold:
                    min_dist = dist
                    best_match = concept

            if best_match is not None:
                best_match.add_prototype(features)
                if exemplar_entity_id:
                    best_match.add_exemplar(exemplar_entity_id)
                self._evaluate_and_update_lifecycle(best_match)
                return best_match
            else:
                # Spawn new emerging concept
                concept_ref = GroundedConceptRef(id=_gen_concept_id())
                new_concept = GroundedConcept(
                    concept_ref=concept_ref,
                    lifecycle_state=ConceptLifecycle.EMERGING,
                    grounding_quality=GroundingQuality.balanced(0.3),
                )
                new_concept.add_prototype(features)
                if exemplar_entity_id:
                    new_concept.add_exemplar(exemplar_entity_id)
                self._concepts[concept_ref.id] = new_concept
                self._evaluate_and_update_lifecycle(new_concept)
                return new_concept

    def update_concept_quality(
        self,
        concept_id: str,
        quality: GroundingQuality,
        prediction_error: float = 0.0,
    ) -> GroundedConcept:
        """Updates grounding quality and evaluates lifecycle transitions."""
        with self._lock:
            concept = self._concepts.get(concept_id)
            if concept is None:
                raise KeyError(f"Concept {concept_id} not found in Working Store")

            concept.update_grounding_quality(quality)
            self._evaluate_and_update_lifecycle(concept, prediction_error=prediction_error)
            return concept

    def bind_linguistic_alias(
        self,
        concept_ref_or_id: str | GroundedConceptRef,
        alias: str,
        confidence: float = 1.0,
    ) -> GroundedConcept:
        """Binds a linguistic alias to a concept, boosting cross-modal support and evaluating lifecycle."""
        cid = (
            concept_ref_or_id.id
            if isinstance(concept_ref_or_id, GroundedConceptRef)
            else concept_ref_or_id
        )
        with self._lock:
            concept = self._concepts.get(cid)
            if concept is None:
                raise KeyError(f"Concept {cid} not found in Working Concept Store")

            concept.add_linguistic_alias(alias, confidence=confidence)

            # Elevate cross-modal support in 7D grounding quality
            q = concept.grounding_quality
            new_cross_modal = min(1.0, max(q.cross_modal_support, confidence * 0.8))
            new_meas = dict(q.measurements)
            new_meas["cross_modal_support"] = QualityMeasurement.measured(new_cross_modal)
            new_quality = GroundingQuality(
                perceptual_stability=q.perceptual_stability,
                temporal_persistence=q.temporal_persistence,
                relational_consistency=q.relational_consistency,
                cross_modal_support=new_cross_modal,
                predictive_utility=q.predictive_utility,
                interaction_support=q.interaction_support,
                boundary_stability=q.boundary_stability,
                measurements=new_meas,
            )
            concept.update_grounding_quality(new_quality)
            self._evaluate_and_update_lifecycle(concept)
            return concept

    def find_concept_by_alias(self, alias: str) -> GroundedConcept | None:
        """Finds the highest-confidence concept associated with a linguistic label."""
        alias_norm = alias.strip().lower()
        candidates = [alias_norm]
        for prefix in ("the ", "a ", "an ", "this ", "that "):
            if alias_norm.startswith(prefix):
                candidates.append(alias_norm[len(prefix) :].strip())

        with self._lock:
            best_concept: GroundedConcept | None = None
            best_conf = -1.0
            for cand in candidates:
                for concept in self._concepts.values():
                    conf = concept.linguistic_aliases.get(cand)
                    if conf is not None and conf > best_conf:
                        best_conf = conf
                        best_concept = concept
                if best_concept is not None:
                    return best_concept

            # Fallback: substring matching
            for concept in self._concepts.values():
                for a_key, conf in concept.linguistic_aliases.items():
                    if (
                        a_key in alias_norm or any(cand in a_key for cand in candidates)
                    ) and conf > best_conf:
                        best_conf = conf
                        best_concept = concept

            return best_concept

    def get_pending_promotion_proposals(self) -> list[ConceptPromotionProposal]:
        with self._lock:
            return list(self._promotion_proposals.values())

    def _evaluate_and_update_lifecycle(
        self,
        concept: GroundedConcept,
        prediction_error: float = 0.0,
    ) -> None:
        """Evaluates lifecycle rules and checks for promotion eligibility."""
        # Progressively evaluate lifecycle transitions until state stabilizes
        for _ in range(5):
            next_state = self._lifecycle_manager.evaluate_transition(
                concept,
                prediction_error=prediction_error,
            )
            if next_state is not None and next_state != concept.lifecycle_state:
                concept.transition_lifecycle(next_state)
            else:
                break

        # Check for promotion proposal
        if self._lifecycle_manager.should_propose_promotion(concept):
            cid = concept.concept_ref.id
            if cid not in self._promotion_proposals:
                proposal_id = f"prop_{uuid.uuid4().hex[:8]}"
                proposal = self._lifecycle_manager.create_promotion_proposal(
                    concept=concept,
                    proposal_id=proposal_id,
                )
                self._promotion_proposals[cid] = proposal

    @staticmethod
    def _compute_feature_distance(features: dict[str, Any], concept: GroundedConcept) -> float:
        """Computes normalized Euclidean distance against concept feature distributions."""
        if not concept.feature_distributions:
            return 1.0

        shared_keys = set(features.keys()) & set(concept.feature_distributions.keys())
        if not shared_keys:
            return float("inf")

        sq_sum = 0.0
        for k in shared_keys:
            val = features[k]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                mean, var = concept.feature_distributions[k]
                std = math.sqrt(var) if var > 1e-6 else 1.0
                z_score = (float(val) - mean) / std
                sq_sum += z_score**2

        return math.sqrt(sq_sum / len(shared_keys))
