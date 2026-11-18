from __future__ import annotations

import time
from dataclasses import dataclass, field

from hbllm.hcir.grounding.concepts.model import GroundedConcept
from hbllm.hcir.grounding.contracts.enums import ConceptLifecycle
from hbllm.hcir.grounding.contracts.quality import GroundingQuality
from hbllm.hcir.grounding.contracts.references import GroundedConceptRef


@dataclass(frozen=True)
class ConceptPromotionProposal:
    """Proposal to promote a consolidated concept into the canonical CognitiveGraph."""

    proposal_id: str
    concept_ref: GroundedConceptRef
    concept_version: int
    quality_summary: GroundingQuality
    exemplar_count: int
    affordances: tuple[str, ...]
    linguistic_aliases: dict[str, float]
    rationale: str
    proposed_at: float = field(default_factory=time.time)


class ConceptLifecycleManager:
    """Evaluates grounding quality dimensions to govern concept lifecycle transitions."""

    def __init__(
        self,
        min_provisional_stability: float = 0.4,
        min_grounded_predictive_utility: float = 0.65,
        min_consolidated_consistency: float = 0.6,
        degradation_threshold: float = 0.35,
    ) -> None:
        self._min_provisional_stability = min_provisional_stability
        self._min_grounded_predictive_utility = min_grounded_predictive_utility
        self._min_consolidated_consistency = min_consolidated_consistency
        self._degradation_threshold = degradation_threshold

    def evaluate_transition(
        self,
        concept: GroundedConcept,
        prediction_error: float = 0.0,
    ) -> ConceptLifecycle | None:
        """Evaluates whether the concept should advance, revise, or degrade."""
        current = concept.lifecycle_state
        q = concept.grounding_quality

        # Check prediction error / degradation trigger
        if prediction_error > 0.6 or q.predictive_utility < self._degradation_threshold:
            if current in (ConceptLifecycle.GROUNDED, ConceptLifecycle.CONSOLIDATED):
                return ConceptLifecycle.REVISED
            elif current == ConceptLifecycle.PROVISIONAL and len(concept.exemplar_entity_ids) > 5:
                return ConceptLifecycle.MERGED_SPLIT_DEPRECATED

        # Forward lifecycle progression
        if current == ConceptLifecycle.EMERGING:
            if (
                len(concept.prototypes) >= 2 or len(concept.exemplar_entity_ids) >= 2
            ) and q.perceptual_stability >= self._min_provisional_stability:
                return ConceptLifecycle.PROVISIONAL

        elif current == ConceptLifecycle.PROVISIONAL:
            if (
                q.predictive_utility >= self._min_grounded_predictive_utility
                and q.temporal_persistence >= 0.4
            ):
                return ConceptLifecycle.GROUNDED

        elif current == ConceptLifecycle.GROUNDED:
            has_affordances_or_relations = (
                len(concept.affordances) > 0 or len(concept.relational_patterns) > 0
            )
            has_consistency = (
                q.relational_consistency >= self._min_consolidated_consistency
                or q.cross_modal_support >= self._min_consolidated_consistency
            )
            if (
                has_affordances_or_relations
                and has_consistency
                and q.boundary_stability >= 0.5
                and q.predictive_utility >= self._min_grounded_predictive_utility
            ):
                return ConceptLifecycle.CONSOLIDATED

        elif current == ConceptLifecycle.REVISED:
            # Check if restabilized
            if (
                q.predictive_utility >= self._min_grounded_predictive_utility
                and q.boundary_stability >= 0.5
            ):
                return ConceptLifecycle.CONSOLIDATED

        return None

    def should_propose_promotion(self, concept: GroundedConcept) -> bool:
        """Determines if a concept is eligible for promotion to authoritative HCIR state."""
        return (
            concept.lifecycle_state == ConceptLifecycle.CONSOLIDATED
            and concept.grounding_quality.predictive_utility
            >= self._min_grounded_predictive_utility
            and len(concept.exemplar_entity_ids) >= 2
        )

    def create_promotion_proposal(
        self,
        concept: GroundedConcept,
        proposal_id: str,
        rationale: str = "",
    ) -> ConceptPromotionProposal:
        """Constructs an immutable ConceptPromotionProposal for the TransactionManager."""
        return ConceptPromotionProposal(
            proposal_id=proposal_id,
            concept_ref=concept.concept_ref,
            concept_version=concept.concept_version,
            quality_summary=concept.grounding_quality,
            exemplar_count=len(concept.exemplar_entity_ids),
            affordances=tuple(sorted(concept.affordances)),
            linguistic_aliases=dict(concept.linguistic_aliases),
            rationale=rationale
            or f"Concept {concept.concept_ref.id} consolidated with high utility",
        )
