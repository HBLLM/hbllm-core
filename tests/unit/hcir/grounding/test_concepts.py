from __future__ import annotations

import pytest

from hbllm.hcir.grounding.concepts import (
    ConceptLifecycleManager,
    ConceptPromotionProposal,
    GroundedConcept,
    WorkingConceptStore,
)
from hbllm.hcir.grounding.contracts import (
    ConceptLifecycle,
    GroundedConceptRef,
    GroundingQuality,
)


def test_welford_feature_distribution_online_update() -> None:
    """Verify Welford's algorithm computes exact running mean and variance."""
    concept = GroundedConcept(
        concept_ref=GroundedConceptRef(id="C-TEST01"),
    )

    # Observations: [10.0, 20.0, 30.0]
    # Mean: 20.0, Sample Variance: 100.0 (std = 10.0)
    concept.update_feature_observation("height", 10.0)
    concept.update_feature_observation("height", 20.0)
    concept.update_feature_observation("height", 30.0)

    mean, var = concept.feature_distributions["height"]
    assert pytest.approx(mean, rel=1e-5) == 20.0
    assert pytest.approx(var, rel=1e-5) == 100.0
    assert concept.feature_counts["height"] == 3


def test_concept_creation_and_prototype_bounds() -> None:
    """Ensure prototypes update distributions and stay bounded within max_prototypes."""
    concept = GroundedConcept(
        concept_ref=GroundedConceptRef(id="C-BOX01"),
    )
    assert concept.concept_version == 1
    assert concept.lifecycle_state == ConceptLifecycle.EMERGING

    # Add 10 prototypes with max bounded at 3
    for i in range(1, 11):
        concept.add_prototype({"width": float(i * 10), "color": "red"}, max_prototypes=3)

    assert len(concept.prototypes) == 3
    # Only last 3 prototypes remain (80, 90, 100)
    assert [p["width"] for p in concept.prototypes] == [80.0, 90.0, 100.0]

    # Distribution was updated across all 10 observations
    assert concept.feature_counts["width"] == 10
    mean, _ = concept.feature_distributions["width"]
    assert pytest.approx(mean, rel=1e-5) == 55.0  # (10 + ... + 100) / 10 = 55.0


def test_dynamic_lifecycle_transitions() -> None:
    """Verify full 6-stage lifecycle progression and revision on prediction error."""
    manager = ConceptLifecycleManager()
    concept = GroundedConcept(
        concept_ref=GroundedConceptRef(id="C-CUP01"),
    )

    # 1. EMERGING: initial state
    assert concept.lifecycle_state == ConceptLifecycle.EMERGING
    assert manager.evaluate_transition(concept) is None

    # Add prototypes & set stability -> transition to PROVISIONAL
    concept.add_prototype({"radius": 4.0})
    concept.add_prototype({"radius": 4.2})
    concept.update_grounding_quality(
        GroundingQuality(
            perceptual_stability=0.6,
            temporal_persistence=0.3,
            relational_consistency=0.2,
            cross_modal_support=0.2,
            predictive_utility=0.3,
            interaction_support=0.1,
            boundary_stability=0.3,
        )
    )
    next_state = manager.evaluate_transition(concept)
    assert next_state == ConceptLifecycle.PROVISIONAL
    concept.transition_lifecycle(next_state)
    assert concept.lifecycle_state == ConceptLifecycle.PROVISIONAL
    assert concept.concept_version == 2

    # 2. PROVISIONAL -> GROUNDED: predictive utility validated
    concept.update_grounding_quality(
        GroundingQuality(
            perceptual_stability=0.7,
            temporal_persistence=0.6,
            relational_consistency=0.4,
            cross_modal_support=0.4,
            predictive_utility=0.85,  # High predictive utility delta
            interaction_support=0.5,
            boundary_stability=0.5,
        )
    )
    next_state = manager.evaluate_transition(concept)
    assert next_state == ConceptLifecycle.GROUNDED
    concept.transition_lifecycle(next_state)
    assert concept.lifecycle_state == ConceptLifecycle.GROUNDED

    # 3. GROUNDED -> CONSOLIDATED: affordances and consistency stabilized
    concept.add_affordance("GRASP")
    concept.add_affordance("CONTAIN")
    concept.relational_patterns.append("SUPPORTED_BY(?, table)")
    concept.add_exemplar("e_cup_1")
    concept.add_exemplar("e_cup_2")
    concept.update_grounding_quality(
        GroundingQuality(
            perceptual_stability=0.85,
            temporal_persistence=0.85,
            relational_consistency=0.75,
            cross_modal_support=0.70,
            predictive_utility=0.90,
            interaction_support=0.80,
            boundary_stability=0.85,
        )
    )
    next_state = manager.evaluate_transition(concept)
    assert next_state == ConceptLifecycle.CONSOLIDATED
    concept.transition_lifecycle(next_state)
    assert concept.lifecycle_state == ConceptLifecycle.CONSOLIDATED

    # 4. CONSOLIDATED -> REVISED: high prediction error trigger
    # e.g. An object predicted to be a cup fails to contain liquid (hole in bottom)
    next_state = manager.evaluate_transition(concept, prediction_error=0.85)
    assert next_state == ConceptLifecycle.REVISED
    concept.transition_lifecycle(next_state, reason="Affordance failure: failed to contain")
    assert concept.lifecycle_state == ConceptLifecycle.REVISED

    # 5. REVISED -> CONSOLIDATED: boundary restored and predictive accuracy recovered
    next_state = manager.evaluate_transition(concept, prediction_error=0.0)
    assert next_state == ConceptLifecycle.CONSOLIDATED
    concept.transition_lifecycle(next_state, reason="Restabilized after boundary revision")
    assert concept.lifecycle_state == ConceptLifecycle.CONSOLIDATED


def test_counterexample_and_boundary_anchoring() -> None:
    """Test anchoring negative counterexamples to prevent semantic drift."""
    concept = GroundedConcept(
        concept_ref=GroundedConceptRef(id="C-BOWL01"),
    )
    concept.add_exemplar("e_cereal_bowl_1")
    concept.add_exemplar("e_salad_bowl_2")

    # Helmet has similar hemispherical shape but different functional identity
    concept.add_counterexample("e_motorcycle_helmet_1")

    assert "e_cereal_bowl_1" in concept.exemplar_entity_ids
    assert "e_salad_bowl_2" in concept.exemplar_entity_ids
    assert "e_motorcycle_helmet_1" in concept.counterexamples
    assert "e_motorcycle_helmet_1" not in concept.exemplar_entity_ids


def test_linguistic_alias_binding_and_lookup() -> None:
    """Verify late-bound linguistic labels with confidence scores."""
    store = WorkingConceptStore()
    concept = GroundedConcept(concept_ref=GroundedConceptRef(id="C-CAB01"))
    concept.add_linguistic_alias("cabinet", confidence=0.95)
    concept.add_linguistic_alias("cupboard", confidence=0.82)
    store.register_concept(concept)

    # Lookup by alias
    found_cab = store.find_concept_by_alias("Cabinet")
    assert found_cab is not None
    assert found_cab.concept_ref == concept.concept_ref

    found_cupboard = store.find_concept_by_alias("CUPBOARD")
    assert found_cupboard is not None
    assert found_cupboard.concept_ref == concept.concept_ref

    assert store.find_concept_by_alias("unknown_label") is None


def test_working_concept_store_clustering_and_promotion_proposal() -> None:
    """Test unsupervised prototype clustering and promotion proposal emission."""
    store = WorkingConceptStore(similarity_threshold=2.0)

    # 1. Ingest first prototype: spawns EMERGING Concept 1
    feat1 = {"width": 10.0, "height": 20.0}
    c1 = store.ingest_feature_prototype(feat1, exemplar_entity_id="e_box_1")
    assert c1.lifecycle_state == ConceptLifecycle.EMERGING

    # 2. Ingest similar prototype: clusters into Concept 1
    feat2 = {"width": 10.5, "height": 20.5}
    c1_updated = store.ingest_feature_prototype(feat2, exemplar_entity_id="e_box_2")
    assert c1_updated.concept_ref.id == c1.concept_ref.id

    # 3. Ingest distant prototype: spawns distinct Concept 2
    feat3 = {"width": 100.0, "height": 200.0}
    c2 = store.ingest_feature_prototype(feat3, exemplar_entity_id="e_table_1")
    assert c2.concept_ref.id != c1.concept_ref.id

    # 4. Advance c1 to CONSOLIDATED
    c1.add_affordance("CONTAIN")
    c1.add_linguistic_alias("box", confidence=0.92)
    store.update_concept_quality(
        concept_id=c1.concept_ref.id,
        quality=GroundingQuality(
            perceptual_stability=0.9,
            temporal_persistence=0.9,
            relational_consistency=0.8,
            cross_modal_support=0.8,
            predictive_utility=0.92,
            interaction_support=0.85,
            boundary_stability=0.88,
        ),
    )

    # Should have emitted a ConceptPromotionProposal
    proposals = store.get_pending_promotion_proposals()
    assert len(proposals) == 1
    prop = proposals[0]
    assert isinstance(prop, ConceptPromotionProposal)
    assert prop.concept_ref == c1.concept_ref
    assert "CONTAIN" in prop.affordances
    assert prop.linguistic_aliases["box"] == 0.92
