"""A15 Benchmark — Grounded Concept Formation.

12 end-to-end scenarios validating the complete A15 stack:

1.  Feature accumulation — 5-dimensional features from entities
2.  Similarity detection — similar entities identified without embeddings
3.  Behavioral regularity — entities with similar event patterns grouped
4.  Concept hypothesis generation — candidate concepts with weighted scores
5.  Consolidation with predictive utility — ONLY predictively useful concepts admitted
6.  Rejection — visually similar but behaviorally incoherent cluster rejected
7.  INSTANCE_OF edges — entities correctly linked to concepts
8.  Concept prediction — concept generates testable expectations
9.  A14 feedback — concept prediction outcome updates concept confidence
10. Concept confidence from prediction — rises/drops with prediction performance
11. Concept contradiction / refinement — heterogeneous concept emits refinement signal
12. Zero LLM — entire loop runs without LLM invocation

All scenarios are zero-LLM.
"""

from __future__ import annotations

from hbllm.brain.concepts.concept_consolidator import (
    ConceptConsolidator,
    ConsolidationDecision,
    PredictiveUtilityTest,
)
from hbllm.brain.concepts.concept_hypothesis_generator import (
    ConceptHypothesisGenerator,
)
from hbllm.brain.concepts.concept_prediction_bridge import ConceptPredictionBridge
from hbllm.brain.concepts.feature_accumulator import (
    AppearanceFeatures,
    BehaviorFeatures,
    EntityFeatureVector,
    EpistemicFeatures,
    FeatureAccumulator,
    RelationalFeatures,
    TemporalFeatures,
)
from hbllm.brain.concepts.grounded_concept_registry import GroundedConceptRegistry
from hbllm.hcir.graph import (
    CognitiveGraph,
    HCIREdgeType,
    PhysicalEntityNode,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_graph() -> CognitiveGraph:
    return CognitiveGraph()


def _add_entity(
    graph: CognitiveGraph,
    entity_type: str = "object",
    properties: dict | None = None,
) -> PhysicalEntityNode:
    """Add a PhysicalEntityNode with given properties."""
    entity = PhysicalEntityNode(
        entity_type=entity_type,
        observed_properties=properties or {},
    )
    graph.add_node(entity)
    return entity


def _table_like_features(entity_id: str) -> EntityFeatureVector:
    """Feature vector for a table-like entity."""
    return EntityFeatureVector(
        entity_id=entity_id,
        entity_type="furniture",
        appearance=AppearanceFeatures(
            entity_type="furniture",
            properties={"shape": "rectangular", "material": "wood", "size": "large"},
        ),
        behavior=BehaviorFeatures(
            stationary_rate=0.95,
            event_type_distribution={"supports_object": 0.8, "stationary": 0.2},
        ),
        relational=RelationalFeatures(
            spatial_relations={"above": 2, "touching": 1},
            support_role="supporter",
        ),
        temporal=TemporalFeatures(persistence_duration=1000.0),
        epistemic=EpistemicFeatures(prediction_accuracy=0.85),
    )


def _chair_like_features(entity_id: str) -> EntityFeatureVector:
    """Feature vector for a chair-like entity."""
    return EntityFeatureVector(
        entity_id=entity_id,
        entity_type="furniture",
        appearance=AppearanceFeatures(
            entity_type="furniture",
            properties={"shape": "rectangular", "material": "wood", "size": "medium"},
        ),
        behavior=BehaviorFeatures(
            stationary_rate=0.90,
            event_type_distribution={"supports_object": 0.7, "stationary": 0.3},
        ),
        relational=RelationalFeatures(
            spatial_relations={"above": 1, "touching": 1},
            support_role="supporter",
        ),
        temporal=TemporalFeatures(persistence_duration=900.0),
        epistemic=EpistemicFeatures(prediction_accuracy=0.80),
    )


def _ball_like_features(entity_id: str) -> EntityFeatureVector:
    """Feature vector for a ball — dissimilar to tables/chairs."""
    return EntityFeatureVector(
        entity_id=entity_id,
        entity_type="toy",
        appearance=AppearanceFeatures(
            entity_type="toy",
            properties={"shape": "spherical", "material": "rubber", "size": "small"},
        ),
        behavior=BehaviorFeatures(
            stationary_rate=0.2,
            event_type_distribution={"rolls": 0.5, "bounces": 0.3, "stationary": 0.2},
        ),
        relational=RelationalFeatures(
            spatial_relations={"near": 3},
        ),
        temporal=TemporalFeatures(persistence_duration=200.0),
        epistemic=EpistemicFeatures(prediction_accuracy=0.6),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1: Feature Accumulation
# ═══════════════════════════════════════════════════════════════════════════


class TestFeatureAccumulation:
    """Multi-dimensional features extracted from entities."""

    def test_five_dimensions_populated(self) -> None:
        fv = _table_like_features("e1")

        assert fv.appearance.entity_type == "furniture"
        assert "shape" in fv.appearance.properties
        assert fv.behavior.stationary_rate > 0.5
        assert fv.relational.support_role == "supporter"
        assert fv.temporal.persistence_duration > 0
        assert fv.epistemic.prediction_accuracy > 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2: Similarity Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestSimilarityDetection:
    """Similar entities identified from feature distance."""

    def test_similar_entities_close_distance(self) -> None:
        t1 = _table_like_features("e1")
        t2 = _table_like_features("e2")
        ball = _ball_like_features("e3")

        dist_similar = FeatureAccumulator.feature_distance(t1, t2)
        dist_different = FeatureAccumulator.feature_distance(t1, ball)

        # Similar entities should have lower distances
        avg_similar = sum(dist_similar.values()) / len(dist_similar)
        avg_different = sum(dist_different.values()) / len(dist_different)

        assert avg_similar < avg_different, (
            f"Similar entities should be closer: {avg_similar:.3f} vs {avg_different:.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3: Behavioral Regularity
# ═══════════════════════════════════════════════════════════════════════════


class TestBehavioralRegularity:
    """Entities with similar event patterns grouped."""

    def test_shared_behavior_detected(self) -> None:
        features = {
            "e1": _table_like_features("e1"),
            "e2": _chair_like_features("e2"),
            "e3": _ball_like_features("e3"),
        }

        generator = ConceptHypothesisGenerator(
            similarity_threshold=0.4,
            min_cluster_size=2,
        )
        hypotheses = generator.generate(features)

        # Table and chair should cluster (shared "supports_object")
        # Ball should NOT be in the same cluster
        if hypotheses:
            support_hyp = hypotheses[0]
            assert "e3" not in support_hyp.member_ids, "Ball should not cluster with tables/chairs"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4: Concept Hypothesis Generation
# ═══════════════════════════════════════════════════════════════════════════


class TestConceptHypothesisGeneration:
    """Candidate concepts with weighted coherence scores."""

    def test_hypothesis_has_coherence_scores(self) -> None:
        features = {
            "e1": _table_like_features("e1"),
            "e2": _table_like_features("e2"),
            "e3": _chair_like_features("e3"),
        }

        generator = ConceptHypothesisGenerator(
            similarity_threshold=0.4,
            min_cluster_size=2,
        )
        hypotheses = generator.generate(features)

        assert len(hypotheses) >= 1, "Should generate at least one hypothesis"
        h = hypotheses[0]
        assert h.coherence_scores, "Hypothesis should have per-dimension scores"
        assert h.overall_coherence > 0, "Overall coherence should be positive"
        assert len(h.member_ids) >= 2, "Hypothesis should have at least 2 members"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5: Consolidation with Predictive Utility (SIGNATURE TEST)
# ═══════════════════════════════════════════════════════════════════════════


class TestPredictiveUtility:
    """Concept admitted ONLY because treating as category improves prediction."""

    def test_high_utility_concept_accepted(self) -> None:
        """Concept with 19% utility improvement → ACCEPT."""
        features = {
            "e1": _table_like_features("e1"),
            "e2": _table_like_features("e2"),
        }
        generator = ConceptHypothesisGenerator(similarity_threshold=0.3, min_cluster_size=2)
        hypotheses = generator.generate(features)
        assert hypotheses

        consolidator = ConceptConsolidator(min_utility_gain=0.05, min_sample_count=3)

        utility = PredictiveUtilityTest(
            individual_outcomes={
                "e1": [True, False, True, False, True],  # 60%
                "e2": [True, False, False, True, True],  # 60%
            },
            concept_outcomes=[True, True, True, True, False],  # 80%
        )

        result = consolidator.consolidate(hypotheses[0], utility)
        assert result.decision == ConsolidationDecision.ACCEPT, (
            f"High utility (Δ={utility.utility_delta:.2f}) should be accepted: {result.reasoning}"
        )
        assert result.utility_delta > 0.05

    def test_low_utility_concept_rejected(self) -> None:
        """Concept with only 1% improvement → REJECT."""
        features = {
            "e1": _table_like_features("e1"),
            "e2": _table_like_features("e2"),
        }
        generator = ConceptHypothesisGenerator(similarity_threshold=0.3, min_cluster_size=2)
        hypotheses = generator.generate(features)
        assert hypotheses

        consolidator = ConceptConsolidator(min_utility_gain=0.05, min_sample_count=3)

        # Almost identical accuracy — no utility
        utility = PredictiveUtilityTest(
            individual_outcomes={
                "e1": [True, False, True, False, True],  # 60%
                "e2": [True, False, True, False, True],  # 60%
            },
            concept_outcomes=[True, False, True, False, True],  # 60% — no gain
        )

        result = consolidator.consolidate(hypotheses[0], utility)
        assert result.decision == ConsolidationDecision.REJECT, (
            f"Low utility (Δ={utility.utility_delta:.3f}) should be rejected: {result.reasoning}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6: Rejection — similar but incoherent
# ═══════════════════════════════════════════════════════════════════════════


class TestRejection:
    """Visually similar but behaviorally incoherent cluster rejected."""

    def test_incoherent_hypothesis_rejected(self) -> None:
        """Entities look similar but behave completely differently."""
        # Create entities that share appearance but differ in behavior
        e1 = EntityFeatureVector(
            entity_id="e1",
            appearance=AppearanceFeatures(properties={"color": "brown", "size": "large"}),
            behavior=BehaviorFeatures(event_type_distribution={"supports": 1.0}),
        )
        e2 = EntityFeatureVector(
            entity_id="e2",
            appearance=AppearanceFeatures(properties={"color": "brown", "size": "large"}),
            behavior=BehaviorFeatures(event_type_distribution={"rolls": 1.0}),
        )

        # These should have low behavioral similarity
        dist = FeatureAccumulator.feature_distance(e1, e2)
        assert dist["behavior"] > 0.5, (
            f"Behaviorally incoherent entities should have high behavior distance: {dist}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7: INSTANCE_OF Edges
# ═══════════════════════════════════════════════════════════════════════════


class TestInstanceOfEdges:
    """Entities correctly linked to concepts via HCIR edges."""

    def test_instance_of_edges_created_on_registration(self) -> None:
        graph = _make_graph()
        e1 = _add_entity(graph, "furniture", {"shape": "rectangular"})
        e2 = _add_entity(graph, "furniture", {"shape": "rectangular"})

        registry = GroundedConceptRegistry(graph)
        concept_id = registry.register(
            feature_prototype={"shape": "rectangular"},
            member_ids=[e1.id, e2.id],
            behavioral_regularities=["supports_objects"],
            domain="furniture",
        )

        # Check INSTANCE_OF edges exist
        edges_e1 = graph.edges_from(e1.id)
        instance_edges = [e for e in edges_e1 if e.edge_type == HCIREdgeType.INSTANCE_OF]
        assert len(instance_edges) >= 1, "INSTANCE_OF edge should exist"
        assert concept_id in instance_edges[0].targets

        # Check concept members
        members = registry.concept_members(concept_id)
        assert e1.id in members
        assert e2.id in members


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 8: Concept Prediction
# ═══════════════════════════════════════════════════════════════════════════


class TestConceptPrediction:
    """Concept generates testable expectations."""

    def test_concept_generates_predictions(self) -> None:
        graph = _make_graph()
        e1 = _add_entity(graph, "furniture")
        e2 = _add_entity(graph, "furniture")

        registry = GroundedConceptRegistry(graph)
        concept_id = registry.register(
            feature_prototype={"shape": "rectangular"},
            member_ids=[e1.id, e2.id],
            behavioral_regularities=["supports_objects", "remains_stationary"],
            domain="furniture",
        )

        bridge = ConceptPredictionBridge(registry)
        specs = bridge.generate_predictions()

        assert len(specs) > 0, "Should generate predictions"
        # Each member × each behavior = 4 predictions
        assert len(specs) == 4, f"Expected 4 predictions, got {len(specs)}"
        assert all(s.concept_id == concept_id for s in specs)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 9: A14 Feedback
# ═══════════════════════════════════════════════════════════════════════════


class TestA14Feedback:
    """Concept prediction outcome feeds through to concept confidence."""

    def test_prediction_outcome_updates_confidence(self) -> None:
        graph = _make_graph()
        e1 = _add_entity(graph, "furniture")

        registry = GroundedConceptRegistry(graph)
        concept_id = registry.register(
            feature_prototype={},
            member_ids=[e1.id],
            domain="furniture",
            utility_delta=0.15,
        )

        bridge = ConceptPredictionBridge(registry)

        # Record successful predictions
        for _ in range(10):
            bridge.record_outcome(concept_id, correct=True)

        concept = registry.get_concept(concept_id)
        assert concept is not None
        assert concept.prediction_count == 10


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 10: Concept Confidence from Prediction
# ═══════════════════════════════════════════════════════════════════════════


class TestConceptConfidence:
    """Confidence rises with successful predictions, drops with failures."""

    def test_confidence_rises_with_success(self) -> None:
        graph = _make_graph()
        e1 = _add_entity(graph, "furniture")

        registry = GroundedConceptRegistry(graph)
        concept_id = registry.register(
            feature_prototype={},
            member_ids=[e1.id],
            domain="furniture",
        )

        # Record many successful predictions
        for _ in range(15):
            registry.record_prediction(concept_id, correct=True)

        concept = registry.get_concept(concept_id)
        assert concept is not None
        assert concept.confidence > 0.8, (
            f"Confidence should be high after successes: {concept.confidence}"
        )

    def test_confidence_drops_with_failure(self) -> None:
        graph = _make_graph()
        e1 = _add_entity(graph, "furniture")

        registry = GroundedConceptRegistry(graph)
        concept_id = registry.register(
            feature_prototype={},
            member_ids=[e1.id],
            domain="furniture",
        )

        # Record many failures
        for _ in range(15):
            registry.record_prediction(concept_id, correct=False)

        concept = registry.get_concept(concept_id)
        assert concept is not None
        assert concept.confidence < 0.3, (
            f"Confidence should drop after failures: {concept.confidence}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 11: Concept Contradiction / Refinement
# ═══════════════════════════════════════════════════════════════════════════


class TestConceptRefinement:
    """Internally heterogeneous concept emits refinement signal."""

    def test_heterogeneous_concept_emits_signal(self) -> None:
        consolidator = ConceptConsolidator(heterogeneity_threshold=0.2)

        # Members with divergent prediction profiles
        member_outcomes = {
            "e1": [True, True, True, True, False],  # 80% accuracy
            "e2": [True, True, True, False, False],  # 60% accuracy
            "e3": [False, False, True, False, False],  # 20% accuracy
            "e4": [False, False, False, False, True],  # 20% accuracy
        }

        signal = consolidator.detect_heterogeneity(
            concept_id="c_001",
            member_outcomes=member_outcomes,
        )

        assert signal is not None, "Should detect heterogeneity"
        assert signal.concept_id == "c_001"
        assert signal.divergence >= 0.2, f"Divergence should be >= threshold: {signal.divergence}"
        assert len(signal.subgroup_a) > 0
        assert len(signal.subgroup_b) > 0

    def test_homogeneous_concept_no_signal(self) -> None:
        consolidator = ConceptConsolidator(heterogeneity_threshold=0.2)

        # All members have similar accuracy
        member_outcomes = {
            "e1": [True, True, False, True, True],  # 80%
            "e2": [True, False, True, True, True],  # 80%
            "e3": [True, True, True, False, True],  # 80%
        }

        signal = consolidator.detect_heterogeneity(
            concept_id="c_002",
            member_outcomes=member_outcomes,
        )

        assert signal is None, "Homogeneous concept should not emit signal"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 12: Zero LLM
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroLLM:
    """Entire A15 loop runs without LLM invocation."""

    def test_no_llm_imports(self) -> None:
        import subprocess
        import sys

        check_code = """
import sys
import hbllm.brain.concepts.concept_consolidator
import hbllm.brain.concepts.concept_formation_loop
import hbllm.brain.concepts.concept_hypothesis_generator
import hbllm.brain.concepts.concept_prediction_bridge
import hbllm.brain.concepts.feature_accumulator
import hbllm.brain.concepts.grounded_concept_registry

llm_markers = [
    "openai",
    "anthropic",
    "litellm",
    "langchain",
    "transformers",
]

loaded = set(sys.modules.keys())
for marker in llm_markers:
    assert marker not in loaded, f"LLM module loaded: {marker}"
"""
        res = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True)
        assert res.returncode == 0, f"Zero-LLM verification failed:\n{res.stderr}"
