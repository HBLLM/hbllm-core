"""Tests for SGE Pipeline Stage 5: ConceptLearner with unsupervised incremental clustering."""

from __future__ import annotations

from hbllm.hcir.grounding.concepts.concept_learner import (
    ConceptLearner,
    ConceptMergeResult,
    ConceptSplitResult,
)
from hbllm.hcir.grounding.concepts.store import WorkingConceptStore
from hbllm.hcir.grounding.contracts.enums import ConceptLifecycle, ModalitySource
from hbllm.hcir.grounding.features.feature_extractor import PerceptualFeatureVector


def _make_feature(
    embedding: tuple[float, ...],
    feature_id: str = "",
    labels: dict[str, float] | None = None,
) -> PerceptualFeatureVector:
    fid = feature_id or f"feat_{hash(embedding) % 100000:05d}"
    return PerceptualFeatureVector(
        feature_id=fid,
        observation_id=f"obs_{fid}",
        modality=ModalitySource.VISION,
        channel_id="cam_front",
        embedding=embedding,
        semantic_labels=labels or {},
    )


# ---------------------------------------------------------------------------
# Basic assignment
# ---------------------------------------------------------------------------


class TestConceptLearnerAssignment:
    def test_first_feature_spawns_new_concept(self) -> None:
        store = WorkingConceptStore()
        learner = ConceptLearner(concept_store=store)

        feat = _make_feature((1.0, 2.0, 3.0))
        assignment = learner.learn(feat)

        assert assignment.is_new_concept is True
        assert assignment.distance == float("inf")
        assert store.get_concept(assignment.concept_ref.id) is not None
        assert learner.total_assignments == 1

    def test_similar_feature_joins_existing_concept(self) -> None:
        store = WorkingConceptStore()
        learner = ConceptLearner(concept_store=store, assignment_threshold=3.0)

        feat1 = _make_feature((1.0, 2.0, 3.0), feature_id="f1")
        feat2 = _make_feature((1.1, 2.1, 3.1), feature_id="f2")

        a1 = learner.learn(feat1)
        a2 = learner.learn(feat2)

        assert a1.is_new_concept is True
        assert a2.is_new_concept is False
        assert a1.concept_ref.id == a2.concept_ref.id

    def test_distant_feature_spawns_separate_concept(self) -> None:
        store = WorkingConceptStore()
        learner = ConceptLearner(concept_store=store, assignment_threshold=2.0)

        feat1 = _make_feature((1.0, 2.0, 3.0), feature_id="f1")
        feat2 = _make_feature((100.0, 200.0, 300.0), feature_id="f2")

        a1 = learner.learn(feat1)
        a2 = learner.learn(feat2)

        assert a1.is_new_concept is True
        assert a2.is_new_concept is True
        assert a1.concept_ref.id != a2.concept_ref.id

    def test_exemplar_entity_tracking(self) -> None:
        store = WorkingConceptStore()
        learner = ConceptLearner(concept_store=store)

        feat = _make_feature((1.0, 2.0), feature_id="f_ex")
        assignment = learner.learn(feat, exemplar_entity_id="ecand_robot_arm")

        concept = store.get_concept(assignment.concept_ref.id)
        assert concept is not None
        assert "ecand_robot_arm" in concept.exemplar_entity_ids

    def test_batch_learning(self) -> None:
        store = WorkingConceptStore()
        learner = ConceptLearner(concept_store=store, assignment_threshold=3.0)

        features = [
            _make_feature((1.0, 1.0), feature_id="b1"),
            _make_feature((1.1, 1.0), feature_id="b2"),
            _make_feature((50.0, 50.0), feature_id="b3"),
        ]
        assignments = learner.learn_batch(features)

        assert len(assignments) == 3
        # First two should join the same concept, third should be separate
        assert assignments[0].concept_ref.id == assignments[1].concept_ref.id
        assert assignments[0].concept_ref.id != assignments[2].concept_ref.id

    def test_deprecated_concepts_are_skipped(self) -> None:
        store = WorkingConceptStore()
        learner = ConceptLearner(concept_store=store)

        feat1 = _make_feature((1.0, 2.0), feature_id="dep1")
        a1 = learner.learn(feat1)

        # Deprecate the concept
        concept = store.get_concept(a1.concept_ref.id)
        assert concept is not None
        concept.transition_lifecycle(ConceptLifecycle.MERGED_SPLIT_DEPRECATED)

        # New similar feature should spawn a new concept since original is deprecated
        feat2 = _make_feature((1.0, 2.0), feature_id="dep2")
        a2 = learner.learn(feat2)
        assert a2.is_new_concept is True
        assert a2.concept_ref.id != a1.concept_ref.id


# ---------------------------------------------------------------------------
# Quality updates
# ---------------------------------------------------------------------------


class TestConceptLearnerQuality:
    def test_quality_improves_with_repeated_observations(self) -> None:
        store = WorkingConceptStore()
        learner = ConceptLearner(concept_store=store, assignment_threshold=5.0)

        for i in range(10):
            feat = _make_feature((1.0 + i * 0.01, 2.0 + i * 0.01), feature_id=f"q{i}")
            learner.learn(feat)

        concepts = store.get_all_concepts()
        assert len(concepts) == 1
        q = concepts[0].grounding_quality
        assert q.perceptual_stability > 0.3  # Should have improved from initial
        assert q.temporal_persistence > 0.1


# ---------------------------------------------------------------------------
# Concept splitting (multimodality detection)
# ---------------------------------------------------------------------------


class TestConceptLearnerSplit:
    def test_split_occurs_for_multimodal_concept(self) -> None:
        store = WorkingConceptStore()
        learner = ConceptLearner(
            concept_store=store,
            assignment_threshold=10.0,  # Very permissive — forces everything into one concept
            split_dispersion_threshold=3.0,
            min_prototypes_for_split=4,
            min_split_gap=1.0,
        )

        # Feed two distinct clusters that are within the assignment threshold
        # (Euclidean distance ~7, well under 10.0) but whose combined dispersion
        # (~7 > 3.0) triggers a split.
        for i in range(4):
            learner.learn(_make_feature((1.0 + i * 0.1, 1.0 + i * 0.1), feature_id=f"cl_a_{i}"))
        for i in range(4):
            learner.learn(_make_feature((8.0 + i * 0.1, 8.0 + i * 0.1), feature_id=f"cl_b_{i}"))

        splits, _ = learner.maintain()

        # Should have detected the multimodal concept and split it
        assert len(splits) >= 1
        assert len(splits[0].child_concept_ids) == 2

        # Parent should be deprecated
        active = [
            c
            for c in store.get_all_concepts()
            if c.lifecycle_state != ConceptLifecycle.MERGED_SPLIT_DEPRECATED
        ]
        assert len(active) >= 2

    def test_no_split_for_tight_cluster(self) -> None:
        store = WorkingConceptStore()
        learner = ConceptLearner(
            concept_store=store,
            assignment_threshold=5.0,
            split_dispersion_threshold=3.0,
            min_prototypes_for_split=4,
        )

        # All points are near each other — no split expected
        for i in range(6):
            learner.learn(_make_feature((1.0 + i * 0.01, 1.0 + i * 0.01), feature_id=f"tight_{i}"))

        splits, _ = learner.maintain()
        assert len(splits) == 0


# ---------------------------------------------------------------------------
# Concept merging (convergence detection)
# ---------------------------------------------------------------------------


class TestConceptLearnerMerge:
    def test_merge_occurs_for_convergent_concepts(self) -> None:
        store = WorkingConceptStore()
        learner = ConceptLearner(
            concept_store=store,
            assignment_threshold=1.0,  # Strict — creates separate concepts easily
            merge_distance_threshold=0.5,
        )

        # Create two concepts that are very close but below assignment threshold
        # by feeding them with slight variance
        learner.learn(_make_feature((1.00, 1.00), feature_id="mg_a1"))
        learner.learn(_make_feature((1.01, 1.01), feature_id="mg_a2"))  # joins a1

        # Force a second concept by using a very different point first, then converging
        b1 = learner.learn(_make_feature((50.0, 50.0), feature_id="mg_b1"))

        # Now manually adjust b1's concept centroid to be very close to a1's
        c_b = store.get_concept(b1.concept_ref.id)
        assert c_b is not None
        # Reset its feature distributions to be near cluster a
        c_b.feature_distributions["dim_0"] = (1.02, 0.001)
        c_b.feature_distributions["dim_1"] = (1.02, 0.001)

        _, merges = learner.maintain()
        assert len(merges) >= 1

        # Source concepts should be deprecated
        deprecated = [
            c
            for c in store.get_all_concepts()
            if c.lifecycle_state == ConceptLifecycle.MERGED_SPLIT_DEPRECATED
        ]
        assert len(deprecated) >= 2

    def test_no_merge_for_distant_concepts(self) -> None:
        store = WorkingConceptStore()
        learner = ConceptLearner(
            concept_store=store,
            assignment_threshold=1.0,
            merge_distance_threshold=0.5,
        )

        learner.learn(_make_feature((1.0, 1.0), feature_id="nm_a"))
        learner.learn(_make_feature((100.0, 100.0), feature_id="nm_b"))

        _, merges = learner.maintain()
        assert len(merges) == 0


# ---------------------------------------------------------------------------
# Audit trails
# ---------------------------------------------------------------------------


class TestConceptLearnerAudit:
    def test_split_log_is_populated(self) -> None:
        store = WorkingConceptStore()
        learner = ConceptLearner(
            concept_store=store,
            assignment_threshold=10.0,
            split_dispersion_threshold=3.0,
            min_prototypes_for_split=4,
            min_split_gap=1.0,
        )

        for i in range(4):
            learner.learn(_make_feature((1.0 + i * 0.05, 1.0), feature_id=f"sl_a_{i}"))
        for i in range(4):
            learner.learn(_make_feature((8.0 + i * 0.05, 8.0), feature_id=f"sl_b_{i}"))

        learner.maintain()

        log = learner.get_split_log()
        assert len(log) >= 1
        assert isinstance(log[0], ConceptSplitResult)
        assert len(log[0].child_concept_ids) == 2

    def test_merge_log_is_populated(self) -> None:
        store = WorkingConceptStore()
        learner = ConceptLearner(
            concept_store=store,
            assignment_threshold=0.5,
            merge_distance_threshold=1.0,
        )

        learner.learn(_make_feature((1.0, 1.0), feature_id="ml_a"))
        learner.learn(_make_feature((50.0, 50.0), feature_id="ml_b"))

        # Force convergence
        all_c = store.get_all_concepts()
        for c in all_c:
            c.feature_distributions["dim_0"] = (5.0, 0.01)
            c.feature_distributions["dim_1"] = (5.0, 0.01)

        learner.maintain()

        log = learner.get_merge_log()
        assert len(log) >= 1
        assert isinstance(log[0], ConceptMergeResult)


# ---------------------------------------------------------------------------
# Integration: embedding_to_dict
# ---------------------------------------------------------------------------


class TestEmbeddingConversion:
    def test_embedding_converted_to_keyed_dict(self) -> None:
        feat = _make_feature(
            (0.5, 0.6, 0.7),
            labels={"RED": 0.9},
        )
        d = ConceptLearner._embedding_to_dict(feat)

        assert d["dim_0"] == 0.5
        assert d["dim_1"] == 0.6
        assert d["dim_2"] == 0.7
        assert d["label_RED"] == 0.9
