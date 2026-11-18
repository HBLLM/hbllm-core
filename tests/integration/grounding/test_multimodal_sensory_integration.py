"""Multi-Modal Sensory Integration Tests.

Validates that grid, kinematic, and text observations all flow through
the same CompositeObservationAdapter → FeatureExtractorRegistry → ConceptLearner
pipeline, and that cross-modal entity identification and concept formation work.
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.grounding.concepts.concept_learner import ConceptLearner
from hbllm.hcir.grounding.concepts.store import WorkingConceptStore
from hbllm.hcir.grounding.contracts.enums import ModalitySource
from hbllm.hcir.grounding.features.composite_observation_adapter import (
    CompositeObservationAdapter,
    GridObservationAdapter,
    RoboticsObservationAdapter,
    TextObservationAdapter,
)
from hbllm.hcir.grounding.features.feature_extractor import PerceptualFeatureVector
from hbllm.hcir.grounding.features.feature_fusion import (
    FeatureFusion,
    MultiModalFeatureBundle,
    group_features_by_entity,
)


class TestGridObservationAdapter:
    def test_adapt_produces_envelope_and_features(self) -> None:
        adapter = GridObservationAdapter()
        grid = np.zeros((5, 5), dtype=int)
        grid[2, 3] = 4
        envelope, features = adapter.adapt(grid, step=1)

        assert envelope.observation_id.startswith("obs_grid_")
        assert envelope.source_modality == ModalitySource.VISION
        assert envelope.metadata["grid_shape"] == [5, 5]

    def test_adapt_increments_sequence(self) -> None:
        adapter = GridObservationAdapter()
        grid = np.zeros((3, 3), dtype=int)
        e1, _ = adapter.adapt(grid)
        e2, _ = adapter.adapt(grid)
        assert e2.sequence_number > e1.sequence_number


class TestRoboticsObservationAdapter:
    def test_adapt_produces_kinematic_envelope(self) -> None:
        adapter = RoboticsObservationAdapter()
        envelope, features = adapter.adapt(
            position=(1.0, 2.0, 3.0),
            joint_angles=(0.5, 1.2),
            velocities=(0.1, 0.2, 0.3),
        )

        assert envelope.observation_id.startswith("obs_robot_")
        assert envelope.source_modality == ModalitySource.PROPRIOCEPTIVE
        assert envelope.metadata["position"] == [1.0, 2.0, 3.0]
        assert envelope.metadata["joint_angles"] == [0.5, 1.2]

    def test_adapt_with_gripper_state(self) -> None:
        adapter = RoboticsObservationAdapter()
        envelope, _ = adapter.adapt(
            position=(0.0, 0.0, 0.0),
            gripper_state=0.7,
        )
        assert envelope.metadata["gripper_state"] == 0.7


class TestTextObservationAdapter:
    def test_adapt_produces_text_envelope(self) -> None:
        adapter = TextObservationAdapter()
        envelope, features = adapter.adapt(
            "The red cube is on the table.",
            named_entities={"OBJECT": ["red cube"], "LOCATION": ["table"]},
        )

        assert envelope.observation_id.startswith("obs_text_")
        assert envelope.source_modality == ModalitySource.TEXT
        assert "red cube" in str(envelope.metadata["named_entities"])

    def test_adapt_with_embeddings(self) -> None:
        adapter = TextObservationAdapter()
        embs = [0.1, 0.2, 0.3, 0.4]
        envelope, _ = adapter.adapt("Hello", embeddings=embs)
        assert envelope.metadata["pre_extracted_embeddings"] == embs


class TestCompositeObservationAdapter:
    def test_dispatches_to_correct_adapter(self) -> None:
        composite = CompositeObservationAdapter()

        grid = np.zeros((4, 4), dtype=int)
        ge, _ = composite.adapt_grid(grid)
        assert ge.source_modality == ModalitySource.VISION

        re, _ = composite.adapt_robotics(position=(1.0, 2.0))
        assert re.source_modality == ModalitySource.PROPRIOCEPTIVE

        te, _ = composite.adapt_text("Hello world")
        assert te.source_modality == ModalitySource.TEXT

    def test_shared_registry(self) -> None:
        composite = CompositeObservationAdapter()
        assert composite.grid_adapter._registry is composite._registry
        assert composite.robotics_adapter._registry is composite._registry
        assert composite.text_adapter._registry is composite._registry


class TestMultiModalFeatureBundle:
    def test_add_feature_groups_by_modality(self) -> None:
        from hbllm.hcir.grounding.features.feature_extractor import PerceptualFeatureVector

        bundle = MultiModalFeatureBundle(entity_id="e_cube_1")

        fv1 = PerceptualFeatureVector(
            feature_id="fv1",
            observation_id="obs1",
            modality=ModalitySource.VISION,
            channel_id="ch1",
            embedding=(1.0, 2.0, 3.0),
        )
        fv2 = PerceptualFeatureVector(
            feature_id="fv2",
            observation_id="obs1",
            modality=ModalitySource.TEXT,
            channel_id="ch2",
            embedding=(4.0, 5.0),
        )

        bundle.add_feature(fv1)
        bundle.add_feature(fv2)

        assert bundle.modality_count == 2
        assert bundle.is_multimodal
        assert bundle.total_features == 2

    def test_single_modality_not_multimodal(self) -> None:
        from hbllm.hcir.grounding.features.feature_extractor import PerceptualFeatureVector

        bundle = MultiModalFeatureBundle(entity_id="e_1")
        fv = PerceptualFeatureVector(
            feature_id="fv1",
            observation_id="obs1",
            modality=ModalitySource.VISION,
            channel_id="ch1",
            embedding=(1.0, 2.0),
        )
        bundle.add_feature(fv)

        assert not bundle.is_multimodal
        assert bundle.modality_count == 1


class TestFeatureFusion:
    def _make_fv(
        self,
        fid: str,
        modality: ModalitySource,
        embedding: tuple[float, ...],
    ) -> PerceptualFeatureVector:
        from hbllm.hcir.grounding.features.feature_extractor import PerceptualFeatureVector

        return PerceptualFeatureVector(
            feature_id=fid,
            observation_id="obs_test",
            modality=modality,
            channel_id="ch_test",
            embedding=embedding,
        )

    def test_concatenation_fusion(self) -> None:
        bundle = MultiModalFeatureBundle(entity_id="e_1")
        bundle.add_feature(self._make_fv("f1", ModalitySource.VISION, (1.0, 2.0)))
        bundle.add_feature(self._make_fv("f2", ModalitySource.TEXT, (3.0, 4.0)))

        fuser = FeatureFusion(default_strategy="concatenation")
        result = fuser.fuse_bundle(bundle)

        assert result.fused_embedding == (1.0, 2.0, 3.0, 4.0)
        assert result.fusion_method == "concatenation"
        assert len(result.contributing_feature_ids) == 2
        assert result.cross_modal_confidence > 0.3

    def test_mean_pool_fusion(self) -> None:
        bundle = MultiModalFeatureBundle(entity_id="e_1")
        bundle.add_feature(self._make_fv("f1", ModalitySource.VISION, (2.0, 4.0)))
        bundle.add_feature(self._make_fv("f2", ModalitySource.TEXT, (6.0, 8.0)))

        fuser = FeatureFusion()
        result = fuser.fuse_bundle(bundle, strategy="mean_pool")

        assert result.fused_embedding == pytest.approx((4.0, 6.0))
        assert result.fusion_method == "mean_pool"

    def test_weighted_mean_fusion(self) -> None:
        bundle = MultiModalFeatureBundle(entity_id="e_1")
        bundle.add_feature(self._make_fv("f1", ModalitySource.VISION, (10.0,)))
        bundle.add_feature(self._make_fv("f2", ModalitySource.TEXT, (20.0,)))

        fuser = FeatureFusion()
        result = fuser.fuse_bundle(bundle, strategy="weighted_mean")
        assert result.fusion_method == "weighted_mean"
        assert len(result.fused_embedding) == 1

    def test_empty_bundle_fusion(self) -> None:
        bundle = MultiModalFeatureBundle(entity_id="e_empty")
        fuser = FeatureFusion()
        result = fuser.fuse_bundle(bundle)
        assert result.fused_embedding == ()
        assert result.cross_modal_confidence == 0.0

    def test_fuse_bundles_batch(self) -> None:
        b1 = MultiModalFeatureBundle(entity_id="e_1")
        b1.add_feature(self._make_fv("f1", ModalitySource.VISION, (1.0,)))
        b2 = MultiModalFeatureBundle(entity_id="e_2")
        b2.add_feature(self._make_fv("f2", ModalitySource.TEXT, (2.0,)))

        fuser = FeatureFusion()
        results = fuser.fuse_bundles([b1, b2])
        assert len(results) == 2
        assert results[0].entity_id == "e_1"
        assert results[1].entity_id == "e_2"


class TestGroupFeaturesByEntity:
    def test_groups_by_observation_id_by_default(self) -> None:
        from hbllm.hcir.grounding.features.feature_extractor import PerceptualFeatureVector

        fv1 = PerceptualFeatureVector(
            feature_id="f1",
            observation_id="obs_A",
            modality=ModalitySource.VISION,
            channel_id="ch",
            embedding=(1.0,),
        )
        fv2 = PerceptualFeatureVector(
            feature_id="f2",
            observation_id="obs_A",
            modality=ModalitySource.TEXT,
            channel_id="ch",
            embedding=(2.0,),
        )
        fv3 = PerceptualFeatureVector(
            feature_id="f3",
            observation_id="obs_B",
            modality=ModalitySource.VISION,
            channel_id="ch",
            embedding=(3.0,),
        )

        bundles = group_features_by_entity([fv1, fv2, fv3])
        assert len(bundles) == 2

        obs_a_bundle = next(b for b in bundles if b.entity_id == "obs_A")
        assert obs_a_bundle.total_features == 2
        assert obs_a_bundle.is_multimodal

    def test_groups_by_explicit_map(self) -> None:
        from hbllm.hcir.grounding.features.feature_extractor import PerceptualFeatureVector

        fv1 = PerceptualFeatureVector(
            feature_id="f1",
            observation_id="obs1",
            modality=ModalitySource.VISION,
            channel_id="ch",
            embedding=(1.0,),
        )
        fv2 = PerceptualFeatureVector(
            feature_id="f2",
            observation_id="obs1",
            modality=ModalitySource.TEXT,
            channel_id="ch",
            embedding=(2.0,),
        )

        entity_map = {"e_cube": ["f1", "f2"]}
        bundles = group_features_by_entity([fv1, fv2], entity_feature_map=entity_map)
        assert len(bundles) == 1
        assert bundles[0].entity_id == "e_cube"
        assert bundles[0].total_features == 2


class TestCrossModalConceptFormation:
    """Integration: multi-modal features → ConceptLearner → concept formation."""

    def test_concept_from_multimodal_features(self) -> None:
        from hbllm.hcir.grounding.features.feature_extractor import PerceptualFeatureVector

        store = WorkingConceptStore()
        learner = ConceptLearner(concept_store=store, assignment_threshold=5.0)

        # Simulate visual features for entity
        vision_fv = PerceptualFeatureVector(
            feature_id="vis_1",
            observation_id="obs1",
            modality=ModalitySource.VISION,
            channel_id="ch_vis",
            embedding=(1.0, 2.0, 3.0),
        )
        # Simulate text features for same entity
        text_fv = PerceptualFeatureVector(
            feature_id="txt_1",
            observation_id="obs1",
            modality=ModalitySource.TEXT,
            channel_id="ch_txt",
            embedding=(1.1, 2.1, 3.1),
        )

        a1 = learner.learn(vision_fv, exemplar_entity_id="e_cube_1")
        a2 = learner.learn(text_fv, exemplar_entity_id="e_cube_1")

        # Both features should be assigned to the same concept (close embeddings)
        assert a1.concept_ref.id == a2.concept_ref.id

        # Concept should have the entity as exemplar
        concept = store.get_concept(a1.concept_ref.id)
        assert concept is not None
        assert "e_cube_1" in concept.exemplar_entity_ids
