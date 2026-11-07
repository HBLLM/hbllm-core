"""Tests for SGE Pipeline Stage 2: FeatureExtractor and PerceptualFeatureVector."""

from __future__ import annotations

import pytest

from hbllm.hcir.grounding.contracts.enums import ModalitySource, TrafficClass
from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope
from hbllm.hcir.grounding.features.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorRegistry,
    KinematicFeatureExtractor,
    PerceptualFeatureVector,
    SemanticTextFeatureExtractor,
    SpatialGridFeatureExtractor,
)

# ---------------------------------------------------------------------------
# PerceptualFeatureVector contract tests
# ---------------------------------------------------------------------------


class TestPerceptualFeatureVector:
    def test_basic_construction(self) -> None:
        fv = PerceptualFeatureVector(
            feature_id="feat_001",
            observation_id="obs_001",
            modality=ModalitySource.VISION,
            channel_id="cam_front",
            embedding=(0.1, 0.2, 0.3),
        )
        assert fv.feature_id == "feat_001"
        assert fv.observation_id == "obs_001"
        assert fv.dimensionality == 3
        assert fv.provenance_hash  # auto-generated

    def test_empty_embedding_rejected(self) -> None:
        with pytest.raises(ValueError, match="embedding must be non-empty"):
            PerceptualFeatureVector(
                feature_id="feat_bad",
                observation_id="obs_001",
                modality=ModalitySource.VISION,
                channel_id="cam_front",
                embedding=(),
            )

    def test_empty_feature_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="feature_id must be non-empty"):
            PerceptualFeatureVector(
                feature_id="",
                observation_id="obs_001",
                modality=ModalitySource.VISION,
                channel_id="cam_front",
                embedding=(1.0,),
            )

    def test_list_embedding_coerced_to_tuple(self) -> None:
        fv = PerceptualFeatureVector(
            feature_id="feat_coerce",
            observation_id="obs_001",
            modality=ModalitySource.VISION,
            channel_id="cam_front",
            embedding=[0.5, 0.6],  # type: ignore[arg-type]
        )
        assert isinstance(fv.embedding, tuple)
        assert fv.embedding == (0.5, 0.6)

    def test_semantic_labels_and_metadata(self) -> None:
        fv = PerceptualFeatureVector(
            feature_id="feat_002",
            observation_id="obs_002",
            modality=ModalitySource.TEXT,
            channel_id="mic_01",
            embedding=(0.1, 0.2),
            semantic_labels={"PERSON": 0.95, "LOCATION": 0.3},
            feature_metadata={"source": "NER"},
        )
        assert fv.semantic_labels["PERSON"] == 0.95
        assert fv.feature_metadata["source"] == "NER"

    def test_frozen_immutability(self) -> None:
        fv = PerceptualFeatureVector(
            feature_id="feat_frozen",
            observation_id="obs_001",
            modality=ModalitySource.VISION,
            channel_id="cam_front",
            embedding=(1.0, 2.0),
        )
        with pytest.raises(AttributeError):
            fv.feature_id = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FeatureExtractor Protocol conformance
# ---------------------------------------------------------------------------


class TestFeatureExtractorProtocol:
    def test_spatial_grid_is_feature_extractor(self) -> None:
        ext = SpatialGridFeatureExtractor()
        assert isinstance(ext, FeatureExtractor)

    def test_kinematic_is_feature_extractor(self) -> None:
        ext = KinematicFeatureExtractor()
        assert isinstance(ext, FeatureExtractor)

    def test_semantic_text_is_feature_extractor(self) -> None:
        ext = SemanticTextFeatureExtractor()
        assert isinstance(ext, FeatureExtractor)


# ---------------------------------------------------------------------------
# SpatialGridFeatureExtractor
# ---------------------------------------------------------------------------


def _make_envelope(
    modality: ModalitySource = ModalitySource.VISION,
    channel_id: str = "cam_front",
    metadata: dict | None = None,
) -> ObservationEnvelope:
    return ObservationEnvelope.create(
        observation_id=f"obs_{id(metadata)}",
        source_modality=modality,
        source_channel_id=channel_id,
        traffic_class=TrafficClass.CLASS_HIGH_FREQ,
        sequence_number=1,
        payload_ref="mem://test_buffer",
        metadata=metadata or {},
    )


class TestSpatialGridFeatureExtractor:
    def test_can_extract_vision_with_entities(self) -> None:
        ext = SpatialGridFeatureExtractor()
        env = _make_envelope(
            metadata={"entities": [{"id": "e1", "position": (5, 3), "color": 2, "size": 4}]}
        )
        assert ext.can_extract(env) is True

    def test_cannot_extract_audio(self) -> None:
        ext = SpatialGridFeatureExtractor()
        env = _make_envelope(modality=ModalitySource.AUDIO, metadata={"entities": []})
        assert ext.can_extract(env) is False

    def test_cannot_extract_without_entities_or_grid(self) -> None:
        ext = SpatialGridFeatureExtractor()
        env = _make_envelope(metadata={"other_key": 123})
        assert ext.can_extract(env) is False

    def test_extract_produces_feature_vectors(self) -> None:
        ext = SpatialGridFeatureExtractor()
        entities = [
            {"id": "seg_0", "position": (3, 7), "color": 5, "size": 9, "grid_h": 30, "grid_w": 30},
            {"id": "seg_1", "position": (10, 2), "color": 1, "size": 2, "grid_h": 30, "grid_w": 30},
        ]
        env = _make_envelope(metadata={"entities": entities})
        results = ext.extract(env)

        assert len(results) == 2
        for fv in results:
            assert isinstance(fv, PerceptualFeatureVector)
            assert fv.modality == ModalitySource.VISION
            assert fv.observation_id == env.observation_id
            assert len(fv.embedding) == 4  # (norm_row, norm_col, norm_color, norm_size)

    def test_extract_with_labels(self) -> None:
        ext = SpatialGridFeatureExtractor()
        entities = [
            {
                "id": "seg_lab",
                "position": (1, 1),
                "color": 3,
                "size": 5,
                "label": "red_block",
                "label_confidence": 0.92,
            },
        ]
        env = _make_envelope(metadata={"entities": entities})
        results = ext.extract(env)
        assert results[0].semantic_labels["red_block"] == 0.92


# ---------------------------------------------------------------------------
# KinematicFeatureExtractor
# ---------------------------------------------------------------------------


class TestKinematicFeatureExtractor:
    def test_can_extract_kinesthetic(self) -> None:
        ext = KinematicFeatureExtractor()
        env = _make_envelope(
            modality=ModalitySource.KINESTHETIC,
            metadata={"entities": [{"id": "arm", "position": (1.0, 2.0), "velocity": (0.5, 0.0)}]},
        )
        assert ext.can_extract(env) is True

    def test_extract_produces_correct_embedding(self) -> None:
        ext = KinematicFeatureExtractor()
        env = _make_envelope(
            modality=ModalitySource.KINESTHETIC,
            metadata={
                "entities": [
                    {"id": "body", "position": (3.0, 4.0), "velocity": (1.0, 0.0), "mass": 10.0},
                ]
            },
        )
        results = ext.extract(env)
        assert len(results) == 1
        fv = results[0]
        # embedding = (pos_x, pos_y, vel_x, vel_y, speed, log1p(mass))
        assert fv.embedding[0] == 3.0
        assert fv.embedding[1] == 4.0
        assert fv.embedding[2] == 1.0
        assert fv.embedding[3] == 0.0
        assert fv.embedding[4] == pytest.approx(1.0, abs=0.01)  # speed
        assert fv.spatial_extent == (3.0, 4.0)


# ---------------------------------------------------------------------------
# SemanticTextFeatureExtractor
# ---------------------------------------------------------------------------


class TestSemanticTextFeatureExtractor:
    def test_can_extract_text_with_embedding(self) -> None:
        ext = SemanticTextFeatureExtractor()
        env = _make_envelope(
            modality=ModalitySource.TEXT,
            metadata={"sentence_embedding": [0.1, 0.2, 0.3]},
        )
        assert ext.can_extract(env) is True

    def test_can_extract_text_with_named_entities(self) -> None:
        ext = SemanticTextFeatureExtractor()
        env = _make_envelope(
            modality=ModalitySource.TEXT,
            metadata={"named_entities": [{"text": "Alice", "type": "PERSON", "confidence": 0.9}]},
        )
        assert ext.can_extract(env) is True

    def test_extract_sentence_embedding(self) -> None:
        ext = SemanticTextFeatureExtractor()
        env = _make_envelope(
            modality=ModalitySource.TEXT,
            metadata={"sentence_embedding": [0.5, 0.6, 0.7, 0.8]},
        )
        results = ext.extract(env)
        assert len(results) == 1
        assert results[0].embedding == (0.5, 0.6, 0.7, 0.8)

    def test_extract_named_entities(self) -> None:
        ext = SemanticTextFeatureExtractor()
        env = _make_envelope(
            modality=ModalitySource.TEXT,
            metadata={
                "named_entities": [
                    {"text": "Paris", "type": "LOCATION", "confidence": 0.95},
                    {"text": "Bob", "type": "PERSON", "confidence": 0.88},
                ]
            },
        )
        results = ext.extract(env)
        assert len(results) == 2
        assert results[0].semantic_labels["LOCATION"] == 0.95
        assert results[1].semantic_labels["PERSON"] == 0.88


# ---------------------------------------------------------------------------
# FeatureExtractorRegistry
# ---------------------------------------------------------------------------


class TestFeatureExtractorRegistry:
    def test_default_registry_has_three_extractors(self) -> None:
        reg = FeatureExtractorRegistry.default()
        assert len(reg._extractors) == 3

    def test_extract_all_dispatches_correctly(self) -> None:
        reg = FeatureExtractorRegistry.default()
        env = _make_envelope(
            modality=ModalitySource.VISION,
            metadata={"entities": [{"id": "x", "position": (1, 2), "color": 3, "size": 1}]},
        )
        results = reg.extract_all(env)
        assert len(results) >= 1
        assert all(isinstance(r, PerceptualFeatureVector) for r in results)

    def test_extract_all_returns_empty_for_unhandled_modality(self) -> None:
        reg = FeatureExtractorRegistry.default()
        env = _make_envelope(
            modality=ModalitySource.LIDAR,
            metadata={"point_cloud": [1, 2, 3]},
        )
        results = reg.extract_all(env)
        assert results == []

    def test_custom_extractor_registration(self) -> None:
        reg = FeatureExtractorRegistry()

        class DummyExtractor:
            @property
            def extractor_name(self) -> str:
                return "dummy"

            def can_extract(self, envelope: ObservationEnvelope) -> bool:
                return True

            def extract(self, envelope: ObservationEnvelope) -> list[PerceptualFeatureVector]:
                return [
                    PerceptualFeatureVector(
                        feature_id="feat_dummy",
                        observation_id=envelope.observation_id,
                        modality=envelope.source_modality,
                        channel_id=envelope.source_channel_id,
                        embedding=(42.0,),
                    )
                ]

        reg.register(DummyExtractor())
        env = _make_envelope(metadata={})
        results = reg.extract_all(env)
        assert len(results) == 1
        assert results[0].embedding == (42.0,)

    def test_multiple_extractors_for_same_envelope(self) -> None:
        """If an envelope matches multiple extractors, all contribute features."""
        reg = FeatureExtractorRegistry.default()
        env = _make_envelope(
            modality=ModalitySource.KINESTHETIC,
            channel_id="imu_01",
            metadata={
                "entities": [
                    {"id": "arm", "position": (1.0, 2.0), "velocity": (0.1, 0.2)},
                ],
            },
        )
        results = reg.extract_all(env)
        assert len(results) >= 1
