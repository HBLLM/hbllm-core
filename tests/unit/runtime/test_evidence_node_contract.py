"""Contract Tests — PerceptualEvidenceNode and EvidenceIntegrationMixin.

Validates the core architectural invariants:
1. PerceptualEvidenceNode and EvidenceNode are siblings, not parent/child.
2. Both share EvidenceIntegrationMixin fields (incorporation_status, etc.).
3. has_evidence_integration() works on both types.
4. Same schema works for audio, vision, IMU, and arbitrary sensors.
5. Proposition.object_value accepts Any (not just str).
6. SpatialContext has structured bounding_box, not string encoding.
7. TemporalValidity distinguishes observed_at from received_at.
8. Existing EvidenceNode users are unaffected.
"""

from __future__ import annotations

import time

from hbllm.hcir.graph import (
    CognitiveGraph,
    EvidenceNode,
    HCIRNode,
    HCIRNodeType,
    PerceptualEvidenceNode,
    has_evidence_integration,
)
from hbllm.hcir.proposition import (
    BoundingBox,
    Proposition,
    SpatialContext,
    TemporalValidity,
)


class TestArchitecturalInvariants:
    """Core sibling/mixin architecture tests."""

    def test_perceptual_evidence_is_not_evidence_subclass(self) -> None:
        """PerceptualEvidenceNode must NOT inherit from EvidenceNode."""
        pe = PerceptualEvidenceNode()
        assert not isinstance(pe, EvidenceNode)

    def test_evidence_is_not_perceptual_evidence_subclass(self) -> None:
        """EvidenceNode must NOT inherit from PerceptualEvidenceNode."""
        de = EvidenceNode()
        assert not isinstance(de, PerceptualEvidenceNode)

    def test_both_are_hcir_nodes(self) -> None:
        """Both types must be HCIRNode subclasses."""
        pe = PerceptualEvidenceNode()
        de = EvidenceNode()
        assert isinstance(pe, HCIRNode)
        assert isinstance(de, HCIRNode)

    def test_both_have_evidence_integration_fields(self) -> None:
        """Both must have incorporation_status, incorporated_transitions, etc."""
        pe = PerceptualEvidenceNode()
        de = EvidenceNode()

        for node in (pe, de):
            assert hasattr(node, "incorporation_status")
            assert hasattr(node, "incorporated_transitions")
            assert hasattr(node, "novelty_score")
            assert hasattr(node, "temporal_pattern")
            assert hasattr(node, "last_incorporated_at")
            assert hasattr(node, "evidence_type")
            assert hasattr(node, "strength")
            assert hasattr(node, "modality")
            assert hasattr(node, "epistemic_profile")
            assert hasattr(node, "candidates")

    def test_has_evidence_integration_both_types(self) -> None:
        """has_evidence_integration() must return True for both."""
        pe = PerceptualEvidenceNode()
        de = EvidenceNode()
        assert has_evidence_integration(pe)
        assert has_evidence_integration(de)

    def test_has_evidence_integration_false_for_other_nodes(self) -> None:
        """has_evidence_integration() must return False for non-evidence nodes."""
        from hbllm.hcir.graph import BeliefNode, GoalNode

        goal = GoalNode()
        belief = BeliefNode()
        assert not has_evidence_integration(goal)
        # BeliefNode has different fields, not the mixin
        # It doesn't have 'incorporated_transitions'
        assert not has_evidence_integration(belief)

    def test_node_types_are_distinct(self) -> None:
        """Each type must have its own HCIRNodeType."""
        pe = PerceptualEvidenceNode()
        de = EvidenceNode()
        assert pe.node_type == HCIRNodeType.PERCEPTUAL_EVIDENCE
        assert de.node_type == HCIRNodeType.EVIDENCE
        assert pe.node_type != de.node_type

    def test_discovery_evidence_has_discovery_fields(self) -> None:
        """EvidenceNode must have discovery-specific fields."""
        de = EvidenceNode(
            claim_id="claim_1",
            methodology="Controlled experiment",
            sample_size=100,
            effect_size=0.5,
            source_uri="https://example.com/paper",
            limitations=["small sample"],
            dataset_refs=["dataset_1"],
            reproducible=True,
        )
        assert de.claim_id == "claim_1"
        assert de.methodology == "Controlled experiment"
        assert de.sample_size == 100
        assert de.effect_size == 0.5

    def test_perceptual_evidence_does_not_have_discovery_fields(self) -> None:
        """PerceptualEvidenceNode must NOT have discovery-specific fields."""
        pe = PerceptualEvidenceNode()
        assert not hasattr(pe, "claim_id")
        assert not hasattr(pe, "methodology")
        assert not hasattr(pe, "sample_size")
        assert not hasattr(pe, "effect_size")
        assert not hasattr(pe, "source_uri")
        assert not hasattr(pe, "dataset_refs")
        assert not hasattr(pe, "reproducible")


class TestProposition:
    """Proposition type tests — object_value accepts Any."""

    def test_string_value(self) -> None:
        p = Proposition(subject="person", predicate="located_at", object_value="kitchen")
        assert p.object_value == "kitchen"
        assert isinstance(p.object_value, str)

    def test_number_value(self) -> None:
        p = Proposition(
            subject="sensor_4",
            predicate="temperature",
            object_value=22.5,
            object_type="celsius",
        )
        assert p.object_value == 22.5
        assert isinstance(p.object_value, float)

    def test_list_value(self) -> None:
        p = Proposition(
            subject="robot",
            predicate="acceleration",
            object_value=[0.1, -0.3, 9.8],
            object_type="vector3",
        )
        assert p.object_value == [0.1, -0.3, 9.8]
        assert isinstance(p.object_value, list)

    def test_dict_value(self) -> None:
        p = Proposition(
            subject="scene",
            predicate="characterized_as",
            object_value={"indoor": True, "noise_level": 0.3},
            object_type="acoustic_scene",
        )
        assert p.object_value["indoor"] is True
        assert p.object_value["noise_level"] == 0.3

    def test_enum_value(self) -> None:
        p = Proposition(
            subject="door_7",
            predicate="state",
            object_value="OPEN",
            object_type="enum",
        )
        assert p.object_value == "OPEN"

    def test_serialization_roundtrip(self) -> None:
        """Pydantic serialization must handle Any-typed values."""
        p = Proposition(
            subject="robot",
            predicate="acceleration",
            object_value=[0.1, -0.3, 9.8],
            object_type="vector3",
        )
        data = p.model_dump()
        p2 = Proposition.model_validate(data)
        assert p2.object_value == [0.1, -0.3, 9.8]


class TestSpatialContext:
    """SpatialContext tests — structured geometry, not strings."""

    def test_bounding_box_properties(self) -> None:
        bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8)
        assert abs(bbox.area - 0.24) < 1e-6
        cx, cy = bbox.center
        assert abs(cx - 0.3) < 1e-6
        assert abs(cy - 0.5) < 1e-6

    def test_spatial_with_bbox_and_depth(self) -> None:
        sp = SpatialContext(
            frame_id="camera_0",
            bounding_box=BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8),
            depth_meters=2.4,
        )
        assert sp.frame_id == "camera_0"
        assert sp.bounding_box is not None
        assert sp.depth_meters == 2.4

    def test_spatial_with_3d_position(self) -> None:
        sp = SpatialContext(
            frame_id="world",
            position=[1.0, 2.0, 3.0],
            orientation=[1.0, 0.0, 0.0, 0.0],
        )
        assert sp.position == [1.0, 2.0, 3.0]
        assert sp.orientation == [1.0, 0.0, 0.0, 0.0]

    def test_spatial_with_polygon(self) -> None:
        sp = SpatialContext(
            frame_id="camera_0",
            polygon=[[0.1, 0.2], [0.5, 0.2], [0.5, 0.8], [0.1, 0.8]],
        )
        assert sp.polygon is not None
        assert len(sp.polygon) == 4


class TestTemporalValidity:
    """TemporalValidity tests — observation vs ingestion distinction."""

    def test_latency_calculation(self) -> None:
        tv = TemporalValidity(
            observed_at=100.0,
            received_at=100.085,
        )
        assert abs(tv.latency_ms - 85.0) < 0.1

    def test_not_expired_when_no_until(self) -> None:
        tv = TemporalValidity()
        assert not tv.is_expired

    def test_expired_when_past(self) -> None:
        tv = TemporalValidity(valid_until=time.time() - 10)
        assert tv.is_expired


class TestMultiModalEvidence:
    """Same PerceptualEvidenceNode schema works for all modalities."""

    def test_vision_evidence(self) -> None:
        node = PerceptualEvidenceNode(
            proposition=Proposition(
                subject="person_17",
                predicate="located_at",
                object_value="kitchen",
            ),
            spatial=SpatialContext(
                frame_id="camera_0",
                bounding_box=BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8),
                depth_meters=2.4,
            ),
            modality="visual",
            strength=0.94,
        )
        assert node.proposition.subject == "person_17"
        assert node.spatial.depth_meters == 2.4
        assert node.modality == "visual"

    def test_audio_speech_evidence(self) -> None:
        node = PerceptualEvidenceNode(
            proposition=Proposition(
                subject="utterance_42",
                predicate="transcribed_as",
                object_value="turn on the lights",
                object_type="transcript",
            ),
            modality="audio",
            strength=0.87,
            payload={"language": "en", "is_partial": False},
        )
        assert node.proposition.object_value == "turn on the lights"
        assert node.modality == "audio"
        assert node.payload["language"] == "en"

    def test_audio_event_evidence(self) -> None:
        node = PerceptualEvidenceNode(
            proposition=Proposition(
                subject="audio_event_82",
                predicate="indicates",
                object_value="doorbell",
                object_type="event_class",
            ),
            modality="audio",
            strength=0.95,
            candidates=[
                {"label": "doorbell", "confidence": 0.95},
                {"label": "knock", "confidence": 0.03},
            ],
        )
        assert node.candidates[0]["label"] == "doorbell"

    def test_imu_evidence(self) -> None:
        node = PerceptualEvidenceNode(
            proposition=Proposition(
                subject="robot",
                predicate="acceleration",
                object_value=[0.1, -0.3, 9.8],
                object_type="vector3",
            ),
            modality="sensor",
            strength=1.0,
            provider_provenance={"provider": "imu_01", "model": "bmi270"},
        )
        assert node.proposition.object_value == [0.1, -0.3, 9.8]
        assert node.modality == "sensor"

    def test_temperature_evidence(self) -> None:
        node = PerceptualEvidenceNode(
            proposition=Proposition(
                subject="sensor_4",
                predicate="temperature",
                object_value=22.5,
                object_type="celsius",
            ),
            modality="sensor",
            strength=0.99,
        )
        assert node.proposition.object_value == 22.5

    def test_door_state_evidence(self) -> None:
        node = PerceptualEvidenceNode(
            proposition=Proposition(
                subject="door_7",
                predicate="state",
                object_value="OPEN",
                object_type="enum",
            ),
            modality="iot",
            strength=1.0,
        )
        assert node.proposition.object_value == "OPEN"


class TestGraphIntegration:
    """PerceptualEvidenceNode can be added to CognitiveGraph."""

    def test_upsert_and_retrieve(self) -> None:
        graph = CognitiveGraph()
        node = PerceptualEvidenceNode(
            proposition=Proposition(
                subject="person_17",
                predicate="located_at",
                object_value="kitchen",
            ),
            modality="visual",
            strength=0.94,
        )
        graph.upsert_node(node)
        retrieved = graph.get_node(node.id)
        assert isinstance(retrieved, PerceptualEvidenceNode)
        assert retrieved.proposition.subject == "person_17"

    def test_incorporation_workflow(self) -> None:
        """Evidence integration fields work for belief revision."""
        node = PerceptualEvidenceNode(
            proposition=Proposition(
                subject="person_17",
                predicate="located_at",
                object_value="kitchen",
            ),
            modality="visual",
            strength=0.94,
        )
        assert node.incorporation_status == "pending"
        assert node.incorporated_transitions == {}
        assert node.novelty_score == 1.0

        # Simulate incorporation
        node.incorporation_status = "incorporated"
        node.incorporated_transitions["belief_123"] = "trans_abc"
        node.last_incorporated_at = time.time()

        assert node.incorporation_status == "incorporated"
        assert "belief_123" in node.incorporated_transitions
