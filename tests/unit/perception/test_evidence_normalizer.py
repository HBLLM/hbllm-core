"""Unit Tests for EvidenceNormalizer."""

from __future__ import annotations

from hbllm.hcir.graph import PerceptualEvidenceNode
from hbllm.hcir.proposition import BoundingBox, SpatialContext
from hbllm.perception.evidence_normalizer import EvidenceNormalizer
from hbllm.perception.providers.audio_evidence import (
    AcousticObservation,
    AcousticSceneEvidence,
    AudioAssessment,
    AudioEpistemicProfile,
    SoundEventEvidence,
    SpeechEvidence,
)
from hbllm.perception.providers.audio_types import (
    AudioEventState,
    ParalinguisticProfile,
    SpeakerIdentification,
    TemporalSpan,
)
from hbllm.perception.providers.evidence import (
    CandidateRanking,
    ConceptCandidate,
    EpistemicEvidenceProfile,
    VisualAssessment,
    VisualEvidence,
)
from hbllm.perception.providers.provider_provenance import ProviderProvenance
from hbllm.perception.providers.types import VisualEmbedding


class TestEvidenceNormalizer:
    def test_normalize_speech(self) -> None:
        normalizer = EvidenceNormalizer()
        speech = SpeechEvidence(
            observation=AcousticObservation(
                observation_id="aobs_123",
                temporal=TemporalSpan(start_time=100.0, end_time=102.5, duration=2.5),
            ),
            transcript="turn on the kitchen light",
            language="en",
            confidence=0.92,
            speaker_ref=SpeakerIdentification(speaker_id="user_1", confidence=0.88),
            paralinguistic=ParalinguisticProfile(
                tone="calm",
                confidence=0.95,
            ),
            provider_provenance=ProviderProvenance(
                provider="whisper",
                model="whisper-base",
                version="1.0",
                device="cpu",
            ),
        )

        node = normalizer.normalize_speech(speech)
        assert isinstance(node, PerceptualEvidenceNode)
        assert node.modality == "audio"
        assert node.proposition.subject == "aobs_123"
        assert node.proposition.predicate == "transcribed_as"
        assert node.proposition.object_value == "turn on the kitchen light"
        assert node.proposition.object_type == "transcript"
        assert node.strength == 0.92
        assert node.payload["language"] == "en"
        assert node.payload["speaker_id"] == "user_1"
        assert node.payload["tone"] == "calm"
        assert node.provider_provenance["provider"] == "whisper"
        assert node.temporal_validity.observed_at == 100.0

    def test_normalize_sound_event(self) -> None:
        normalizer = EvidenceNormalizer()
        event = SoundEventEvidence(
            observation=AcousticObservation(
                observation_id="aobs_456",
                temporal=TemporalSpan(start_time=200.0, duration=1.0),
            ),
            event_type="doorbell",
            confidence=0.96,
            is_critical=True,
            event_state=AudioEventState.INSTANTANEOUS,
            top_classes=[("doorbell", 0.96), ("chime", 0.03)],
            provider_provenance=ProviderProvenance(provider="yamnet", model="v1"),
        )

        node = normalizer.normalize_sound_event(event)
        assert isinstance(node, PerceptualEvidenceNode)
        assert node.proposition.subject == "aobs_456"
        assert node.proposition.predicate == "indicates"
        assert node.proposition.object_value == "doorbell"
        assert node.proposition.object_type == "event_class"
        assert node.strength == 0.96
        assert len(node.candidates) == 2
        assert node.candidates[0]["label"] == "doorbell"
        assert node.payload["is_critical"] is True

    def test_normalize_acoustic_scene(self) -> None:
        normalizer = EvidenceNormalizer()
        scene = AcousticSceneEvidence(
            observation=AcousticObservation(observation_id="aobs_789"),
            indoor=True,
            speech_present=True,
            noise_level=0.2,
            estimated_activity=0.5,
            scene_tags=["office", "quiet"],
            confidence=0.85,
        )

        node = normalizer.normalize_acoustic_scene(scene)
        assert isinstance(node, PerceptualEvidenceNode)
        assert node.proposition.subject == "aobs_789"
        assert node.proposition.predicate == "characterized_as"
        assert node.proposition.object_type == "acoustic_scene"
        assert node.proposition.object_value["indoor"] is True
        assert node.proposition.object_value["noise_level"] == 0.2

    def test_normalize_audio_assessment_compound(self) -> None:
        normalizer = EvidenceNormalizer()
        speech = SpeechEvidence(
            observation=AcousticObservation(observation_id="aobs_1"),
            transcript="hello world",
            confidence=0.9,
        )
        event = SoundEventEvidence(
            observation=AcousticObservation(observation_id="aobs_2"),
            event_type="knock",
            confidence=0.8,
        )
        assessment = AudioAssessment(
            speech=speech,
            events=[event],
            epistemic_profile=AudioEpistemicProfile(perceptual_confidence=0.9),
        )

        nodes = normalizer.normalize_audio_assessment(assessment)
        assert len(nodes) == 2
        assert nodes[0].proposition.predicate == "transcribed_as"
        assert nodes[1].proposition.predicate == "indicates"

    def test_normalize_visual(self) -> None:
        normalizer = EvidenceNormalizer()
        vis_ev = VisualEvidence(
            embedding=VisualEmbedding(
                vector=[0.1, 0.2, 0.3],
                model_id="siglip",
                space_id="siglip-base",
                embedding_type="semantic",
                dimensions=3,
            ),
            image_hash="sha256_hash_123",
        )
        candidate = ConceptCandidate(
            concept_node_id="vc_123",
            label="screwdriver",
            mean_similarity=0.85,
            best_similarity=0.91,
            matching_observations=3,
        )
        assessment = VisualAssessment(
            evidence=vis_ev,
            candidate_concepts=[candidate],
            ranking=CandidateRanking(best_score=0.91, second_score=0.4, margin=0.51),
            epistemic_profile=EpistemicEvidenceProfile(
                perceptual_similarity=0.91,
                evidence_strength=0.8,
                source_reliability=0.95,
            ),
        )

        bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8)
        node = normalizer.normalize_visual(
            assessment,
            frame_id="camera_front",
            bounding_box=bbox,
            depth_meters=1.5,
        )

        assert isinstance(node, PerceptualEvidenceNode)
        assert node.modality == "visual"
        assert node.proposition.predicate == "classified_as"
        assert node.proposition.object_value == "screwdriver"
        assert node.strength == 0.91
        assert node.spatial is not None
        assert node.spatial.frame_id == "camera_front"
        assert node.spatial.bounding_box.x1 == 0.1
        assert node.spatial.depth_meters == 1.5
        assert node.epistemic_profile is not None
        assert node.epistemic_profile.sensory_clarity == 0.91

    def test_normalize_sensor(self) -> None:
        normalizer = EvidenceNormalizer()
        sp = SpatialContext(frame_id="robot_base", position=[0.0, 0.0, 0.5])
        node = normalizer.normalize_sensor(
            sensor_id="imu_main",
            predicate="angular_velocity",
            value=[0.01, -0.02, 0.05],
            value_type="rad_per_s_vec3",
            modality="sensor",
            confidence=0.99,
            spatial=sp,
            provider_provenance=ProviderProvenance(provider="imu_driver"),
        )

        assert isinstance(node, PerceptualEvidenceNode)
        assert node.proposition.subject == "imu_main"
        assert node.proposition.predicate == "angular_velocity"
        assert node.proposition.object_value == [0.01, -0.02, 0.05]
        assert node.proposition.object_type == "rad_per_s_vec3"
        assert node.modality == "sensor"
        assert node.spatial.frame_id == "robot_base"
        assert node.provider_provenance["provider"] == "imu_driver"
