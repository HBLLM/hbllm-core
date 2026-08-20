"""Tests for Wave A6 — Temporal & Cross-Modal Integration.

Tests integration between AudioEvidence / AudioObservationNode,
TemporalFuser (candidate patterns), PerceptionFuser (cross-modal fusion),
and WorldStateEngine (dual-source gradual migration).
"""

from __future__ import annotations

import time

import pytest

from hbllm.hcir.graph import (
    AudioObservationNode,
    CognitiveGraph,
    _new_id,
)
from hbllm.perception.audio_perception_runtime import AudioPerceptionRuntime
from hbllm.perception.perception_fuser import PerceptionEvent, PerceptionFuser
from hbllm.perception.providers.audio_evidence import (
    AcousticObservation,
    AudioAssessment,
    SoundEventEvidence,
    SpeechEvidence,
)
from hbllm.perception.providers.audio_types import (
    TemporalSpan,
)
from hbllm.perception.providers.mock_audio_provider import MockAudioProvider
from hbllm.perception.temporal_fuser import (
    PerceptionSnapshot,
    TemporalFuser,
    TemporalPatternCandidate,
)
from hbllm.perception.world_state import WorldStateEngine


@pytest.fixture
def mock_provider() -> MockAudioProvider:
    return MockAudioProvider()


@pytest.fixture
def runtime(mock_provider: MockAudioProvider) -> AudioPerceptionRuntime:
    return AudioPerceptionRuntime(
        speech=mock_provider,
        events=mock_provider,
        scene=mock_provider,
        speaker=mock_provider,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Fuser Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTemporalFuserAudioIntegration:
    """Tests for TemporalFuser candidate pattern detection with audio."""

    def test_doorbell_motion_candidate(self) -> None:
        """Doorbell audio followed by motion generates a TemporalPatternCandidate."""
        fuser = TemporalFuser(window_s=60.0)
        now = time.time()

        # 1. Doorbell audio observation
        doorbell_obs = AudioObservationNode(
            id=_new_id("aobs"),
            event_type="doorbell",
            start_time=now - 5.0,
        )
        c1 = fuser.ingest_audio_observation(doorbell_obs)
        assert len(c1) == 0  # Incomplete sequence

        # 2. Motion event
        motion_snapshot = PerceptionSnapshot(
            event_type="iot.motion",
            sub_type="detected",
            timestamp=now - 1.0,
            room="front_door",
        )
        c2 = fuser.ingest_candidates(motion_snapshot)
        assert len(c2) >= 1
        candidate = c2[0]
        assert isinstance(candidate, TemporalPatternCandidate)
        assert candidate.pattern_name == "doorbell_visitor"
        assert candidate.confidence > 0.0
        assert len(candidate.observations) == 2

    def test_audio_assessment_ingestion(self, runtime: AudioPerceptionRuntime) -> None:
        """Ingesting full AudioAssessment populates temporal events."""
        fuser = TemporalFuser(window_s=60.0)
        assessment = AudioAssessment(
            observation=AcousticObservation(
                observation_id="aobs_123",
                temporal=TemporalSpan(start_time=time.time()),
            ),
            speech=SpeechEvidence(
                transcript="Hello there",
                confidence=0.95,
            ),
            events=[
                SoundEventEvidence(
                    event_type="doorbell",
                    confidence=0.91,
                ),
            ],
        )
        candidates = fuser.ingest_audio_assessment(assessment)
        assert isinstance(candidates, list)
        assert fuser._events_processed >= 2  # Speech + event


# ═══════════════════════════════════════════════════════════════════════════
# Perception Fuser Cross-Modal Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPerceptionFuserCrossModal:
    """Tests for multimodal cross-modal fusion (audio + visual)."""

    def test_audio_visual_fusion(self) -> None:
        """Visual detection + audio speech within window generates FusedContext."""
        fuser = PerceptionFuser(window_seconds=5.0, min_modalities=2)

        # 1. Ingest visual event
        fuser.ingest_event(
            PerceptionEvent(
                modality="visual",
                content="person approaching door",
                confidence=0.92,
            ),
        )

        # 2. Ingest audio observation
        audio_obs = AudioObservationNode(
            id=_new_id("aobs"),
            transcript="I have a delivery",
            event_type="speech",
            start_time=time.time(),
        )
        fused = fuser.ingest_audio_observation(audio_obs)

        assert fused is not None
        assert fused.is_multimodal
        assert "audio" in fused.modalities
        assert "visual" in fused.modalities
        assert len(fused.events) == 2

    def test_audio_assessment_cross_modal_fusion(self) -> None:
        """Ingesting AudioAssessment with previous visual event fuses cleanly."""
        fuser = PerceptionFuser(window_seconds=5.0, min_modalities=2)

        # Visual event
        fuser.ingest_event(
            PerceptionEvent(
                modality="visual",
                content="front door open",
                confidence=0.95,
            ),
        )

        assessment = AudioAssessment(
            observation=AcousticObservation(
                observation_id="aobs_456",
                temporal=TemporalSpan(start_time=time.time()),
            ),
            events=[
                SoundEventEvidence(
                    event_type="footsteps",
                    confidence=0.88,
                ),
            ],
        )
        fused = fuser.ingest_audio_assessment(assessment)
        assert fused is not None
        assert fused.is_multimodal


# ═══════════════════════════════════════════════════════════════════════════
# World State Engine Dual-Source Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWorldStateDualSource:
    """Tests for WorldStateEngine updates from typed audio and HCIR graph."""

    def test_update_from_audio_assessment(self) -> None:
        engine = WorldStateEngine()
        assessment = AudioAssessment(
            observation=AcousticObservation(
                observation_id="aobs_789",
                energy_db=-22.5,
            ),
            events=[
                SoundEventEvidence(
                    event_type="smoke_detector",
                    confidence=0.97,
                    is_critical=True,
                ),
            ],
        )
        engine.update_from_audio_assessment(assessment)
        state = engine.get_state()
        audio_env = state.get("audio_environment", {})
        assert audio_env.get("sound_class") == "smoke_detector"
        assert audio_env.get("is_critical") is True
        assert audio_env.get("confidence") == pytest.approx(0.97)

    def test_update_from_hcir_graph(self) -> None:
        engine = WorldStateEngine()
        graph = CognitiveGraph()

        obs_node = AudioObservationNode(
            id=_new_id("aobs"),
            event_type="speech",
            transcript="Turn on the living room lights",
            start_time=time.time(),
        )
        graph.add_node(obs_node)

        engine.update_from_hcir(graph)
        state = engine.get_state()
        audio_env = state.get("audio_environment", {})
        assert audio_env.get("sound_class") == "speech"
        assert audio_env.get("transcript") == "Turn on the living room lights"
