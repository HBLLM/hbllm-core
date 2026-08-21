"""Tests for Audio Perception Contracts — Wave A1.

Tests the type system, protocols, evidence composition model,
temporal identity, and policy configurations.
"""

from __future__ import annotations

import pytest

from hbllm.perception.providers.audio_base import (
    AcousticEventProvider,
    AcousticSceneProvider,
    AudioProvider,
    SoundLocalizationProvider,
    SpeakerProvider,
    SpeechProvider,
)
from hbllm.perception.providers.audio_evidence import (
    AcousticObservation,
    AcousticSceneEvidence,
    AudioAssessment,
    AudioEpistemicProfile,
    SoundEventEvidence,
    SoundSourceEvidence,
    SpeechEvidence,
)
from hbllm.perception.providers.audio_policy import AudioRecognitionPolicy
from hbllm.perception.providers.audio_types import (
    AcousticSceneResult,
    AudioEmbedding,
    AudioEventState,
    ParalinguisticProfile,
    SoundEventResult,
    SoundLocalizationResult,
    SpeakerIdentification,
    SpeechResult,
    TemporalSpan,
)

# ═══════════════════════════════════════════════════════════════════════════
# Temporal Identity
# ═══════════════════════════════════════════════════════════════════════════


class TestTemporalSpan:
    """Tests for TemporalSpan — observation/event/segment identity."""

    def test_default_ids_generated(self) -> None:
        span = TemporalSpan()
        assert span.observation_id.startswith("aobs_")
        assert span.event_id.startswith("aevt_")
        assert span.segment_id.startswith("aseg_")

    def test_ids_unique(self) -> None:
        s1 = TemporalSpan()
        s2 = TemporalSpan()
        assert s1.observation_id != s2.observation_id
        assert s1.event_id != s2.event_id

    def test_same_event_different_observations(self) -> None:
        """Multiple observations can share an event_id."""
        event_id = "aevt_doorbell_001"
        obs1 = TemporalSpan(
            event_id=event_id,
            state=AudioEventState.STARTED,
            start_time=100.0,
        )
        obs2 = TemporalSpan(
            event_id=event_id,
            state=AudioEventState.ENDED,
            start_time=101.7,
            end_time=101.7,
        )
        assert obs1.event_id == obs2.event_id
        assert obs1.observation_id != obs2.observation_id
        assert obs1.state == AudioEventState.STARTED
        assert obs2.state == AudioEventState.ENDED

    def test_event_states(self) -> None:
        assert AudioEventState.STARTED == "started"
        assert AudioEventState.CONTINUED == "continued"
        assert AudioEventState.ENDED == "ended"
        assert AudioEventState.INSTANTANEOUS == "instantaneous"

    def test_duration(self) -> None:
        span = TemporalSpan(start_time=10.0, end_time=11.5, duration=1.5)
        assert span.duration == pytest.approx(1.5)


# ═══════════════════════════════════════════════════════════════════════════
# Speaker
# ═══════════════════════════════════════════════════════════════════════════


class TestSpeakerIdentification:
    """Tests for SpeakerIdentification — structured, not a string."""

    def test_unknown_speaker(self) -> None:
        speaker = SpeakerIdentification()
        assert speaker.speaker_id is None
        assert not speaker.is_identified
        assert not speaker.is_enrolled

    def test_identified_speaker(self) -> None:
        speaker = SpeakerIdentification(
            speaker_id="alice",
            confidence=0.92,
            is_enrolled=True,
            embedding_ref="spk_alice_001",
        )
        assert speaker.is_identified
        assert speaker.speaker_id == "alice"

    def test_low_confidence_not_identified(self) -> None:
        speaker = SpeakerIdentification(
            speaker_id="alice",
            confidence=0.3,
        )
        assert not speaker.is_identified

    def test_voice_characteristics(self) -> None:
        speaker = SpeakerIdentification(
            speaker_id="bob",
            confidence=0.85,
            voice_characteristics={"pitch_hz": 120.0, "timbre": 0.6},
        )
        assert speaker.voice_characteristics["pitch_hz"] == 120.0


# ═══════════════════════════════════════════════════════════════════════════
# Paralinguistic
# ═══════════════════════════════════════════════════════════════════════════


class TestParalinguisticProfile:
    """Tests for ParalinguisticProfile — probabilistic, NOT factual."""

    def test_default_neutral(self) -> None:
        profile = ParalinguisticProfile()
        assert profile.tone == "neutral"
        assert profile.confidence == 0.0

    def test_urgent_tone(self) -> None:
        profile = ParalinguisticProfile(
            tone="urgent",
            confidence=0.78,
            pitch_mean=250.0,
            speech_rate=180.0,
            energy_level=0.9,
        )
        assert profile.tone == "urgent"
        assert profile.energy_level == pytest.approx(0.9)


# ═══════════════════════════════════════════════════════════════════════════
# Provider Results (pre-evidence)
# ═══════════════════════════════════════════════════════════════════════════


class TestSpeechResult:
    """Tests for SpeechResult — raw STT output."""

    def test_basic_result(self) -> None:
        result = SpeechResult(
            transcript="Hello, how are you?",
            language="en",
            confidence=0.95,
        )
        assert result.transcript == "Hello, how are you?"
        assert result.language == "en"
        assert result.confidence == pytest.approx(0.95)

    def test_with_speaker(self) -> None:
        result = SpeechResult(
            transcript="Turn off the lights",
            speaker=SpeakerIdentification(
                speaker_id="alice",
                confidence=0.91,
                is_enrolled=True,
            ),
        )
        assert result.speaker is not None
        assert result.speaker.is_identified

    def test_with_temporal(self) -> None:
        result = SpeechResult(
            transcript="Test",
            temporal=TemporalSpan(
                start_time=10.0,
                end_time=12.5,
                duration=2.5,
            ),
        )
        assert result.temporal.duration == pytest.approx(2.5)


class TestSoundEventResult:
    """Tests for SoundEventResult — raw classification output."""

    def test_doorbell(self) -> None:
        result = SoundEventResult(
            event_type="doorbell",
            confidence=0.91,
            is_critical=False,
            top_classes=[("doorbell", 0.91), ("phone_ring", 0.05)],
        )
        assert result.event_type == "doorbell"
        assert not result.is_critical

    def test_critical_event(self) -> None:
        result = SoundEventResult(
            event_type="smoke_detector",
            confidence=0.94,
            is_critical=True,
        )
        assert result.is_critical


class TestAcousticSceneResult:
    """Tests for AcousticSceneResult — raw scene characterization."""

    def test_indoor_quiet(self) -> None:
        result = AcousticSceneResult(
            indoor=True,
            speech_present=False,
            noise_level=0.1,
            estimated_activity=0.2,
            scene_tags=["indoor", "quiet", "residential"],
        )
        assert result.indoor
        assert not result.speech_present
        assert "quiet" in result.scene_tags


class TestAudioEmbedding:
    """Tests for AudioEmbedding — immutable embedding vector."""

    def test_creation(self) -> None:
        emb = AudioEmbedding(
            vector=[0.1, 0.2, 0.3],
            model_id="yamnet",
            space_id="yamnet-v1",
            dimensions=3,
        )
        assert len(emb.vector) == 3
        assert emb.model_id == "yamnet"

    def test_frozen(self) -> None:
        emb = AudioEmbedding(
            vector=[0.1],
            model_id="test",
            space_id="test",
            dimensions=1,
        )
        with pytest.raises(AttributeError):
            emb.model_id = "changed"  # type: ignore[misc]


class TestSoundLocalizationResult:
    """Tests for SoundLocalizationResult — direction/distance."""

    def test_basic(self) -> None:
        result = SoundLocalizationResult(
            direction_degrees=72.5,
            distance_estimate=3.0,
            confidence=0.8,
        )
        assert result.direction_degrees == pytest.approx(72.5)
        assert result.distance_estimate == pytest.approx(3.0)


# ═══════════════════════════════════════════════════════════════════════════
# Evidence (composition model)
# ═══════════════════════════════════════════════════════════════════════════


class TestAcousticObservation:
    """Tests for AcousticObservation — 'the microphone received this'."""

    def test_default_id(self) -> None:
        obs = AcousticObservation()
        assert obs.observation_id.startswith("aobs_")
        assert obs.embedding_ref is None

    def test_with_embedding_ref(self) -> None:
        obs = AcousticObservation(
            embedding_ref="emb_abc123",
            embedding_space="yamnet-v1",
            energy_db=-25.3,
        )
        assert obs.embedding_ref == "emb_abc123"
        assert obs.energy_db == pytest.approx(-25.3)


class TestSpeechEvidence:
    """Tests for SpeechEvidence — composition with AcousticObservation."""

    def test_observation_is_composed_not_inherited(self) -> None:
        """SpeechEvidence CONTAINS an observation, not IS an observation."""
        evidence = SpeechEvidence(
            transcript="Hello world",
            confidence=0.95,
        )
        assert hasattr(evidence, "observation")
        assert isinstance(evidence.observation, AcousticObservation)
        assert evidence.observation.observation_id.startswith("aobs_")

    def test_multiple_evidence_same_observation(self) -> None:
        """Multiple evidence types can share the same observation."""
        obs = AcousticObservation(energy_db=-20.0)

        speech = SpeechEvidence(
            observation=obs,
            transcript="Come in!",
            confidence=0.92,
        )
        event = SoundEventEvidence(
            observation=obs,
            event_type="doorbell",
            confidence=0.91,
        )
        scene = AcousticSceneEvidence(
            observation=obs,
            indoor=True,
            speech_present=True,
        )

        # All reference the same observation
        assert speech.observation is event.observation
        assert event.observation is scene.observation
        assert speech.observation.observation_id == event.observation.observation_id

    def test_with_speaker_ref(self) -> None:
        evidence = SpeechEvidence(
            transcript="Turn on the light",
            speaker_ref=SpeakerIdentification(
                speaker_id="alice",
                confidence=0.88,
                is_enrolled=True,
            ),
        )
        assert evidence.speaker_ref is not None
        assert evidence.speaker_ref.is_identified

    def test_with_paralinguistic(self) -> None:
        evidence = SpeechEvidence(
            transcript="Help!",
            paralinguistic=ParalinguisticProfile(
                tone="urgent",
                confidence=0.82,
            ),
        )
        assert evidence.paralinguistic is not None
        assert evidence.paralinguistic.tone == "urgent"


class TestSoundEventEvidence:
    """Tests for SoundEventEvidence — event classification evidence."""

    def test_instantaneous_event(self) -> None:
        evidence = SoundEventEvidence(
            event_type="knock",
            confidence=0.87,
            event_state=AudioEventState.INSTANTANEOUS,
        )
        assert evidence.event_state == AudioEventState.INSTANTANEOUS

    def test_critical_event(self) -> None:
        evidence = SoundEventEvidence(
            event_type="glass_breaking",
            confidence=0.93,
            is_critical=True,
        )
        assert evidence.is_critical

    def test_event_with_alternatives(self) -> None:
        evidence = SoundEventEvidence(
            event_type="doorbell",
            confidence=0.91,
            top_classes=[("doorbell", 0.91), ("phone_ring", 0.05)],
        )
        assert len(evidence.top_classes) == 2


class TestSoundSourceEvidence:
    """Tests for SoundSourceEvidence — what and where."""

    def test_with_direction(self) -> None:
        evidence = SoundSourceEvidence(
            source_class="vehicle",
            direction_degrees=270.0,
            distance_estimate=15.0,
            confidence=0.85,
        )
        assert evidence.source_class == "vehicle"
        assert evidence.direction_degrees == pytest.approx(270.0)


class TestAcousticSceneEvidence:
    """Tests for AcousticSceneEvidence — NOT itself an event."""

    def test_scene_not_event(self) -> None:
        """Scene is a separate interpretation, not a subclass of event."""
        scene = AcousticSceneEvidence(
            indoor=True,
            speech_present=True,
            noise_level=0.3,
        )
        assert not isinstance(scene, SoundEventEvidence)

    def test_scene_tags(self) -> None:
        scene = AcousticSceneEvidence(
            indoor=True,
            scene_tags=["residential", "quiet", "daytime"],
        )
        assert "residential" in scene.scene_tags


# ═══════════════════════════════════════════════════════════════════════════
# Epistemic Profile
# ═══════════════════════════════════════════════════════════════════════════


class TestAudioEpistemicProfile:
    """Tests for AudioEpistemicProfile — multi-dimensional confidence."""

    def test_default_combined(self) -> None:
        profile = AudioEpistemicProfile()
        # source_reliability defaults to 1.0, all others 0.0
        # combined = 0.20 * 1.0 = 0.20
        assert profile.combined == pytest.approx(0.2)

    def test_full_confidence(self) -> None:
        profile = AudioEpistemicProfile(
            perceptual_confidence=1.0,
            classification_confidence=1.0,
            source_reliability=1.0,
            label_provenance=1.0,
            temporal_confidence=1.0,
        )
        assert profile.combined == pytest.approx(1.0)

    def test_weighted_combination(self) -> None:
        profile = AudioEpistemicProfile(
            perceptual_confidence=0.8,
            classification_confidence=0.9,
            source_reliability=0.7,
        )
        expected = 0.25 * 0.8 + 0.30 * 0.9 + 0.20 * 0.7
        assert profile.combined == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════════
# Assessment
# ═══════════════════════════════════════════════════════════════════════════


class TestAudioAssessment:
    """Tests for AudioAssessment — full perception output."""

    def test_empty_assessment(self) -> None:
        assessment = AudioAssessment()
        assert assessment.speech is None
        assert assessment.events == []
        assert assessment.scene is None
        assert assessment.source is None

    def test_full_assessment(self) -> None:
        obs = AcousticObservation(energy_db=-18.0)

        assessment = AudioAssessment(
            observation=obs,
            speech=SpeechEvidence(
                observation=obs,
                transcript="Someone's at the door",
                confidence=0.93,
            ),
            events=[
                SoundEventEvidence(
                    observation=obs,
                    event_type="doorbell",
                    confidence=0.91,
                ),
            ],
            scene=AcousticSceneEvidence(
                observation=obs,
                indoor=True,
                speech_present=True,
            ),
        )
        assert assessment.speech is not None
        assert assessment.speech.transcript == "Someone's at the door"
        assert len(assessment.events) == 1
        assert assessment.events[0].event_type == "doorbell"
        assert assessment.scene is not None
        assert assessment.scene.indoor


# ═══════════════════════════════════════════════════════════════════════════
# Policy
# ═══════════════════════════════════════════════════════════════════════════


class TestAudioRecognitionPolicy:
    """Tests for AudioRecognitionPolicy — configurable thresholds."""

    def test_default_thresholds(self) -> None:
        policy = AudioRecognitionPolicy()
        assert policy.speech_confidence_threshold == pytest.approx(0.5)
        assert policy.event_confidence_threshold == pytest.approx(0.6)
        assert policy.min_energy_db == pytest.approx(-40.0)

    def test_strict_policy(self) -> None:
        policy = AudioRecognitionPolicy.strict()
        assert policy.speech_confidence_threshold > 0.5
        assert policy.event_confidence_threshold > 0.6

    def test_permissive_policy(self) -> None:
        policy = AudioRecognitionPolicy.permissive()
        assert policy.speech_confidence_threshold < 0.5
        assert policy.event_confidence_threshold < 0.6

    def test_critical_lower_than_normal(self) -> None:
        policy = AudioRecognitionPolicy()
        assert policy.critical_event_threshold < policy.event_confidence_threshold

    def test_frozen(self) -> None:
        policy = AudioRecognitionPolicy()
        with pytest.raises(AttributeError):
            policy.speech_confidence_threshold = 0.9  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# Protocol Checks
# ═══════════════════════════════════════════════════════════════════════════


class TestProtocolChecks:
    """Tests that protocols are runtime-checkable."""

    def test_audio_provider_is_runtime_checkable(self) -> None:
        assert hasattr(AudioProvider, "__protocol_attrs__") or isinstance(
            AudioProvider,
            type,
        )

    def test_speech_provider_is_runtime_checkable(self) -> None:
        assert hasattr(SpeechProvider, "__protocol_attrs__") or isinstance(
            SpeechProvider,
            type,
        )

    def test_event_provider_is_runtime_checkable(self) -> None:
        assert hasattr(AcousticEventProvider, "__protocol_attrs__") or isinstance(
            AcousticEventProvider,
            type,
        )

    def test_scene_provider_is_runtime_checkable(self) -> None:
        assert hasattr(AcousticSceneProvider, "__protocol_attrs__") or isinstance(
            AcousticSceneProvider,
            type,
        )

    def test_speaker_provider_is_runtime_checkable(self) -> None:
        assert hasattr(SpeakerProvider, "__protocol_attrs__") or isinstance(
            SpeakerProvider,
            type,
        )

    def test_localization_provider_is_runtime_checkable(self) -> None:
        assert hasattr(SoundLocalizationProvider, "__protocol_attrs__") or isinstance(
            SoundLocalizationProvider,
            type,
        )
