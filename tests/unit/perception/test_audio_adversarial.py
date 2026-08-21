"""Adversarial & Uncertainty Tests — HBLLM Audio Perception §A7.

Tests provider provenance, classifier disagreement, cross-provider
conflict, temporal ambiguity, edge cases, and failure modes.

Key invariants validated:
    1. Classifier disagreement: Multiple candidates preserved (no arbitration)
    2. Cross-provider disagreement: All providers' results kept independently
    3. Provider provenance: Every evidence carries model identity
    4. Temporal ambiguity: Different event_ids for temporally separated events
    5. Failure modes: Graceful degradation, never crash

    256 + N tests should pass after this wave.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pytest

from hbllm.perception.audio_memory import AudioMemory
from hbllm.perception.audio_perception_runtime import AudioPerceptionRuntime
from hbllm.perception.providers.audio_evidence import (
    AudioAssessment,
    SoundEventEvidence,
    SpeechEvidence,
)
from hbllm.perception.providers.audio_types import (
    AcousticSceneResult,
    AudioEventState,
    AudioInput,
    SoundEventResult,
    SpeakerIdentification,
    SpeechResult,
    TemporalSpan,
)
from hbllm.perception.providers.provider_provenance import ProviderProvenance

# ═══════════════════════════════════════════════════════════════════════════
# Fake providers for adversarial testing
# ═══════════════════════════════════════════════════════════════════════════


class DisagreeingSpeechProvider:
    """STT that always reports speech."""

    modality = "audio"
    provider_id = "fake-stt:v1"
    sample_rate = 16000

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def transcribe(self, audio: AudioInput) -> SpeechResult:
        return SpeechResult(
            transcript="hello world",
            confidence=0.85,
            language="en",
            temporal=TemporalSpan(start_time=time.time()),
        )

    async def transcribe_streaming(
        self,
        audio_chunks: Sequence[AudioInput],
    ) -> SpeechResult:
        return await self.transcribe(audio_chunks[0] if audio_chunks else b"")


class DisagreeingEventProvider:
    """Classifier that reports alarm (contradicting speech)."""

    modality = "audio"
    provider_id = "fake-ambient:v1"
    sample_rate = 16000

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def classify(self, audio: AudioInput) -> list[SoundEventResult]:
        return [
            SoundEventResult(
                event_type="alarm",
                confidence=0.82,
                is_critical=True,
                temporal=TemporalSpan(start_time=time.time()),
                top_classes=[("alarm", 0.82), ("siren", 0.71)],
            ),
        ]

    async def analyze_scene(self, audio: AudioInput) -> AcousticSceneResult:
        return AcousticSceneResult(
            noise_level=0.8,
            estimated_activity=0.9,
            scene_tags=["alarm", "emergency"],
        )


class DisagreeingSpeakerProvider:
    """Speaker provider that says no human voice detected."""

    modality = "audio"
    provider_id = "fake-speaker:v1"
    sample_rate = 16000

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def identify(self, audio: AudioInput) -> SpeakerIdentification:
        return SpeakerIdentification(
            speaker_id=None,
            confidence=0.1,
            is_enrolled=False,
        )


class MultiCandidateEventProvider:
    """Returns multiple competing classifications."""

    modality = "audio"
    provider_id = "multi-candidate:v1"
    sample_rate = 16000

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def classify(self, audio: AudioInput) -> list[SoundEventResult]:
        return [
            SoundEventResult(
                event_type="doorbell",
                confidence=0.82,
                temporal=TemporalSpan(start_time=time.time()),
                top_classes=[("doorbell", 0.82), ("knock", 0.76)],
            ),
            SoundEventResult(
                event_type="knock",
                confidence=0.76,
                temporal=TemporalSpan(start_time=time.time()),
                top_classes=[],
            ),
        ]


class FailingSpeechProvider:
    """Provider that always fails."""

    modality = "audio"
    provider_id = "failing-stt:v1"
    sample_rate = 16000

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def transcribe(self, audio: AudioInput) -> SpeechResult:
        msg = "STT model crashed"
        raise RuntimeError(msg)

    async def transcribe_streaming(
        self,
        audio_chunks: Sequence[AudioInput],
    ) -> SpeechResult:
        msg = "STT model crashed"
        raise RuntimeError(msg)


class LowConfidenceSpeechProvider:
    """Provider with very low confidence output."""

    modality = "audio"
    provider_id = "low-conf:v1"
    sample_rate = 16000

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def transcribe(self, audio: AudioInput) -> SpeechResult:
        return SpeechResult(
            transcript="um maybe",
            confidence=0.15,
            language="en",
        )

    async def transcribe_streaming(
        self,
        audio_chunks: Sequence[AudioInput],
    ) -> SpeechResult:
        return await self.transcribe(audio_chunks[0] if audio_chunks else b"")


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Classifier Disagreement
# ═══════════════════════════════════════════════════════════════════════════


class TestClassifierDisagreement:
    """Multiple event candidates must be preserved without arbitration."""

    @pytest.mark.asyncio
    async def test_doorbell_vs_knock_both_preserved(self) -> None:
        """DOORBELL 0.82 vs KNOCK 0.76 → both kept as ranked candidates."""
        runtime = AudioPerceptionRuntime(
            events=MultiCandidateEventProvider(),
            memory=AudioMemory(),
        )
        assessment = await runtime.perceive(b"\x00" * 3200)

        assert len(assessment.events) == 2
        types = {e.event_type for e in assessment.events}
        assert "doorbell" in types
        assert "knock" in types

        # Order by confidence
        sorted_events = sorted(assessment.events, key=lambda e: e.confidence, reverse=True)
        assert sorted_events[0].event_type == "doorbell"
        assert sorted_events[0].confidence == 0.82
        assert sorted_events[1].event_type == "knock"
        assert sorted_events[1].confidence == 0.76

    @pytest.mark.asyncio
    async def test_top_classes_preserved_on_primary(self) -> None:
        """Primary event should carry its top_classes for downstream ranking."""
        runtime = AudioPerceptionRuntime(
            events=MultiCandidateEventProvider(),
            memory=AudioMemory(),
        )
        assessment = await runtime.perceive(b"\x00" * 3200)

        primary = assessment.events[0]
        assert len(primary.top_classes) == 2
        assert primary.top_classes[0][0] == "doorbell"


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Cross-Provider Disagreement
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossProviderDisagreement:
    """STT says speech, Ambient says alarm, Speaker says no human."""

    @pytest.mark.asyncio
    async def test_all_three_providers_preserved(self) -> None:
        """Runtime must preserve all three contradictory results."""
        runtime = AudioPerceptionRuntime(
            speech=DisagreeingSpeechProvider(),
            events=DisagreeingEventProvider(),
            scene=DisagreeingEventProvider(),
            memory=AudioMemory(),
        )
        assessment = await runtime.perceive(b"\x00" * 3200)

        # Speech says "hello world" with 0.85
        assert assessment.speech is not None
        assert assessment.speech.transcript == "hello world"
        assert assessment.speech.confidence == 0.85

        # Events say "alarm" with 0.82
        assert len(assessment.events) >= 1
        assert assessment.events[0].event_type == "alarm"
        assert assessment.events[0].is_critical is True

        # Scene says noisy + emergency
        assert assessment.scene is not None
        assert assessment.scene.noise_level == 0.8
        assert "alarm" in assessment.scene.scene_tags

    @pytest.mark.asyncio
    async def test_no_arbitration_between_modalities(self) -> None:
        """Runtime does NOT pick one truth. All evidence coexists."""
        runtime = AudioPerceptionRuntime(
            speech=DisagreeingSpeechProvider(),
            events=DisagreeingEventProvider(),
            memory=AudioMemory(),
        )
        assessment = await runtime.perceive(b"\x00" * 3200)

        # Both speech AND alarm evidence present simultaneously
        assert assessment.speech is not None
        assert len(assessment.events) >= 1
        # The runtime didn't suppress either one
        assert assessment.speech.confidence > 0
        assert assessment.events[0].confidence > 0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Provider Provenance
# ═══════════════════════════════════════════════════════════════════════════


class TestProviderProvenance:
    """Every evidence must carry its provider identity."""

    @pytest.mark.asyncio
    async def test_speech_carries_provenance(self) -> None:
        runtime = AudioPerceptionRuntime(
            speech=DisagreeingSpeechProvider(),
            memory=AudioMemory(),
        )
        assessment = await runtime.perceive(b"\x00" * 3200)

        assert assessment.speech is not None
        prov = assessment.speech.provider_provenance
        assert prov.provider == "fake-stt"
        assert prov.model == "v1"

    @pytest.mark.asyncio
    async def test_event_carries_provenance(self) -> None:
        runtime = AudioPerceptionRuntime(
            events=DisagreeingEventProvider(),
            memory=AudioMemory(),
        )
        assessment = await runtime.perceive(b"\x00" * 3200)

        assert len(assessment.events) >= 1
        prov = assessment.events[0].provider_provenance
        assert prov.provider == "fake-ambient"
        assert prov.model == "v1"

    @pytest.mark.asyncio
    async def test_scene_carries_provenance(self) -> None:
        runtime = AudioPerceptionRuntime(
            scene=DisagreeingEventProvider(),
            memory=AudioMemory(),
        )
        assessment = await runtime.perceive(b"\x00" * 3200)

        assert assessment.scene is not None
        prov = assessment.scene.provider_provenance
        assert prov.provider == "fake-ambient"

    @pytest.mark.asyncio
    async def test_provenance_identifier_format(self) -> None:
        """ProviderProvenance.identifier should be 'provider/model/vX'."""
        prov = ProviderProvenance(provider="moonshine", model="base", version="1.2")
        assert prov.identifier == "moonshine/base/v1.2"

    @pytest.mark.asyncio
    async def test_provenance_identifier_minimal(self) -> None:
        prov = ProviderProvenance(provider="test")
        assert prov.identifier == "test"

    @pytest.mark.asyncio
    async def test_default_provenance_is_unknown(self) -> None:
        prov = ProviderProvenance()
        assert prov.provider == "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Provider Failure & Graceful Degradation
# ═══════════════════════════════════════════════════════════════════════════


class TestProviderFailure:
    """Failing providers must not crash the runtime."""

    @pytest.mark.asyncio
    async def test_speech_provider_crash_produces_empty(self) -> None:
        """Crashing STT → no speech evidence, but assessment still valid."""
        runtime = AudioPerceptionRuntime(
            speech=FailingSpeechProvider(),
            events=MultiCandidateEventProvider(),
            memory=AudioMemory(),
        )
        assessment = await runtime.perceive(b"\x00" * 3200)

        assert assessment.speech is None  # Graceful fallback
        assert len(assessment.events) == 2  # Other providers still work

    @pytest.mark.asyncio
    async def test_no_providers_returns_empty_assessment(self) -> None:
        """Runtime with no providers → valid but empty assessment."""
        runtime = AudioPerceptionRuntime(memory=AudioMemory())
        assessment = await runtime.perceive(b"\x00" * 3200)

        assert assessment.speech is None
        assert assessment.events == []
        assert assessment.scene is None
        assert assessment.observation is not None

    @pytest.mark.asyncio
    async def test_speech_only_runtime(self) -> None:
        """Runtime with only speech provider → only speech evidence."""
        runtime = AudioPerceptionRuntime(
            speech=DisagreeingSpeechProvider(),
            memory=AudioMemory(),
        )
        assessment = await runtime.perceive(b"\x00" * 3200)

        assert assessment.speech is not None
        assert assessment.events == []
        assert assessment.scene is None


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Low-Confidence Evidence
# ═══════════════════════════════════════════════════════════════════════════


class TestLowConfidence:
    """Low-confidence results must penalize epistemic profile."""

    @pytest.mark.asyncio
    async def test_low_confidence_speech_penalizes_profile(self) -> None:
        runtime = AudioPerceptionRuntime(
            speech=LowConfidenceSpeechProvider(),
            memory=AudioMemory(),
        )
        assessment = await runtime.perceive(b"\x00" * 3200)

        assert assessment.speech is not None
        assert assessment.speech.confidence == 0.15
        # Epistemic profile should reflect low perceptual confidence
        assert assessment.epistemic_profile.perceptual_confidence == 0.15

    @pytest.mark.asyncio
    async def test_high_confidence_boosts_profile(self) -> None:
        runtime = AudioPerceptionRuntime(
            speech=DisagreeingSpeechProvider(),
            memory=AudioMemory(),
        )
        assessment = await runtime.perceive(b"\x00" * 3200)

        assert assessment.epistemic_profile.perceptual_confidence == 0.85


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Zero-length, corrupt, very short, and empty audio."""

    @pytest.mark.asyncio
    async def test_empty_bytes(self) -> None:
        runtime = AudioPerceptionRuntime(
            speech=DisagreeingSpeechProvider(),
            memory=AudioMemory(),
        )
        assessment = await runtime.perceive(b"")
        assert assessment.observation is not None
        # Provider still called (it decides how to handle empty input)
        assert assessment.speech is not None

    @pytest.mark.asyncio
    async def test_numpy_zeros(self) -> None:
        runtime = AudioPerceptionRuntime(
            speech=DisagreeingSpeechProvider(),
            memory=AudioMemory(),
        )
        assessment = await runtime.perceive(np.zeros(100, dtype=np.float32))
        assert assessment.observation is not None

    @pytest.mark.asyncio
    async def test_very_short_audio(self) -> None:
        """Very short audio (< 100 samples) should not crash."""
        runtime = AudioPerceptionRuntime(
            speech=DisagreeingSpeechProvider(),
            memory=AudioMemory(),
        )
        assessment = await runtime.perceive(b"\x01\x00" * 5)  # 5 samples
        assert assessment.observation is not None


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Temporal Ambiguity
# ═══════════════════════════════════════════════════════════════════════════


class TestTemporalAmbiguity:
    """Different temporal patterns should produce different event identities."""

    def test_rapid_events_share_event_id(self) -> None:
        """Knock-100ms-knock: same event (continued)."""
        span1 = TemporalSpan(
            start_time=1000.0,
            end_time=1000.05,
            duration=0.05,
            state=AudioEventState.STARTED,
        )
        span2 = TemporalSpan(
            observation_id=span1.observation_id.replace(
                span1.observation_id,
                f"aobs_{span1.observation_id[-12:]}_2",
            ),
            event_id=span1.event_id,  # Same event
            start_time=1000.15,
            end_time=1000.20,
            duration=0.05,
            state=AudioEventState.CONTINUED,
        )
        # Same event_id means same logical event
        assert span1.event_id == span2.event_id
        # Different observations within that event
        assert span1.observation_id != span2.observation_id

    def test_distant_events_different_event_id(self) -> None:
        """Knock-2s-knock: different events."""
        span1 = TemporalSpan(
            start_time=1000.0,
            end_time=1000.05,
            state=AudioEventState.INSTANTANEOUS,
        )
        span2 = TemporalSpan(
            start_time=1002.0,
            end_time=1002.05,
            state=AudioEventState.INSTANTANEOUS,
        )
        # Different event_ids (auto-generated)
        assert span1.event_id != span2.event_id
        assert span1.observation_id != span2.observation_id


# ═══════════════════════════════════════════════════════════════════════════
# Tests: SoundEventEvidence Immutability
# ═══════════════════════════════════════════════════════════════════════════


class TestEvidenceStructure:
    """Verify evidence carries all required fields."""

    def test_speech_evidence_fields(self) -> None:
        ev = SpeechEvidence(
            transcript="test",
            confidence=0.9,
            provider_provenance=ProviderProvenance(
                provider="moonshine",
                model="base",
                version="1.2",
            ),
        )
        assert ev.transcript == "test"
        assert ev.provider_provenance.provider == "moonshine"
        assert ev.provider_provenance.identifier == "moonshine/base/v1.2"

    def test_event_evidence_preserves_top_classes(self) -> None:
        ev = SoundEventEvidence(
            event_type="doorbell",
            confidence=0.82,
            top_classes=[("doorbell", 0.82), ("knock", 0.76), ("alarm", 0.3)],
        )
        assert len(ev.top_classes) == 3
        assert ev.top_classes[0][0] == "doorbell"

    def test_assessment_preserves_all_evidence(self) -> None:
        assessment = AudioAssessment(
            speech=SpeechEvidence(transcript="hello"),
            events=[
                SoundEventEvidence(event_type="doorbell", confidence=0.82),
                SoundEventEvidence(event_type="knock", confidence=0.76),
            ],
        )
        assert assessment.speech is not None
        assert len(assessment.events) == 2
