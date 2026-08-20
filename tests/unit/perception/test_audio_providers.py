"""Tests for Audio Providers — Wave A2.

Tests the MockAudioProvider and protocol compliance.
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.perception.providers.audio_base import (
    AcousticEventProvider,
    AcousticSceneProvider,
    SpeakerProvider,
    SpeechProvider,
)
from hbllm.perception.providers.mock_audio_provider import MockAudioProvider


@pytest.fixture
def provider() -> MockAudioProvider:
    return MockAudioProvider()


# ═══════════════════════════════════════════════════════════════════════════
# Protocol Compliance
# ═══════════════════════════════════════════════════════════════════════════


class TestProtocolCompliance:
    """Verify MockAudioProvider satisfies all audio protocols."""

    def test_is_speech_provider(self, provider: MockAudioProvider) -> None:
        assert isinstance(provider, SpeechProvider)

    def test_is_event_provider(self, provider: MockAudioProvider) -> None:
        assert isinstance(provider, AcousticEventProvider)

    def test_is_scene_provider(self, provider: MockAudioProvider) -> None:
        assert isinstance(provider, AcousticSceneProvider)

    def test_is_speaker_provider(self, provider: MockAudioProvider) -> None:
        assert isinstance(provider, SpeakerProvider)

    def test_provider_id(self, provider: MockAudioProvider) -> None:
        assert provider.provider_id == "mock-audio-v1"

    def test_modality(self, provider: MockAudioProvider) -> None:
        assert provider.modality == "audio"

    def test_sample_rate(self, provider: MockAudioProvider) -> None:
        assert provider.sample_rate == 16000


# ═══════════════════════════════════════════════════════════════════════════
# Speech Provider
# ═══════════════════════════════════════════════════════════════════════════


class TestMockSpeechProvider:
    """Tests for MockAudioProvider speech transcription."""

    @pytest.mark.asyncio
    async def test_transcribe_returns_speech_result(
        self, provider: MockAudioProvider,
    ) -> None:
        result = await provider.transcribe(b"audio data")
        assert result.transcript != ""
        assert result.language == "en"
        assert 0.0 < result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_determinism_same_input(
        self, provider: MockAudioProvider,
    ) -> None:
        r1 = await provider.transcribe(b"same audio")
        r2 = await provider.transcribe(b"same audio")
        assert r1.transcript == r2.transcript
        assert r1.confidence == r2.confidence

    @pytest.mark.asyncio
    async def test_different_inputs_different_results(
        self, provider: MockAudioProvider,
    ) -> None:
        r1 = await provider.transcribe(b"audio one")
        r2 = await provider.transcribe(b"audio two")
        # Very unlikely same transcript from different inputs
        assert r1.transcript != r2.transcript or r1.confidence != r2.confidence

    @pytest.mark.asyncio
    async def test_speaker_ref_present(
        self, provider: MockAudioProvider,
    ) -> None:
        result = await provider.transcribe(b"with speaker")
        assert result.speaker is not None
        assert result.speaker.speaker_id is not None
        assert result.speaker.embedding_ref != ""

    @pytest.mark.asyncio
    async def test_paralinguistic_present(
        self, provider: MockAudioProvider,
    ) -> None:
        result = await provider.transcribe(b"emotional")
        assert result.paralinguistic is not None
        assert result.paralinguistic.tone in ("neutral", "urgent", "calm", "excited")

    @pytest.mark.asyncio
    async def test_temporal_span(
        self, provider: MockAudioProvider,
    ) -> None:
        result = await provider.transcribe(b"temporal test")
        assert result.temporal.start_time > 0
        assert result.temporal.end_time >= result.temporal.start_time
        assert result.temporal.duration > 0

    @pytest.mark.asyncio
    async def test_streaming_transcribe(
        self, provider: MockAudioProvider,
    ) -> None:
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        result = await provider.transcribe_streaming(chunks)
        assert result.transcript != ""

    @pytest.mark.asyncio
    async def test_string_path_input(
        self, provider: MockAudioProvider,
    ) -> None:
        result = await provider.transcribe("/path/to/audio.wav")
        assert result.transcript != ""

    @pytest.mark.asyncio
    async def test_numpy_input(
        self, provider: MockAudioProvider,
    ) -> None:
        audio = np.random.randn(16000).astype(np.float32)
        result = await provider.transcribe(audio)
        assert result.transcript != ""


# ═══════════════════════════════════════════════════════════════════════════
# Acoustic Event Provider
# ═══════════════════════════════════════════════════════════════════════════


class TestMockEventProvider:
    """Tests for MockAudioProvider acoustic event classification."""

    @pytest.mark.asyncio
    async def test_classify_returns_list(
        self, provider: MockAudioProvider,
    ) -> None:
        events = await provider.classify(b"event audio")
        assert isinstance(events, list)
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_event_has_type_and_confidence(
        self, provider: MockAudioProvider,
    ) -> None:
        events = await provider.classify(b"event audio")
        event = events[0]
        assert event.event_type != ""
        assert 0.0 < event.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_event_has_top_classes(
        self, provider: MockAudioProvider,
    ) -> None:
        events = await provider.classify(b"classify me")
        assert len(events[0].top_classes) >= 1

    @pytest.mark.asyncio
    async def test_determinism(
        self, provider: MockAudioProvider,
    ) -> None:
        e1 = await provider.classify(b"same audio")
        e2 = await provider.classify(b"same audio")
        assert e1[0].event_type == e2[0].event_type


# ═══════════════════════════════════════════════════════════════════════════
# Acoustic Scene Provider
# ═══════════════════════════════════════════════════════════════════════════


class TestMockSceneProvider:
    """Tests for MockAudioProvider scene analysis."""

    @pytest.mark.asyncio
    async def test_scene_result(
        self, provider: MockAudioProvider,
    ) -> None:
        scene = await provider.analyze_scene(b"scene audio")
        assert isinstance(scene.indoor, bool)
        assert isinstance(scene.speech_present, bool)
        assert 0.0 <= scene.noise_level <= 1.0

    @pytest.mark.asyncio
    async def test_scene_tags(
        self, provider: MockAudioProvider,
    ) -> None:
        scene = await provider.analyze_scene(b"tagged scene")
        assert len(scene.scene_tags) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Speaker Provider
# ═══════════════════════════════════════════════════════════════════════════


class TestMockSpeakerProvider:
    """Tests for MockAudioProvider speaker identification."""

    @pytest.mark.asyncio
    async def test_identify_returns_structured(
        self, provider: MockAudioProvider,
    ) -> None:
        speaker = await provider.identify(b"voice sample")
        assert speaker.speaker_id is not None
        assert speaker.embedding_ref != ""
        assert speaker.confidence > 0

    @pytest.mark.asyncio
    async def test_enroll_succeeds(
        self, provider: MockAudioProvider,
    ) -> None:
        result = await provider.enroll("alice", b"enrollment audio")
        assert result is True

    @pytest.mark.asyncio
    async def test_voice_characteristics(
        self, provider: MockAudioProvider,
    ) -> None:
        speaker = await provider.identify(b"voice features")
        assert "pitch_hz" in speaker.voice_characteristics


# ═══════════════════════════════════════════════════════════════════════════
# Embedding
# ═══════════════════════════════════════════════════════════════════════════


class TestMockEmbedding:
    """Tests for MockAudioProvider embedding generation."""

    def test_embedding_dimensions(self, provider: MockAudioProvider) -> None:
        emb = provider.get_embedding(b"test audio")
        assert len(emb.vector) == 128
        assert emb.dimensions == 128

    def test_embedding_l2_normalized(self, provider: MockAudioProvider) -> None:
        emb = provider.get_embedding(b"normalized")
        vec = np.array(emb.vector)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_determinism(self, provider: MockAudioProvider) -> None:
        e1 = provider.get_embedding(b"same")
        e2 = provider.get_embedding(b"same")
        assert e1.vector == e2.vector

    def test_different_inputs(self, provider: MockAudioProvider) -> None:
        e1 = provider.get_embedding(b"audio one")
        e2 = provider.get_embedding(b"audio two")
        assert e1.vector != e2.vector

    def test_custom_dimensions(self) -> None:
        provider = MockAudioProvider(embedding_dim=64)
        emb = provider.get_embedding(b"custom")
        assert len(emb.vector) == 64

    def test_compatibility(self, provider: MockAudioProvider) -> None:
        assert provider.is_compatible("mock-audio-v1")
        assert not provider.is_compatible("yamnet-v1")
