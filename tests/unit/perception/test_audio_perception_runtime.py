"""Tests for Audio Perception Runtime — Wave A3.

Tests the evidence-only runtime and audio memory.
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.perception.audio_memory import AudioMemory
from hbllm.perception.audio_perception_runtime import AudioPerceptionRuntime
from hbllm.perception.providers.audio_evidence import (
    AcousticObservation,
    AcousticSceneEvidence,
    AudioAssessment,
    SoundEventEvidence,
    SpeechEvidence,
)
from hbllm.perception.providers.mock_audio_provider import MockAudioProvider

# ═══════════════════════════════════════════════════════════════════════════
# Audio Memory
# ═══════════════════════════════════════════════════════════════════════════


class TestAudioMemory:
    """Tests for AudioMemory — observation-first index."""

    def _make_embedding(self, seed: int = 0, dim: int = 8):
        from hbllm.perception.providers.audio_types import AudioEmbedding

        rng = np.random.RandomState(seed)  # noqa: NPY002
        vec = rng.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return AudioEmbedding(
            vector=vec.tolist(),
            model_id="test",
            space_id="test",
            dimensions=dim,
        )

    def test_store_and_search(self) -> None:
        memory = AudioMemory()
        emb = self._make_embedding(seed=42)
        memory.store_observation("obs1", emb, concept_label="doorbell")

        results = memory.search_observations(emb, top_k=5)
        assert len(results) == 1
        assert results[0].observation_id == "obs1"
        assert results[0].similarity > 0.99

    def test_search_sorted(self) -> None:
        memory = AudioMemory()
        for i in range(5):
            memory.store_observation(f"obs{i}", self._make_embedding(seed=i))

        query = self._make_embedding(seed=2)
        results = memory.search_observations(query, top_k=3)
        assert len(results) == 3
        # First result should be the exact match
        assert results[0].observation_id == "obs2"
        assert results[0].similarity > results[1].similarity

    def test_concept_filter(self) -> None:
        memory = AudioMemory()
        memory.store_observation("obs1", self._make_embedding(0), "doorbell")
        memory.store_observation("obs2", self._make_embedding(1), "alarm")

        results = memory.search_observations(
            self._make_embedding(0),
            concept_filter="doorbell",
        )
        assert len(results) == 1
        assert results[0].concept_label == "doorbell"

    def test_prototype_store_and_search(self) -> None:
        memory = AudioMemory()
        emb = self._make_embedding(seed=10)
        memory.store_prototype("siren", emb)

        results = memory.search_prototypes(emb)
        assert len(results) == 1
        assert results[0].concept_label == "siren"
        assert results[0].similarity > 0.99

    def test_prototype_running_average(self) -> None:
        memory = AudioMemory()
        e1 = self._make_embedding(seed=10)
        e2 = self._make_embedding(seed=11)
        memory.store_prototype("test", e1)
        memory.update_prototype("test", e2)

        assert memory._prototypes["test"].observation_count == 2

    def test_exemplar_diversity(self) -> None:
        memory = AudioMemory()
        emb = self._make_embedding(seed=0)
        memory.store_prototype("test", emb)
        memory._prototypes["test"].exemplar_refs = []

        # Add first exemplar
        obs_emb = self._make_embedding(seed=0)
        memory.store_observation("obs0", obs_emb)
        added = memory.add_exemplar("test", "obs0", obs_emb)
        assert added

        # Try adding identical — too similar
        obs_emb2 = self._make_embedding(seed=0)
        memory.store_observation("obs1", obs_emb2)
        added2 = memory.add_exemplar("test", "obs1", obs_emb2)
        assert not added2

    def test_observation_count(self) -> None:
        memory = AudioMemory()
        assert memory.observation_count == 0
        memory.store_observation("obs1", self._make_embedding(0))
        assert memory.observation_count == 1

    def test_concept_count(self) -> None:
        memory = AudioMemory()
        assert memory.concept_count == 0
        memory.store_prototype("doorbell", self._make_embedding(0))
        assert memory.concept_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# Audio Perception Runtime
# ═══════════════════════════════════════════════════════════════════════════


class TestAudioPerceptionRuntime:
    """Tests for AudioPerceptionRuntime — evidence-only."""

    @pytest.fixture
    def provider(self) -> MockAudioProvider:
        return MockAudioProvider()

    @pytest.fixture
    def runtime(self, provider: MockAudioProvider) -> AudioPerceptionRuntime:
        return AudioPerceptionRuntime(
            speech=provider,
            events=provider,
            scene=provider,
            speaker=provider,
        )

    @pytest.mark.asyncio
    async def test_perceive_returns_assessment(
        self,
        runtime: AudioPerceptionRuntime,
    ) -> None:
        assessment = await runtime.perceive(b"test audio")
        assert isinstance(assessment, AudioAssessment)
        assert isinstance(assessment.observation, AcousticObservation)

    @pytest.mark.asyncio
    async def test_speech_evidence_populated(
        self,
        runtime: AudioPerceptionRuntime,
    ) -> None:
        assessment = await runtime.perceive(b"speech test")
        assert assessment.speech is not None
        assert isinstance(assessment.speech, SpeechEvidence)
        assert assessment.speech.transcript != ""

    @pytest.mark.asyncio
    async def test_event_evidence_populated(
        self,
        runtime: AudioPerceptionRuntime,
    ) -> None:
        assessment = await runtime.perceive(b"event test")
        assert len(assessment.events) >= 1
        assert isinstance(assessment.events[0], SoundEventEvidence)

    @pytest.mark.asyncio
    async def test_scene_evidence_populated(
        self,
        runtime: AudioPerceptionRuntime,
    ) -> None:
        assessment = await runtime.perceive(b"scene test")
        assert assessment.scene is not None
        assert isinstance(assessment.scene, AcousticSceneEvidence)

    @pytest.mark.asyncio
    async def test_shared_observation(
        self,
        runtime: AudioPerceptionRuntime,
    ) -> None:
        """All evidence types share the same AcousticObservation."""
        assessment = await runtime.perceive(b"shared obs test")
        obs = assessment.observation
        if assessment.speech:
            assert assessment.speech.observation is obs
        for event in assessment.events:
            assert event.observation is obs
        if assessment.scene:
            assert assessment.scene.observation is obs

    @pytest.mark.asyncio
    async def test_epistemic_profile(
        self,
        runtime: AudioPerceptionRuntime,
    ) -> None:
        assessment = await runtime.perceive(b"profile test")
        profile = assessment.epistemic_profile
        assert profile.perceptual_confidence > 0
        assert profile.temporal_confidence > 0

    @pytest.mark.asyncio
    async def test_perceive_with_label(
        self,
        runtime: AudioPerceptionRuntime,
    ) -> None:
        assessment = await runtime.perceive(b"labeled", label="doorbell")
        assert assessment.proposed_label == "doorbell"

    @pytest.mark.asyncio
    async def test_perceive_speech_only(
        self,
        runtime: AudioPerceptionRuntime,
    ) -> None:
        speech = await runtime.perceive_speech(b"speech only")
        assert isinstance(speech, SpeechEvidence)
        assert speech.transcript != ""

    @pytest.mark.asyncio
    async def test_perceive_speech_no_provider(self) -> None:
        runtime = AudioPerceptionRuntime()
        with pytest.raises(RuntimeError, match="No speech provider"):
            await runtime.perceive_speech(b"no provider")

    @pytest.mark.asyncio
    async def test_partial_providers(self) -> None:
        """Runtime works with only some providers configured."""
        provider = MockAudioProvider()
        runtime = AudioPerceptionRuntime(speech=provider)
        assessment = await runtime.perceive(b"partial")
        assert assessment.speech is not None
        assert assessment.events == []
        assert assessment.scene is None

    @pytest.mark.asyncio
    async def test_does_not_mutate_memory(
        self,
        runtime: AudioPerceptionRuntime,
    ) -> None:
        """Perceive must not store anything in memory."""
        memory = runtime._memory
        count_before = memory.observation_count
        await runtime.perceive(b"no mutation")
        assert memory.observation_count == count_before
