"""Tests for Visual Perception Runtime — V2."""

from __future__ import annotations

import pytest

from hbllm.perception.providers.mock_provider import MockVisionProvider
from hbllm.perception.visual_memory import VisualMemory
from hbllm.perception.visual_perception_runtime import VisualPerceptionRuntime


class TestVisualPerceptionRuntime:
    @pytest.fixture
    def runtime(self) -> VisualPerceptionRuntime:
        provider = MockVisionProvider()
        memory = VisualMemory()
        return VisualPerceptionRuntime(provider, memory)

    @pytest.mark.asyncio
    async def test_perceive_empty_memory(self, runtime: VisualPerceptionRuntime) -> None:
        """Perceive with empty memory → no candidates, evidence produced."""
        assessment = await runtime.perceive(b"test_image")

        assert assessment.evidence.embedding.vector is not None
        assert len(assessment.evidence.embedding.vector) == 384
        assert assessment.evidence.image_hash != ""
        assert assessment.candidate_observations == []
        assert assessment.candidate_concepts == []
        assert assessment.ranking.best_score == 0.0
        assert assessment.proposed_label is None

    @pytest.mark.asyncio
    async def test_perceive_does_not_mutate_hcir(
        self,
        runtime: VisualPerceptionRuntime,
    ) -> None:
        """Runtime perceive() must NOT create HCIR nodes."""
        # Perceive does NOT add to memory or HCIR — it's read-only
        assessment = await runtime.perceive(b"test_image")
        assert assessment.evidence is not None
        # Memory should still be empty (runtime doesn't store)
        assert runtime.memory.observation_count == 0

    @pytest.mark.asyncio
    async def test_perceive_with_stored_observation(
        self,
        runtime: VisualPerceptionRuntime,
    ) -> None:
        """After manually storing an observation, perceive should find it."""
        # Store an observation directly in memory
        emb = await runtime.provider.encode(b"cup_image")
        await runtime.memory.store_observation(emb, "vcpt_1", "cup")

        # Now perceive the same image
        assessment = await runtime.perceive(b"cup_image")

        assert len(assessment.candidate_observations) == 1
        assert assessment.candidate_observations[0].label == "cup"
        assert assessment.candidate_observations[0].similarity > 0.99

    @pytest.mark.asyncio
    async def test_perceive_with_label(
        self,
        runtime: VisualPerceptionRuntime,
    ) -> None:
        """perceive_with_label sets label_provenance = 1.0."""
        assessment = await runtime.perceive_with_label(
            b"screwdriver_image",
            "screwdriver",
            "workshop",
        )
        assert assessment.proposed_label == "screwdriver"
        assert assessment.proposed_context == "workshop"
        assert assessment.epistemic_profile.label_provenance == 1.0

    @pytest.mark.asyncio
    async def test_epistemic_profile_populated(
        self,
        runtime: VisualPerceptionRuntime,
    ) -> None:
        """Epistemic dimensions should be populated independently."""
        assessment = await runtime.perceive(b"test")
        profile = assessment.epistemic_profile

        # Empty memory → perceptual_similarity = 0, evidence_strength = 0
        assert profile.perceptual_similarity == 0.0
        assert profile.evidence_strength == 0.0
        assert profile.source_reliability == 1.0
        assert profile.label_provenance == 0.0

    @pytest.mark.asyncio
    async def test_evidence_immutable_across_perceive(
        self,
        runtime: VisualPerceptionRuntime,
    ) -> None:
        """Evidence should be a separate object from assessment."""
        assessment = await runtime.perceive(b"test")
        evidence = assessment.evidence

        # Modifying assessment shouldn't affect evidence
        assessment.proposed_label = "something"
        assert not hasattr(evidence, "proposed_label")

    @pytest.mark.asyncio
    async def test_provenance_tracks_provider(
        self,
        runtime: VisualPerceptionRuntime,
    ) -> None:
        assessment = await runtime.perceive(b"test")
        prov = assessment.evidence.provenance
        assert "visual_perception:" in prov.created_by
        assert "mock" in prov.created_by
        assert prov.source_type == "observed"
