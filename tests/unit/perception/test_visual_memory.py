"""Tests for Visual Memory — V2."""

from __future__ import annotations

import pytest

from hbllm.perception.providers.policy import RecognitionPolicy
from hbllm.perception.providers.types import VisualEmbedding
from hbllm.perception.visual_memory import VisualMemory


def _make_embedding(
    vector: list[float],
    space_id: str = "test-space",
    model_id: str = "test-model",
) -> VisualEmbedding:
    return VisualEmbedding(
        vector=vector,
        model_id=model_id,
        space_id=space_id,
        embedding_type="semantic",
        dimensions=len(vector),
    )


class TestVisualMemory:
    @pytest.fixture
    def memory(self) -> VisualMemory:
        return VisualMemory()

    @pytest.mark.asyncio
    async def test_store_and_search_observation(self, memory: VisualMemory) -> None:
        emb = _make_embedding([1.0, 0.0, 0.0])
        ref = await memory.store_observation(emb, concept_node_id="vcpt_1", label="cup")
        assert ref.startswith("vobs_")
        assert memory.observation_count == 1

        matches = await memory.search_observations(emb, top_k=5)
        assert len(matches) == 1
        assert matches[0].similarity > 0.99
        assert matches[0].concept_node_id == "vcpt_1"
        assert matches[0].label == "cup"

    @pytest.mark.asyncio
    async def test_search_returns_sorted(self, memory: VisualMemory) -> None:
        await memory.store_observation(
            _make_embedding([1.0, 0.0, 0.0]),
            "vcpt_1",
            "cup",
        )
        await memory.store_observation(
            _make_embedding([0.9, 0.1, 0.0]),
            "vcpt_1",
            "cup",
        )
        await memory.store_observation(
            _make_embedding([0.0, 1.0, 0.0]),
            "vcpt_2",
            "bottle",
        )

        query = _make_embedding([0.95, 0.05, 0.0])
        matches = await memory.search_observations(query, top_k=10)
        # The two cup embeddings should be most similar
        assert matches[0].label == "cup"
        assert matches[1].label == "cup"
        assert matches[0].similarity >= matches[1].similarity
        assert matches[2].label == "bottle"

    @pytest.mark.asyncio
    async def test_cross_space_filter(self, memory: VisualMemory) -> None:
        await memory.store_observation(
            _make_embedding([1.0, 0.0, 0.0], space_id="space-a"),
            "vcpt_1",
            "cup",
        )
        query = _make_embedding([1.0, 0.0, 0.0], space_id="space-b")
        matches = await memory.search_observations(query, top_k=10)
        assert len(matches) == 0  # Different space → filtered out

    @pytest.mark.asyncio
    async def test_store_and_search_prototype(self, memory: VisualMemory) -> None:
        emb = _make_embedding([1.0, 0.0, 0.0])
        ref = await memory.store_prototype("vcpt_1", [1.0, 0.0, 0.0], emb)
        assert ref.startswith("vproto_")
        assert memory.prototype_count == 1

        matches = await memory.search_prototypes(emb, top_k=5)
        assert len(matches) == 1
        assert matches[0].concept_node_id == "vcpt_1"

    @pytest.mark.asyncio
    async def test_derive_concept_candidates(self, memory: VisualMemory) -> None:
        await memory.store_observation(
            _make_embedding([1.0, 0.0, 0.0]),
            "vcpt_1",
            "cup",
        )
        await memory.store_observation(
            _make_embedding([0.95, 0.05, 0.0]),
            "vcpt_1",
            "cup",
        )
        await memory.store_observation(
            _make_embedding([0.0, 1.0, 0.0]),
            "vcpt_2",
            "bottle",
        )

        query = _make_embedding([0.98, 0.02, 0.0])
        matches = await memory.search_observations(query, top_k=10)
        candidates, ranking = memory.derive_concept_candidates(matches)

        assert len(candidates) == 2
        assert candidates[0].label == "cup"  # Higher similarity
        assert candidates[0].matching_observations == 2
        assert ranking.best_score > ranking.second_score
        assert ranking.margin > 0

    @pytest.mark.asyncio
    async def test_derive_empty(self, memory: VisualMemory) -> None:
        candidates, ranking = memory.derive_concept_candidates([])
        assert candidates == []
        assert ranking.best_score == 0.0

    @pytest.mark.asyncio
    async def test_derive_unassigned_observations_ignored(
        self,
        memory: VisualMemory,
    ) -> None:
        await memory.store_observation(
            _make_embedding([1.0, 0.0, 0.0]),
            None,
            "unknown",
        )
        query = _make_embedding([1.0, 0.0, 0.0])
        matches = await memory.search_observations(query, top_k=10)
        candidates, _ = memory.derive_concept_candidates(matches)
        assert len(candidates) == 0

    @pytest.mark.asyncio
    async def test_exemplar_diversity(self, memory: VisualMemory) -> None:
        policy = RecognitionPolicy(
            exemplar_diversity_threshold=0.99,
            exemplar_limit=5,
        )
        emb1 = _make_embedding([1.0, 0.0, 0.0])
        await memory.store_observation(emb1, "vcpt_1", "cup")

        # Near-duplicate should be rejected
        emb2 = _make_embedding([1.0, 0.0, 0.0])
        ref = await memory.add_exemplar("vcpt_1", emb2, policy)
        assert ref is None

        # Sufficiently different should be accepted
        emb3 = _make_embedding([0.7, 0.7, 0.0])
        ref = await memory.add_exemplar("vcpt_1", emb3, policy)
        assert ref is not None

    @pytest.mark.asyncio
    async def test_exemplar_limit(self, memory: VisualMemory) -> None:
        policy = RecognitionPolicy(exemplar_limit=2, exemplar_diversity_threshold=0.5)
        await memory.store_observation(
            _make_embedding([1.0, 0.0, 0.0]),
            "vcpt_1",
            "cup",
        )
        await memory.store_observation(
            _make_embedding([0.0, 1.0, 0.0]),
            "vcpt_1",
            "cup",
        )

        # Third should be rejected (limit = 2)
        ref = await memory.add_exemplar(
            "vcpt_1",
            _make_embedding([0.0, 0.0, 1.0]),
            policy,
        )
        assert ref is None

    @pytest.mark.asyncio
    async def test_update_prototype(self, memory: VisualMemory) -> None:
        emb = _make_embedding([1.0, 0.0, 0.0])
        await memory.store_prototype("vcpt_1", [1.0, 0.0, 0.0], emb)

        new_emb = _make_embedding([0.0, 1.0, 0.0])
        await memory.update_prototype("vcpt_1", new_emb, count=2)

        # Prototype should now be shifted toward [0, 1, 0]
        matches = await memory.search_prototypes(new_emb, top_k=1)
        assert len(matches) == 1
        # Similarity should be higher than pure [1,0,0] would give
        assert matches[0].similarity > 0.5
