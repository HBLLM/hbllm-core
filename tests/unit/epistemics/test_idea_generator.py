"""Tests for IdeaGenerator — raw creative generation + memory filtering."""

from __future__ import annotations

import pytest

from hbllm.brain.epistemics.epistemic_memory import EpistemicMemory
from hbllm.brain.epistemics.idea_generator import IdeaGenerator
from hbllm.brain.epistemics.workspace import DiscoveryWorkspace
from hbllm.hcir.graph import CognitiveGraph, ContradictionNode, ObservationNode


class TestGenerateFromUnknown:
    """Test generate_from_unknown() template fallback."""

    @pytest.mark.asyncio
    async def test_basic_generation(
        self,
        graph: CognitiveGraph,
        workspace: DiscoveryWorkspace,
    ) -> None:
        prog = workspace.create_program("Test", "Why X?")
        obj = workspace.add_objective(prog.program_id, "Find")
        q_id = workspace.add_question(prog.program_id, obj, "Why X?", importance=0.8)

        gen = IdeaGenerator(graph=graph)
        ideas = await gen.generate_from_unknown(q_id)

        assert len(ideas) > 0
        assert all(hasattr(idea, "claim") for idea in ideas)
        assert all(hasattr(idea, "plausibility") for idea in ideas)

    @pytest.mark.asyncio
    async def test_nonexistent_unknown(self, graph: CognitiveGraph) -> None:
        gen = IdeaGenerator(graph=graph)
        ideas = await gen.generate_from_unknown("nonexistent")
        assert ideas == []


class TestGenerateFromContradiction:
    """Test generate_from_contradiction() template fallback."""

    @pytest.mark.asyncio
    async def test_basic_contradiction(self, graph: CognitiveGraph) -> None:
        node = ContradictionNode(
            claim_a_id="claim_a",
            claim_b_id="claim_b",
            contradiction_type="logical",
        )
        graph.upsert_node(node)

        gen = IdeaGenerator(graph=graph)
        ideas = await gen.generate_from_contradiction(node.id)

        assert len(ideas) == 2  # Template generates 2 ideas
        assert "hidden variable" in ideas[0].claim.lower()

    @pytest.mark.asyncio
    async def test_nonexistent_contradiction(self, graph: CognitiveGraph) -> None:
        gen = IdeaGenerator(graph=graph)
        ideas = await gen.generate_from_contradiction("nonexistent")
        assert ideas == []


class TestGenerateFromAnomaly:
    """Test generate_from_anomaly() template fallback."""

    @pytest.mark.asyncio
    async def test_basic_anomaly(self, graph: CognitiveGraph) -> None:
        node = ObservationNode(
            description="Unexpected temperature spike",
            tags=["anomaly", "temperature"],
        )
        graph.upsert_node(node)

        gen = IdeaGenerator(graph=graph)
        ideas = await gen.generate_from_anomaly(node.id)

        assert len(ideas) == 2  # Template generates 2 ideas


class TestMemoryFiltering:
    """Test _filter_known_failures() memory integration."""

    @pytest.mark.asyncio
    async def test_no_memory_passthrough(self, graph: CognitiveGraph) -> None:
        """Without memory, all ideas pass through."""
        gen = IdeaGenerator(graph=graph, memory=None)
        node = ContradictionNode(
            claim_a_id="a",
            claim_b_id="b",
            contradiction_type="logical",
        )
        graph.upsert_node(node)
        ideas = await gen.generate_from_contradiction(node.id)
        assert len(ideas) == 2

    @pytest.mark.asyncio
    async def test_memory_filters_known_failures(
        self,
        graph: CognitiveGraph,
        memory: EpistemicMemory,
    ) -> None:
        """Ideas matching past failures should be filtered out."""
        # Record a past failure with keywords matching template ideas
        await memory.record_hypothesis_outcome(
            "h_old",
            "falsified",
            "Proved wrong",
            claim="A hidden variable may explain the contradiction",
        )

        gen = IdeaGenerator(graph=graph, memory=memory)
        node = ContradictionNode(
            claim_a_id="a",
            claim_b_id="b",
            contradiction_type="logical",
        )
        graph.upsert_node(node)
        ideas = await gen.generate_from_contradiction(node.id)

        # At least one idea should be filtered (hidden variable template)
        assert len(ideas) < 2

    @pytest.mark.asyncio
    async def test_memory_empty_no_filtering(
        self,
        graph: CognitiveGraph,
        memory: EpistemicMemory,
    ) -> None:
        """Empty memory should not filter anything."""
        gen = IdeaGenerator(graph=graph, memory=memory)
        node = ContradictionNode(
            claim_a_id="a",
            claim_b_id="b",
            contradiction_type="logical",
        )
        graph.upsert_node(node)
        ideas = await gen.generate_from_contradiction(node.id)
        assert len(ideas) == 2
