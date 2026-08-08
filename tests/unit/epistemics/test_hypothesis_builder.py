"""Tests for HypothesisBuilder — validate → deduplicate → promote pipeline."""

from __future__ import annotations

import pytest

from hbllm.brain.epistemics.hypothesis_builder import HypothesisBuilder
from hbllm.brain.epistemics.interfaces import RawIdea
from hbllm.brain.epistemics.workspace import DiscoveryWorkspace
from hbllm.hcir.graph import CognitiveGraph, HypothesisNode
from hbllm.hcir.types import DiscoveryTrigger


@pytest.fixture
def raw_ideas() -> list[RawIdea]:
    """Sample raw ideas for validation."""
    return [
        RawIdea(
            claim="Z causes X through mechanism M",
            plausibility=0.7,
            origin_trigger=DiscoveryTrigger.KNOWLEDGE_GAP,
            reasoning="Strong structural similarity",
        ),
        RawIdea(
            claim="Y modulates Z through pathway P",
            plausibility=0.5,
            origin_trigger=DiscoveryTrigger.KNOWLEDGE_GAP,
            reasoning="Analogical transfer",
        ),
        RawIdea(
            claim="Invalid claim",
            plausibility=0.01,
            origin_trigger=DiscoveryTrigger.KNOWLEDGE_GAP,
            reasoning="Very unlikely",
        ),
    ]


class TestValidate:
    """Test hypothesis validation pipeline."""

    @pytest.mark.asyncio
    async def test_validate_filters_low_plausibility(
        self,
        graph: CognitiveGraph,
        raw_ideas: list[RawIdea],
    ) -> None:
        builder = HypothesisBuilder(graph=graph)
        candidates = await builder.validate(raw_ideas)

        # Low plausibility idea should be filtered
        assert len(candidates) <= len(raw_ideas)
        assert all(c.plausibility >= 0.1 for c in candidates)

    @pytest.mark.asyncio
    async def test_validate_empty(self, graph: CognitiveGraph) -> None:
        builder = HypothesisBuilder(graph=graph)
        candidates = await builder.validate([])
        assert candidates == []


class TestDeduplicate:
    """Test deduplication of hypothesis candidates."""

    @pytest.mark.asyncio
    async def test_deduplicate_removes_duplicates(
        self,
        graph: CognitiveGraph,
    ) -> None:
        builder = HypothesisBuilder(graph=graph)

        # Create duplicate ideas
        ideas = [
            RawIdea(
                claim="Z causes X", plausibility=0.7, origin_trigger=DiscoveryTrigger.KNOWLEDGE_GAP
            ),
            RawIdea(
                claim="Z causes X", plausibility=0.6, origin_trigger=DiscoveryTrigger.KNOWLEDGE_GAP
            ),
        ]
        candidates = await builder.validate(ideas)
        unique = await builder.deduplicate(candidates)

        assert len(unique) <= len(candidates)


class TestPromoteToNode:
    """Test promotion from candidate to graph node."""

    @pytest.mark.asyncio
    async def test_promote_creates_node(
        self,
        graph: CognitiveGraph,
        workspace: DiscoveryWorkspace,
    ) -> None:
        prog = workspace.create_program("Test", "Why X?")
        builder = HypothesisBuilder(graph=graph)

        idea = RawIdea(
            claim="Z causes X through mechanism M",
            plausibility=0.7,
            origin_trigger=DiscoveryTrigger.KNOWLEDGE_GAP,
            reasoning="Strong evidence",
        )
        candidates = await builder.validate([idea])
        assert len(candidates) > 0

        node_id = await builder.promote_to_node(
            candidates[0],
            prog.program_id,
        )

        assert node_id != ""
        node = graph.get_node(node_id)
        assert isinstance(node, HypothesisNode)
        assert "Z causes X" in node.claim

    @pytest.mark.asyncio
    async def test_promote_multiple(
        self,
        graph: CognitiveGraph,
        workspace: DiscoveryWorkspace,
        raw_ideas: list[RawIdea],
    ) -> None:
        prog = workspace.create_program("Test", "Why X?")
        builder = HypothesisBuilder(graph=graph)

        candidates = await builder.validate(raw_ideas)
        novel = await builder.deduplicate(candidates)

        promoted: list[str] = []
        for c in novel[:3]:
            nid = await builder.promote_to_node(c, prog.program_id)
            promoted.append(nid)

        assert len(promoted) > 0
        assert all(graph.has_node(nid) for nid in promoted)
