"""Tests for CuriosityEngine — self-directed investigation."""

from __future__ import annotations

import pytest

from hbllm.brain.epistemics.curiosity_engine import CuriosityEngine
from hbllm.brain.epistemics.workspace import DiscoveryWorkspace
from hbllm.hcir.graph import CognitiveGraph, HypothesisNode


class TestPrioritizeInvestigations:
    """Test investigation prioritization."""

    @pytest.mark.asyncio
    async def test_empty_graph_no_signals(
        self,
        graph: CognitiveGraph,
    ) -> None:
        engine = CuriosityEngine(graph=graph)
        signals = await engine.prioritize_investigations()
        assert signals == []

    @pytest.mark.asyncio
    async def test_unknowns_generate_signals(
        self,
        graph: CognitiveGraph,
        workspace: DiscoveryWorkspace,
    ) -> None:
        prog = workspace.create_program("Curiosity Test", "Why X?")
        obj = workspace.add_objective(prog.program_id, "Find")
        workspace.add_question(
            prog.program_id,
            obj,
            "Why X?",
            importance=0.9,
        )

        engine = CuriosityEngine(graph=graph)
        signals = await engine.prioritize_investigations()
        assert len(signals) >= 1

    @pytest.mark.asyncio
    async def test_signals_sorted_by_score(
        self,
        graph: CognitiveGraph,
        workspace: DiscoveryWorkspace,
    ) -> None:
        prog = workspace.create_program("Sort Test", "Q")
        obj = workspace.add_objective(prog.program_id, "Find")
        workspace.add_question(prog.program_id, obj, "High?", importance=0.9)
        workspace.add_question(prog.program_id, obj, "Low?", importance=0.3)

        engine = CuriosityEngine(graph=graph)
        signals = await engine.prioritize_investigations()

        if len(signals) >= 2:
            # Higher importance should score higher
            assert signals[0].estimated_info_gain >= signals[1].estimated_info_gain


class TestEstimateValueOfKnowing:
    """Test value estimation."""

    @pytest.mark.asyncio
    async def test_estimate_for_unknown(
        self,
        graph: CognitiveGraph,
        workspace: DiscoveryWorkspace,
    ) -> None:
        prog = workspace.create_program("Value Test", "Q")
        obj = workspace.add_objective(prog.program_id, "Find")
        q_id = workspace.add_question(
            prog.program_id,
            obj,
            "Why X?",
            importance=0.8,
        )

        engine = CuriosityEngine(graph=graph)
        value = await engine.estimate_value_of_knowing(q_id)
        assert value >= 0.0


class TestGenerateSpontaneousUnknowns:
    """Test spontaneous unknown generation."""

    @pytest.mark.asyncio
    async def test_spontaneous_unknowns_with_hypotheses(
        self,
        graph: CognitiveGraph,
    ) -> None:
        # Add some untested hypotheses
        for i in range(3):
            h = HypothesisNode(claim=f"Hypothesis {i}")
            graph.upsert_node(h)

        engine = CuriosityEngine(graph=graph)
        unknowns = await engine.generate_spontaneous_unknowns()
        assert isinstance(unknowns, list)

    @pytest.mark.asyncio
    async def test_spontaneous_unknowns_empty_graph(
        self,
        graph: CognitiveGraph,
    ) -> None:
        engine = CuriosityEngine(graph=graph)
        unknowns = await engine.generate_spontaneous_unknowns()
        assert unknowns == []
