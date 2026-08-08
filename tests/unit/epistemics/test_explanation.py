"""Tests for ExplanationEngine — provenance chain generation."""

from __future__ import annotations

from typing import Any

import pytest

from hbllm.brain.epistemics.explanation import ExplanationEngine
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    EvidenceNode,
    HCIREdge,
    HCIREdgeType,
    ObservationNode,
)
from hbllm.hcir.types import BeliefConfidence, EvidenceStrength


class TestExplainBelief:
    """Test explain_belief() provenance chain."""

    @pytest.mark.asyncio
    async def test_chain_with_evidence(
        self,
        populated_graph: dict[str, Any],
    ) -> None:
        graph = populated_graph["graph"]
        engine = ExplanationEngine(graph=graph)

        chain = await engine.explain_belief(populated_graph["belief_id"])

        assert chain.belief_id == populated_graph["belief_id"]
        assert len(chain.steps) > 0

    @pytest.mark.asyncio
    async def test_chain_without_evidence(self, graph: CognitiveGraph) -> None:
        belief = BeliefNode(
            claim="Unsupported",
            belief_confidence=BeliefConfidence(evidence_quality=0.3),
        )
        graph.upsert_node(belief)

        engine = ExplanationEngine(graph=graph)
        chain = await engine.explain_belief(belief.id)

        assert chain.belief_id == belief.id
        assert len(chain.steps) == 0

    @pytest.mark.asyncio
    async def test_nonexistent_belief(self, graph: CognitiveGraph) -> None:
        engine = ExplanationEngine(graph=graph)
        chain = await engine.explain_belief("nonexistent")
        assert chain.steps == []


class TestExplainConfidence:
    """Test explain_confidence() breakdown."""

    @pytest.mark.asyncio
    async def test_confidence_breakdown(
        self,
        populated_graph: dict[str, Any],
    ) -> None:
        graph = populated_graph["graph"]
        engine = ExplanationEngine(graph=graph)

        explanation = await engine.explain_confidence(
            populated_graph["belief_id"],
        )
        assert explanation is not None
        assert isinstance(explanation, dict)


class TestTraceToObservations:
    """Test tracing belief back to raw observations."""

    @pytest.mark.asyncio
    async def test_trace_finds_observations(
        self,
        graph: CognitiveGraph,
    ) -> None:
        # Build: observation → evidence → belief
        obs = ObservationNode(
            description="Measured temperature spike",
            tags=["temperature"],
        )
        graph.upsert_node(obs)

        ev = EvidenceNode(
            evidence_type=EvidenceStrength.EXPERIMENTAL,
            methodology="Sensor data",
        )
        graph.upsert_node(ev)

        belief = BeliefNode(
            claim="Temperature affects X",
            belief_confidence=BeliefConfidence(evidence_quality=0.7),
        )
        graph.upsert_node(belief)

        graph.add_edge(
            HCIREdge(
                sources=[obs.id],
                targets=[ev.id],
                edge_type=HCIREdgeType.SUPPORTS,
            )
        )
        graph.add_edge(
            HCIREdge(
                sources=[ev.id],
                targets=[belief.id],
                edge_type=HCIREdgeType.SUPPORTS,
            )
        )

        engine = ExplanationEngine(graph=graph)
        observations = await engine.trace_to_observations(belief.id)

        assert len(observations) >= 1
