"""Tests for ContradictionEngine — proactive contradiction scanning."""

from __future__ import annotations

import pytest

from hbllm.brain.epistemics.contradiction_engine import ContradictionEngine
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    ContradictionNode,
    HCIREdge,
    HCIREdgeType,
)
from hbllm.hcir.types import BeliefConfidence


class TestScanForContradictions:
    """Test contradiction scanning."""

    @pytest.mark.asyncio
    async def test_empty_graph_no_contradictions(
        self,
        graph: CognitiveGraph,
    ) -> None:
        engine = ContradictionEngine(graph=graph)
        results = await engine.scan_for_contradictions()
        assert results == []

    @pytest.mark.asyncio
    async def test_explicit_contradiction_edge(
        self,
        graph: CognitiveGraph,
    ) -> None:
        b1 = BeliefNode(
            claim="X causes Y",
            belief_confidence=BeliefConfidence(evidence_quality=0.7),
        )
        b2 = BeliefNode(
            claim="X does NOT cause Y",
            belief_confidence=BeliefConfidence(evidence_quality=0.6),
        )
        graph.upsert_node(b1)
        graph.upsert_node(b2)
        graph.add_edge(
            HCIREdge(
                sources=[b1.id],
                targets=[b2.id],
                edge_type=HCIREdgeType.CONTRADICTS,
            )
        )

        engine = ContradictionEngine(graph=graph)
        results = await engine.scan_for_contradictions()
        assert len(results) >= 1


class TestScanForAnomalies:
    """Test anomaly scanning."""

    @pytest.mark.asyncio
    async def test_no_anomalies_empty(self, graph: CognitiveGraph) -> None:
        engine = ContradictionEngine(graph=graph)
        results = await engine.scan_for_anomalies()
        assert results == []


class TestAnalyzeContradiction:
    """Test contradiction analysis."""

    @pytest.mark.asyncio
    async def test_analyze_existing_contradiction(
        self,
        graph: CognitiveGraph,
    ) -> None:
        c = ContradictionNode(
            claim_a_id="claim_a",
            claim_b_id="claim_b",
            contradiction_type="logical",
        )
        graph.upsert_node(c)

        engine = ContradictionEngine(graph=graph)
        analysis = await engine.analyze_contradiction(c.id)
        assert analysis is not None

    @pytest.mark.asyncio
    async def test_analyze_nonexistent(self, graph: CognitiveGraph) -> None:
        engine = ContradictionEngine(graph=graph)
        analysis = await engine.analyze_contradiction("nonexistent")
        assert analysis is not None  # Should return a default/empty analysis


class TestDetectUnexpectedOutcomes:
    """Test unexpected outcome detection."""

    @pytest.mark.asyncio
    async def test_no_unexpected_empty(self, graph: CognitiveGraph) -> None:
        engine = ContradictionEngine(graph=graph)
        results = await engine.detect_unexpected_outcomes()
        assert results == []
