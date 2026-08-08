"""Tests for EvidenceEvaluator — evidence quality scoring."""

from __future__ import annotations

import pytest

from hbllm.brain.epistemics.evidence_evaluator import EvidenceEvaluator
from hbllm.hcir.graph import CognitiveGraph, EvidenceNode
from hbllm.hcir.types import EvidenceStrength


class TestEvaluateEvidence:
    """Test evidence evaluation pipeline."""

    @pytest.mark.asyncio
    async def test_evaluate_experimental(self, graph: CognitiveGraph) -> None:
        ev = EvidenceNode(
            evidence_type=EvidenceStrength.EXPERIMENTAL,
            methodology="RCT n=200",
            sample_size=200,
            reproducible=True,
        )
        graph.upsert_node(ev)

        evaluator = EvidenceEvaluator(graph=graph)
        assessment = await evaluator.evaluate(ev.id)

        assert assessment is not None
        assert assessment.quality_score > 0.0
        assert 0.0 <= assessment.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_observational(self, graph: CognitiveGraph) -> None:
        ev = EvidenceNode(
            evidence_type=EvidenceStrength.OBSERVATIONAL,
            methodology="Case study",
            sample_size=5,
            reproducible=False,
        )
        graph.upsert_node(ev)

        evaluator = EvidenceEvaluator(graph=graph)
        assessment = await evaluator.evaluate(ev.id)

        assert assessment is not None
        # Observational should score lower than experimental
        assert assessment.quality_score <= 0.7

    @pytest.mark.asyncio
    async def test_evaluate_nonexistent(self, graph: CognitiveGraph) -> None:
        evaluator = EvidenceEvaluator(graph=graph)
        assessment = await evaluator.evaluate("nonexistent")
        assert assessment is not None
        # Nonexistent nodes get default quality score
        assert assessment.quality_score == 0.5


class TestBiasDetection:
    """Test evidence bias detection."""

    @pytest.mark.asyncio
    async def test_bias_flags(self, graph: CognitiveGraph) -> None:
        ev = EvidenceNode(
            evidence_type=EvidenceStrength.ANECDOTAL,
            methodology="Single case report",
            sample_size=1,
            reproducible=False,
        )
        graph.upsert_node(ev)

        evaluator = EvidenceEvaluator(graph=graph)
        assessment = await evaluator.evaluate(ev.id)

        assert isinstance(assessment.bias_flags, list)
