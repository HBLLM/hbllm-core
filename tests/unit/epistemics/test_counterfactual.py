"""Tests for CounterfactualReasoner — 'What if...' epistemic analysis."""

from __future__ import annotations

from typing import Any

import pytest

from hbllm.brain.epistemics.counterfactual import CounterfactualReasoner
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
)
from hbllm.hcir.types import BeliefConfidence


class TestWhatIfHypothesisWrong:
    """Test what_if_hypothesis_wrong()."""

    @pytest.mark.asyncio
    async def test_falsify_hypothesis(
        self,
        populated_graph: dict[str, Any],
    ) -> None:
        graph = populated_graph["graph"]
        cf = CounterfactualReasoner(graph=graph)

        result = await cf.what_if_hypothesis_wrong(populated_graph["hypothesis_id"])

        assert result.mutation_type == "falsify_hypothesis"
        assert len(result.affected_beliefs) == 1
        assert populated_graph["belief_id"] in result.affected_beliefs
        assert result.structural_impact > 0

    @pytest.mark.asyncio
    async def test_falsify_nonexistent(self, graph: CognitiveGraph) -> None:
        cf = CounterfactualReasoner(graph=graph)
        result = await cf.what_if_hypothesis_wrong("nonexistent")
        assert result.affected_beliefs == []


class TestWhatIfEvidenceRemoved:
    """Test what_if_evidence_removed()."""

    @pytest.mark.asyncio
    async def test_remove_evidence(
        self,
        populated_graph: dict[str, Any],
    ) -> None:
        graph = populated_graph["graph"]
        cf = CounterfactualReasoner(graph=graph)

        result = await cf.what_if_evidence_removed(populated_graph["evidence_ids"][0])

        assert result.mutation_type == "remove_evidence"
        assert len(result.affected_beliefs) >= 1
        assert len(result.confidence_deltas) >= 1
        # Confidence should decrease
        for delta in result.confidence_deltas.values():
            assert delta < 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, graph: CognitiveGraph) -> None:
        cf = CounterfactualReasoner(graph=graph)
        result = await cf.what_if_evidence_removed("nonexistent")
        assert result.affected_beliefs == []


class TestWhatIfEvidenceQuality:
    """Test what_if_evidence_quality()."""

    @pytest.mark.asyncio
    async def test_increase_quality(
        self,
        populated_graph: dict[str, Any],
    ) -> None:
        graph = populated_graph["graph"]
        cf = CounterfactualReasoner(graph=graph)

        result = await cf.what_if_evidence_quality(
            populated_graph["evidence_ids"][0],
            new_quality=1.0,
        )
        assert result.mutation_type == "change_evidence_quality"


class TestWhatIfNewEvidence:
    """Test what_if_new_evidence()."""

    @pytest.mark.asyncio
    async def test_supporting_evidence(
        self,
        populated_graph: dict[str, Any],
    ) -> None:
        graph = populated_graph["graph"]
        cf = CounterfactualReasoner(graph=graph)

        result = await cf.what_if_new_evidence(
            populated_graph["belief_id"],
            evidence_quality=0.9,
            direction="supporting",
        )

        assert populated_graph["belief_id"] in result.confidence_deltas
        assert result.confidence_deltas[populated_graph["belief_id"]] > 0

    @pytest.mark.asyncio
    async def test_contradicting_evidence(
        self,
        populated_graph: dict[str, Any],
    ) -> None:
        graph = populated_graph["graph"]
        cf = CounterfactualReasoner(graph=graph)

        result = await cf.what_if_new_evidence(
            populated_graph["belief_id"],
            evidence_quality=0.9,
            direction="contradicting",
        )

        assert result.confidence_deltas[populated_graph["belief_id"]] < 0


class TestSensitivityAnalysis:
    """Test sensitivity_analysis()."""

    @pytest.mark.asyncio
    async def test_sensitivity(
        self,
        populated_graph: dict[str, Any],
    ) -> None:
        graph = populated_graph["graph"]
        cf = CounterfactualReasoner(graph=graph)

        sensitivity = await cf.sensitivity_analysis(populated_graph["belief_id"])

        assert len(sensitivity) == 2  # Two evidence nodes
        # Values should be positive
        for impact in sensitivity.values():
            assert impact > 0

    @pytest.mark.asyncio
    async def test_sensitivity_no_evidence(self, graph: CognitiveGraph) -> None:
        belief = BeliefNode(
            claim="Unsupported belief",
            belief_confidence=BeliefConfidence(evidence_quality=0.5),
        )
        graph.upsert_node(belief)
        cf = CounterfactualReasoner(graph=graph)

        sensitivity = await cf.sensitivity_analysis(belief.id)
        assert sensitivity == {}
