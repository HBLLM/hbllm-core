"""Tests for DiscoveryBeliefManager — Bayesian belief revision."""

from __future__ import annotations

import pytest

from hbllm.brain.epistemics.belief_manager import BayesianConfig, DiscoveryBeliefManager
from hbllm.brain.epistemics.interfaces import PredictionOutcome
from hbllm.hcir.graph import BeliefNode, CognitiveGraph, EvidenceNode
from hbllm.hcir.types import BeliefConfidence, EvidenceStrength, FalsificationStatus


@pytest.fixture
def belief_graph(graph: CognitiveGraph) -> tuple[CognitiveGraph, str, str]:
    """Graph with one belief and one evidence node."""
    belief = BeliefNode(
        claim="X causes Y",
        belief_confidence=BeliefConfidence(evidence_quality=0.6),
    )
    belief.uncertainty.confidence = 0.5
    graph.upsert_node(belief)

    evidence = EvidenceNode(
        evidence_type=EvidenceStrength.EXPERIMENTAL,
        methodology="RCT",
        sample_size=100,
    )
    graph.upsert_node(evidence)

    return graph, belief.id, evidence.id


class TestReviseBelief:
    """Test evidence-based revision."""

    @pytest.mark.asyncio
    async def test_supporting_evidence_increases_confidence(
        self,
        belief_graph: tuple[CognitiveGraph, str, str],
    ) -> None:
        graph, belief_id, evidence_id = belief_graph
        mgr = DiscoveryBeliefManager(graph=graph)

        revision = await mgr.revise_belief(
            belief_id,
            evidence_id,
            direction="supporting",
            evidence_strength="experimental",
        )

        assert revision.new_confidence > revision.old_confidence
        assert "supporting" in revision.reason

    @pytest.mark.asyncio
    async def test_contradicting_evidence_decreases_confidence(
        self,
        belief_graph: tuple[CognitiveGraph, str, str],
    ) -> None:
        graph, belief_id, evidence_id = belief_graph
        mgr = DiscoveryBeliefManager(graph=graph)

        revision = await mgr.revise_belief(
            belief_id,
            evidence_id,
            direction="contradicting",
            evidence_strength="experimental",
        )

        assert revision.new_confidence < revision.old_confidence

    @pytest.mark.asyncio
    async def test_unknown_direction_returns_error(
        self,
        belief_graph: tuple[CognitiveGraph, str, str],
    ) -> None:
        graph, belief_id, evidence_id = belief_graph
        mgr = DiscoveryBeliefManager(graph=graph)

        revision = await mgr.revise_belief(
            belief_id,
            evidence_id,
            direction="neutral",
        )
        assert "Unknown direction" in revision.reason

    @pytest.mark.asyncio
    async def test_nonexistent_belief_returns_error(
        self,
        graph: CognitiveGraph,
    ) -> None:
        mgr = DiscoveryBeliefManager(graph=graph)
        revision = await mgr.revise_belief(
            "nonexistent",
            "ev1",
            direction="supporting",
        )
        assert "not a BeliefNode" in revision.reason

    @pytest.mark.asyncio
    async def test_falsification_at_low_confidence(
        self,
        belief_graph: tuple[CognitiveGraph, str, str],
    ) -> None:
        graph, belief_id, evidence_id = belief_graph

        # Use aggressive config
        config = BayesianConfig(max_contradict_delta=0.5)
        mgr = DiscoveryBeliefManager(graph=graph, config=config)

        # Contradict multiple times to drive below threshold
        for _ in range(10):
            await mgr.revise_belief(
                belief_id,
                evidence_id,
                direction="contradicting",
                evidence_strength="replicated",
            )

        node = graph.get_node(belief_id)
        assert isinstance(node, BeliefNode)
        assert node.falsification_status == FalsificationStatus.FALSIFIED


class TestReviseFromPrediction:
    """Test prediction-based revision."""

    @pytest.mark.asyncio
    async def test_correct_prediction_raises_confidence(
        self,
        belief_graph: tuple[CognitiveGraph, str, str],
    ) -> None:
        graph, belief_id, _ = belief_graph
        mgr = DiscoveryBeliefManager(graph=graph)

        outcome = PredictionOutcome(
            prediction_id="p1",
            hypothesis_id="h1",
            predicted="x",
            observed="x",
            correct=True,
        )
        revision = await mgr.revise_from_prediction(belief_id, outcome)
        assert revision.new_confidence > revision.old_confidence

    @pytest.mark.asyncio
    async def test_wrong_prediction_lowers_confidence(
        self,
        belief_graph: tuple[CognitiveGraph, str, str],
    ) -> None:
        graph, belief_id, _ = belief_graph
        mgr = DiscoveryBeliefManager(graph=graph)

        outcome = PredictionOutcome(
            prediction_id="p2",
            hypothesis_id="h1",
            predicted="x",
            observed="y",
            correct=False,
        )
        revision = await mgr.revise_from_prediction(belief_id, outcome)
        assert revision.new_confidence < revision.old_confidence


class TestFalsificationCandidates:
    """Test falsification candidate identification."""

    @pytest.mark.asyncio
    async def test_candidates_in_range(self, graph: CognitiveGraph) -> None:
        mgr = DiscoveryBeliefManager(graph=graph)

        for i, conf in enumerate([0.2, 0.5, 0.5, 0.8, 0.95]):
            b = BeliefNode(
                claim=f"Belief {i}",
                belief_confidence=BeliefConfidence(evidence_quality=conf),
            )
            b.uncertainty.confidence = conf
            graph.upsert_node(b)

        candidates = await mgr.get_falsification_candidates(
            min_confidence=0.3,
            max_confidence=0.9,
        )
        # 0.5, 0.5, 0.8 are in range (0.2 too low, 0.95 too high)
        assert len(candidates) == 3


class TestBeliefSummary:
    """Test belief summary generation."""

    @pytest.mark.asyncio
    async def test_summary_fields(
        self,
        belief_graph: tuple[CognitiveGraph, str, str],
    ) -> None:
        graph, belief_id, evidence_id = belief_graph
        mgr = DiscoveryBeliefManager(graph=graph)

        await mgr.revise_belief(
            belief_id,
            evidence_id,
            direction="supporting",
        )

        summary = await mgr.get_belief_summary(belief_id)
        assert summary["belief_id"] == belief_id
        assert summary["revision_count"] == 1
        assert summary["evidence_count"] == 1
