"""Tests for evidence idempotency — (evidence_id, proposition_id) invariant.

Covers:
- Double-submit produces exactly one transition
- Multi-proposition: same evidence can update multiple beliefs
- Incorporated evidence returns no-op transition with delta == 0.0
"""

from __future__ import annotations

import time

import pytest

from hbllm.brain.epistemics.belief_manager import DiscoveryBeliefManager
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    EvidenceNode,
    Provenance,
)
from hbllm.hcir.types import PropositionLikelihood, UncertaintyVector

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


def _make_graph_with_evidence_and_belief() -> tuple[CognitiveGraph, BeliefNode, EvidenceNode]:
    graph = CognitiveGraph()
    belief = BeliefNode(
        id="belief_1",
        claim="person is present",
        tags=["Person present"],
        uncertainty=UncertaintyVector(confidence=0.5),
    )
    graph.upsert_node(belief)

    evidence = EvidenceNode(
        id="ev_1",
        tags=["person detected"],
        modality="visual",
        provenance=Provenance(timestamp=time.time()),
    )
    graph.upsert_node(evidence)

    return graph, belief, evidence


def _make_informative_likelihood(belief_id: str, evidence_id: str) -> PropositionLikelihood:
    return PropositionLikelihood(
        belief_id=belief_id,
        evidence_id=evidence_id,
        p_e_given_h=0.9,
        p_e_given_not_h=0.2,
        likelihood_ratio=4.5,
        raw_likelihood_ratio=4.5,
        effective_likelihood_ratio=4.5,
        novelty_discount=1.0,
        status="informative",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDoubleSubmitIdempotency:
    """Verify that submitting the same evidence twice for the same belief
    produces exactly one BeliefTransitionNode."""

    @pytest.mark.asyncio
    async def test_double_submit_single_transition(self):
        graph, belief, evidence = _make_graph_with_evidence_and_belief()
        manager = DiscoveryBeliefManager(graph)
        likelihood = _make_informative_likelihood("belief_1", "ev_1")

        # First revision
        t1 = await manager.revise("belief_1", likelihood)
        assert t1.transition_id != ""
        assert t1.delta != 0.0

        # Confidence should have changed
        node = graph.get_node("belief_1")
        assert isinstance(node, BeliefNode)
        conf_after_first = node.uncertainty.confidence
        assert conf_after_first != 0.5

        # Second revision with same evidence — should be no-op
        t2 = await manager.revise("belief_1", likelihood)
        assert t2.transition_id == ""
        assert t2.delta == 0.0

        # Confidence should be unchanged
        node = graph.get_node("belief_1")
        assert isinstance(node, BeliefNode)
        assert node.uncertainty.confidence == conf_after_first

    @pytest.mark.asyncio
    async def test_incorporated_evidence_has_transition_id(self):
        graph, belief, evidence = _make_graph_with_evidence_and_belief()
        manager = DiscoveryBeliefManager(graph)
        likelihood = _make_informative_likelihood("belief_1", "ev_1")

        t1 = await manager.revise("belief_1", likelihood)

        # Evidence should now have the transition recorded
        ev = graph.get_node("ev_1")
        assert isinstance(ev, EvidenceNode)
        assert "belief_1" in ev.incorporated_transitions
        assert ev.incorporated_transitions["belief_1"] == t1.transition_id
        assert ev.incorporation_status == "incorporated"

    @pytest.mark.asyncio
    async def test_redundant_evidence_returns_noop_transition(self):
        graph, belief, evidence = _make_graph_with_evidence_and_belief()
        manager = DiscoveryBeliefManager(graph)
        likelihood = _make_informative_likelihood("belief_1", "ev_1")

        await manager.revise("belief_1", likelihood)
        t2 = await manager.revise("belief_1", likelihood)

        assert t2.delta == 0.0
        assert t2.novelty_score == 0.0
        assert t2.effective_likelihood_ratio == 1.0
        assert "already incorporated" in t2.rationale


class TestMultiPropositionIncorporation:
    """Same evidence may legitimately update multiple independent propositions."""

    @pytest.mark.asyncio
    async def test_same_evidence_two_propositions(self):
        graph = CognitiveGraph()

        b1 = BeliefNode(
            id="belief_room_occupied",
            claim="someone is in the room",
            tags=["Room occupied"],
            uncertainty=UncertaintyVector(confidence=0.5),
        )
        b2 = BeliefNode(
            id="belief_sensor_working",
            claim="human presence sensor is functioning",
            tags=["Sensor working"],
            uncertainty=UncertaintyVector(confidence=0.5),
        )
        evidence = EvidenceNode(
            id="ev_person",
            tags=["person detected"],
            modality="visual",
            provenance=Provenance(timestamp=time.time()),
        )
        graph.upsert_node(b1)
        graph.upsert_node(b2)
        graph.upsert_node(evidence)

        manager = DiscoveryBeliefManager(graph)

        lr1 = PropositionLikelihood(
            belief_id="belief_room_occupied",
            evidence_id="ev_person",
            p_e_given_h=0.9,
            p_e_given_not_h=0.2,
            likelihood_ratio=4.5,
            raw_likelihood_ratio=4.5,
            effective_likelihood_ratio=4.5,
            novelty_discount=1.0,
            status="informative",
        )
        lr2 = PropositionLikelihood(
            belief_id="belief_sensor_working",
            evidence_id="ev_person",
            p_e_given_h=0.8,
            p_e_given_not_h=0.4,
            likelihood_ratio=2.0,
            raw_likelihood_ratio=2.0,
            effective_likelihood_ratio=2.0,
            novelty_discount=1.0,
            status="informative",
        )

        t1 = await manager.revise("belief_room_occupied", lr1)
        t2 = await manager.revise("belief_sensor_working", lr2)

        # Both should succeed
        assert t1.transition_id != ""
        assert t2.transition_id != ""
        assert t1.delta != 0.0
        assert t2.delta != 0.0

        # Evidence should track both
        ev = graph.get_node("ev_person")
        assert isinstance(ev, EvidenceNode)
        assert "belief_room_occupied" in ev.incorporated_transitions
        assert "belief_sensor_working" in ev.incorporated_transitions
        assert ev.incorporated_transitions["belief_room_occupied"] == t1.transition_id
        assert ev.incorporated_transitions["belief_sensor_working"] == t2.transition_id

    @pytest.mark.asyncio
    async def test_double_submit_for_one_proposition_still_blocked(self):
        """After updating two propositions, re-submitting for the first should be blocked."""
        graph = CognitiveGraph()
        b1 = BeliefNode(
            id="b1",
            claim="claim 1",
            tags=["B1"],
            uncertainty=UncertaintyVector(confidence=0.5),
        )
        b2 = BeliefNode(
            id="b2",
            claim="claim 2",
            tags=["B2"],
            uncertainty=UncertaintyVector(confidence=0.5),
        )
        ev = EvidenceNode(
            id="ev_1",
            tags=["detected"],
            modality="visual",
            provenance=Provenance(timestamp=time.time()),
        )

        graph.upsert_node(b1)
        graph.upsert_node(b2)
        graph.upsert_node(ev)

        manager = DiscoveryBeliefManager(graph)
        lr = PropositionLikelihood(
            belief_id="b1",
            evidence_id="ev_1",
            p_e_given_h=0.9,
            p_e_given_not_h=0.2,
            likelihood_ratio=4.5,
            raw_likelihood_ratio=4.5,
            effective_likelihood_ratio=4.5,
            novelty_discount=1.0,
            status="informative",
        )

        await manager.revise("b1", lr)
        # Trying to re-submit for b1 should be blocked
        t_noop = await manager.revise("b1", lr)
        assert t_noop.delta == 0.0
