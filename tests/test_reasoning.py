"""
Tests for Memory-Driven Reasoning (v3 Step 3).

Tests cover:
  - ReasoningNetwork: chain evaluation, confidence output
  - CausalReasoner: single-hop, multi-hop, pruning, ranking
  - Integration: ComprehensionStream with CausalReasoner
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from hbllm.brain.causality.causal_graph import CausalGraph, CausalLink
from hbllm.brain.snn.reasoning.reasoner import CausalChain, CausalReasoner
from hbllm.brain.snn.reasoning.reasoning_network import ReasoningNetwork

# ═══════════════════════════════════════════════════════════════════════════
# ReasoningNetwork Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestReasoningNetwork:
    """Test the SNN-based chain evaluator."""

    def test_high_confidence_for_strong_chain(self) -> None:
        """Strong, short, recent chain → high confidence."""
        net = ReasoningNetwork()
        score = net.evaluate(
            {
                "chain_probability": 0.95,
                "chain_length": 1.0,  # 1 hop
                "recency": 0.99,
                "diversity": 0.8,
            }
        )
        assert 0.0 <= score <= 1.0
        # Strong features should produce reasonable confidence
        assert score > 0.3

    def test_low_confidence_for_weak_chain(self) -> None:
        """Weak, long chain → lower confidence."""
        net = ReasoningNetwork()
        strong = net.evaluate(
            {
                "chain_probability": 0.9,
                "chain_length": 1.0,
                "recency": 0.9,
                "diversity": 0.8,
            }
        )
        weak = net.evaluate(
            {
                "chain_probability": 0.1,
                "chain_length": 0.33,  # 3 hops
                "recency": 0.2,
                "diversity": 0.3,
            }
        )
        # Strong should outrank weak
        assert strong >= weak

    def test_zero_features_returns_valid(self) -> None:
        """All zero features → valid confidence."""
        net = ReasoningNetwork()
        score = net.evaluate(
            {
                "chain_probability": 0.0,
                "chain_length": 0.0,
                "recency": 0.0,
                "diversity": 0.0,
            }
        )
        assert 0.0 <= score <= 1.0

    def test_missing_features_handled(self) -> None:
        """Missing feature keys default to 0.0."""
        net = ReasoningNetwork()
        score = net.evaluate({})
        assert 0.0 <= score <= 1.0

    def test_network_accessible(self) -> None:
        """Underlying SpikingNetwork is accessible."""
        from hbllm.brain.snn.network import SpikingNetwork

        net = ReasoningNetwork()
        assert isinstance(net.network, SpikingNetwork)
        assert "evidence" in net.network.layer_names
        assert "evaluation" in net.network.layer_names
        assert "confidence" in net.network.layer_names


# ═══════════════════════════════════════════════════════════════════════════
# CausalReasoner Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCausalReasoner:
    """Test multi-hop causal graph traversal."""

    @pytest.fixture
    def graph(self, tmp_path):
        """Create a CausalGraph with test data."""
        g = CausalGraph(data_dir=str(tmp_path / "causal"))

        # Build a chain: A → B → C → D
        now = time.time()
        g._insert(
            CausalLink(
                link_id="link_ab",
                source_id="concept_A",
                target_id="concept_B",
                probability=0.9,
                created_at=now,
            )
        )
        g._insert(
            CausalLink(
                link_id="link_bc",
                source_id="concept_B",
                target_id="concept_C",
                probability=0.8,
                created_at=now,
            )
        )
        g._insert(
            CausalLink(
                link_id="link_cd",
                source_id="concept_C",
                target_id="concept_D",
                probability=0.7,
                created_at=now,
            )
        )

        # Also: A → E (separate branch)
        g._insert(
            CausalLink(
                link_id="link_ae",
                source_id="concept_A",
                target_id="concept_E",
                probability=0.95,
                created_at=now,
            )
        )

        return g

    def test_single_hop(self, graph) -> None:
        """Find direct effects from a concept."""
        reasoner = CausalReasoner(graph, max_depth=1)
        chains = reasoner.reason(["concept_A"])

        assert len(chains) > 0
        # All chains should be depth 1
        for chain in chains:
            assert chain.depth == 1

    def test_multi_hop(self, graph) -> None:
        """Find multi-hop chains."""
        reasoner = CausalReasoner(graph, max_depth=3)
        chains = reasoner.reason(["concept_A"])

        # Should find chains of different depths
        depths = {c.depth for c in chains}
        assert len(depths) > 1  # at least depth 1 and 2

    def test_chain_probability(self, graph) -> None:
        """Combined probability is product of link probabilities."""
        reasoner = CausalReasoner(graph, max_depth=2, min_probability=0.1)
        chains = reasoner.reason(["concept_A"])

        for chain in chains:
            expected_prob = 1.0
            for link in chain.links:
                expected_prob *= link.probability
            assert abs(chain.combined_probability - expected_prob) < 1e-6

    def test_pruning_weak_chains(self, graph) -> None:
        """Chains below min_probability are pruned."""
        reasoner = CausalReasoner(graph, max_depth=3, min_probability=0.99)
        chains = reasoner.reason(["concept_A"])

        # Only A→E has prob 0.95, still below 0.99
        # All chains should be above the threshold
        for chain in chains:
            assert chain.combined_probability >= 0.99

    def test_top_k_limits(self, graph) -> None:
        """Top-K limits the number of returned chains."""
        reasoner = CausalReasoner(graph, max_depth=3, min_probability=0.1, top_k=2)
        chains = reasoner.reason(["concept_A"])

        assert len(chains) <= 2

    def test_no_chains_from_unknown_concept(self, graph) -> None:
        """Unknown concept returns empty chains."""
        reasoner = CausalReasoner(graph)
        chains = reasoner.reason(["nonexistent_concept"])
        assert chains == []

    def test_reason_between(self, graph) -> None:
        """Find chains between two specific concepts."""
        reasoner = CausalReasoner(graph, max_depth=3, min_probability=0.1)
        chains = reasoner.reason_between("concept_A", "concept_C")

        assert len(chains) > 0
        for chain in chains:
            assert chain.source_concept == "concept_A"
            assert chain.conclusion == "concept_C"

    def test_reason_between_no_path(self, graph) -> None:
        """No path between disconnected concepts."""
        reasoner = CausalReasoner(graph, max_depth=3)
        chains = reasoner.reason_between("concept_E", "concept_A")
        assert chains == []

    def test_snn_confidence_present(self, graph) -> None:
        """All chains have SNN confidence scores."""
        reasoner = CausalReasoner(graph, max_depth=2, min_probability=0.1)
        chains = reasoner.reason(["concept_A"])

        for chain in chains:
            assert 0.0 <= chain.snn_confidence <= 1.0

    def test_chain_serialization(self, graph) -> None:
        """CausalChain serializes correctly."""
        reasoner = CausalReasoner(graph, max_depth=1, min_probability=0.1)
        chains = reasoner.reason(["concept_A"])

        if chains:
            d = chains[0].to_dict()
            assert "depth" in d
            assert "combined_probability" in d
            assert "snn_confidence" in d
            assert "links" in d


# ═══════════════════════════════════════════════════════════════════════════
# Integration: ComprehensionStream + CausalReasoner
# ═══════════════════════════════════════════════════════════════════════════


class TestStreamCausalIntegration:
    """Test ComprehensionStream with CausalReasoner wired in."""

    @pytest.fixture
    def causal_graph(self, tmp_path):
        g = CausalGraph(data_dir=str(tmp_path / "causal"))
        now = time.time()
        g._insert(
            CausalLink(
                link_id="link_1",
                source_id="test concept",
                target_id="effect_1",
                probability=0.9,
                created_at=now,
            )
        )
        return g

    def _make_stream(self, causal_graph=None, with_reasoner=True):
        from hbllm.brain.snn.comprehension import (
            ComprehensionEnsemble,
            ComprehensionStream,
            LexicalBuffer,
        )

        ensemble = ComprehensionEnsemble(domain="general")
        lexical_buffer = LexicalBuffer()
        def encoder(text):
            return np.random.randn(384)
        domain_centroids = {"general": np.random.randn(384)}

        reasoner = None
        if with_reasoner and causal_graph is not None:
            reasoner = CausalReasoner(causal_graph)

        return ComprehensionStream(
            ensemble=ensemble,
            lexical_buffer=lexical_buffer,
            encoder=encoder,
            domain_centroids=domain_centroids,
            causal_reasoner=reasoner,
        )

    @pytest.mark.asyncio
    async def test_stream_with_causal_reasoner(self, causal_graph) -> None:
        stream = self._make_stream(causal_graph, with_reasoner=True)

        state = await stream.comprehend("My Laravel API returns 500 errors on auth routes")

        assert hasattr(state, "causal_chains")
        assert isinstance(state.causal_chains, list)

    @pytest.mark.asyncio
    async def test_stream_without_causal_reasoner(self) -> None:
        stream = self._make_stream(with_reasoner=False)

        state = await stream.comprehend("simple test query")

        assert state.causal_chains == []

    @pytest.mark.asyncio
    async def test_understanding_state_has_causal_chains_field(self) -> None:
        from hbllm.brain.snn.comprehension.models import UnderstandingState

        state = UnderstandingState()
        assert hasattr(state, "causal_chains")
        assert state.causal_chains == []
