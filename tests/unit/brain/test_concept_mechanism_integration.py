"""Tests for Concept ↔ Mechanism ↔ Belief integration.

Tests the Phase 2 abstraction hierarchy:
    Mechanisms → Beliefs → Concepts
"""

from __future__ import annotations

import pytest

from hbllm.brain.emotion.mechanism_store import MechanismStore
from hbllm.brain.reasoning.belief_store import BeliefStore, BeliefType
from hbllm.brain.reasoning.concept_formation import ConceptFormationEngine


@pytest.fixture
def mechanism_store(tmp_path):
    return MechanismStore(data_dir=str(tmp_path / "mechs"))


@pytest.fixture
def belief_store(tmp_path):
    return BeliefStore(data_dir=str(tmp_path / "beliefs"))


@pytest.fixture
def engine(tmp_path, mechanism_store, belief_store):
    return ConceptFormationEngine(
        mechanism_store=mechanism_store,
        belief_store=belief_store,
        data_dir=str(tmp_path / "concepts"),
    )


class TestMechanismPatternDiscovery:
    """Test discover_mechanism_patterns()."""

    @pytest.mark.asyncio
    async def test_no_store_returns_empty(self, tmp_path):
        engine = ConceptFormationEngine(data_dir=str(tmp_path))
        result = await engine.discover_mechanism_patterns()
        assert result == []

    @pytest.mark.asyncio
    async def test_too_few_mechanisms(self, engine, mechanism_store):
        mechanism_store.create(
            description="Only one",
            preconditions=["a"],
            process_steps=["validate input"],
            expected_outcomes=["safe"],
        )
        result = await engine.discover_mechanism_patterns()
        assert result == []

    @pytest.mark.asyncio
    async def test_discovers_pattern(self, engine, mechanism_store):
        """Two mechanisms with shared steps should form a pattern."""
        mechanism_store.create(
            description="SQL injection defense",
            preconditions=["user input"],
            process_steps=["validate input", "sanitize characters", "parameterize query"],
            expected_outcomes=["safe query"],
            domain="security",
        )
        mechanism_store.create(
            description="XSS defense",
            preconditions=["user input"],
            process_steps=["validate input", "sanitize characters", "encode output"],
            expected_outcomes=["safe output"],
            domain="security",
        )

        result = await engine.discover_mechanism_patterns(domain="security")
        assert len(result) >= 1
        assert engine._mechanism_patterns_found >= 1

    @pytest.mark.asyncio
    async def test_creates_meta_mechanism(self, engine, mechanism_store):
        """Should create a meta-mechanism in the store."""
        mechanism_store.create(
            description="Auth check A",
            preconditions=["credentials"],
            process_steps=["verify token", "check permissions"],
            expected_outcomes=["access granted"],
        )
        mechanism_store.create(
            description="Auth check B",
            preconditions=["api key"],
            process_steps=["verify token", "check permissions"],
            expected_outcomes=["access allowed"],
        )

        initial_count = len(mechanism_store.list_all())
        await engine.discover_mechanism_patterns()
        new_count = len(mechanism_store.list_all())

        # Should have created at least one meta-mechanism
        assert new_count > initial_count


class TestBeliefAbstraction:
    """Test discover_belief_abstractions()."""

    @pytest.mark.asyncio
    async def test_no_store_returns_empty(self, tmp_path):
        engine = ConceptFormationEngine(data_dir=str(tmp_path))
        result = await engine.discover_belief_abstractions()
        assert result == []

    @pytest.mark.asyncio
    async def test_too_few_beliefs(self, engine, belief_store):
        belief_store.store_belief(
            claim="X causes Y",
            concept="test",
            belief_type=BeliefType.CAUSAL,
            confidence=0.8,
        )
        result = await engine.discover_belief_abstractions()
        assert result == []

    @pytest.mark.asyncio
    async def test_abstracts_similar_beliefs(self, engine, belief_store):
        """3+ similar causal beliefs should form an abstraction."""
        # Claims must be distinct enough to avoid Jaccard dedup (>0.5)
        # but share the same concept for clustering
        claims = [
            "SQL parameterization prevents database injection attacks",
            "output encoding blocks cross-site scripting exploitation",
            "shell escaping mitigates operating system command execution",
        ]
        for claim in claims:
            belief_store.store_belief(
                claim=claim,
                concept="injection_attacks",
                belief_type=BeliefType.CAUSAL,
                confidence=0.7,
                domain="security",
            )

        result = await engine.discover_belief_abstractions(domain="security")
        assert len(result) >= 1
        assert engine._belief_abstractions_found >= 1

    @pytest.mark.asyncio
    async def test_filters_by_domain(self, engine, belief_store):
        """Should only abstract beliefs from the specified domain."""
        for i in range(4):
            belief_store.store_belief(
                claim=f"security thing {i} causes vulnerability {i}",
                concept=f"sec_{i}",
                belief_type=BeliefType.CAUSAL,
                confidence=0.7,
                domain="security",
            )
        belief_store.store_belief(
            claim="unrelated causes things",
            concept="math",
            belief_type=BeliefType.CAUSAL,
            confidence=0.7,
            domain="math",
        )

        result = await engine.discover_belief_abstractions(domain="security")
        # Math belief should not be included
        for concept in result:
            assert concept.domain == "security"


class TestStats:
    """Test stats include new fields."""

    def test_stats_include_stores(self, engine):
        stats = engine.stats()
        assert stats["has_mechanism_store"] is True
        assert stats["has_belief_store"] is True
        assert "mechanism_patterns_found" in stats
        assert "belief_abstractions_found" in stats


class TestLinkMechanisms:
    """Test link_mechanisms_to_concepts()."""

    def test_no_store_returns_zero(self, tmp_path):
        engine = ConceptFormationEngine(data_dir=str(tmp_path))
        assert engine.link_mechanisms_to_concepts() == 0
