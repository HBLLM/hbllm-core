"""Tests for BeliefStore — persistent belief and contradiction storage."""

from __future__ import annotations

import pytest

from hbllm.brain.belief_store import (
    Belief,
    BeliefStatus,
    BeliefStore,
    BeliefType,
)


@pytest.fixture
def store(tmp_path):
    return BeliefStore(data_dir=str(tmp_path))


class TestBeliefDataclass:
    """Test Belief dataclass behavior."""

    def test_reinforce(self):
        b = Belief(claim="X", confidence=0.5)
        b.reinforce("src_1", delta=0.1)
        assert b.confidence == 0.6
        assert b.reinforcement_count == 1
        assert "src_1" in b.evidence_sources

    def test_weaken(self):
        b = Belief(claim="X", confidence=0.5)
        b.weaken("src_1", delta=0.1)
        assert b.confidence == 0.4

    def test_weaken_below_threshold_decays(self):
        b = Belief(claim="X", confidence=0.05)
        b.weaken("src_1", delta=0.1)
        assert b.confidence == 0.0
        assert b.status == BeliefStatus.DECAYED

    def test_link_contradiction(self):
        b = Belief(claim="X", confidence=0.5)
        b.link_contradiction("ctr_abc")
        assert "ctr_abc" in b.contradiction_ids
        assert b.status == BeliefStatus.CONTESTED
        # Idempotent
        b.link_contradiction("ctr_abc")
        assert b.contradiction_ids.count("ctr_abc") == 1

    def test_serialization(self):
        b = Belief(
            claim="SQL injection exploits user input",
            concept="injection",
            belief_type=BeliefType.CAUSAL,
            confidence=0.8,
            domain="security",
        )
        d = b.to_dict()
        restored = Belief.from_dict(d)
        assert restored.claim == b.claim
        assert restored.belief_type == BeliefType.CAUSAL
        assert restored.confidence == 0.8


class TestBeliefStorage:
    """Test BeliefStore CRUD."""

    def test_store_and_retrieve(self, store):
        b = store.store_belief(
            claim="Python uses reference counting",
            concept="python_gc",
            confidence=0.9,
            source="research:session_1",
            belief_type=BeliefType.FACTUAL,
            domain="programming",
        )
        assert b.belief_id.startswith("blf_")

        retrieved = store.get_belief(b.belief_id)
        assert retrieved is not None
        assert retrieved.claim == "Python uses reference counting"
        assert retrieved.confidence == 0.9

    def test_store_duplicate_reinforces(self, store):
        """Storing a similar claim should reinforce, not duplicate."""
        b1 = store.store_belief(
            claim="Python uses reference counting for memory",
            concept="python_gc",
            confidence=0.7,
            source="src_1",
        )
        b2 = store.store_belief(
            claim="Python uses reference counting for memory management",
            concept="python_gc",
            confidence=0.8,
            source="src_2",
        )
        # Should be same belief reinforced
        assert b2.belief_id == b1.belief_id
        assert b2.reinforcement_count == 1
        assert "src_2" in b2.evidence_sources

    def test_get_beliefs_for_concept(self, store):
        store.store_belief(claim="X is true", concept="test")
        store.store_belief(claim="Y is also true", concept="test")
        store.store_belief(claim="Z is irrelevant", concept="other")

        beliefs = store.get_beliefs_for_concept("test")
        assert len(beliefs) == 2

    def test_get_beliefs_by_domain(self, store):
        store.store_belief(claim="A", concept="c1", domain="security")
        store.store_belief(claim="B", concept="c2", domain="security")
        store.store_belief(claim="C", concept="c3", domain="math")

        beliefs = store.get_beliefs_by_domain("security")
        assert len(beliefs) == 2

    def test_get_beliefs_by_type(self, store):
        store.store_belief(
            claim="X causes Y",
            concept="c1",
            belief_type=BeliefType.CAUSAL,
        )
        store.store_belief(
            claim="Z is true",
            concept="c2",
            belief_type=BeliefType.FACTUAL,
        )
        store.store_belief(
            claim="A enables B",
            concept="c3",
            belief_type=BeliefType.CAUSAL,
        )

        causal = store.get_beliefs_by_type(BeliefType.CAUSAL)
        assert len(causal) == 2

    def test_update_belief(self, store):
        b = store.store_belief(claim="X", concept="test", confidence=0.5)
        b.confidence = 0.9
        store.update_belief(b)

        retrieved = store.get_belief(b.belief_id)
        assert retrieved.confidence == 0.9


class TestContradictionPersistence:
    """Test contradiction storage and priority retrieval."""

    def test_store_contradiction(self, store):
        ctr_id = store.store_contradiction(
            claim_a="X is true",
            claim_b="X is false",
            concept="test",
            severity=0.8,
        )
        assert ctr_id.startswith("ctr_")

    def test_unresolved_contradictions(self, store):
        store.store_contradiction("A", "B", concept="c1", severity=0.5)
        store.store_contradiction("C", "D", concept="c2", severity=0.9)

        unresolved = store.get_unresolved_contradictions()
        assert len(unresolved) == 2
        # Ordered by severity descending
        assert unresolved[0]["severity"] == 0.9

    def test_resolve_contradiction(self, store):
        ctr_id = store.store_contradiction("A", "B", concept="c1", severity=0.5)
        store.resolve_contradiction(ctr_id, "A was correct based on new evidence")

        unresolved = store.get_unresolved_contradictions()
        assert len(unresolved) == 0

    def test_contradiction_links_to_belief(self, store):
        """Storing a contradiction should link to matching beliefs."""
        b = store.store_belief(claim="X is true", concept="test")
        store.store_contradiction(
            claim_a="X is true",
            claim_b="X is false",
            concept="test",
            severity=0.7,
        )

        updated = store.get_belief(b.belief_id)
        assert len(updated.contradiction_ids) == 1
        assert updated.status == BeliefStatus.CONTESTED

    def test_priority_ordering(self, store):
        store.store_contradiction("A", "B", concept="c1", severity=0.3)
        store.store_contradiction("C", "D", concept="c2", severity=0.9)

        prioritized = store.get_contradictions_by_priority()
        assert len(prioritized) == 2
        assert prioritized[0]["severity"] == 0.9

    def test_get_contested_beliefs(self, store):
        b = store.store_belief(claim="X is true", concept="test")
        store.store_contradiction("X is true", "X is false", concept="test")

        contested = store.get_contested_beliefs()
        assert len(contested) == 1
        assert contested[0].belief_id == b.belief_id


class TestDecay:
    """Test belief decay."""

    def test_decay_reduces_confidence(self, store):
        store.store_belief(claim="A", concept="c1", confidence=0.5)
        store.store_belief(claim="B", concept="c2", confidence=0.03)

        decayed = store.decay_beliefs(rate=0.01, threshold=0.05)
        assert decayed == 1  # B should have decayed below threshold


class TestStats:
    """Test stats reporting."""

    def test_stats(self, store):
        store.store_belief(claim="A", concept="c1")
        store.store_contradiction("X", "Y", concept="c2")

        stats = store.stats()
        assert stats["total_beliefs"] == 1
        assert stats["active_beliefs"] == 1
        assert stats["total_contradictions"] == 1
        assert stats["unresolved_contradictions"] == 1
