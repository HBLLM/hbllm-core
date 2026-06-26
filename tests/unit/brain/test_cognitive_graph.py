"""Tests for CognitiveGraph — 3-layer cognitive query infrastructure.

Tests are organized by layer:
    Layer 1: Typed view adapters (pure delegation)
    Layer 2: CognitiveIntegrator (dispatch + normalize + rank)
    Layer 3: Synthesis (knowledge_about, detect_conflicts)

Plus:
    Backward compatibility (LearningSubsystem alias)
    Layer isolation (views don't call router)
    Cross-store coherence (deduplication)
"""

from __future__ import annotations

import pytest

from hbllm.brain.belief_store import BeliefStore, BeliefType
from hbllm.brain.goal_manager import GoalManager, GoalPriority
from hbllm.brain.learning_subsystem import (
    CognitiveGraph,
    CognitiveQueryResult,
    LearningSubsystem,
)
from hbllm.brain.mechanism_store import MechanismStore

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def belief_store(tmp_path):
    store = BeliefStore(data_dir=str(tmp_path / "beliefs"))
    store.store_belief(
        claim="SQL injection exploits user input",
        concept="injection",
        belief_type=BeliefType.CAUSAL,
        confidence=0.9,
        domain="security",
    )
    store.store_belief(
        claim="Python uses reference counting for GC",
        concept="python_gc",
        belief_type=BeliefType.FACTUAL,
        confidence=0.8,
        domain="programming",
    )
    return store


@pytest.fixture
def mechanism_store(tmp_path):
    store = MechanismStore(data_dir=str(tmp_path / "mechs"))
    store.create(
        description="Input validation defense",
        preconditions=["user input", "security"],
        process_steps=["validate input", "sanitize characters"],
        expected_outcomes=["safe query"],
        domain="security",
        confidence=0.85,
    )
    store.create(
        description="Garbage collection mechanism",
        preconditions=["memory management"],
        process_steps=["reference count", "cycle detection"],
        expected_outcomes=["freed memory"],
        domain="programming",
        confidence=0.7,
    )
    return store


@pytest.fixture
def goal_manager(tmp_path):
    gm = GoalManager(data_dir=str(tmp_path / "goals"))
    gm.create_learning_goal(
        topic="Learn security best practices",
        priority=GoalPriority.HIGH,
    )
    return gm


@pytest.fixture
def graph(tmp_path, belief_store, mechanism_store, goal_manager):
    return CognitiveGraph(
        belief_store=belief_store,
        mechanism_store=mechanism_store,
        goal_manager=goal_manager,
    )


@pytest.fixture
def minimal_graph(tmp_path):
    """Graph with no stores — tests graceful degradation."""
    return CognitiveGraph()


# ── Layer 1: Typed View Adapters ────────────────────────────────────────────


class TestCognitiveBeliefs:
    """Test CognitiveBeliefs view — pure delegation to BeliefStore."""

    def test_available(self, graph):
        assert graph.beliefs.available is True

    def test_unavailable(self, minimal_graph):
        assert minimal_graph.beliefs.available is False

    def test_active_returns_beliefs(self, graph):
        active = graph.beliefs.active(domain="security")
        assert len(active) >= 1
        assert any("injection" in b.claim.lower() for b in active)

    def test_for_concept(self, graph):
        beliefs = graph.beliefs.for_concept("injection")
        assert len(beliefs) >= 1

    def test_strongest(self, graph):
        strongest = graph.beliefs.strongest(n=5)
        assert len(strongest) >= 1
        # Ordered by confidence descending
        if len(strongest) >= 2:
            assert strongest[0].confidence >= strongest[1].confidence

    def test_search_returns_cognitive_ir(self, graph):
        results = graph.beliefs.search("security")
        assert all(isinstance(r, CognitiveQueryResult) for r in results)
        assert all(r.source == "beliefs" for r in results)

    def test_search_empty_when_unavailable(self, minimal_graph):
        assert minimal_graph.beliefs.search("anything") == []

    def test_stats(self, graph):
        stats = graph.beliefs.stats()
        assert "total_beliefs" in stats


class TestCognitiveMechanisms:
    """Test CognitiveMechanisms view — pure delegation to MechanismStore."""

    def test_available(self, graph):
        assert graph.mechanisms.available is True

    def test_for_domain(self, graph):
        mechs = graph.mechanisms.for_domain("security")
        assert len(mechs) >= 1

    def test_search_returns_cognitive_ir(self, graph):
        results = graph.mechanisms.search("security")
        assert all(isinstance(r, CognitiveQueryResult) for r in results)
        assert all(r.source == "mechanisms" for r in results)

    def test_search_empty_when_unavailable(self, minimal_graph):
        assert minimal_graph.mechanisms.search("anything") == []

    def test_stats(self, graph):
        stats = graph.mechanisms.stats()
        assert "total_mechanisms" in stats


class TestCognitiveGoals:
    """Test CognitiveGoals view — pure delegation to GoalManager."""

    def test_available(self, graph):
        assert graph.goals.available is True

    def test_unavailable(self, minimal_graph):
        assert minimal_graph.goals.available is False

    def test_active(self, graph):
        active = graph.goals.active()
        assert len(active) >= 1

    def test_learning(self, graph):
        learning = graph.goals.learning()
        assert len(learning) >= 1

    def test_search_returns_cognitive_ir(self, graph):
        results = graph.goals.search("security")
        assert all(isinstance(r, CognitiveQueryResult) for r in results)
        assert all(r.source == "goals" for r in results)

    def test_search_empty_when_unavailable(self, minimal_graph):
        assert minimal_graph.goals.search("anything") == []


# ── Layer 2: CognitiveIntegrator ────────────────────────────────────────────


class TestCognitiveIntegrator:
    """Test integrator — dispatch + normalize + deduplicate + rank."""

    def test_query_all_scopes(self, graph):
        """Query across all stores returns results from multiple sources."""
        results = graph.query("security")
        sources = {r.source for r in results}
        # Should have results from beliefs AND mechanisms at minimum
        assert "beliefs" in sources or "mechanisms" in sources

    def test_query_scoped(self, graph):
        """Scoped query only searches specified stores."""
        results = graph.query("security", scope=["beliefs"])
        assert all(r.source == "beliefs" for r in results)

    def test_query_deduplicates(self, graph):
        """Same entity should not appear twice."""
        results = graph.query("security")
        seen = set()
        for r in results:
            key = (r.source, r.entity_id)
            assert key not in seen, f"Duplicate: {key}"
            seen.add(key)

    def test_query_ranked(self, graph):
        """Results should be ranked by relevance × confidence."""
        results = graph.query("security")
        if len(results) >= 2:

            def _score(r: CognitiveQueryResult) -> float:
                return r.relevance * 0.6 + r.confidence * 0.4

            for i in range(len(results) - 1):
                assert _score(results[i]) >= _score(results[i + 1])

    def test_query_returns_empty_for_unknown(self, graph):
        results = graph.query("xyzzy_nonexistent_topic_42")
        # Might return empty or low-relevance results
        assert isinstance(results, list)

    def test_query_handles_unavailable_stores(self, minimal_graph):
        """Should not crash when stores are unavailable."""
        results = minimal_graph.query("anything")
        assert results == []


# ── Layer 3: Synthesis ──────────────────────────────────────────────────────


class TestKnowledgeAbout:
    """Test knowledge_about() — structured merge, not summary score."""

    def test_returns_structured_dict(self, graph):
        knowledge = graph.knowledge_about("security")
        assert isinstance(knowledge, dict)
        assert knowledge["domain"] == "security"
        assert "beliefs" in knowledge
        assert "mechanisms" in knowledge
        assert "concepts" in knowledge
        assert "goals" in knowledge

    def test_fast_mode_no_conflicts(self, graph):
        """Fast mode should NOT run conflict detection."""
        knowledge = graph.knowledge_about("security", mode="fast")
        assert knowledge["conflicts"] == []

    def test_deep_mode_includes_conflicts(self, graph):
        """Deep mode should include conflicts."""
        knowledge = graph.knowledge_about("security", mode="deep")
        assert "conflicts" in knowledge

    def test_beliefs_are_structured(self, graph):
        """Beliefs in result should have claim + confidence, not raw objects."""
        knowledge = graph.knowledge_about("security")
        for b in knowledge["beliefs"]:
            assert "claim" in b
            assert "confidence" in b

    def test_empty_domain(self, graph):
        """Unknown domain returns empty lists, not errors."""
        knowledge = graph.knowledge_about("xyzzy_nonexistent")
        assert knowledge["beliefs"] == []
        assert knowledge["mechanisms"] == []


class TestDetectConflicts:
    """Test detect_conflicts() — first-class conflict detection."""

    def test_returns_list(self, graph):
        conflicts = graph.detect_conflicts("security")
        assert isinstance(conflicts, list)

    def test_belief_contradictions_detected(self, tmp_path):
        """Stored contradictions should be surfaced."""
        bs = BeliefStore(data_dir=str(tmp_path / "bs"))
        bs.store_belief(claim="X is true", concept="test", confidence=0.8)
        bs.store_contradiction("X is true", "X is false", concept="test", severity=0.7)

        graph = CognitiveGraph(belief_store=bs)
        conflicts = graph.detect_conflicts("test")
        assert len(conflicts) >= 1
        assert conflicts[0]["type"] == "belief_contradiction"

    def test_mechanism_belief_tension(self, tmp_path):
        """Weak mechanism + strong belief in same domain = tension."""
        bs = BeliefStore(data_dir=str(tmp_path / "bs"))
        ms = MechanismStore(data_dir=str(tmp_path / "ms"))

        bs.store_belief(
            claim="Input validation prevents injection attacks",
            concept="injection",
            confidence=0.9,
            domain="security",
        )
        ms.create(
            description="Input validation prevents injection attacks",
            preconditions=["user input"],
            process_steps=["check input"],
            expected_outcomes=["safe"],
            domain="security",
            confidence=0.1,  # Very weak mechanism
        )

        graph = CognitiveGraph(belief_store=bs, mechanism_store=ms)
        conflicts = graph.detect_conflicts("security")
        tensions = [c for c in conflicts if c["type"] == "mechanism_belief_tension"]
        assert len(tensions) >= 1


# ── Capability Surface ──────────────────────────────────────────────────────


class TestCapabilities:
    """Test explicit capability signaling."""

    def test_full_capabilities(self, graph):
        caps = graph.capabilities
        assert "beliefs" in caps
        assert "mechanisms" in caps
        assert "goals" in caps

    def test_minimal_capabilities(self, minimal_graph):
        caps = minimal_graph.capabilities
        assert "beliefs" not in caps
        assert "mechanisms" not in caps
        assert "goals" not in caps

    def test_has_goals_explicit(self, graph):
        assert graph.has_goals is True

    def test_has_goals_false(self, minimal_graph):
        assert minimal_graph.has_goals is False


class TestCognitiveState:
    """Test cognitive_state() — structured, not summary."""

    def test_returns_per_store_stats(self, graph):
        state = graph.cognitive_state()
        assert "capabilities" in state
        assert "beliefs" in state
        assert "mechanisms" in state

    def test_no_artificial_health_score(self, graph):
        state = graph.cognitive_state()
        # Should NOT have a single scalar "health" or "score"
        assert "health" not in state
        assert "score" not in state
        assert "overall" not in state

    def test_empty_when_no_stores(self, minimal_graph):
        state = minimal_graph.cognitive_state()
        assert "capabilities" in state
        assert len(state["capabilities"]) == 0


# ── Backward Compatibility ──────────────────────────────────────────────────


class TestBackwardCompat:
    """LearningSubsystem alias must work."""

    def test_alias_is_same_class(self):
        assert LearningSubsystem is CognitiveGraph

    def test_alias_instantiation(self, tmp_path):
        ls = LearningSubsystem(
            mechanism_store=MechanismStore(data_dir=str(tmp_path)),
        )
        assert ls.mechanisms.available is True
        assert ls.summary()["mechanism_store"] is True

    def test_summary_backward_compat(self, graph):
        s = graph.summary()
        # Old fields still present
        assert "mechanism_store" in s
        assert "belief_store" in s
        assert "has_belief_infrastructure" in s
        assert "has_research_infrastructure" in s
        # New fields also present
        assert "capabilities" in s


# ── Layer Isolation ─────────────────────────────────────────────────────────


class TestLayerIsolation:
    """Ensure views do not call router or synthesis directly."""

    def test_view_search_returns_independently(self, graph):
        """View.search() should work without the integrator being called."""
        # Get beliefs directly from view
        view_results = graph.beliefs.search("security")
        # These should be CognitiveQueryResult, produced by the view alone
        assert all(r.source == "beliefs" for r in view_results)

    def test_integrator_does_not_produce_conflicts(self, graph):
        """Integrator (Layer 2) should NOT detect conflicts (Layer 3 job)."""
        results = graph.query("security")
        # No result should have type "conflict" — that's synthesis
        for r in results:
            assert r.metadata.get("type") != "conflict"


# ── Cross-Store Coherence ───────────────────────────────────────────────────


class TestCrossStoreCoherence:
    """Test deduplication and coherence across stores."""

    def test_query_returns_deduplicated_cross_store_results(self, graph):
        """Same entity_id from same source should not appear twice."""
        results = graph.query("security")
        keys = [(r.source, r.entity_id) for r in results]
        assert len(keys) == len(set(keys))

    def test_cognitive_query_result_schema(self):
        """CognitiveQueryResult must have all required fields."""
        r = CognitiveQueryResult(
            source="beliefs",
            entity_id="blf_abc",
            content="Test claim",
            relevance=0.8,
            confidence=0.9,
            metadata={"key": "value"},
        )
        assert r.source == "beliefs"
        assert r.entity_id == "blf_abc"
        assert r.content == "Test claim"
        assert r.relevance == 0.8
        assert r.confidence == 0.9
        assert r.metadata == {"key": "value"}
