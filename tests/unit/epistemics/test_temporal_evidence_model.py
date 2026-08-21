"""Tests for TemporalEvidenceModel — multidimensional novelty, identity, and dependence correction.

Covers:
- Correlated stream bounded information (50 consecutive frames, confidence plateau)
- Independent evidence accumulates more than correlated stream
- Identical frames have novelty → 0
- Large time gap recovers novelty
- State transition overrides temporal decay
- Temporal pattern classification (PERSISTENT, TRANSITION, TRANSIENT)
- Half-life formula correctness: n_t = 1 − 2^(−Δt / T½)
"""

from __future__ import annotations

import math

from hbllm.brain.epistemics.temporal_evidence_model import (
    LabelStateChangeDetector,
    TemporalEvidenceModel,
)
from hbllm.hcir.graph import BeliefNode, CognitiveGraph, EvidenceNode
from hbllm.hcir.types import (
    EvidenceTemporalPattern,
    NoveltyPolicy,
    UncertaintyVector,
)

# ═══════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════


def _make_graph() -> CognitiveGraph:
    return CognitiveGraph()


def _make_belief(graph: CognitiveGraph, belief_id: str = "belief_1") -> BeliefNode:
    node = BeliefNode(
        id=belief_id,
        claim="person is present",
        tags=["Person is present"],
        uncertainty=UncertaintyVector(confidence=0.5),
    )
    graph.upsert_node(node)
    return node


def _make_evidence(
    graph: CognitiveGraph,
    evidence_id: str,
    label: str = "person detected",
    timestamp: float = 0.0,
    candidates: list | None = None,
) -> EvidenceNode:
    from hbllm.hcir.graph import Provenance

    node = EvidenceNode(
        id=evidence_id,
        tags=[label],
        modality="visual",
        provenance=Provenance(timestamp=timestamp),
        candidates=candidates or [{"label": label, "score": 0.9}],
    )
    graph.upsert_node(node)
    return node


# ═══════════════════════════════════════════════════════════════════════════
# Half-Life Formula Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHalfLifeFormula:
    """Test that temporal novelty uses proper half-life: n_t = 1 − 2^(−Δt/T½)."""

    def test_zero_delta_returns_zero_novelty(self):
        policy = NoveltyPolicy(half_life_seconds=5.0)
        assert policy.compute_temporal_novelty(0.0) == 0.0

    def test_negative_delta_returns_zero_novelty(self):
        policy = NoveltyPolicy(half_life_seconds=5.0)
        assert policy.compute_temporal_novelty(-1.0) == 0.0

    def test_at_half_life_novelty_is_half(self):
        """At Δt = T½, novelty should be 0.5."""
        policy = NoveltyPolicy(half_life_seconds=5.0)
        novelty = policy.compute_temporal_novelty(5.0)
        assert abs(novelty - 0.5) < 1e-10

    def test_at_two_half_lives_novelty_is_three_quarters(self):
        """At Δt = 2T½, novelty should be 0.75."""
        policy = NoveltyPolicy(half_life_seconds=5.0)
        novelty = policy.compute_temporal_novelty(10.0)
        assert abs(novelty - 0.75) < 1e-10

    def test_large_delta_approaches_one(self):
        """At Δt >> T½, novelty should approach 1.0."""
        policy = NoveltyPolicy(half_life_seconds=5.0)
        novelty = policy.compute_temporal_novelty(100.0)
        assert novelty > 0.99

    def test_formula_matches_2_power(self):
        """Verify n_t = 1 − 2^(−Δt/T½) explicitly."""
        policy = NoveltyPolicy(half_life_seconds=3.0)
        delta_t = 7.5
        expected = 1.0 - math.pow(2.0, -delta_t / 3.0)
        actual = policy.compute_temporal_novelty(delta_t)
        assert abs(actual - expected) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# Identity & Idempotency Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestIdentityCheck:
    """Test (evidence_id, proposition_id) idempotency."""

    def test_fresh_evidence_not_incorporated(self):
        graph = _make_graph()
        belief = _make_belief(graph)
        evidence = _make_evidence(graph, "ev_1", timestamp=100.0)
        model = TemporalEvidenceModel(graph)

        assert not model.check_identity(evidence, belief)

    def test_incorporated_evidence_is_detected(self):
        graph = _make_graph()
        belief = _make_belief(graph)
        evidence = _make_evidence(graph, "ev_1", timestamp=100.0)

        # Simulate incorporation
        evidence.incorporated_transitions["belief_1"] = "trans_001"
        graph.upsert_node(evidence)

        model = TemporalEvidenceModel(graph)
        assert model.check_identity(evidence, belief)

    def test_same_evidence_different_proposition_not_incorporated(self):
        """Same evidence may legitimately affect multiple propositions."""
        graph = _make_graph()
        belief_1 = _make_belief(graph, "belief_1")
        belief_2 = _make_belief(graph, "belief_2")
        evidence = _make_evidence(graph, "ev_1", timestamp=100.0)

        # Incorporate for belief_1 only
        evidence.incorporated_transitions["belief_1"] = "trans_001"
        graph.upsert_node(evidence)

        model = TemporalEvidenceModel(graph)
        assert model.check_identity(evidence, belief_1)
        assert not model.check_identity(evidence, belief_2)

    def test_incorporated_evidence_gets_zero_novelty(self):
        graph = _make_graph()
        belief = _make_belief(graph)
        evidence = _make_evidence(graph, "ev_1", timestamp=100.0)
        evidence.incorporated_transitions["belief_1"] = "trans_001"
        graph.upsert_node(evidence)

        model = TemporalEvidenceModel(graph)
        assessment = model.assess(evidence, belief)

        assert assessment.already_incorporated
        assert assessment.composite_novelty == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Correlated Stream Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCorrelatedStreamBounding:
    """Test that correlated evidence from streaming perception is bounded."""

    def test_fifty_consecutive_frames_bounded(self):
        """50 consecutive 'person detected' at 30fps should have diminishing novelty."""
        graph = _make_graph()
        belief = _make_belief(graph)
        policy = NoveltyPolicy(half_life_seconds=5.0)
        model = TemporalEvidenceModel(graph, policy=policy)

        novelties = []
        base_time = 1000.0
        frame_interval = 1.0 / 30.0  # 30fps

        for i in range(50):
            t = base_time + i * frame_interval
            ev = _make_evidence(graph, f"ev_{i}", label="person detected", timestamp=t)
            assessment = model.assess(ev, belief)
            novelties.append(assessment.composite_novelty)

            # Simulate incorporation
            ev.incorporated_transitions["belief_1"] = f"trans_{i}"
            ev.last_incorporated_at = t
            graph.upsert_node(ev)

        # First evidence should have high novelty (no prior)
        assert novelties[0] > 0.5

        # Later evidence should have very low novelty (correlated)
        assert novelties[-1] < 0.15

        # Novelty should generally decrease
        assert novelties[0] > novelties[10] > novelties[49]

    def test_independent_evidence_accumulates_more(self):
        """Independent observations should accumulate significantly more novelty."""
        graph_corr = _make_graph()
        belief_corr = _make_belief(graph_corr)
        policy = NoveltyPolicy(half_life_seconds=5.0)
        model_corr = TemporalEvidenceModel(graph_corr, policy=policy)

        # Correlated: 10 frames in rapid succession (30fps)
        corr_total = 0.0
        base_time = 1000.0
        for i in range(10):
            t = base_time + i / 30.0
            ev = _make_evidence(graph_corr, f"ev_corr_{i}", label="person detected", timestamp=t)
            assessment = model_corr.assess(ev, belief_corr)
            corr_total += assessment.composite_novelty
            ev.incorporated_transitions["belief_1"] = f"trans_{i}"
            ev.last_incorporated_at = t
            graph_corr.upsert_node(ev)

        # Independent: 10 observations with large time gaps
        graph_ind = _make_graph()
        belief_ind = _make_belief(graph_ind)
        model_ind = TemporalEvidenceModel(graph_ind, policy=policy)

        ind_total = 0.0
        for i in range(10):
            t = base_time + i * 60.0  # 1 minute apart
            ev = _make_evidence(graph_ind, f"ev_ind_{i}", label="person detected", timestamp=t)
            assessment = model_ind.assess(ev, belief_ind)
            ind_total += assessment.composite_novelty
            ev.incorporated_transitions["belief_1"] = f"trans_{i}"
            ev.last_incorporated_at = t
            graph_ind.upsert_node(ev)

        # Independent evidence should accumulate significantly more novelty
        assert ind_total > corr_total * 1.5


# ═══════════════════════════════════════════════════════════════════════════
# State Change Override Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStateChangeOverride:
    """Test that state transitions override temporal decay."""

    def test_state_transition_at_short_interval_high_novelty(self):
        """'person sitting' → 'person standing' at Δt=100ms should get high novelty."""
        graph = _make_graph()
        belief = _make_belief(graph)
        policy = NoveltyPolicy(half_life_seconds=5.0, state_change_override=True)
        model = TemporalEvidenceModel(graph, policy=policy)

        base_time = 1000.0
        # First: person sitting
        ev1 = _make_evidence(graph, "ev_1", label="person sitting", timestamp=base_time)
        model.assess(ev1, belief)
        ev1.incorporated_transitions["belief_1"] = "trans_1"
        ev1.last_incorporated_at = base_time
        graph.upsert_node(ev1)

        # Second: person standing (100ms later - very short interval)
        ev2 = _make_evidence(graph, "ev_2", label="person standing", timestamp=base_time + 0.1)
        assessment = model.assess(ev2, belief)

        # Despite tiny time interval, state change should give high novelty
        assert assessment.composite_novelty > 0.5
        assert assessment.state_change_novelty > 0.5

    def test_same_label_low_novelty_at_short_interval(self):
        """Same label at short interval should have low novelty."""
        graph = _make_graph()
        belief = _make_belief(graph)
        policy = NoveltyPolicy(half_life_seconds=5.0)
        model = TemporalEvidenceModel(graph, policy=policy)

        base_time = 1000.0
        ev1 = _make_evidence(graph, "ev_1", label="person detected", timestamp=base_time)
        model.assess(ev1, belief)
        ev1.incorporated_transitions["belief_1"] = "trans_1"
        ev1.last_incorporated_at = base_time
        graph.upsert_node(ev1)

        ev2 = _make_evidence(graph, "ev_2", label="person detected", timestamp=base_time + 0.033)
        assessment = model.assess(ev2, belief)

        # Same content, very short interval → low novelty
        assert assessment.composite_novelty < 0.15


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Pattern Classification Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTemporalPatternClassification:
    """Test classification into PERSISTENT, TRANSITION, TRANSIENT, etc."""

    def test_insufficient_history_returns_unknown(self):
        graph = _make_graph()
        model = TemporalEvidenceModel(graph)
        evidence = _make_evidence(graph, "ev_1")
        assert model.classify_pattern(evidence, []) == EvidenceTemporalPattern.UNKNOWN
        assert model.classify_pattern(evidence, [evidence]) == EvidenceTemporalPattern.UNKNOWN

    def test_persistent_pattern_detected(self):
        """4+ consecutive identical labels → PERSISTENT."""
        graph = _make_graph()
        model = TemporalEvidenceModel(graph)

        history = []
        for i in range(4):
            ev = _make_evidence(graph, f"ev_{i}", label="person detected")
            history.append(ev)

        current = _make_evidence(graph, "ev_current", label="person detected")
        assert model.classify_pattern(current, history) == EvidenceTemporalPattern.PERSISTENT

    def test_transition_pattern_detected(self):
        """Label change at most recent → TRANSITION."""
        graph = _make_graph()
        model = TemporalEvidenceModel(graph)

        history = [
            _make_evidence(graph, "ev_0", label="person sitting"),
            _make_evidence(graph, "ev_1", label="person sitting"),
            _make_evidence(graph, "ev_2", label="person sitting"),
        ]

        current = _make_evidence(graph, "ev_current", label="person running")
        assert model.classify_pattern(current, history) == EvidenceTemporalPattern.TRANSITION


# ═══════════════════════════════════════════════════════════════════════════
# Semantic Novelty Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSemanticNovelty:
    """Test Jaccard-based semantic novelty."""

    def test_identical_tags_zero_novelty(self):
        detector = LabelStateChangeDetector()
        graph = _make_graph()
        ev1 = _make_evidence(graph, "ev_1", label="person standing")
        ev2 = _make_evidence(graph, "ev_2", label="person standing")

        assessment = detector.detect(ev1, ev2)
        # Same label → no transition
        assert not assessment.is_transition

    def test_different_tags_detected(self):
        detector = LabelStateChangeDetector()
        graph = _make_graph()
        ev1 = _make_evidence(graph, "ev_1", label="person standing")
        ev2 = _make_evidence(graph, "ev_2", label="cat running")

        assessment = detector.detect(ev1, ev2)
        assert assessment.is_transition
        assert assessment.change_magnitude > 0.5

    def test_partial_overlap_moderate_novelty(self):
        detector = LabelStateChangeDetector()
        graph = _make_graph()
        ev1 = _make_evidence(graph, "ev_1", label="person standing outside")
        ev2 = _make_evidence(graph, "ev_2", label="person running outside")

        assessment = detector.detect(ev1, ev2)
        # Partial overlap → moderate change
        assert assessment.change_magnitude > 0.0
        assert assessment.change_magnitude < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Large Time Gap Recovery Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLargeTimeGapRecovery:
    """Test that large time gaps recover novelty to near 1.0."""

    def test_sixty_second_gap_recovers_novelty(self):
        graph = _make_graph()
        belief = _make_belief(graph)
        policy = NoveltyPolicy(half_life_seconds=5.0)
        model = TemporalEvidenceModel(graph, policy=policy)

        base_time = 1000.0
        ev1 = _make_evidence(graph, "ev_1", label="person detected", timestamp=base_time)
        model.assess(ev1, belief)
        ev1.incorporated_transitions["belief_1"] = "trans_1"
        ev1.last_incorporated_at = base_time
        graph.upsert_node(ev1)

        # 60 seconds later (12 half-lives → novelty ≈ 0.9998)
        ev2 = _make_evidence(graph, "ev_2", label="person detected", timestamp=base_time + 60.0)
        assessment = model.assess(ev2, belief)

        assert assessment.temporal_novelty > 0.99
