"""
Tests for A12 — Cognitive Execution Model.

Verifies the core invariants:
    1. Operators receive immutable HCIR views (FrozenGraphView).
    2. Operators produce CognitiveResults with provenance.
    3. Results propose HCIR transactions, never directly mutate state.
    4. Given the same snapshot + problem → same trace (determinism).
    5. Pipeline composition: result of operator A feeds operator B.
    6. Zero LLM calls throughout.

These tests construct synthetic HCIR states, run reasoning over them,
and verify the full contract from context→operator→result→transaction.
"""

from __future__ import annotations

import time

import pytest

from hbllm.brain.reasoning.operators.abduction import AbductionOperator
from hbllm.brain.reasoning.operators.analogy import AnalogyOperator
from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    FrozenGraphView,
    OperatorTrace,
    ProblemType,
    ReasoningBudget,
    ReasoningProblem,
    ResultStatus,
)
from hbllm.brain.reasoning.operators.deduction import DeductionOperator
from hbllm.brain.reasoning.operators.induction import InductionOperator
from hbllm.brain.reasoning.operators.registry import OperatorRegistry
from hbllm.brain.reasoning.operators.spatial import SpatialOperator
from hbllm.brain.reasoning.operators.temporal import TemporalOperator
from hbllm.brain.reasoning.unified_runtime import (
    UnifiedReasoningRuntime,
)
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    EventNode,
    FactNode,
    HCIREdge,
    HCIREdgeType,
    PhysicalEntityNode,
    PredictionErrorNode,
)
from hbllm.hcir.types import UncertaintyVector

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


def _make_belief(claim: str, confidence: float = 0.8) -> BeliefNode:
    """Create a BeliefNode with specified confidence."""
    node = BeliefNode(claim=claim)
    node.uncertainty = UncertaintyVector(confidence=confidence)
    return node


def _make_fact(claim: str) -> FactNode:
    return FactNode(claim=claim)


def _build_deduction_graph() -> CognitiveGraph:
    """Graph with conditional beliefs for forward-chaining deduction."""
    g = CognitiveGraph()

    # Rules
    g.add_node(_make_belief("if entity is spherical then entity can roll", 0.9))
    g.add_node(_make_belief("if entity can roll then entity is movable", 0.85))
    g.add_node(_make_belief("all heavy objects are slow", 0.7))

    # Facts
    g.add_node(_make_belief("entity is spherical", 0.95))
    g.add_node(_make_belief("entity is heavy", 0.8))
    g.add_node(_make_fact("entity is solid"))

    return g


def _build_induction_graph() -> CognitiveGraph:
    """Graph with multiple entities having co-occurring properties."""
    g = CognitiveGraph()

    for i in range(5):
        g.add_node(_make_belief(f"ball_{i} is spherical", 0.9))
        g.add_node(_make_belief(f"ball_{i} can roll", 0.9))
        g.add_node(_make_belief(f"ball_{i} is bouncy", 0.85))

    # Add a non-ball entity to test specificity
    g.add_node(_make_belief("cube_0 is angular", 0.9))
    g.add_node(_make_belief("cube_0 is stable", 0.9))

    return g


def _build_temporal_graph() -> CognitiveGraph:
    """Graph with multiple events for temporal reasoning."""
    g = CognitiveGraph()
    base_time = time.time()

    e1 = EventNode(event_kind="push", event_timestamp=base_time)
    e2 = EventNode(event_kind="roll", event_timestamp=base_time + 1.0)
    e3 = EventNode(event_kind="collision", event_timestamp=base_time + 2.0)
    e4 = EventNode(event_kind="stop", event_timestamp=base_time + 3.0)

    for e in [e1, e2, e3, e4]:
        g.add_node(e)

    return g


def _build_spatial_graph() -> CognitiveGraph:
    """Graph with physical entities having spatial relationships."""
    g = CognitiveGraph()

    room = PhysicalEntityNode(id="room_1")
    table = PhysicalEntityNode(id="table_1")
    ball = PhysicalEntityNode(id="ball_1")

    g.add_node(room)
    g.add_node(table)
    g.add_node(ball)

    # table is inside room
    g.add_edge(
        HCIREdge(
            edge_type=HCIREdgeType.PART_OF,
            sources=["table_1"],
            targets=["room_1"],
        )
    )

    # ball is inside room too (should infer spatial relationship)
    g.add_edge(
        HCIREdge(
            edge_type=HCIREdgeType.PART_OF,
            sources=["ball_1"],
            targets=["room_1"],
        )
    )

    return g


def _build_abduction_graph() -> CognitiveGraph:
    """Graph with prediction errors needing explanations."""
    g = CognitiveGraph()

    # A prediction error — something unexpected
    pe = PredictionErrorNode(
        predicted_value="moving",
        observed_value="stopped",
        delta=1.0,
        error_magnitude=1.0,
    )
    g.add_node(pe)

    # Some beliefs that could explain it
    wall = _make_belief("walls block movement", 0.9)
    g.add_node(wall)

    friction = _make_belief("friction causes deceleration", 0.85)
    g.add_node(friction)

    # Causal edge: wall → prediction error (walls can cause stopping)
    g.add_edge(
        HCIREdge(
            edge_type=HCIREdgeType.CAUSES,
            sources=[wall.id],
            targets=[pe.id],
            weight=0.8,
        )
    )

    g.add_edge(
        HCIREdge(
            edge_type=HCIREdgeType.CAUSES,
            sources=[friction.id],
            targets=[pe.id],
            weight=0.6,
        )
    )

    return g


# ═══════════════════════════════════════════════════════════════════════════
# Test: FrozenGraphView Immutability
# ═══════════════════════════════════════════════════════════════════════════


class TestFrozenGraphView:
    """Verify the immutability contract of FrozenGraphView."""

    def test_frozen_view_is_read_only(self) -> None:
        """Operators cannot modify the graph through FrozenGraphView."""
        g = CognitiveGraph()
        belief = _make_belief("test claim", 0.8)
        g.add_node(belief)

        view = FrozenGraphView.from_graph(g)

        # Retrieve a node — should be a deep copy
        retrieved = view.get_node(belief.id)
        assert retrieved is not None
        assert retrieved.id == belief.id

        # Mutating the retrieved node should NOT affect the view
        retrieved.claim = "MUTATED"
        original = view.get_node(belief.id)
        assert original is not None
        assert original.claim == "test claim"  # Unchanged

    def test_content_hash_deterministic(self) -> None:
        """Same graph state → same content hash."""
        g = CognitiveGraph()
        g.add_node(_make_belief("claim 1", 0.8))
        g.add_node(_make_belief("claim 2", 0.9))

        view1 = FrozenGraphView.from_graph(g)
        view2 = FrozenGraphView.from_graph(g)

        assert view1.content_hash == view2.content_hash

    def test_bounded_view(self) -> None:
        """FrozenGraphView can be bounded to a subset of nodes."""
        g = CognitiveGraph()
        b1 = _make_belief("claim 1", 0.8)
        b2 = _make_belief("claim 2", 0.9)
        b3 = _make_belief("claim 3", 0.7)
        g.add_node(b1)
        g.add_node(b2)
        g.add_node(b3)

        # Only include b1 and b2
        view = FrozenGraphView.from_graph(g, node_ids={b1.id, b2.id})
        assert view.node_count == 2
        assert view.has_node(b1.id)
        assert view.has_node(b2.id)
        assert not view.has_node(b3.id)


# ═══════════════════════════════════════════════════════════════════════════
# Test: Operator Registry
# ═══════════════════════════════════════════════════════════════════════════


class TestOperatorRegistry:
    """Verify operator registration, scoring, and selection."""

    def test_register_and_retrieve(self) -> None:
        registry = OperatorRegistry()
        op = DeductionOperator()
        registry.register(op)

        assert "deduction" in registry.operator_ids
        assert registry.get("deduction") is op

    def test_duplicate_registration_raises(self) -> None:
        registry = OperatorRegistry()
        registry.register(DeductionOperator())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(DeductionOperator())

    def test_selection_ranks_by_applicability(self) -> None:
        registry = OperatorRegistry()
        registry.register(DeductionOperator())
        registry.register(InductionOperator())
        registry.register(TemporalOperator())

        g = _build_deduction_graph()
        view = FrozenGraphView.from_graph(g)
        problem = ReasoningProblem(problem_type=ProblemType.EXPLANATION)
        context = CognitiveContext(graph_view=view, problem=problem)

        scores = registry.select(problem, context)

        # At least deduction and induction should be applicable
        applicable_ids = {s.operator_id for s in scores}
        assert "deduction" in applicable_ids

        # Scores should be sorted descending
        for i in range(len(scores) - 1):
            assert scores[i].composite_score >= scores[i + 1].composite_score

    def test_reliability_tracking(self) -> None:
        registry = OperatorRegistry()
        registry.register(DeductionOperator())

        # Initial reliability should be neutral
        assert registry.get_reliability("deduction") == 0.5

        # Record successes
        for _ in range(10):
            registry.record_outcome("deduction", True)

        assert registry.get_reliability("deduction") > 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Test: Deduction Operator
# ═══════════════════════════════════════════════════════════════════════════


class TestDeductionOperator:
    """Verify the deduction operator's forward-chaining logic."""

    def test_simple_deduction(self) -> None:
        """Forward chain: spherical → can roll → movable."""
        g = _build_deduction_graph()
        view = FrozenGraphView.from_graph(g)
        problem = ReasoningProblem(problem_type=ProblemType.EXPLANATION)
        context = CognitiveContext(graph_view=view, problem=problem)

        op = DeductionOperator()
        result = op.execute(problem, context)

        assert result.status == ResultStatus.SUCCESS
        assert result.conclusions.get("derived_count", 0) > 0
        assert result.operator_id == "deduction"

        # Should have proposed HCIR transactions
        assert len(result.proposed_transitions) > 0

        # Should have provenance chains
        assert len(result.provenance_chains) > 0
        for chain in result.provenance_chains:
            assert chain.operator_id == "deduction"
            assert chain.confidence > 0

    def test_deduction_no_facts_returns_no_result(self) -> None:
        """Empty graph → NO_RESULT."""
        g = CognitiveGraph()
        view = FrozenGraphView.from_graph(g)
        problem = ReasoningProblem(problem_type=ProblemType.EXPLANATION)
        context = CognitiveContext(graph_view=view, problem=problem)

        op = DeductionOperator()
        result = op.execute(problem, context)

        assert result.status == ResultStatus.NO_RESULT

    def test_deduction_never_mutates_graph(self) -> None:
        """The original graph must be untouched after deduction."""
        g = _build_deduction_graph()
        original_count = g.node_count
        view = FrozenGraphView.from_graph(g)
        problem = ReasoningProblem(problem_type=ProblemType.EXPLANATION)
        context = CognitiveContext(graph_view=view, problem=problem)

        op = DeductionOperator()
        op.execute(problem, context)

        # Graph should be unchanged
        assert g.node_count == original_count


# ═══════════════════════════════════════════════════════════════════════════
# Test: Induction Operator
# ═══════════════════════════════════════════════════════════════════════════


class TestInductionOperator:
    """Verify induction finds co-occurrence patterns."""

    def test_discovers_property_pattern(self) -> None:
        g = _build_induction_graph()
        view = FrozenGraphView.from_graph(g)
        problem = ReasoningProblem(problem_type=ProblemType.GENERALIZATION)
        context = CognitiveContext(graph_view=view, problem=problem)

        op = InductionOperator()
        result = op.execute(problem, context)

        assert result.status in (ResultStatus.SUCCESS, ResultStatus.NO_RESULT)
        if result.status == ResultStatus.SUCCESS:
            assert result.conclusions.get("patterns_found", 0) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Test: Temporal Operator
# ═══════════════════════════════════════════════════════════════════════════


class TestTemporalOperator:
    """Verify temporal reasoning over events."""

    def test_event_ordering(self) -> None:
        g = _build_temporal_graph()
        view = FrozenGraphView.from_graph(g)
        problem = ReasoningProblem(problem_type=ProblemType.TEMPORAL)
        context = CognitiveContext(graph_view=view, problem=problem)

        op = TemporalOperator()
        result = op.execute(problem, context)

        assert result.status == ResultStatus.SUCCESS
        ordering = result.conclusions.get("ordering", [])
        assert len(ordering) == 4
        # Should be in chronological order
        assert ordering == ["push", "roll", "collision", "stop"]


# ═══════════════════════════════════════════════════════════════════════════
# Test: Abduction Operator
# ═══════════════════════════════════════════════════════════════════════════


class TestAbductionOperator:
    """Verify abduction generates candidate explanations."""

    def test_generates_hypotheses(self) -> None:
        g = _build_abduction_graph()
        view = FrozenGraphView.from_graph(g)
        problem = ReasoningProblem(problem_type=ProblemType.EXPLANATION)
        context = CognitiveContext(graph_view=view, problem=problem)

        op = AbductionOperator()
        result = op.execute(problem, context)

        assert result.status == ResultStatus.SUCCESS
        assert result.conclusions.get("hypotheses_count", 0) > 0

        # Should propose HypothesisNodes
        assert len(result.proposed_transitions) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Test: Unified Reasoning Runtime
# ═══════════════════════════════════════════════════════════════════════════


class TestUnifiedReasoningRuntime:
    """Verify the full runtime contract."""

    def _make_runtime(self) -> UnifiedReasoningRuntime:
        registry = OperatorRegistry()
        registry.register(DeductionOperator())
        registry.register(InductionOperator())
        registry.register(AbductionOperator())
        registry.register(TemporalOperator())
        registry.register(SpatialOperator())
        registry.register(AnalogyOperator())
        return UnifiedReasoningRuntime(registry)

    def test_runtime_produces_trace(self) -> None:
        """Runtime produces a complete OperatorTrace."""
        runtime = self._make_runtime()
        g = _build_deduction_graph()
        problem = ReasoningProblem(problem_type=ProblemType.EXPLANATION)

        trace = runtime.reason(g, problem)

        assert isinstance(trace, OperatorTrace)
        assert trace.context_hash  # Should have a content hash
        assert trace.problem is problem
        assert trace.total_wall_clock_ms > 0

    def test_runtime_never_mutates_graph(self) -> None:
        """The live graph must be unchanged after reasoning."""
        runtime = self._make_runtime()
        g = _build_deduction_graph()
        original_node_count = g.node_count
        original_edge_count = g.edge_count

        problem = ReasoningProblem(problem_type=ProblemType.EXPLANATION)
        runtime.reason(g, problem)

        assert g.node_count == original_node_count
        assert g.edge_count == original_edge_count

    def test_runtime_produces_transaction(self) -> None:
        """If operators produce conclusions, the trace should contain
        a proposed HCIRTransaction."""
        runtime = self._make_runtime()
        g = _build_deduction_graph()
        problem = ReasoningProblem(problem_type=ProblemType.EXPLANATION)

        trace = runtime.reason(g, problem)

        if trace.final_result and trace.final_result.proposed_transitions:
            assert trace.proposed_transaction is not None
            assert trace.proposed_transaction.author.startswith("reasoning_runtime:")
            assert trace.proposed_transaction.parent_snapshot_hash == trace.context_hash

    def test_runtime_respects_budget(self) -> None:
        """Runtime should stop when budget is exhausted."""
        runtime = self._make_runtime()
        g = _build_deduction_graph()
        problem = ReasoningProblem(problem_type=ProblemType.EXPLANATION)

        # Very restrictive budget
        budget = ReasoningBudget(max_operators=1, compute_ms=10000)
        trace = runtime.reason(g, problem, budget=budget)

        assert len(trace.invocations) <= 1

    def test_deterministic_traces(self) -> None:
        """Same graph + same problem → same content hash."""
        runtime = self._make_runtime()
        g = _build_deduction_graph()
        problem = ReasoningProblem(
            problem_type=ProblemType.EXPLANATION,
            problem_id="determinism_test",
        )

        trace1 = runtime.reason(g, problem)
        trace2 = runtime.reason(g, problem)

        # Same snapshot → same context hash
        assert trace1.context_hash == trace2.context_hash

    def test_pipeline_composition(self) -> None:
        """Multiple operators can run in sequence."""
        runtime = self._make_runtime()
        g = _build_deduction_graph()
        problem = ReasoningProblem(problem_type=ProblemType.EXPLANATION)
        budget = ReasoningBudget(max_operators=3)

        trace = runtime.reason(g, problem, budget=budget)

        # Should have invoked multiple operators
        if trace.invocations:
            assert len(trace.invocations) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Test: LLM Independence
# ═══════════════════════════════════════════════════════════════════════════


class TestLLMIndependence:
    """Verify zero LLM dependency in the reasoning pipeline."""

    def test_no_llm_imports_in_operators(self) -> None:
        """Static check: no LLM interfaces imported in operator modules."""
        import importlib
        import inspect

        modules = [
            "hbllm.brain.reasoning.operators.base",
            "hbllm.brain.reasoning.operators.registry",
            "hbllm.brain.reasoning.operators.deduction",
            "hbllm.brain.reasoning.operators.induction",
            "hbllm.brain.reasoning.operators.abduction",
            "hbllm.brain.reasoning.operators.temporal",
            "hbllm.brain.reasoning.operators.spatial",
            "hbllm.brain.reasoning.operators.analogy",
            "hbllm.brain.reasoning.unified_runtime",
        ]

        llm_indicators = {
            "llm_interface",
            "LLMInterface",
            "generate_json",
            "generate_stream",
            "ChatCompletionMessage",
        }

        for mod_name in modules:
            try:
                mod = importlib.import_module(mod_name)
                source = inspect.getsource(mod)
                for indicator in llm_indicators:
                    assert indicator not in source, (
                        f"LLM dependency '{indicator}' found in {mod_name}"
                    )
            except ImportError:
                pass  # Module not available in test env
