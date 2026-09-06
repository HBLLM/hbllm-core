"""Unit tests verifying end-to-end contradiction detection and epistemics without an LLM."""

import pytest

from hbllm.brain.epistemics.contradiction_engine import ContradictionEngine
from hbllm.brain.epistemics.hypothesis_builder import HypothesisBuilder
from hbllm.brain.epistemics.idea_generator import IdeaGenerator
from hbllm.brain.epistemics.prediction_tracker import PredictionTracker
from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    FrozenGraphView,
    ProblemType,
    ReasoningProblem,
)
from hbllm.brain.reasoning.operators.contradiction import ContradictionOperator
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    ContradictionNode,
    HCIREdgeType,
)
from hbllm.hcir.types import UncertaintyVector


@pytest.mark.asyncio
async def test_contradiction_engine_detects_belief_conflict_without_llm():
    """Verify that ContradictionEngine detects contradictions with llm=None and updates the graph."""
    graph = CognitiveGraph()

    b1 = BeliefNode(
        id="belief_door_locked",
        claim="the front door is locked",
        uncertainty=UncertaintyVector(confidence=0.95),
    )
    b2 = BeliefNode(
        id="belief_door_unlocked",
        claim="the front door is unlocked",
        uncertainty=UncertaintyVector(confidence=0.95),
    )
    graph.upsert_node(b1)
    graph.upsert_node(b2)

    # Strictly no LLM provided (the no-LLM deployment target)
    engine = ContradictionEngine(graph=graph, llm=None)
    reports = await engine.scan_for_contradictions()

    # Must find the contradiction
    assert len(reports) >= 1
    report = next(r for r in reports if r.contradiction_type == "belief_conflict")
    assert {report.claim_a_id, report.claim_b_id} == {b1.id, b2.id}
    assert report.investigation_priority > 0.7

    # Must have committed a ContradictionNode to the graph
    contra_nodes = [n for n in graph.all_nodes() if isinstance(n, ContradictionNode)]
    assert len(contra_nodes) >= 1
    contra = contra_nodes[0]
    assert {contra.claim_a_id, contra.claim_b_id} == {b1.id, b2.id}

    # Must have committed an HCIREdgeType.CONTRADICTS edge to the graph
    edges = [e for e in graph.all_edges() if e.edge_type == HCIREdgeType.CONTRADICTS]
    assert len(edges) >= 1
    edge = edges[0]
    assert (b1.id in edge.sources and b2.id in edge.targets) or (
        b2.id in edge.sources and b1.id in edge.targets
    )


@pytest.mark.asyncio
async def test_contradiction_operator_detects_negation_without_llm():
    """Verify ContradictionOperator finds structural negation contradictions."""
    graph = CognitiveGraph()
    b1 = BeliefNode(
        id="b1",
        claim="the reactor core is safe",
        uncertainty=UncertaintyVector(confidence=0.9),
    )
    b2 = BeliefNode(
        id="b2",
        claim="the reactor core is unsafe",
        uncertainty=UncertaintyVector(confidence=0.9),
    )
    graph.upsert_node(b1)
    graph.upsert_node(b2)

    problem = ReasoningProblem(
        problem_id="p1",
        problem_type=ProblemType.CONTRADICTION,
        description="Scan for contradictions",
    )
    view = FrozenGraphView.from_graph(graph)
    context = CognitiveContext(graph_view=view, problem=problem)
    operator = ContradictionOperator()

    result = operator.execute(problem, context)
    assert result.conclusions["contradictions_found"] >= 1
    assert any("reactor core" in c["description"] for c in result.conclusions["contradictions"])


@pytest.mark.asyncio
async def test_prediction_tracker_conflict_detection_without_llm():
    """Verify PredictionTracker detects conflicting outcomes using consolidated contradiction utils."""
    graph = CognitiveGraph()
    tracker = PredictionTracker(graph=graph, llm=None)

    # Direct conflict via opposites
    assert tracker._outcomes_conflict("increase", "decrease")
    assert tracker._outcomes_conflict("system is locked", "system is unlocked")
    assert not tracker._outcomes_conflict("increase", "increase")
    assert not tracker._outcomes_conflict("system is locked", "door is locked")

    # Matching logic: contradictory outcome must not match prediction
    assert not await tracker._outcomes_match("increase", "decrease")
    assert not await tracker._outcomes_match("the valve is open", "the valve is closed")
    assert await tracker._outcomes_match("the valve is open", "the valve is open")


@pytest.mark.asyncio
async def test_idea_generator_template_generates_concrete_resolutions():
    """Verify IdeaGenerator produces concrete resolution hypotheses from contradiction explanations."""
    graph = CognitiveGraph()
    b1 = BeliefNode(id="b1", claim="motor speed is high")
    b2 = BeliefNode(id="b2", claim="motor speed is low")
    graph.upsert_node(b1)
    graph.upsert_node(b2)

    contra_node = ContradictionNode(
        id="contra_test",
        claim_a_id="b1",
        claim_b_id="b2",
        contradiction_type="belief_conflict",
        possible_explanations=["Antonym conflict: 'high' vs 'low' on shared subject 'motor speed'"],
    )
    graph.upsert_node(contra_node)

    gen = IdeaGenerator(graph=graph, llm=None)
    ideas = gen._template_generate_from_contradiction(contra_node)

    assert len(ideas) >= 2
    # At least one idea must reference the actual resolution/explanation
    assert any("Resolve conflict" in idea.claim and "motor speed" in idea.claim for idea in ideas)


@pytest.mark.asyncio
async def test_hypothesis_builder_does_not_discard_contradictory_hypotheses():
    """Verify HypothesisBuilder._is_duplicate does NOT drop opposing/contradictory hypotheses."""
    graph = CognitiveGraph()
    builder = HypothesisBuilder(graph=graph, llm=None)

    existing = ["the front door is locked"]
    new_opposing_claim = "the front door is unlocked"

    # Even though word overlap is high, they are opposite hypotheses and must NOT be deduplicated away
    is_dup = await builder._is_duplicate(new_opposing_claim, existing)
    assert not is_dup
