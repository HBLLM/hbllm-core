"""Unit tests for physical deadlock contradiction detection in ContradictionOperator."""

from __future__ import annotations

from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    FrozenGraphView,
    ProblemType,
    ReasoningProblem,
)
from hbllm.brain.reasoning.operators.contradiction import ContradictionOperator
from hbllm.hcir.graph import CognitiveGraph, GoalNode, PhysicalEntityNode, PredictionNode


def test_contradiction_operator_detects_physical_deadlock() -> None:
    """Verify ContradictionOperator emits CONTRADICTS when an entity is physically deadlocked."""
    graph = CognitiveGraph()

    # Goal: reach target
    graph.upsert_node(GoalNode(id="g_sokoban", description="Deliver box to target"))

    # Entity trapped in deadlock
    graph.upsert_node(
        PhysicalEntityNode(
            id="box_1",
            entity_name="box",
            status="deadlocked",
            properties={"deadlock": True},
        )
    )

    operator = ContradictionOperator()
    problem = ReasoningProblem(
        problem_id="p1",
        problem_type=ProblemType.CONTRADICTION,
        description="Check consistency",
    )
    context = CognitiveContext(graph_view=FrozenGraphView.from_graph(graph), problem=problem)

    result = operator.execute(problem, context)

    assert result.status.value == "success"
    assert result.conclusions["contradictions_found"] >= 1
    found_deadlock_contradiction = any(
        "deadlock" in c["description"].lower() for c in result.conclusions["contradictions"]
    )
    assert found_deadlock_contradiction is True
    assert result.confidence == 1.0


def test_contradiction_operator_detects_prediction_deadlock() -> None:
    """Verify ContradictionOperator emits CONTRADICTS when a prediction indicates fatal deadlock."""
    graph = CognitiveGraph()

    # Goal: reach target
    graph.upsert_node(GoalNode(id="g_sokoban", description="Deliver box to target"))

    # Prediction indicating fatal deadlock outcome
    graph.upsert_node(
        PredictionNode(
            id="pred_move_fatal",
            claim="Outcome of MOVE_UP",
            predicted_outcome="{'spatial_outcome': {'deadlock': True}}",
            properties={
                "predicted_state": {
                    "spatial_outcome": {"deadlock": True, "deadlocked_entities": ["box_1"]}
                }
            },
        )
    )

    operator = ContradictionOperator()
    problem = ReasoningProblem(
        problem_id="p1",
        problem_type=ProblemType.CONTRADICTION,
        description="Check consistency",
    )
    context = CognitiveContext(graph_view=FrozenGraphView.from_graph(graph), problem=problem)

    result = operator.execute(problem, context)

    assert result.status.value == "success"
    assert result.conclusions["contradictions_found"] >= 1
    found_deadlock_contradiction = any(
        "deadlock" in c["description"].lower() for c in result.conclusions["contradictions"]
    )
    assert found_deadlock_contradiction is True
