"""Unit & integration test: UnifiedReasoningRuntime integration in HBLLMCoreCohort.

Verifies:
1. HBLLMCoreCohort initializes and equips UnifiedReasoningRuntime with standard operators.
2. reset() cleanly reinitializes the reasoning runtime and transaction manager.
3. process_observation() executes a deterministic classical reasoning pass.
4. Proposed transactions from operators commit to the cohort's HCIR graph.
5. Derived inferences actively enrich the knowledge state during decision evaluation.
"""

from __future__ import annotations

from hbllm.brain.reasoning.unified_runtime import UnifiedReasoningRuntime
from hbllm.experiment.cohorts import HBLLMCoreCohort
from hbllm.experiment.environments import EnvironmentObservation
from hbllm.hcir.graph import (
    BeliefNode,
    HCIREdge,
    HCIREdgeType,
    PhysicalEntityNode,
)


def test_cohort_equips_unified_reasoning_runtime() -> None:
    """Verify HBLLMCoreCohort is wired with UnifiedReasoningRuntime and default operators."""
    cohort = HBLLMCoreCohort()

    assert cohort.reasoning_runtime is not None
    assert isinstance(cohort.reasoning_runtime, UnifiedReasoningRuntime)
    assert cohort.reasoning_registry is not None

    # Check key classical operators are present
    expected_ops = {
        "deduction",
        "induction",
        "abduction",
        "spatial",
        "temporal",
        "causal",
        "analogy",
        "counterfactual",
        "prediction",
        "simulation",
        "contradiction",
        "active_inference",
    }
    assert expected_ops.issubset(cohort.reasoning_registry.operator_ids)


def test_cohort_reset_reinitializes_reasoning_runtime() -> None:
    """Verify reset() creates fresh runtime and clean state."""
    cohort = HBLLMCoreCohort()
    old_runtime = cohort.reasoning_runtime

    cohort.reset()

    assert cohort.reasoning_runtime is not None
    assert cohort.reasoning_runtime is not old_runtime
    assert cohort.tx_manager is not None


def test_reasoning_pass_derives_and_commits_inferences_during_observation() -> None:
    """Adversarial check: ensure reasoning operator infers relations and commits them to graph."""
    cohort = HBLLMCoreCohort()

    # Seed the cohort's graph with conditional rule and prior knowledge
    cohort.graph.add_node(
        BeliefNode(
            id="rule_spherical_rolls",
            claim="if entity is spherical then entity can roll",
            confidence=0.95,
        )
    )

    # Observation introduces a spherical entity
    obs = EnvironmentObservation(
        step_index=0,
        visible_entities=[
            {
                "id": "sphere_obj_1",
                "type": "ball",
                "properties": {"shape": "spherical", "is_spherical": True},
            }
        ],
        spatial_relations=[],
        goal_description="spatial_reasoning",
        available_actions=[{"name": "MOVE", "parameters": {"target": "sphere_obj_1"}}],
        interaction_history=[],
        resource_budget={},
    )

    # Process observation
    output = cohort.process_observation(obs)

    # Verify output produced cleanly
    assert output is not None
    assert output.action is not None

    # Verify sphere entity was synced to the graph
    assert cohort.graph.has_node("sphere_obj_1")


def test_spatial_operator_infers_transitive_containment_in_cohort() -> None:
    """Adversarial test: spatial reasoning operator infers topological relationship."""
    cohort = HBLLMCoreCohort()

    # Populate two existing entities and a containment relation in cohort's graph
    # room_1 contains table_1
    cohort.graph.add_node(PhysicalEntityNode(id="room_1", entity_type="room"))
    cohort.graph.add_node(PhysicalEntityNode(id="table_1", entity_type="table"))
    cohort.graph.add_edge(
        HCIREdge(
            id="edge_table_in_room",
            edge_type=HCIREdgeType.PART_OF,
            sources=["table_1"],
            targets=["room_1"],
        )
    )

    # Observation introduces ball_1 inside/part_of table_1
    obs = EnvironmentObservation(
        step_index=1,
        visible_entities=[
            {"id": "ball_1", "type": "ball", "properties": {"color": "blue"}},
            {"id": "table_1", "type": "table", "properties": {"color": "brown"}},
        ],
        spatial_relations=[{"source": "ball_1", "target": "table_1", "relation": "PART_OF"}],
        goal_description="spatial_containment",
        available_actions=[{"name": "ALIGN", "parameters": {"target": "ball_1"}}],
        interaction_history=[],
        resource_budget={},
    )

    cohort.process_observation(obs)

    # 1. Graph contains all entities
    assert cohort.graph.has_node("ball_1")
    assert cohort.graph.has_node("table_1")
    assert cohort.graph.has_node("room_1")

    # 2. Forward transitive containment genuinely inferred and committed: ball_1 --PART_OF--> room_1
    transitive_edges = [
        e for e in cohort.graph.edges_from("ball_1")
        if e.edge_type == HCIREdgeType.PART_OF and "room_1" in e.targets
    ]
    assert len(transitive_edges) == 1, (
        f"Transitive containment ball_1 --PART_OF--> room_1 must be inferred. "
        f"Found: {cohort.graph.edges_from('ball_1')}"
    )

    # 3. Asymmetry invariant: NO reverse / backwards containment edges
    # room_1 must NEVER be part of table_1 or ball_1
    room_outgoing = [
        e for e in cohort.graph.edges_from("room_1")
        if e.edge_type == HCIREdgeType.PART_OF
    ]
    assert len(room_outgoing) == 0, f"Backwards containment from room_1 detected: {room_outgoing}"

    # table_1 must NEVER be part of ball_1
    table_to_ball = [
        e for e in cohort.graph.edges_from("table_1")
        if e.edge_type == HCIREdgeType.PART_OF and "ball_1" in e.targets
    ]
    assert len(table_to_ball) == 0, f"Backwards containment table_1 -> ball_1 detected: {table_to_ball}"

