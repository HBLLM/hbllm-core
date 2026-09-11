"""Unit tests for Developmental Perception Invariance and Object Permanence."""

from __future__ import annotations

import pytest

from hbllm.hcir.graph import CognitiveGraph, PhysicalEntityNode
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.perception import (
    DevelopmentalPerceptionAdapter,
    SemanticLeakageViolationError,
)


def test_neutral_ids_and_zero_semantic_labels():
    env = BabyWorldEnvironment(seed=42)
    obs = env.reset("confounded_train_world")

    adapter = DevelopmentalPerceptionAdapter()
    graph = adapter.observe(obs)

    # Node IDs should be neutral (e.g. entity_001, entity_002, agent_effector)
    for node in graph.all_nodes():
        assert node.id == "agent_effector" or node.id.startswith("entity_")

    # Invariant check passes without raising SemanticLeakageViolationError
    adapter.validate_perception_invariance(graph)


def test_semantic_leakage_detection():
    adapter = DevelopmentalPerceptionAdapter()
    graph = CognitiveGraph()

    # Intentionally inject forbidden semantic key
    bad_node = PhysicalEntityNode(
        id="entity_999",
        properties={
            "shape": "sphere",
            "concept": "BALL",  # FORBIDDEN
        },
    )
    graph.add_node(bad_node)

    with pytest.raises(SemanticLeakageViolationError) as exc_info:
        adapter.validate_perception_invariance(graph)
    assert "Semantic leakage detected" in str(exc_info.value)


def test_affordance_leakage_detection():
    adapter = DevelopmentalPerceptionAdapter()
    graph = CognitiveGraph()

    # Intentionally inject forbidden affordance property
    bad_node = PhysicalEntityNode(
        id="entity_888",
        properties={
            "shape": "box",
            "affordance": "ROLLABLE",  # FORBIDDEN
        },
    )
    graph.add_node(bad_node)

    with pytest.raises(SemanticLeakageViolationError) as exc_info:
        adapter.validate_perception_invariance(graph)
    assert "affordance" in str(exc_info.value)


def test_object_permanence_in_perception():
    env = BabyWorldEnvironment(seed=42)
    adapter = DevelopmentalPerceptionAdapter()

    # Step 1: Observe ball in train world (it becomes known)
    obs_1 = env.reset("confounded_train_world")
    graph_1 = adapter.observe(obs_1)
    assert any(
        n.properties.get("is_observed") is True
        for n in graph_1.all_nodes()
        if n.id != "agent_effector"
    )

    # Step 2: Switch to occlusion world where ball is hidden
    obs_2 = env.reset("occlusion_permanence_world")
    # Manually register the hidden ball into adapter beliefs to simulate it moving behind the screen
    adapter._unobserved_beliefs["obj_hidden_ball"] = {
        "neutral_id": "entity_hidden",
        "last_position": (1.8, 0.0),
        "shape": "ball",
        "color": "green",
        "mass_sensation": 1.0,
        "last_seen_step": 0,
    }

    graph_2 = adapter.observe(obs_2)
    # The hidden ball should exist in the graph with is_observed=False, occluded=True
    assert graph_2.has_node("entity_hidden")
    hidden_node = graph_2.get_node("entity_hidden")
    assert hidden_node is not None
    assert hidden_node.properties.get("is_observed") is False
    assert hidden_node.properties.get("occluded") is True
