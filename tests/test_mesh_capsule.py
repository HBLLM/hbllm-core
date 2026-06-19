"""Tests for Contextual Cognitive Capsules and Locality Engine."""

import time

from hbllm.brain.mesh.capsule import TaskCapsule
from hbllm.brain.mesh.locality import CognitiveLocalityEngine
from hbllm.brain.mesh.registry import TaskPriorityClass


def test_capsule_validity_and_loop_prevention():
    capsule = TaskCapsule()
    assert capsule.is_valid is True

    # Simulate delegation hops
    capsule.add_hop("node_A")
    capsule.add_hop("node_B")
    capsule.add_hop("node_C")

    assert capsule.delegation_depth == 3
    # Max depth reached
    assert capsule.is_valid is False

    # Test expiration
    capsule2 = TaskCapsule(expires_at=time.time() - 10.0)
    assert capsule2.is_valid is False


def test_cognitive_locality_engine():
    engine = CognitiveLocalityEngine(local_node_id="phone_1")

    world_state = {"location": "home", "biometric_heart_rate": 75, "temperature": 22}

    capsule = engine.create_task_capsule(
        goal_id="g1",
        target_node_id="cloud_server_1",
        authority_node="phone_1",
        priority=TaskPriorityClass.BACKGROUND,
        world_state_subgraph=world_state,
        causal_edges=[],
        utility_constraints={},
        permissions_scope=[],
    )

    assert capsule.ownership.origin_node == "phone_1"
    assert capsule.ownership.execution_node == "cloud_server_1"

    # Verify locality filtering (biometrics stripped for cloud)
    assert "location" in capsule.required_entities
    assert "temperature" in capsule.required_entities
    assert "biometric_heart_rate" not in capsule.required_entities
