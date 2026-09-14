"""Unit tests for Stage D4: Affordance Discovery Engine."""

from __future__ import annotations

from plugins.developmental_adapter.affordance_discovery import AffordanceDiscoveryEngine
from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.perception import DevelopmentalPerceptionAdapter


def test_affordance_discovery_blank_brain():
    """Verify that a blank brain discovers ROLLABLE, SLIDABLE, and GRASPABLE affordances."""
    env = BabyWorldEnvironment(seed=42)
    env.reset("affordance_discovery_world")

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = AffordanceDiscoveryEngine(substrate, perception, env)

    # 1. Hypotheses generation across observed shapes
    hypotheses = engine.observe_and_generate_hypotheses()
    assert len(hypotheses) > 0
    shapes = {h.entity_shape for h in hypotheses}
    assert "ball" in shapes
    assert "block" in shapes

    # 2. Interventional discovery
    confirmed = engine.discover_affordances(max_interventions=20)
    assert "ball" in confirmed
    assert "block" in confirmed

    # Balls afford ROLLABLE and SLIDABLE and GRASPABLE
    assert "ROLLABLE" in confirmed["ball"]
    assert "SLIDABLE" in confirmed["ball"]
    assert "GRASPABLE" in confirmed["ball"]

    # Blocks afford SLIDABLE and GRASPABLE, but NOT ROLLABLE
    assert "SLIDABLE" in confirmed["block"]
    assert "ROLLABLE" not in confirmed["block"]

    # Check that substrate affordances was populated
    assert "ball" in substrate.affordances
    assert "ROLLABLE" in substrate.affordances["ball"]


def test_affordance_novel_entity_transfer():
    """Verify that acquired affordances transfer zero-shot to novel held-out entities."""
    env = BabyWorldEnvironment(seed=42)
    env.reset("affordance_discovery_world")

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = AffordanceDiscoveryEngine(substrate, perception, env)
    engine.discover_affordances(max_interventions=20)

    # Novel test entities
    held_out_entities = [
        {
            "id": "unseen_sphere_1",
            "shape": "ellipsoid",
            "base_shape": "ball",
            "ground_truth_affordances": ["ROLLABLE", "SLIDABLE", "GRASPABLE"],
        },
        {
            "id": "unseen_cube_1",
            "shape": "polyhedron",
            "base_shape": "block",
            "ground_truth_affordances": ["SLIDABLE", "GRASPABLE"],
        },
    ]

    acc, records = engine.evaluate_novel_entity_transfer(held_out_entities)
    assert acc == 1.0
    assert len(records) == 2
    assert all(r["correct"] for r in records)
