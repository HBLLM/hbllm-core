"""Unit tests for Stage D2: Spatial Containment & Transport Engine."""

from __future__ import annotations

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.perception import DevelopmentalPerceptionAdapter
from plugins.developmental_adapter.spatial_containment import SpatialContainmentEngine
from plugins.developmental_adapter.types import BabyRelationType


def test_spatial_relation_detection():
    """Verify that INSIDE and NEAR relations are extracted from geometric observations."""
    env = BabyWorldEnvironment(seed=42)
    obs = env.reset("containment_world")

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = SpatialContainmentEngine(substrate, perception, env)

    facts = engine.detect_spatial_relations(obs)
    inside_facts = [f for f in facts if f.relation == BabyRelationType.INSIDE]
    near_facts = [f for f in facts if f.relation == BabyRelationType.NEAR]

    assert len(inside_facts) >= 2
    inside_subjects = {f.subject_id for f in inside_facts}
    assert "obj_toy_ball" in inside_subjects
    assert "obj_toy_cube" in inside_subjects
    assert "obj_outside_ball" not in inside_subjects
    assert len(near_facts) > 0


def test_containment_transport_schema_discovery():
    """Verify discovery of containment transport invariance: MOVE(container) => MOVE(contents)."""
    env = BabyWorldEnvironment(seed=42)
    env.reset("containment_world")

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = SpatialContainmentEngine(substrate, perception, env)

    schema = engine.discover_containment_transport_schema(
        container_id="obj_container_box",
        contained_id="obj_toy_ball",
        outside_id="obj_outside_ball",
    )

    assert schema["confirmed"] is True
    assert schema["invariant"] == "SYNCHRONOUS_TRANSPORT"
    assert schema["contained_displacement"] > 0.1
    assert schema["outside_displacement"] < 0.05
    assert len(substrate.spatial_schemas) >= 1


def test_object_permanence_under_containment_and_transport():
    """Verify that moving an opaque container updates mental location of hidden contents."""
    env = BabyWorldEnvironment(seed=42)
    env.reset("containment_world")

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = SpatialContainmentEngine(substrate, perception, env)

    perm_res = engine.verify_object_permanence_during_transport(
        container_id="obj_container_box",
        contained_id="obj_toy_ball",
    )

    assert perm_res["permanence_preserved"] is True
    assert perm_res["prediction_error"] < 0.05
