"""Unit tests for Core HCIR Spatial Containment & Transport Engine."""

from __future__ import annotations

from hbllm.hcir.world.causal_discovery import BeliefTransitionType
from hbllm.hcir.world.spatial_containment import (
    BaseSpatialContainmentEngine,
)


def test_spatial_relation_detection() -> None:
    engine = BaseSpatialContainmentEngine()

    percepts = [
        {
            "percept_id": "box_1",
            "spatial_coordinates": (1.0, 1.0),
        },
        {
            "percept_id": "ball_1",
            "contained_in": "box_1",
            "spatial_coordinates": (1.1, 1.1),
        },
        {
            "percept_id": "toy_far",
            "spatial_coordinates": (5.0, 5.0),
        },
    ]

    facts = engine.detect_spatial_relations(percepts, near_threshold=0.5)

    inside_facts = [f for f in facts if f.relation == "INSIDE"]
    assert len(inside_facts) == 1
    assert inside_facts[0].subject_id == "ball_1"
    assert inside_facts[0].object_id == "box_1"

    near_facts = [f for f in facts if f.relation == "NEAR"]
    assert len(near_facts) == 1  # box_1 and ball_1 are close (~0.14m <= 0.5m)
    assert {near_facts[0].subject_id, near_facts[0].object_id} == {"box_1", "ball_1"}


def test_containment_transport_invariance() -> None:
    # 1. Nominal synchronous transport
    confirmed, schema = BaseSpatialContainmentEngine.evaluate_containment_transport_invariance(
        container_moved=True,
        container_disp=0.5,
        inside_disp=0.51,  # inside moved ~0.5m with container
        outside_disp=0.01,  # outside stayed static
        tolerance=0.05,
    )
    assert confirmed is True
    assert schema["confirmed"] is True
    assert schema["invariant"] == "SYNCHRONOUS_TRANSPORT"

    # 2. Refutation: inside object did not move
    failed_inside, schema_failed = (
        BaseSpatialContainmentEngine.evaluate_containment_transport_invariance(
            container_moved=True,
            container_disp=0.5,
            inside_disp=0.0,
            outside_disp=0.0,
            tolerance=0.05,
        )
    )
    assert failed_inside is False
    assert schema_failed["confirmed"] is False

    # 3. Refutation: outside object moved too (no containment specificity)
    failed_outside, _ = BaseSpatialContainmentEngine.evaluate_containment_transport_invariance(
        container_moved=True,
        container_disp=0.5,
        inside_disp=0.5,
        outside_disp=0.45,
        tolerance=0.05,
    )
    assert failed_outside is False


def test_object_permanence_evaluation() -> None:
    # Accurate mental tracking
    res_correct = BaseSpatialContainmentEngine.evaluate_object_permanence(
        predicted_pos=(2.0, 3.0),
        actual_pos=(2.02, 2.99),
        tolerance=0.05,
    )
    assert res_correct["permanence_preserved"] is True
    assert res_correct["prediction_error"] < 0.05

    # Erroneous mental tracking
    res_drift = BaseSpatialContainmentEngine.evaluate_object_permanence(
        predicted_pos=(2.0, 3.0),
        actual_pos=(2.2, 3.2),
        tolerance=0.05,
    )
    assert res_drift["permanence_preserved"] is False
    assert res_drift["prediction_error"] > 0.05


def test_record_spatial_schema() -> None:
    engine = BaseSpatialContainmentEngine()
    store: list[dict] = []

    schema = {"schema_id": "test_schema", "confirmed": True}
    event = engine.record_spatial_schema(
        schema=schema,
        container_id="box_1",
        target_store=store,
        step_index=3,
    )

    assert len(engine.discovered_spatial_schemas) == 1
    assert len(store) == 1
    assert event.event_type == BeliefTransitionType.SPATIAL_SCHEMA_INDUCED
    assert event.posterior_confidence == 1.0
