"""Tests for the Copy-On-Write State Projector."""

from __future__ import annotations

import time

import pytest

from hbllm.brain.simulation.models import PredictionOrigin
from hbllm.brain.simulation.projector import ProjectedState
from hbllm.brain.world_state import EntityState, WorldStateEngine
from hbllm.perception.reality_bus import PerceptionEvent


@pytest.fixture
def base_state():
    engine = WorldStateEngine()
    event = PerceptionEvent(
        entity_id="device_1", event_type="status", payload={"battery": 100, "state": "idle"}
    )
    # Simulate an async call locally for test setup
    import asyncio

    asyncio.run(engine.handle_normalized_event(event))
    return engine


def test_projected_state_read(base_state):
    """Ensure projector can read from base state."""
    projector = ProjectedState(base_state)
    entity = projector.get_entity_state("device_1")
    assert entity is not None
    assert entity.properties["battery"] == 100


def test_projected_state_mutation_isolation(base_state):
    """Ensure mutating projected state does not affect base state."""
    projector = ProjectedState(base_state)

    # Mutate in simulation
    projector.update_entity("device_1", {"battery": 80, "state": "active"}, confidence=0.8)

    # Check projection
    sim_entity = projector.get_entity_state("device_1")
    assert sim_entity.properties["battery"] == 80
    assert sim_entity.properties["state"] == "active"

    # Check base (should remain untouched)
    base_entity = base_state.get_entity_state("device_1")
    assert base_entity.properties["battery"] == 100
    assert base_entity.properties["state"] == "idle"


def test_projector_finalize(base_state):
    """Ensure finalize produces a valid FutureWorldState."""
    projector = ProjectedState(base_state)
    projector.update_entity("device_1", {"battery": 50})

    future = projector.finalize(PredictionOrigin.INFERRED)

    assert future.prediction_origin == PredictionOrigin.INFERRED
    assert "device_1" in future.mutations
    assert future.mutations["device_1"]["properties"]["battery"] == 50
