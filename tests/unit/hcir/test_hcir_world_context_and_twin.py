"""
Unit tests for Commit 1 of Phase 11 — Predictive World Kernel Integration.
Tests WorldContext, WorldStateSnapshot, DigitalTwinRegistry, WorldStateInterpreter, and WorldBeliefGraph.
"""

from __future__ import annotations

from hbllm.hcir.world.digital_twin import DigitalTwinRegistry
from hbllm.hcir.world.world_belief import WorldBeliefGraph
from hbllm.hcir.world.world_context import BranchMode, WorldContext, WorldModelScope
from hbllm.hcir.world.world_state_interpreter import WorldStateInterpreter
from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot


def test_world_context_and_scope():
    ctx = WorldContext(world_id="factory_alpha", branch_mode=BranchMode.REALITY)
    assert ctx.is_reality()
    assert ctx.allows_reality_mutation()

    sim_ctx = WorldContext(world_id="factory_alpha", branch_mode=BranchMode.SIMULATION)
    assert not sim_ctx.is_reality()
    assert not sim_ctx.allows_reality_mutation()

    scope = WorldModelScope(tenant_id="t1", world_id="w1", device_class="robotics", domain="motion")
    assert scope.to_key() == "t1:w1:robotics:motion:state_transition"


def test_digital_twin_and_snapshot():
    twin = DigitalTwinRegistry(world_id="greenhouse_01")
    twin.register_entity("pump_1", "Water Pump", status="active")
    twin.sync_sensor_telemetry("temp_celsius", 82.5, source="sensor_board_1")

    assert twin.get_variable("temp_celsius") == 82.5
    assert twin.get_entity("pump_1").status == "active"

    snapshot = twin.create_snapshot()
    assert isinstance(snapshot, WorldStateSnapshot)
    assert snapshot.world_id == "greenhouse_01"
    assert snapshot.variables["temp_celsius"] == 82.5
    assert len(snapshot.state_hash) == 16


def test_interpreter_and_belief_graph():
    snapshot = WorldStateSnapshot(
        world_id="greenhouse_01",
        variables={"temp_celsius": 85.0, "vibration_g": 0.08, "pressure_bar": 2.1},
        entity_states={"pump_1": "active"},
    )

    interpreter = WorldStateInterpreter()
    hypotheses = interpreter.interpret_snapshot(snapshot)
    assert len(hypotheses) == 3

    belief_graph = WorldBeliefGraph(world_id="greenhouse_01")
    belief_graph.ingest_hypotheses(hypotheses)

    overheating_belief = belief_graph.get_belief("b_temp_celsius_status")
    assert overheating_belief is not None
    assert overheating_belief.value == "overheating_warning"
    assert overheating_belief.confidence == 0.88
