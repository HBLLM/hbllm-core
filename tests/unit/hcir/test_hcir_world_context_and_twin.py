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

    pred_ctx = ctx.to_causal_predicate()
    assert pred_ctx.variable == "world_context"
    assert pred_ctx.value == "reality"

    tx_ctx = ctx.to_transaction_operations()
    assert len(tx_ctx) == 1
    assert tx_ctx[0].node_id == "context_factory_alpha"

    pred_scope = scope.to_causal_predicate()
    assert pred_scope.variable == "world_model_scope"
    assert pred_scope.value == scope.to_key()


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

    twin_preds = twin.export_causal_predicates()
    assert len(twin_preds) == 2
    assert any(p.variable == "temp_celsius" for p in twin_preds)

    twin_txs = twin.to_transaction_operations()
    assert len(twin_txs) == 1
    assert twin_txs[0].node_id == "phys_pump_1"


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

    # Causal predicate & world theory exports
    causal_preds = interpreter.export_causal_predicates(snapshot)
    assert len(causal_preds) == 3
    assert any(p.variable == "temp_celsius_status" for p in causal_preds)

    world_theory = interpreter.to_world_theory(snapshot)
    assert len(world_theory.goal_predicates) == 3

    # WorldBeliefGraph exports
    belief_preds = belief_graph.export_causal_predicates()
    assert len(belief_preds) == 3
    belief_txs = belief_graph.to_transaction_operations()
    assert len(belief_txs) == 3
    assert any(tx.node_id == "b_temp_celsius_status" for tx in belief_txs)

    belief_theory = belief_graph.to_world_theory()
    assert len(belief_theory.goal_predicates) == 3
