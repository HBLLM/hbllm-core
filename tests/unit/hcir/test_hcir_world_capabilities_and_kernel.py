"""
Unit tests for Commit 7 of Phase 11 — Predictive World Kernel Integration.
Tests World Capabilities and WorldKernel integration.
"""

from __future__ import annotations

from hbllm.hcir.capabilities.world_capabilities import WORLD_MANIFESTS, execute_world_capability
from hbllm.hcir.graph import ActionNode
from hbllm.hcir.kernel.cognitive_abi import CapabilityCall
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world_kernel import WorldKernel


def test_world_capability_manifests_and_execution():
    assert len(WORLD_MANIFESTS) == 5
    manifest_names = [m.name for m in WORLD_MANIFESTS]
    assert "world.predict" in manifest_names
    assert "world.select_action" in manifest_names

    call_pred = CapabilityCall(
        capability_name="world.predict", arguments={"action_intent": "reduce_speed"}
    )
    res = execute_world_capability(call_pred)
    assert res.status == "SUCCESS"
    assert "predicted_outcome" in res.output

    call_select = CapabilityCall(
        capability_name="world.select_action",
        arguments={"candidate_intents": ["reduce_speed", "ignore"]},
    )
    res_select = execute_world_capability(call_select)
    assert res_select.status == "SUCCESS"
    assert res_select.output["selected_action_intent"] == "reduce_speed"


def test_world_kernel_integration():
    ws = HCIRWorkspaceState()
    kernel = WorldKernel(ws)

    state = kernel.get_current_world_state()
    assert state.environment_name == "default_env"

    action = ActionNode(intent="optimize_cooling")
    pred_node = kernel.predict(action)
    assert pred_node.id.startswith("pred_")
    assert "optimize_cooling" in pred_node.claim


async def test_world_kernel_governed_capability_execution():
    from hbllm.hcir.capabilities.world_capabilities import (
        WorldKernelCapabilityExecutor,
        execute_world_capability,
    )

    ws = HCIRWorkspaceState()
    kernel = WorldKernel(ws)

    call_pred = CapabilityCall(
        capability_name="world.predict", arguments={"action_intent": "reduce_speed"}
    )
    res = execute_world_capability(call_pred, kernel=kernel)
    assert res.status == "SUCCESS"
    assert "reduce_speed" in res.output["predicted_outcome"]
    assert res.output["confidence"] > 0.0

    call_twin = CapabilityCall(
        capability_name="world.sync_twin",
        arguments={"variable_name": "temp", "value": 72.5},
    )
    res_twin = execute_world_capability(call_twin, kernel=kernel)
    assert res_twin.status == "SUCCESS"
    assert res_twin.output["synced"] is True

    executor = WorldKernelCapabilityExecutor(kernel)
    assert executor.is_available is True
    out = await executor.execute({"action_intent": "throttle_down"})
    assert out["status"] == "simulated"
    assert "throttle_down" in out["claim"]
