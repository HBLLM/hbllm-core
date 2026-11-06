"""Unit tests for newly interconnected HCIR capabilities, debugging, and compilation pipelines."""

from __future__ import annotations

import pytest

from hbllm.hcir.adapters.sensor_adapter import SensorCapabilityAdapter, SensorReading
from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.capabilities.planning_capabilities import (
    PLANNING_MANIFESTS,
    execute_planning_capability,
)
from hbllm.hcir.capabilities.routing_capabilities import (
    ROUTING_MANIFESTS,
)
from hbllm.hcir.compiler import HCIRCompiler, IntentType, SemanticAST
from hbllm.hcir.counterfactual_planner import CounterfactualPlanner
from hbllm.hcir.graph import GoalNode
from hbllm.hcir.kernel.cognitive_abi import CapabilityCall
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.replay_debugger import CognitiveBreakpoint, ReplayDebugger
from hbllm.hcir.subgoal_decomposer import HierarchicalGoalDecomposer
from hbllm.hcir.workspace import HCIRWorkspaceState


class TestPlanningCapabilitiesInterconnection:
    def test_planning_manifests_present(self):
        names = {m.name for m in PLANNING_MANIFESTS}
        assert "planning.generate_candidates" in names
        assert "planning.validate" in names
        assert "planning.expand" in names

    def test_generate_candidates_with_planner(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)
        planner = CounterfactualPlanner(ws, services)
        goal = GoalNode(id="goal_1", description="Reach objective")

        call = CapabilityCall(
            capability_name="planning.generate_candidates",
            arguments={"goal": goal},
        )
        res = execute_planning_capability(call, planner=planner, workspace=ws)
        assert res.status == "SUCCESS"
        assert res.output["candidates_generated"] >= 4
        assert "selected_plan" in res.output

    def test_planning_validate(self):
        call_valid = CapabilityCall(
            capability_name="planning.validate",
            arguments={"action_id": "act_1", "intent": "NAVIGATE"},
        )
        res_valid = execute_planning_capability(call_valid)
        assert res_valid.status == "SUCCESS"
        assert res_valid.output["valid"] is True

        call_invalid = CapabilityCall(
            capability_name="planning.validate",
            arguments={"action_id": "act_2", "collision": True},
        )
        res_invalid = execute_planning_capability(call_invalid)
        assert res_invalid.status == "VALIDATION_FAILED"
        assert res_invalid.output["valid"] is False
        assert len(res_invalid.output["reasons"]) > 0

    def test_planning_expand_with_decomposer(self):
        ws = HCIRWorkspaceState()
        decomposer = HierarchicalGoalDecomposer()
        primary_goal = GoalNode(
            id="g_prim",
            description="Clear barrier",
            properties={"target_position": (8, 8)},
        )

        call = CapabilityCall(
            capability_name="planning.expand",
            arguments={
                "primary_goal": primary_goal,
                "avatar_pos": (0, 0),
                "grid_shape": (10, 10),
            },
        )
        res = execute_planning_capability(call, decomposer=decomposer, workspace=ws)
        assert res.status == "SUCCESS"
        assert "active_subgoal_id" in res.output

    @pytest.mark.asyncio
    async def test_planning_capability_executor_via_resolver(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)

        res = await services.capability_resolver.resolve_and_execute(
            "planning.generate_candidates",
            {"goal": {"id": "g_test", "description": "test goal"}},
        )
        assert res["status"] == "SUCCESS"
        assert res["candidates_generated"] >= 4


class TestRoutingCapabilitiesInterconnection:
    def test_routing_manifests_present(self):
        names = {m.name for m in ROUTING_MANIFESTS}
        assert "routing.resolve_route" in names
        assert "execution.sandbox_run" in names

    @pytest.mark.asyncio
    async def test_routing_resolve_route_via_resolver(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)

        res = await services.capability_resolver.resolve_and_execute(
            "routing.resolve_route",
            {"target_capability": "world.predict_outcome"},
        )
        assert res["status"] == "SUCCESS"
        assert res["resolved_route"] == "kernel_world.predict_outcome"
        assert res["priority"] == 5

    @pytest.mark.asyncio
    async def test_execution_sandbox_run(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)

        res = await services.capability_resolver.resolve_and_execute(
            "execution.sandbox_run",
            {"action": "read_state", "implementation_id": "test_impl"},
        )
        assert res["status"] == "SUCCESS"
        assert res["sandboxed"] is True
        assert res["allowed"] is True


class TestReplayDebuggerBreakpoints:
    @pytest.mark.asyncio
    async def test_breakpoint_tracking_and_hit_count(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)
        debugger = ReplayDebugger(services)

        bp = CognitiveBreakpoint(opcode="ASSERT")
        debugger.add_breakpoint(bp)

        stream = InstructionStream(
            author="tester",
            instructions=[
                Instruction(opcode=Opcode.ASSERT, params={"condition": "ok"}),
                Instruction(opcode=Opcode.QUERY, params={"node_type": "goal"}),
                Instruction(opcode=Opcode.ASSERT, params={"condition": "ok2"}),
            ],
        )

        steps = await debugger.replay_stream(stream)
        assert len(steps) == 3
        assert bp.hit_count == 2
        assert len(steps[0].hit_breakpoints) == 1
        assert len(steps[1].hit_breakpoints) == 0
        assert len(steps[2].hit_breakpoints) == 1

    @pytest.mark.asyncio
    async def test_stop_on_breakpoint(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)
        debugger = ReplayDebugger(services)

        bp = CognitiveBreakpoint(opcode="QUERY")
        debugger.add_breakpoint(bp)

        stream = InstructionStream(
            author="tester",
            instructions=[
                Instruction(opcode=Opcode.ASSERT, params={"condition": "ok"}),
                Instruction(opcode=Opcode.QUERY, params={"node_type": "goal"}),
                Instruction(opcode=Opcode.ASSERT, params={"condition": "never_reached"}),
            ],
        )

        steps = await debugger.replay_stream(stream, stop_on_breakpoint=True)
        assert len(steps) == 2
        assert steps[-1].instruction.opcode == Opcode.QUERY


class TestCompilerOptimizerInterconnection:
    def test_compiler_with_optimizer(self):
        compiler = HCIRCompiler()
        ast = SemanticAST(
            intent=IntentType.INVESTIGATE,
            subject="temperature_sensor",
            action="diagnose",
        )
        stream_unopt = compiler.compile(ast, optimize=False)
        stream_opt = compiler.compile(ast, optimize=True)

        assert stream_unopt.length >= 1
        assert stream_opt.length >= 1


class TestSensorAdapterInterconnection:
    def test_sensor_ingestion_publishes_event_and_syncs_twin(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)
        adapter = SensorCapabilityAdapter(services)

        events_received = []

        def event_handler(event):
            events_received.append(event)

        services.event_bus.subscribe("sensor.reading_ingested", event_handler)

        reading = SensorReading(
            sensor_id="thermo_1",
            sensor_type="thermocouple",
            variable_name="core_temp",
            value=42.5,
            unit="C",
        )
        tx = adapter.ingest_reading(reading)
        assert tx is not None
        assert len(events_received) == 1
        assert events_received[0].data["sensor_id"] == "thermo_1"
        assert events_received[0].data["value"] == 42.5


class TestWorldKernelCapabilitiesInterconnection:
    @pytest.mark.asyncio
    async def test_world_predict_outcome_via_resolver(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)
        res = await services.capability_resolver.resolve_and_execute(
            "world.predict_outcome",
            {"action_intent": "probe_switch"},
        )
        assert res.get("status") in ("success", "simulated")
        assert "predicted_outcome" in res or "claim" in res
        assert "confidence" in res

    @pytest.mark.asyncio
    async def test_world_query_state_via_resolver(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)
        res = await services.capability_resolver.resolve_and_execute(
            "world.query_state",
            {},
        )
        assert res.get("status") == "success"
        assert "environment_name" in res
        assert "variables" in res
        assert "entities" in res

    @pytest.mark.asyncio
    async def test_world_evaluate_prediction_error_via_resolver(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)
        res = await services.capability_resolver.resolve_and_execute(
            "world.evaluate_prediction_error",
            {"expected": {"pos": [1, 2]}, "actual": {"pos": [1, 3]}},
        )
        assert res.get("status") == "success"
        assert "surprise_score" in res
        assert "is_surprising" in res

    @pytest.mark.asyncio
    async def test_world_sync_digital_twin_via_resolver(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)
        res = await services.capability_resolver.resolve_and_execute(
            "world.sync_digital_twin",
            {"variable_name": "temp", "value": 37.0},
        )
        assert res.get("status") == "success"
        assert res.get("synced") is True
        assert services.world_kernel.digital_twin._variables.get("temp") == 37.0

    def test_autonomous_epistemic_engine_receives_world_kernel(self):
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
        from hbllm.hcir.world.cortex_assimilator import EpistemicPhase

        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)
        engine = AutonomousEpistemicEngine(services=services)
        assert engine.world_kernel is not None
        assert engine.world_kernel is services.world_kernel

        engine.transition_to_phase(EpistemicPhase.MENTAL_SIMULATION, reason="test_phase_persist")
        node = ws.graph.get_node(f"phase_{engine.world_id}")
        assert node is not None
        assert getattr(node, "event_data", {}).get("phase") == "mental_simulation"

        # Verify re-attaching services preserves and updates world_kernel
        ws2 = HCIRWorkspaceState()
        services2 = KernelServices.create_default(ws2)
        engine.attach_kernel_services(services2)
        assert engine.world_kernel is services2.world_kernel

    @pytest.mark.asyncio
    async def test_frontier_faculties_via_capability_resolver(self):
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)

        # 1. Odometry capability
        res_odo = await services.capability_resolver.resolve_and_execute(
            "world.path_integrate_odometry",
            {"delta_r": 1.0, "delta_c": 0.0},
        )
        assert res_odo.get("status") == "success"
        assert "integrated_position" in res_odo

        # 2. Predictive coding capability
        res_pc = await services.capability_resolver.resolve_and_execute(
            "world.predictive_coding_cycle",
            {"sensory_inputs": {"pos_r": 2.0}, "abstract_priors": {"pos_r": 1.0}},
        )
        assert res_pc.get("status") == "success"
        assert "total_free_energy" in res_pc

        # 3. Metacognitive agency capability
        res_meta = await services.capability_resolver.resolve_and_execute(
            "world.assess_metacognitive_agency",
            {
                "action_id": 1,
                "raw_confidence": 0.9,
                "predicted_avatar_pos": [2, 3],
                "observed_avatar_pos": [2, 3],
            },
        )
        assert res_meta.get("status") == "success"
        assert "agency_attribution_score" in res_meta
        assert res_meta["agency_attribution_score"] >= 0.9

        # 4. Superposition rollout capability
        res_sp = await services.capability_resolver.resolve_and_execute(
            "world.superposition_rollout",
            {"candidate_actions": [1, 2]},
        )
        assert res_sp.get("status") == "success"
        assert "best_action" in res_sp

        # 5. Skill synthesis capability
        res_synth = await services.capability_resolver.resolve_and_execute(
            "world.synthesize_skills",
            {"min_support": 2},
        )
        assert res_synth.get("status") == "success"
        assert "synthesized_count" in res_synth
