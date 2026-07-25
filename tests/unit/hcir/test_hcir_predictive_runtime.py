"""Unit tests for HCIR Predictive Cognitive Runtime & World Kernel."""

import pytest

from hbllm.hcir.counterfactual_planner import CounterfactualPlanner
from hbllm.hcir.graph import (
    ActionNode,
    EnvironmentStateNode,
    GoalNode,
    HCIRNodeType,
    PhysicalEntityNode,
    PredictionNode,
    WorldVariableNode,
)
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.scheduler import CognitiveScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world_kernel import WorldKernel

# ═══════════════════════════════════════════════════════════════════════════
# World Model Graph Node Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWorldModelNodes:
    def test_world_variable_node(self):
        node = WorldVariableNode(
            id="wv_temp",
            variable_name="temperature",
            value=24.5,
            unit="Celsius",
            min_value=0.0,
            max_value=50.0,
        )
        assert node.node_type == HCIRNodeType.WORLD_VARIABLE
        assert node.value == 24.5

    def test_physical_entity_node(self):
        node = PhysicalEntityNode(
            id="pe_solar",
            entity_name="solar_dehydrator_01",
            entity_type="dehydrator",
            status="operational",
        )
        assert node.node_type == HCIRNodeType.PHYSICAL_ENTITY
        assert node.status == "operational"

    def test_environment_state_node(self):
        node = EnvironmentStateNode(
            id="env_greenhouse",
            environment_name="Greenhouse_Alpha",
            active_variables=["temperature", "humidity"],
            overall_status="nominal",
        )
        assert node.node_type == HCIRNodeType.ENVIRONMENT_STATE
        assert len(node.active_variables) == 2


# ═══════════════════════════════════════════════════════════════════════════
# World Kernel Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWorldKernel:
    def test_get_current_world_state(self):
        ws = HCIRWorkspaceState()
        ws.add_node(
            EnvironmentStateNode(id="env_1", environment_name="Lab", overall_status="optimal")
        )
        ws.add_node(WorldVariableNode(id="wv_1", variable_name="co2", value=420))
        ws.add_node(PhysicalEntityNode(id="pe_1", entity_name="pump_01", status="active"))

        world_kernel = WorldKernel(ws)
        state = world_kernel.get_current_world_state()

        assert state.environment_name == "Lab"
        assert state.variables["co2"] == 420
        assert state.entities["pump_01"] == "active"

    def test_predict(self):
        ws = HCIRWorkspaceState()
        world_kernel = WorldKernel(ws)

        action = ActionNode(id="act_irrigate", intent="increase_irrigation")
        prediction = world_kernel.predict(action, time_horizon_ms=1800000)

        assert isinstance(prediction, PredictionNode)
        assert prediction.time_horizon_ms == 1800000
        assert ws.get_node(prediction.id) is not None

    def test_compare_outcomes(self):
        act1 = ActionNode(id="a1", intent="slow_option", estimated_cost=10)
        pred1 = PredictionNode(id="p1", claim="c1")
        pred1.uncertainty.confidence = 0.9

        act2 = ActionNode(id="a2", intent="costly_option", estimated_cost=500)
        pred2 = PredictionNode(id="p2", claim="c2")
        pred2.uncertainty.confidence = 0.5

        best_action, best_pred = WorldKernel.compare_outcomes(
            [
                (act1, pred1),
                (act2, pred2),
            ]
        )
        assert best_action.id == "a1"


# ═══════════════════════════════════════════════════════════════════════════
# Counterfactual Planner Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCounterfactualPlanner:
    @pytest.mark.asyncio
    async def test_evaluate_and_select_candidates(self):
        ws = HCIRWorkspaceState()
        services = KernelServices(
            workspace=ws,
            transaction_manager=TransactionManager(ws),
            capability_resolver=CapabilityResolver(),
            scheduler=CognitiveScheduler(),
        )

        planner = CounterfactualPlanner(ws, services)
        goal = GoalNode(id="g_dehydrate", description="Optimize solar dehydrator")

        candidates = [
            ActionNode(id="a_copper", intent="use_copper_tubing", estimated_cost=10),
            ActionNode(id="a_aluminum", intent="use_aluminum_fins", estimated_cost=50),
        ]

        best_result = await planner.evaluate_and_select(goal, candidates)

        assert best_result is not None
        assert best_result.action.id == "a_copper"
        assert best_result.utility_score > 0
        assert best_result.receipt.success is True
