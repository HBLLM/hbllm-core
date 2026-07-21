"""Unit tests for ExecutiveController and UnifiedCognitiveState."""

import pytest

from hbllm.hcir.cognitive_state import UnifiedCognitiveState
from hbllm.hcir.graph import ActionNode, GoalNode
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.executive_controller import ExecutiveController
from hbllm.hcir.kernel.scheduler import CognitiveScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.workspace import HCIRWorkspaceState


class TestExecutiveController:
    @pytest.mark.asyncio
    async def test_executive_run_cycle(self):
        ws = HCIRWorkspaceState()
        ws.add_node(GoalNode(id="g_main", description="Optimize energy usage", priority=0.85))

        services = KernelServices(
            workspace=ws,
            transaction_manager=TransactionManager(ws),
            capability_resolver=CapabilityResolver(),
            scheduler=CognitiveScheduler(),
        )

        executive = ExecutiveController(services)
        candidates = [
            ActionNode(id="act_low_power", intent="enable_low_power_mode", estimated_cost=10),
            ActionNode(id="act_off", intent="power_off", estimated_cost=90),
        ]

        result = await executive.run_cycle(candidate_actions=candidates)

        assert result.cycle_index == 1
        assert result.goal_evaluated == "g_main"
        assert result.selected_action == "act_low_power"
        assert isinstance(result.state_snapshot, UnifiedCognitiveState)
        assert result.state_snapshot.node_count >= 1
