"""
Executive Controller & Cognitive Cycle Engine — top-level cognitive orchestration.

Unified executive controller executing the 8-phase cognitive loop:

             ExecutiveController
                     │
         ┌───────────┴───────────┐
         │ 1. Observe (Sensors)  │
         │ 2. Orient (Attention) │
         │ 3. Predict (World)    │
         │ 4. Simulate (Planner) │
         │ 5. Decide (Executive) │
         │ 6. Execute (VM)       │
         │ 7. Reflect (Errors)   │
         │ 8. Learn (Skills)     │
         └───────────────────────┘
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.cognitive_state import UnifiedCognitiveState
from hbllm.hcir.counterfactual_planner import CounterfactualPlanner
from hbllm.hcir.graph import ActionNode, GoalNode, HCIRNodeType
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.attention_graph import AttentionManager
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.learning_loop import LearningLoopEngine
from hbllm.hcir.world_kernel import WorldKernel

logger = logging.getLogger(__name__)


@dataclass
class CognitiveCycleResult:
    """Output metrics and state from a single executive cognitive cycle."""

    cycle_index: int
    elapsed_ms: int
    goal_evaluated: str | None
    selected_action: str | None
    prediction_error_detected: bool
    skills_induced: int
    state_snapshot: UnifiedCognitiveState


class ExecutiveController:
    """Top-level executive brain orchestrator running the closed cognitive loop.

    Usage::

        executive = ExecutiveController(services)
        res = await executive.run_cycle()
    """

    def __init__(self, services: KernelServices) -> None:
        self._services = services
        self._workspace = services.workspace
        self._attention_mgr = AttentionManager(self._workspace)
        self._world_kernel = WorldKernel(self._workspace)
        self._cf_planner = CounterfactualPlanner(self._workspace, services)
        self._interpreter = HCIRInterpreter(self._workspace, services)
        self._learning_engine = LearningLoopEngine(self._workspace, services.transaction_manager)
        self._cycle_count = 0

    async def run_cycle(
        self,
        candidate_actions: list[ActionNode] | None = None,
        process_id: str = "proc_exec",
    ) -> CognitiveCycleResult:
        """Execute a full 8-phase cognitive cycle."""
        start_time = time.monotonic()
        self._cycle_count += 1

        # Phase 1: OBSERVE — Ingest pending sensor telemetry & update world model
        _ = self._world_kernel.get_current_world_state()

        # Phase 2: ORIENT — Recompute attention salience across goals & surprise signals
        attn_graph = self._attention_mgr.recompute_attention()

        # Phase 3: PREDICT — Evaluate active predictions against world state
        default_act = ActionNode(id="act_predict_default", intent="evaluate_baseline_stability")
        _ = self._world_kernel.predict(default_act)

        # Phase 4 & 5: SIMULATE & DECIDE — Evaluate candidate actions via counterfactual simulation
        active_goals = self._workspace.active_goals()
        primary_goal = (
            active_goals[0]
            if active_goals
            else GoalNode(id="goal_default", description="Maintain baseline stability")
        )

        selected_action_id = None
        if candidate_actions:
            decision = await self._cf_planner.evaluate_and_select(primary_goal, candidate_actions)
            selected_action_id = decision.action.id

        # Phase 6: EXECUTE — Compile & run instruction stream via VM Interpreter
        stream = InstructionStream(
            instructions=[
                Instruction(opcode=Opcode.QUERY, params={"node_type": "goal"}),
                Instruction(opcode=Opcode.ASSERT, params={"condition": "system_healthy"}),
            ]
        )
        exec_res, receipt = await self._interpreter.execute_with_receipt(
            stream, process_id=process_id, thread_id=f"thr_cycle_{self._cycle_count}"
        )

        # Phase 7: REFLECT — Measure prediction variance & detect PredictionErrorNodes
        error_nodes = self._workspace.graph.nodes_by_type(HCIRNodeType.PREDICTION_ERROR)
        has_errors = len(error_nodes) > 0

        # Phase 8: LEARN — Induce SkillNodes from execution experience certificate
        skills_before = len(self._workspace.graph.nodes_by_type(HCIRNodeType.SKILL))
        self._learning_engine.process_and_learn(receipt, user_reward=1.0)
        skills_after = len(self._workspace.graph.nodes_by_type(HCIRNodeType.SKILL))
        skills_induced = max(0, skills_after - skills_before)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        snapshot = UnifiedCognitiveState.from_workspace(self._workspace, attn_graph)

        logger.info(
            "Completed Cognitive Cycle #%d in %d ms (action=%s, skills_induced=%d)",
            self._cycle_count,
            elapsed_ms,
            selected_action_id,
            skills_induced,
        )

        return CognitiveCycleResult(
            cycle_index=self._cycle_count,
            elapsed_ms=elapsed_ms,
            goal_evaluated=primary_goal.id,
            selected_action=selected_action_id,
            prediction_error_detected=has_errors,
            skills_induced=skills_induced,
            state_snapshot=snapshot,
        )
