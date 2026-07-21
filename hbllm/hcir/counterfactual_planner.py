"""
Counterfactual Planner — predictive simulation & candidate branch evaluation.

Implements the counterfactual planning loop:

    Goal
      ↓
    Generate Candidate Plans / Actions
      ↓
    FORK Simulation Branch per candidate
      ↓
    Simulate Candidate Execution & Forward Prediction
      ↓
    Evaluate Execution Receipts & Outcome Utility
      ↓
    MERGE Best Branch to main workspace
      ↓
    ROLLBACK / Drop discarded simulation branches

This transforms planning from static search into executable predictive simulation.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import ActionNode, GoalNode
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.receipt import ExecutionReceipt
from hbllm.hcir.world_kernel import WorldKernel
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


@dataclass
class CandidatePlanResult:
    """Evaluation result for a single candidate simulation branch."""

    candidate_id: str
    action: ActionNode
    branch_name: str
    receipt: ExecutionReceipt
    utility_score: float


class CounterfactualPlanner:
    """Predictive counterfactual planner using simulation branch forking.

    Usage::

        planner = CounterfactualPlanner(workspace, services)
        best_candidate = await planner.evaluate_and_select(
            goal=GoalNode(description="Optimize solar dehydrator efficiency"),
            candidate_actions=[
                ActionNode(intent="use_copper_tubing"),
                ActionNode(intent="use_aluminum_fins"),
            ],
        )
    """

    def __init__(
        self,
        workspace: HCIRWorkspaceState,
        services: KernelServices,
    ) -> None:
        self._workspace = workspace
        self._services = services
        self._world_kernel = WorldKernel(workspace)
        self._interpreter = HCIRInterpreter(workspace, services)

    async def evaluate_and_select(
        self,
        goal: GoalNode,
        candidate_actions: list[ActionNode],
        author: str = "counterfactual_planner",
    ) -> CandidatePlanResult:
        """Run counterfactual simulation for candidate actions and merge the best branch."""
        if not candidate_actions:
            raise ValueError("Candidate actions list cannot be empty")

        results: list[CandidatePlanResult] = []

        for action in candidate_actions:
            branch_name = f"sim_{uuid.uuid4().hex[:6]}"
            # 1. Create simulation branch fork
            self._workspace.fork_branch(branch_name)

            # 2. Build simulation instruction stream (FORK -> ASSERT -> EXECUTE -> MERGE)
            stream = InstructionStream(
                author=author,
                description=f"Simulate candidate: {action.intent}",
                instructions=[
                    Instruction(
                        opcode=Opcode.ASSERT,
                        params={"node_data": action.model_dump(), "author": author},
                    ),
                    Instruction(
                        opcode=Opcode.EXECUTE,
                        params={
                            "capability": "world_prediction",
                            "params": {"action": action.intent, "branch": branch_name},
                        },
                    ),
                ],
            )

            # 3. Execute stream within simulation branch and get receipt
            res, receipt = await self._interpreter.execute_with_receipt(
                stream, process_id=f"proc_{branch_name}", thread_id=f"thr_{branch_name}"
            )

            # 4. Predict forward outcome using WorldKernel
            prediction = self._world_kernel.predict(
                action=action,
                confidence=0.85 if res.success else 0.2,
                author=author,
            )

            # 5. Compute utility score
            utility_score = (
                prediction.uncertainty.confidence
                * (1.0 if res.success else 0.0)
                * (1.0 / (1.0 + action.estimated_cost * 0.01))
            )

            results.append(
                CandidatePlanResult(
                    candidate_id=action.id,
                    action=action,
                    branch_name=branch_name,
                    receipt=receipt,
                    utility_score=utility_score,
                )
            )

        # 6. Rank candidates by utility score
        best_candidate = max(results, key=lambda r: r.utility_score)

        logger.info(
            "Counterfactual planner selected candidate '%s' (branch '%s') with utility score %.3f",
            best_candidate.action.intent,
            best_candidate.branch_name,
            best_candidate.utility_score,
        )

        # 7. Merge best branch to main workspace state and cleanup unused branches
        self._workspace.merge_branch(best_candidate.branch_name)
        for r in results:
            if r.branch_name != best_candidate.branch_name:
                self._workspace.drop_branch(r.branch_name)

        return best_candidate
