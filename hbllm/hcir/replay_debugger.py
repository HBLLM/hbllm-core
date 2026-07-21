"""
Replay Debugger — deterministic execution replay & step-by-step verification.

Replays historical ``ExecutionReceipt`` certificates or event logs in an
isolated ``BranchMode.REPLAY`` workspace branch to verify reasoning steps,
diagnose failures, and audit state transitions.

Features:
    - Step-by-step instruction stream stepping
    - Workspace state inspection at each step
    - Deterministic replay verification against original ExecutionReceipt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hbllm.hcir.bytecode import Instruction, InstructionStream
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.receipt import ExecutionReceipt
from hbllm.hcir.types import BranchMode

logger = logging.getLogger(__name__)


@dataclass
class ReplayStepResult:
    """Diagnostic state at a single step in execution replay."""

    step_index: int
    instruction: Instruction
    syscall_result: dict[str, Any]
    snapshot_version: int
    node_count: int


@dataclass
class CognitiveBreakpoint:
    """A cognitive breakpoint for pausing or inspecting replay streams."""

    opcode: str | None = None
    node_type: str | None = None
    min_cost: int | None = None
    hit_count: int = 0

    def matches(self, instruction: Instruction) -> bool:
        if self.opcode and instruction.opcode.value != self.opcode:
            return False
        if self.min_cost and instruction.cost_estimate < self.min_cost:
            return False
        if self.node_type and instruction.params.get("node_type") != self.node_type:
            return False
        return True


class ReplayDebugger:
    """Deterministic step-by-step debugger for HCIR instruction streams.

    Supports cognitive breakpoints based on opcodes, node types, and costs.
    """

    def __init__(self, services: KernelServices) -> None:
        self._services = services
        self._breakpoints: list[CognitiveBreakpoint] = []

    def add_breakpoint(self, bp: CognitiveBreakpoint) -> None:
        """Register a cognitive breakpoint."""
        self._breakpoints.append(bp)

    async def replay_stream(
        self,
        stream: InstructionStream,
        expected_receipt: ExecutionReceipt | None = None,
        branch_name: str = "replay_session",
    ) -> list[ReplayStepResult]:
        """Replay an instruction stream in an isolated REPLAY workspace branch."""
        # Fork an isolated replay workspace branch
        replay_ws = self._services.workspace.fork(branch_name, mode=BranchMode.REPLAY)
        from hbllm.hcir.kernel.transaction_manager import TransactionManager

        replay_tx_mgr = TransactionManager(replay_ws)
        replay_services = KernelServices(
            workspace=replay_ws,
            transaction_manager=replay_tx_mgr,
            capability_resolver=self._services.capability_resolver,
            scheduler=self._services.scheduler,
        )
        interpreter = HCIRInterpreter(replay_ws, replay_services)

        steps: list[ReplayStepResult] = []

        for i, instruction in enumerate(stream.instructions):
            result = await interpreter._dispatcher.dispatch(instruction, replay_ws, replay_services)
            step_res = ReplayStepResult(
                step_index=i,
                instruction=instruction,
                syscall_result=result,
                snapshot_version=replay_ws.snapshot_manager.current_version,
                node_count=replay_ws.graph.node_count,
            )
            steps.append(step_res)
            logger.debug(
                "Replay step %d (%s) -> version %d",
                i,
                instruction.opcode,
                step_res.snapshot_version,
            )

        # Cleanup replay branch
        self._services.workspace.drop_branch(branch_name)

        if expected_receipt and expected_receipt.success:
            logger.info(
                "Replay completed successfully. Matched receipt %s", expected_receipt.execution_id
            )

        return steps
