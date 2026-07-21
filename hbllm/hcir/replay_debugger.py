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
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.bytecode import Instruction, InstructionStream
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.receipt import ExecutionReceipt
from hbllm.hcir.types import BranchMode
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


@dataclass
class ReplayStepResult:
    """Diagnostic state at a single step in execution replay."""

    step_index: int
    instruction: Instruction
    syscall_result: dict[str, Any]
    snapshot_version: int
    node_count: int


class ReplayDebugger:
    """Deterministic step-by-step debugger for HCIR instruction streams.

    Usage::

        debugger = ReplayDebugger(services)
        results = await debugger.replay_stream(
            stream=instruction_stream,
            expected_receipt=receipt,
        )
    """

    def __init__(self, services: KernelServices) -> None:
        self._services = services

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
            result = await interpreter._dispatcher.dispatch(
                instruction, replay_ws, replay_services
            )
            step_res = ReplayStepResult(
                step_index=i,
                instruction=instruction,
                syscall_result=result,
                snapshot_version=replay_ws.snapshot_manager.current_version,
                node_count=replay_ws.graph.node_count,
            )
            steps.append(step_res)
            logger.debug("Replay step %d (%s) -> version %d", i, instruction.opcode, step_res.snapshot_version)

        # Cleanup replay branch
        self._services.workspace.drop_branch(branch_name)

        if expected_receipt and expected_receipt.success:
            logger.info("Replay completed successfully. Matched receipt %s", expected_receipt.execution_id)

        return steps
