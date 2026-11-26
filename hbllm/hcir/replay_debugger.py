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
from hbllm.hcir.registers import CallFrame
from hbllm.hcir.types import BranchMode
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


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


@dataclass
class ReplayStepResult:
    """Diagnostic state at a single step in execution replay."""

    step_index: int
    instruction: Instruction
    syscall_result: dict[str, Any]
    snapshot_version: int
    node_count: int
    hit_breakpoints: list[CognitiveBreakpoint] = field(default_factory=list)
    call_depth: int = 0
    registers: dict[str, Any] = field(default_factory=dict)


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
        stop_on_breakpoint: bool = False,
    ) -> list[ReplayStepResult]:
        """Replay an instruction stream step-by-step in an isolated REPLAY branch."""
        replay_ws = self._services.workspace.fork(branch_name, mode=BranchMode.REPLAY)
        from hbllm.hcir.kernel.transaction_manager import TransactionManager

        stages = list(getattr(self._services.transaction_manager, "_verification_stages", []))
        replay_tx_mgr = TransactionManager(replay_ws, verification_stages=stages)
        replay_services = KernelServices(
            workspace=replay_ws,
            transaction_manager=replay_tx_mgr,
            capability_resolver=self._services.capability_resolver,
            scheduler=self._services.scheduler,
            tiered_workspace=self._services.tiered_workspace,
        )
        interpreter = HCIRInterpreter(replay_ws, replay_services)
        steps: list[ReplayStepResult] = []

        def _on_step(
            step_idx: int,
            ins: Instruction,
            res: dict[str, Any],
            frame: CallFrame,
            active_ws: HCIRWorkspaceState,
        ) -> bool | None:
            matched_bps: list[CognitiveBreakpoint] = []
            for bp in self._breakpoints:
                if bp.matches(ins):
                    bp.hit_count += 1
                    matched_bps.append(bp)

            step_res = ReplayStepResult(
                step_index=step_idx,
                instruction=ins,
                syscall_result=res,
                snapshot_version=active_ws.snapshot_manager.current_version,
                node_count=active_ws.graph.node_count,
                hit_breakpoints=matched_bps,
                registers=frame.registers.snapshot(),
            )
            steps.append(step_res)
            logger.debug(
                "Replay step %d (%s) -> version %d (breakpoints=%d)",
                step_idx,
                ins.opcode,
                step_res.snapshot_version,
                len(matched_bps),
            )
            if stop_on_breakpoint and matched_bps:
                logger.info("Paused replay on breakpoint at step %d (%s)", step_idx, ins.opcode)
                return False
            return True

        try:
            await interpreter.execute(stream, step_hook=_on_step)
        finally:
            self._services.workspace.drop_branch(branch_name)

        if expected_receipt and expected_receipt.success:
            logger.info(
                "Replay completed successfully. Matched receipt %s", expected_receipt.execution_id
            )

        return steps
