"""HCIR Learning Loop — Experience -> Reflection -> Homoiconic Skill Extraction."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import NodeLifecycle, SkillNode
from hbllm.hcir.receipt import ExecutionReceipt
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance, Scope
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


@dataclass
class CognitiveReflectionOutcome:
    """Outcome of reflecting on an ExecutionReceipt."""

    receipt_id: str
    utility_score: float
    should_extract_skill: bool = False
    learned_skill_name: str = ""
    description: str = ""


class SkillCompiler:
    """Compiles successful execution receipts into parameterized homoiconic `SkillNode`s."""

    def compile_receipt_to_skill(
        self,
        receipt: ExecutionReceipt,
        skill_name: str,
        description: str = "",
        tenant_id: str = "default",
        source_stream: InstructionStream | None = None,
    ) -> SkillNode:
        """Compile a receipt into a parameterized `SkillNode` with executable `bytecode_stream`."""
        from hbllm.hcir.interpreter import SyscallAllowlist

        allowlist = SyscallAllowlist(strict=True)
        body: list[Instruction] = []
        if source_stream is not None and source_stream.instructions:
            body = [
                ins.model_copy(deep=True)
                for ins in source_stream.instructions
                if ins.opcode != Opcode.RETURN
            ]
        else:
            body = [
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={"capability": cap, "params": {"target": "$r0"}},
                    cost_estimate=20,
                )
                for cap in receipt.capabilities_used
                if allowlist.classify(cap) == "allowed"
            ]
        if not any(i.opcode == Opcode.UNIFY for i in body):
            body.insert(
                0,
                Instruction(
                    opcode=Opcode.UNIFY,
                    params={"pattern": {"id": "?target"}, "bind": "$r0"},
                    cost_estimate=10,
                ),
            )
        body.append(Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}))

        compiled_stream = InstructionStream(
            author="skill_compiler",
            description=description
            or f"Homoiconic skill compiled from receipt {receipt.execution_id}",
            instructions=body,
        )
        return SkillNode(
            id=f"skill_{uuid.uuid4().hex[:8]}",
            skill_name=skill_name,
            description=description or f"Auto-compiled skill from receipt {receipt.execution_id}",
            parameters=[{"name": "target", "required": False, "default": None}],
            precondition_pattern={"capabilities": list(receipt.capabilities_used)},
            postcondition_pattern={"skill_name": skill_name, "target": "?target"},
            bytecode_stream=compiled_stream.model_dump(),
            alpha_success=2.0 if receipt.success else 1.0,
            beta_failure=1.0 if receipt.success else 2.0,
            success_rate=1.0 if receipt.success else 0.0,
            invocation_count=1,
            lifecycle=NodeLifecycle.ACTIVE,
            provenance=Provenance(
                created_by="skill_compiler", reasoning_step=receipt.final_snapshot_version
            ),
            scope=Scope(tenant_id=tenant_id),
            tags=["compiled_skill", skill_name],
        )


class LearningLoopEngine:
    """Orchestrates the cognitive reflection and skill induction loop."""

    def __init__(
        self, workspace: HCIRWorkspaceState, transaction_manager: Any | None = None
    ) -> None:
        from hbllm.hcir.kernel.transaction_manager import TransactionManager

        self._workspace = workspace
        self._tx_manager = transaction_manager or TransactionManager(workspace)
        self._skill_compiler = SkillCompiler()
        self._history: list[CognitiveReflectionOutcome] = []

    def evaluate_receipt(
        self,
        receipt: ExecutionReceipt,
        user_reward: float | None = None,
        *,
        source_stream: InstructionStream | None = None,
    ) -> CognitiveReflectionOutcome:
        """Evaluate an ExecutionReceipt's utility score."""
        if not receipt.success:
            score = -1.0
        else:
            rejection_penalty = len(receipt.transactions_rejected) * 0.2
            eff = max(0.0, 1.0 - (receipt.metrics.tokens_consumed / 2000.0))
            score = 0.8 - rejection_penalty + (eff * 0.2)
        if user_reward is not None:
            score = (score + user_reward) / 2.0
        score = max(-1.0, min(1.0, score))
        has_program = len(receipt.capabilities_used) > 0 or (
            source_stream is not None and source_stream.length > 0
        )
        should_extract = score > 0.6 and receipt.success and has_program
        outcome = CognitiveReflectionOutcome(
            receipt_id=receipt.execution_id,
            utility_score=score,
            should_extract_skill=should_extract,
            learned_skill_name=f"macro_{receipt.author}_{receipt.execution_id[:6]}"
            if should_extract
            else "",
            description=f"Auto-induced macro using capabilities: {receipt.capabilities_used}",
        )
        self._history.append(outcome)
        return outcome

    def process_and_learn(
        self,
        receipt: ExecutionReceipt,
        user_reward: float | None = None,
        author: str = "learning_loop",
        *,
        source_stream: InstructionStream | None = None,
    ) -> HCIRTransaction | None:
        """Evaluate receipt and commit a homoiconic SkillNode transaction."""
        outcome = self.evaluate_receipt(
            receipt, user_reward=user_reward, source_stream=source_stream
        )
        if not outcome.should_extract_skill:
            return None

        skill_node = self._skill_compiler.compile_receipt_to_skill(
            receipt=receipt,
            skill_name=outcome.learned_skill_name,
            description=outcome.description,
            tenant_id=receipt.author or "default",
            source_stream=source_stream,
        )
        tx = HCIRTransaction(
            author=author,
            operations=[
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_id=skill_node.id,
                    node_data=skill_node.model_dump(),
                )
            ],
            provenance=Provenance(created_by=author),
        )
        res = self._tx_manager.commit(tx)
        return tx if res.is_committed else None
