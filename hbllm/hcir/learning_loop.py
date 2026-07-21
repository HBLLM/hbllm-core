"""
HCIR Learning Loop — Experience → Reflection → Skill Extraction → Future Compilation.

Completes the self-improving cognitive loop:

    ExecutionReceipt
           ↓
    Outcome Monitor & Utility Evaluation
           ↓
    SkillCompiler (extracts reusable macros)
           ↓
    Graph Mutation (SkillNode / ProcedureNode)
           ↓
    Improved Future Executions
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

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
    utility_score: float  # -1.0 (failure) to 1.0 (success)
    should_extract_skill: bool = False
    learned_skill_name: str = ""
    description: str = ""


class SkillCompiler:
    """Extracts reusable SkillNode or ProcedureNode from successful receipts."""

    def compile_receipt_to_skill(
        self,
        receipt: ExecutionReceipt,
        skill_name: str,
        description: str = "",
        tenant_id: str = "default",
    ) -> SkillNode:
        """Compile a successful receipt's capabilities into a SkillNode."""
        return SkillNode(
            id=f"skill_{uuid.uuid4().hex[:8]}",
            skill_name=skill_name,
            description=description or f"Auto-compiled skill from receipt {receipt.execution_id}",
            success_rate=1.0 if receipt.success else 0.0,
            invocation_count=1,
            lifecycle=NodeLifecycle.ACTIVE,
            provenance=Provenance(
                created_by="skill_compiler",
                reasoning_step=receipt.final_snapshot_version,
            ),
            scope=Scope(tenant_id=tenant_id),
            tags=["compiled_skill", skill_name],
        )


class LearningLoopEngine:
    """Orchestrates the cognitive reflection and skill induction loop."""

    def __init__(
        self,
        workspace: HCIRWorkspaceState,
        transaction_manager: Any | None = None,
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
    ) -> CognitiveReflectionOutcome:
        """Evaluate an ExecutionReceipt's utility score.

        Utility score combines:
            - Execution success (1.0 vs -1.0)
            - Resource efficiency (tokens consumed vs limit)
            - Verification errors (penalty per rejected transaction)
            - Explicit user reward (if available)
        """
        if not receipt.success:
            score = -1.0
        else:
            base_score = 0.8
            # Penalty for rejected transactions
            rejection_penalty = len(receipt.transactions_rejected) * 0.2
            # Token efficiency bonus/penalty
            eff = max(0.0, 1.0 - (receipt.metrics.tokens_consumed / 2000.0))
            score = base_score - rejection_penalty + (eff * 0.2)

        if user_reward is not None:
            score = (score + user_reward) / 2.0

        score = max(-1.0, min(1.0, score))
        should_extract = score > 0.6 and receipt.success and len(receipt.capabilities_used) > 0

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
    ) -> HCIRTransaction | None:
        """Evaluate receipt and emit a graph transaction to store learned skills/procedures."""
        outcome = self.evaluate_receipt(receipt, user_reward=user_reward)

        if not outcome.should_extract_skill:
            logger.debug(
                "Receipt %s utility score %.2f insufficient for skill extraction",
                receipt.execution_id,
                outcome.utility_score,
            )
            return None

        skill_node = self._skill_compiler.compile_receipt_to_skill(
            receipt=receipt,
            skill_name=outcome.learned_skill_name,
            description=outcome.description,
            tenant_id=receipt.author or "default",
        )

        # Commit via transaction
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
        if res.is_committed:
            logger.info("Successfully learned and committed skill node: %s", skill_node.id)
            return tx
        return None
