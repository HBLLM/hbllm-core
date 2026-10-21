"""World Kernel Transaction — Immutable Security & Identity Transaction Envelope.

Immutable transaction envelope binding execution context, world context, and snapshot state,
lifted to declarative CausalPredicates and atomic TransactionOperations.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from hbllm.hcir.context import HCIRExecutionContext
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.world_context import BranchMode, WorldContext, WorldModelScope


@dataclass(frozen=True)
class WorldKernelTransaction:
    """Immutable transaction envelope binding execution context, world context, and snapshot state."""

    transaction_id: str = field(default_factory=lambda: f"wtx_{uuid.uuid4().hex[:8]}")
    execution_context: HCIRExecutionContext = field(default_factory=HCIRExecutionContext)
    world_context: WorldContext = field(default_factory=WorldContext)
    model_scope: WorldModelScope = field(default_factory=WorldModelScope)
    snapshot_hash: str = "0000000000000000"
    branch_mode: BranchMode = BranchMode.REALITY
    prediction_id: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export transaction envelope metadata as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"kernel_tx:{self.transaction_id}",
            operator="==",
            value=self.snapshot_hash,
            relation=self.branch_mode.value,
            target_entity=self.world_context.world_id,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction committing transaction envelope to audit log."""
        return [
            TransactionOperation(
                op=TransactionOp.ADD_NODE,
                node_id=self.transaction_id,
                node_data={
                    "transaction_id": self.transaction_id,
                    "world_id": self.world_context.world_id,
                    "tenant_id": self.world_context.tenant_id,
                    "snapshot_hash": self.snapshot_hash,
                    "branch_mode": self.branch_mode.value,
                    "prediction_id": self.prediction_id,
                    "created_at": self.created_at,
                },
            )
        ]
