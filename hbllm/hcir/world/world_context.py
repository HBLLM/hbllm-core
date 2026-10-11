"""World Context & World Model Scope — Multi-tenant Physical World Identity Containers.

Provides identity isolation for physical environments, devices, entities, and simulation branches,
lifted to declarative CausalPredicates and atomic TransactionOperations.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate


class BranchMode(StrEnum):
    """Execution branch mode for reality vs simulation isolation."""

    REALITY = "reality"
    SIMULATION = "simulation"
    COUNTERFACTUAL = "counterfactual"
    REPLAY = "replay"


@dataclass(frozen=True)
class WorldModelScope:
    """Multi-tenant isolation scope for world models and predictor weights."""

    tenant_id: str = "default_tenant"
    world_id: str = "default_world"
    device_class: str = "generic_device"
    domain: str = "general"
    prediction_type: str = "state_transition"

    def to_key(self) -> str:
        """Construct deterministic lookup key for predictor weight management."""
        return f"{self.tenant_id}:{self.world_id}:{self.device_class}:{self.domain}:{self.prediction_type}"

    def to_causal_predicate(self) -> CausalPredicate:
        """Export world model scope as a declarative causal predicate."""
        return CausalPredicate(
            variable="world_model_scope",
            operator="==",
            value=self.to_key(),
            relation=self.domain,
            target_entity=self.world_id,
        )


@dataclass(frozen=True)
class WorldContext:
    """Context container isolating physical world state and simulation branch execution."""

    world_id: str = "default_world"
    tenant_id: str = "default_tenant"
    device_id: str | None = None
    entity_ids: Sequence[str] = field(default_factory=tuple)
    branch_mode: BranchMode = BranchMode.REALITY
    timestamp: float = field(default_factory=time.time)

    def is_reality(self) -> bool:
        """Return True if execution mode is live physical reality."""
        return self.branch_mode == BranchMode.REALITY

    def allows_reality_mutation(self) -> bool:
        """Enforce strict branch isolation invariant: only REALITY branch may mutate live state."""
        return self.branch_mode == BranchMode.REALITY

    def to_causal_predicate(self) -> CausalPredicate:
        """Export world execution context as a declarative causal predicate."""
        return CausalPredicate(
            variable="world_context",
            operator="==",
            value=self.branch_mode.value,
            relation=self.tenant_id,
            target_entity=self.world_id,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction establishing world context isolation node."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"context_{self.world_id}",
                node_data={
                    "world_id": self.world_id,
                    "tenant_id": self.tenant_id,
                    "device_id": self.device_id,
                    "entity_ids": list(self.entity_ids),
                    "branch_mode": self.branch_mode.value,
                    "timestamp": self.timestamp,
                },
            )
        ]
