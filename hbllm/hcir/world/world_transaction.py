"""
World Kernel Transaction — Immutable Security & Identity Transaction Envelope.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from hbllm.hcir.context import HCIRExecutionContext
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
