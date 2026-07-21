"""
Kernel Transaction Envelope & Cognitive Authority Chain — HCIR §10.

Cryptographic-style transaction envelope wrapping identity context, authority chain,
and capability call metadata before committing graph transactions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.context import HCIRExecutionContext
from hbllm.hcir.kernel.governance.policies.migration_policy import MigrationMode
from hbllm.hcir.workspace import BranchMode


@dataclass
class CognitiveAuthorityChain:
    """Explicit provenance chain tracking authorization across cognitive layers."""

    requestor: str = "user_default"
    tenant: str = "default"
    executive: str = "ExecutiveController"
    kernel: str = "CognitiveKernel"
    capability: str = "core.execution"
    actuator: str = "internal_state"


@dataclass
class KernelTransactionEnvelope:
    """Transaction envelope required at the TransactionManager.commit() boundary."""

    execution_context: HCIRExecutionContext
    authority_chain: CognitiveAuthorityChain = field(default_factory=CognitiveAuthorityChain)
    capability_name: str = "system.transaction"
    branch_mode: BranchMode = BranchMode.LIVE
    migration_mode: MigrationMode = MigrationMode.HYBRID
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
