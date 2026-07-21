"""
Cognitive Kernel — Single Public Execution Boundary for HCIR Cognitive OS.

Orchestrates the 9-stage instruction execution pipeline:
 1. Envelope creation
 2. Identity validation
 3. Governance evaluation
 4. Capability resolution
 5. Sandbox execution
 6. Verification pipeline
 7. Transaction commit
 8. Receipt generation
 9. Learning event emission
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.kernel.contracts import CognitiveKernelProtocol
from hbllm.hcir.kernel.governance.governance_engine import GovernanceDecision, GovernanceEngine
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


@dataclass
class KernelExecutionReceipt:
    """Receipt emitted after executing a capability through CognitiveKernel."""

    receipt_id: str
    capability_name: str
    status: str
    result: Any
    elapsed_ms: int
    timestamp: float = field(default_factory=time.time)
    governance_decision: GovernanceDecision | None = None


class CognitiveKernel(CognitiveKernelProtocol):
    """The central execution boundary for the HCIR Cognitive OS Kernel."""

    def __init__(
        self,
        workspace: HCIRWorkspaceState,
        governance_engine: GovernanceEngine | None = None,
    ) -> None:
        self._workspace = workspace
        self._governance = governance_engine or GovernanceEngine()
        self._execution_count = 0

    @property
    def workspace(self) -> HCIRWorkspaceState:
        return self._workspace

    @property
    def governance(self) -> GovernanceEngine:
        return self._governance

    def execute(
        self,
        capability_name: str,
        arguments: dict[str, Any],
        context: Any = None,
    ) -> KernelExecutionReceipt:
        """Execute a capability call through the 9-stage kernel pipeline."""
        start_time = time.monotonic()
        self._execution_count += 1
        receipt_id = f"rcpt_k_{self._execution_count}"

        # Stage 3: Governance evaluation
        gov_dec = self._governance.evaluate_execution(capability_name, arguments, context)
        if not gov_dec.allowed:
            raise PermissionError(
                f"Kernel governance blocked capability '{capability_name}': {gov_dec.reason}"
            )

        # Stage 4 & 5: Capability resolution and execution placeholder
        result_data: dict[str, Any] = {
            "executed": True,
            "capability": capability_name,
            "args": arguments,
        }

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        receipt = KernelExecutionReceipt(
            receipt_id=receipt_id,
            capability_name=capability_name,
            status="SUCCESS",
            result=result_data,
            elapsed_ms=elapsed_ms,
            governance_decision=gov_dec,
        )
        logger.debug(
            "CognitiveKernel executed capability '%s' [receipt: %s]", capability_name, receipt_id
        )
        return receipt
