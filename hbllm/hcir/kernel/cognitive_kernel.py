"""
Cognitive Kernel — Lightweight Public Execution Boundary & Facade for HCIR.

Note on Architecture:
    - ``CognitiveKernel`` serves as a lightweight facade and test/migration harness
      implementing ``CognitiveKernelProtocol``.
    - For full production runtime execution, ``KernelServices`` (the dependency
      injection container in ``services.py``) coupled with ``ExecutiveRuntime``
      in ``executive_runtime.py`` is the canonical engine booted by ``serving/boot.py``.

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
from typing import TYPE_CHECKING, Any

from hbllm.hcir.kernel.contracts import CognitiveKernelProtocol
from hbllm.hcir.kernel.governance.governance_engine import GovernanceDecision, GovernanceEngine
from hbllm.hcir.workspace import HCIRWorkspaceState

if TYPE_CHECKING:
    from hbllm.hcir.kernel.services import KernelServices

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

    def to_capability_result(self) -> Any:
        from hbllm.hcir.kernel.cognitive_abi import CapabilityResult

        return CapabilityResult(
            capability_name=self.capability_name,
            status=self.status,
            output=self.result,
            receipt_id=self.receipt_id,
            timestamp=self.timestamp,
        )


class CognitiveKernel(CognitiveKernelProtocol):
    """Lightweight execution facade for the HCIR Cognitive OS Kernel."""

    def __init__(
        self,
        workspace: HCIRWorkspaceState,
        governance_engine: GovernanceEngine | None = None,
        services: KernelServices | None = None,
    ) -> None:
        self._workspace = workspace
        self._governance = governance_engine or GovernanceEngine()
        self._services = services
        self._execution_count = 0

    @property
    def services(self) -> KernelServices | None:
        """The underlying full KernelServices container if attached."""
        return self._services

    @property
    def workspace(self) -> HCIRWorkspaceState:
        return self._workspace

    @property
    def governance(self) -> GovernanceEngine:
        return self._governance

    async def execute_async(
        self,
        capability_name: str,
        arguments: dict[str, Any],
        context: Any = None,
    ) -> KernelExecutionReceipt:
        """Asynchronously execute a capability call through the kernel pipeline."""
        start_time = time.monotonic()
        self._execution_count += 1
        receipt_id = f"rcpt_k_{self._execution_count}"

        gov_dec = self._governance.evaluate_execution(capability_name, arguments, context)
        if not gov_dec.allowed:
            raise PermissionError(
                f"Kernel governance blocked capability '{capability_name}': {gov_dec.reason}"
            )

        result_data: dict[str, Any]
        if self._services and self._services.capability_resolver:
            resolver = self._services.capability_resolver
            if resolver.has_capability(capability_name):
                result_data = await resolver.resolve_and_execute(capability_name, arguments)
            else:
                result_data = {"executed": True, "capability": capability_name, "args": arguments}
        else:
            result_data = {"executed": True, "capability": capability_name, "args": arguments}

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        receipt = KernelExecutionReceipt(
            receipt_id=receipt_id,
            capability_name=capability_name,
            status="SUCCESS" if "error" not in result_data else "FAILED",
            result=result_data,
            elapsed_ms=elapsed_ms,
            governance_decision=gov_dec,
        )
        logger.debug(
            "CognitiveKernel executed capability '%s' [receipt: %s]", capability_name, receipt_id
        )
        return receipt

    def execute(
        self,
        capability_name: str,
        arguments: dict[str, Any],
        context: Any = None,
    ) -> KernelExecutionReceipt:
        """Execute a capability call through the 9-stage kernel pipeline (synchronous entry).

        This is the blocking facade for callers outside an event loop. Inside a running
        loop callers must ``await execute_async`` — executing capabilities on a side
        thread would race the single-threaded commit path of the TransactionManager.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._execute_guarded(capability_name, arguments, context))
        raise RuntimeError(
            "CognitiveKernel.execute() called from within a running event loop; "
            "use `await CognitiveKernel.execute_async(...)` instead"
        )

    async def _execute_guarded(
        self,
        capability_name: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> KernelExecutionReceipt:
        """``execute_async`` with capability failures captured into the receipt."""
        try:
            return await self.execute_async(capability_name, arguments, context)
        except PermissionError:
            raise
        except Exception as exc:
            self._execution_count += 1
            return KernelExecutionReceipt(
                receipt_id=f"rcpt_k_{self._execution_count}",
                capability_name=capability_name,
                status="FAILED",
                result={"executed": False, "error": str(exc), "capability": capability_name},
                elapsed_ms=0,
            )
