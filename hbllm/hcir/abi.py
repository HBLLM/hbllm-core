"""
Cognitive Node ABI — the stable interface contract.

Every reasoning node, plugin, or external solver communicates
with the kernel across this strict ABI contract.  Kernel internals
can evolve without breaking existing nodes.

ABI Contract:
    - Declare supported HCIR versions
    - Declare required kernel services
    - Declare provided capabilities
    - Implement execute(transaction, workspace, services) → ExecutionResult
    - Version negotiation: kernel selects highest compatible version
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.transactions import HCIRDelta, TransactionAnnotation
from hbllm.hcir.types import Timestamp

# Current stable HCIR version
HCIR_VERSION = "1.0.0"


# ═══════════════════════════════════════════════════════════════════════════
# Execution Result
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ExecutionMetrics:
    """Resource consumption metrics from a node execution."""

    elapsed_ms: int = 0
    tokens_consumed: int = 0
    api_calls: int = 0
    memory_bytes: int = 0


@dataclass
class ExecutionResult:
    """Return type from cognitive node execution.

    Contains proposed graph deltas, annotations, emitted events,
    and resource consumption metrics.
    """

    delta: HCIRDelta = field(default_factory=HCIRDelta)
    annotations: list[TransactionAnnotation] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    success: bool = True
    error: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
# ABI Contract
# ═══════════════════════════════════════════════════════════════════════════


class ICognitiveNodeABI(ABC):
    """The stable ABI contract that every cognitive node must satisfy.

    Usage::

        class MyPlannerNode(ICognitiveNodeABI):
            supported_hcir_versions = ["1.0.0"]
            required_kernel_services = ["TransactionManager"]
            declared_capabilities = ["planning"]

            async def execute(self, transaction, workspace, services):
                # ... produce deltas ...
                return ExecutionResult(delta=my_delta)
    """

    # ── Metadata declarations (set by subclasses) ────────────────────

    supported_hcir_versions: list[str] = [HCIR_VERSION]
    required_kernel_services: list[str] = []
    declared_capabilities: list[str] = []

    # ── Execution contract ───────────────────────────────────────────

    @abstractmethod
    async def execute(
        self,
        transaction: Any,  # HCIRTransaction
        workspace: Any,  # HCIRWorkspaceState
        services: Any,  # KernelServices
    ) -> ExecutionResult:
        """Execute a cognitive computation.

        Must capture all resource usage and errors.
        Returns an ExecutionResult containing proposed deltas.
        """
        ...

    # ── Version Negotiation ──────────────────────────────────────────

    def negotiate_version(self, kernel_versions: list[str]) -> str | None:
        """Find the highest mutually compatible HCIR version.

        Returns None if no compatible version exists.
        """
        node_set = set(self.supported_hcir_versions)
        # Find intersection and return highest version
        compatible = node_set & set(kernel_versions)
        if not compatible:
            return None
        return max(compatible)

    def is_compatible(self, kernel_version: str) -> bool:
        """Check if this node is compatible with a kernel version."""
        return kernel_version in self.supported_hcir_versions
