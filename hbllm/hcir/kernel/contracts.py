"""
Cognitive Kernel Protocol & Contracts — HCIR §10.

Defines the abstract ``CognitiveKernelProtocol`` interface contract that all
kernel implementations (LocalKernel, CloudKernel, SwarmKernel, SimulationKernel)
must satisfy.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CognitiveKernelProtocol(Protocol):
    """Abstract protocol for Cognitive OS kernel implementations."""

    def execute(
        self,
        capability_name: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> Any:
        """Execute a capability call through the 9-stage kernel pipeline."""
        ...
