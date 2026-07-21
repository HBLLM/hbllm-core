"""
Capability Resolver — declarative action dispatch.

Planners never pick plugins or specific tools.  They declare
a need for a *Capability* (e.g., ``execute_python``,
``image_segmentation``).  The ``CapabilityResolver`` maps these
requests to concrete registered implementations at runtime.

    Capability → Implementation → Executor

Multiple implementations can serve the same capability.
Selection is based on priority, availability, and resource cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Capability Executor Interface
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class ICapabilityExecutor(Protocol):
    """Interface for concrete capability implementations.

    Implementations can be local Python, Docker containers,
    remote APIs, MCP servers, or other HBLLM nodes.
    """

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the capability with given parameters."""
        ...

    @property
    def is_available(self) -> bool:
        """Whether this executor is currently available."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Capability Implementation Registration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CapabilityImplementation:
    """A concrete implementation that satisfies a capability.

    Multiple implementations can serve the same capability.
    The resolver selects based on priority and availability.
    """

    capability_name: str
    implementation_id: str
    executor: ICapabilityExecutor
    priority: int = 0  # Higher = preferred
    estimated_cost: int = 0  # Tokens
    description: str = ""
    tags: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Capability Resolver
# ═══════════════════════════════════════════════════════════════════════════


class CapabilityResolver:
    """Resolves declarative action capabilities to concrete executors.

    Usage::

        resolver = CapabilityResolver()
        resolver.register(CapabilityImplementation(
            capability_name="execute_python",
            implementation_id="local_sandbox",
            executor=LocalPythonExecutor(),
            priority=10,
        ))

        executor = await resolver.resolve("execute_python")
        result = await executor.execute({"code": "print('hello')"})
    """

    def __init__(self) -> None:
        # capability_name → list of implementations (sorted by priority desc)
        self._registry: dict[str, list[CapabilityImplementation]] = {}

    def register(self, impl: CapabilityImplementation) -> None:
        """Register a capability implementation."""
        impls = self._registry.setdefault(impl.capability_name, [])
        impls.append(impl)
        # Sort by priority descending (highest first)
        impls.sort(key=lambda x: x.priority, reverse=True)
        logger.info(
            "Registered capability '%s' implementation '%s' (priority=%d)",
            impl.capability_name, impl.implementation_id, impl.priority,
        )

    def unregister(self, capability_name: str, implementation_id: str) -> bool:
        """Unregister a specific implementation."""
        impls = self._registry.get(capability_name, [])
        for i, impl in enumerate(impls):
            if impl.implementation_id == implementation_id:
                impls.pop(i)
                return True
        return False

    async def resolve(self, capability_name: str) -> ICapabilityExecutor | None:
        """Resolve a capability to its best available executor.

        Selects the highest-priority implementation that reports
        itself as available.
        """
        impls = self._registry.get(capability_name, [])
        for impl in impls:
            if impl.executor.is_available:
                logger.debug(
                    "Resolved capability '%s' → '%s'",
                    capability_name, impl.implementation_id,
                )
                return impl.executor

        logger.warning("No available executor for capability '%s'", capability_name)
        return None

    def list_capabilities(self) -> list[str]:
        """Return all registered capability names."""
        return list(self._registry.keys())

    def list_implementations(self, capability_name: str) -> list[CapabilityImplementation]:
        """Return all implementations for a capability."""
        return list(self._registry.get(capability_name, []))

    def has_capability(self, capability_name: str) -> bool:
        """Check if any implementation is registered for a capability."""
        return bool(self._registry.get(capability_name))
