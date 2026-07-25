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
    estimated_latency_ms: int = 0  # Expected latency
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
        self._total_cost: int = 0

    def register(self, impl: CapabilityImplementation) -> None:
        """Register a capability implementation."""
        impls = self._registry.setdefault(impl.capability_name, [])
        impls.append(impl)
        # Sort by priority descending (highest first)
        impls.sort(key=lambda x: x.priority, reverse=True)
        logger.info(
            "Registered capability '%s' implementation '%s' (priority=%d)",
            impl.capability_name,
            impl.implementation_id,
            impl.priority,
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
                    capability_name,
                    impl.implementation_id,
                )
                return impl.executor

        logger.warning("No available executor for capability '%s'", capability_name)
        return None

    async def resolve_cheapest(
        self,
        capability_name: str,
        max_cost: int | None = None,
        max_latency_ms: int | None = None,
    ) -> CapabilityImplementation | None:
        """Resolve to the cheapest available implementation within constraints.

        Market-based selection: sort by estimated_cost (ascending),
        filter by max_cost and max_latency_ms if provided.
        """
        impls = self._registry.get(capability_name, [])
        candidates = [
            impl
            for impl in impls
            if impl.executor.is_available
            and (max_cost is None or impl.estimated_cost <= max_cost)
            and (max_latency_ms is None or impl.estimated_latency_ms <= max_latency_ms)
        ]
        if not candidates:
            return None
        # Sort by cost ascending (cheapest first)
        candidates.sort(key=lambda x: x.estimated_cost)
        return candidates[0]

    async def resolve_and_execute(
        self,
        capability_name: str,
        params: dict[str, Any],
        budget: int | None = None,
    ) -> dict[str, Any]:
        """Resolve and execute a capability in a single call.

        Combines resolution + execution for convenience.
        Tracks budget consumption if a budget is provided.

        Returns:
            The executor result dict, or {"error": ...} on failure.
        """
        if budget is not None:
            impl = await self.resolve_cheapest(capability_name, max_cost=budget)
            if impl is None:
                return {
                    "error": f"No implementation for '{capability_name}' within budget {budget}"
                }
            try:
                result = await impl.executor.execute(params)
                self._total_cost += impl.estimated_cost
                return result
            except Exception as exc:
                return {"error": f"Execution failed: {exc}"}
        else:
            executor = await self.resolve(capability_name)
            if executor is None:
                return {"error": f"No available executor for '{capability_name}'"}
            try:
                result = await executor.execute(params)
                return result
            except Exception as exc:
                return {"error": f"Execution failed: {exc}"}

    def list_capabilities(self) -> list[str]:
        """Return all registered capability names."""
        return list(self._registry.keys())

    def list_implementations(self, capability_name: str) -> list[CapabilityImplementation]:
        """Return all implementations for a capability."""
        return list(self._registry.get(capability_name, []))

    def has_capability(self, capability_name: str) -> bool:
        """Check if any implementation is registered for a capability."""
        return bool(self._registry.get(capability_name))

    @property
    def total_cost(self) -> int:
        """Total estimated cost consumed across all executions."""
        return self._total_cost
