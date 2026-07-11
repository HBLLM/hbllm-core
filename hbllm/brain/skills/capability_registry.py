"""
Capability Registry — Dynamic cognitive service discovery.

Allows cognitive subsystems to register their capabilities and
discover other services dynamically, eliminating hardcoded
dependency chains.

Instead of::

    planner = PlannerNode(simulation=sim, memory=mem, reasoner=rsn)

Components query::

    simulators = registry.find("simulation")
    memories = registry.find("memory_retrieval")

This becomes critical as the number of cognitive services grows,
and especially when users can enable/disable components.

Usage::

    from hbllm.brain.skills.capability_registry import CapabilityRegistry

    registry = CapabilityRegistry()
    registry.register("planner", planner_node, ["planning", "graph_of_thoughts"])
    registry.register("sim_engine", sim_engine, ["simulation", "deliberation"])

    # Later: discover services by capability
    simulators = registry.find("simulation")
    # → [sim_engine]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Registration Entry
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ServiceEntry:
    """A registered service and its capabilities.

    Attributes:
        name: Unique service name.
        provider: The service instance.
        capabilities: List of capability tags this service provides.
        metadata: Additional service metadata.
    """

    name: str
    provider: Any
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# CapabilityRegistry
# ═══════════════════════════════════════════════════════════════════════════


class CapabilityRegistry:
    """Dynamic service registry for cognitive capability discovery.

    Services register themselves with capability tags. Other services
    discover providers by querying for capabilities.

    Thread-safe for read operations. Registration should happen at
    bootstrap time.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ServiceEntry] = {}
        self._capability_index: dict[str, list[str]] = {}

    def register(
        self,
        name: str,
        provider: Any,
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a service with its capabilities.

        Args:
            name: Unique service name.
            provider: The service instance.
            capabilities: List of capability tags.
            metadata: Optional metadata.

        Raises:
            ValueError: If a service with this name is already registered.
        """
        if name in self._entries:
            raise ValueError(f"Service '{name}' is already registered")

        caps = capabilities or []
        entry = ServiceEntry(
            name=name,
            provider=provider,
            capabilities=caps,
            metadata=metadata or {},
        )
        self._entries[name] = entry

        # Update capability index
        for cap in caps:
            if cap not in self._capability_index:
                self._capability_index[cap] = []
            self._capability_index[cap].append(name)

        logger.debug(
            "Registered service '%s' with capabilities: %s",
            name,
            caps,
        )

    def unregister(self, name: str) -> bool:
        """Unregister a service.

        Args:
            name: Service name to remove.

        Returns:
            True if the service was found and removed.
        """
        entry = self._entries.pop(name, None)
        if entry is None:
            return False

        # Clean up capability index
        for cap in entry.capabilities:
            if cap in self._capability_index:
                self._capability_index[cap] = [n for n in self._capability_index[cap] if n != name]
                if not self._capability_index[cap]:
                    del self._capability_index[cap]

        logger.debug("Unregistered service '%s'", name)
        return True

    def find(self, capability: str) -> list[Any]:
        """Find all providers that offer a capability.

        Args:
            capability: The capability tag to search for.

        Returns:
            List of provider instances.
        """
        names = self._capability_index.get(capability, [])
        return [self._entries[n].provider for n in names if n in self._entries]

    def find_one(self, capability: str) -> Any | None:
        """Find a single provider for a capability.

        Args:
            capability: The capability tag to search for.

        Returns:
            The first matching provider, or None.
        """
        providers = self.find(capability)
        return providers[0] if providers else None

    def get(self, name: str) -> Any | None:
        """Get a service by name.

        Args:
            name: Service name.

        Returns:
            The provider instance, or None.
        """
        entry = self._entries.get(name)
        return entry.provider if entry else None

    def has(self, name: str) -> bool:
        """Check if a service is registered.

        Args:
            name: Service name.

        Returns:
            True if the service exists.
        """
        return name in self._entries

    def has_capability(self, capability: str) -> bool:
        """Check if any service offers a capability.

        Args:
            capability: The capability tag.

        Returns:
            True if at least one provider exists.
        """
        return bool(self._capability_index.get(capability))

    @property
    def all_capabilities(self) -> list[str]:
        """List all registered capabilities."""
        return list(self._capability_index.keys())

    @property
    def all_services(self) -> list[str]:
        """List all registered service names."""
        return list(self._entries.keys())

    def stats(self) -> dict[str, Any]:
        """Registry statistics."""
        return {
            "services": len(self._entries),
            "capabilities": len(self._capability_index),
            "index": {cap: len(names) for cap, names in self._capability_index.items()},
        }
