"""HCIR Capabilities — Domain capability backends and executors for the Cognitive OS."""

from __future__ import annotations

from hbllm.hcir.capabilities.planning_capabilities import (
    PLANNING_MANIFESTS,
    PlanningCapabilityExecutor,
    execute_planning_capability,
)
from hbllm.hcir.capabilities.routing_capabilities import (
    ROUTING_MANIFESTS,
    RoutingCapabilityExecutor,
    execute_routing_capability,
)
from hbllm.hcir.capabilities.world_capabilities import (
    WORLD_MANIFESTS,
    WorldKernelCapabilityExecutor,
    execute_world_capability,
)

__all__ = [
    # World
    "WORLD_MANIFESTS",
    "WorldKernelCapabilityExecutor",
    "execute_world_capability",
    # Planning
    "PLANNING_MANIFESTS",
    "PlanningCapabilityExecutor",
    "execute_planning_capability",
    # Routing & Execution
    "ROUTING_MANIFESTS",
    "RoutingCapabilityExecutor",
    "execute_routing_capability",
]
