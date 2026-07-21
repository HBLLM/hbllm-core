"""
Capability Registry — structured capability metadata and discovery.

Extends the ``CapabilityResolver`` with rich metadata per capability,
enabling the scheduler to make cost/latency/accuracy-aware decisions.

    Capability Registry
        │
        ├── image_understanding
        │     ├── qwen-vl     (cost=0.03, latency=2s, gpu=true)
        │     ├── clip-local   (cost=0.00, latency=0.5s, gpu=true)
        │     └── human-label  (cost=5.00, latency=300s, gpu=false)
        │
        └── code_execution
              ├── local-sandbox (cost=0.00, latency=0.1s)
              └── docker-sandbox (cost=0.01, latency=1s)

The scheduler queries the registry to select the optimal provider
based on resource constraints, latency budgets, and accuracy requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.kernel.capability_resolver import (
    CapabilityImplementation,
    CapabilityResolver,
    ICapabilityExecutor,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Capability Metadata
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ResourceRequirements:
    """Hardware/software requirements for a capability provider."""

    gpu: bool = False
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 0
    memory_gb: float = 0.0
    network: bool = False
    battery_ok: bool = True  # False = requires wall power


@dataclass
class PerformanceProfile:
    """Performance characteristics of a capability provider."""

    cost_per_call: float = 0.0  # USD or token cost
    avg_latency_ms: int = 0
    max_latency_ms: int = 0
    accuracy: float = 0.0  # 0.0 – 1.0
    reliability: float = 1.0  # Uptime / success rate
    throughput_rps: float = 0.0  # Requests per second


@dataclass
class CapabilitySpec:
    """Full specification of a registered capability provider.

    Enriches ``CapabilityImplementation`` with structured metadata
    for intelligent scheduling decisions.
    """

    capability_name: str
    provider_id: str
    executor: ICapabilityExecutor
    priority: int = 0
    description: str = ""
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    performance: PerformanceProfile = field(default_factory=PerformanceProfile)


# ═══════════════════════════════════════════════════════════════════════════
# Capability Registry
# ═══════════════════════════════════════════════════════════════════════════


class CapabilityRegistry:
    """Structured capability registry with metadata-driven selection.

    Wraps ``CapabilityResolver`` and adds:
        - Rich metadata per provider (cost, latency, accuracy, requirements)
        - Query capabilities by constraints
        - Device-aware provider filtering

    Usage::

        registry = CapabilityRegistry()
        registry.register(CapabilitySpec(
            capability_name="image_understanding",
            provider_id="qwen-vl",
            executor=QwenVLExecutor(),
            requirements=ResourceRequirements(gpu=True, gpu_memory_gb=8),
            performance=PerformanceProfile(cost_per_call=0.03, accuracy=0.92),
        ))

        # Find cheapest image understanding provider
        spec = registry.find_best(
            "image_understanding",
            strategy="cheapest",
        )
    """

    def __init__(self) -> None:
        self._specs: dict[str, list[CapabilitySpec]] = {}
        self._resolver = CapabilityResolver()

    def register(self, spec: CapabilitySpec) -> None:
        """Register a capability provider with full metadata."""
        specs = self._specs.setdefault(spec.capability_name, [])
        specs.append(spec)
        specs.sort(key=lambda s: s.priority, reverse=True)

        # Also register with the underlying resolver
        self._resolver.register(
            CapabilityImplementation(
                capability_name=spec.capability_name,
                implementation_id=spec.provider_id,
                executor=spec.executor,
                priority=spec.priority,
                description=spec.description,
                tags=spec.tags,
            )
        )

        logger.info(
            "Registered capability '%s' provider '%s' (cost=%.3f, latency=%dms, accuracy=%.2f)",
            spec.capability_name,
            spec.provider_id,
            spec.performance.cost_per_call,
            spec.performance.avg_latency_ms,
            spec.performance.accuracy,
        )

    def unregister(self, capability_name: str, provider_id: str) -> bool:
        """Remove a specific provider."""
        specs = self._specs.get(capability_name, [])
        for i, spec in enumerate(specs):
            if spec.provider_id == provider_id:
                specs.pop(i)
                self._resolver.unregister(capability_name, provider_id)
                return True
        return False

    def find_best(
        self,
        capability_name: str,
        strategy: str = "priority",
        max_cost: float | None = None,
        max_latency_ms: int | None = None,
        min_accuracy: float | None = None,
        require_gpu: bool | None = None,
    ) -> CapabilitySpec | None:
        """Find the best provider matching constraints.

        Strategies:
            "priority"  — highest priority (default)
            "cheapest"  — lowest cost_per_call
            "fastest"   — lowest avg_latency_ms
            "accurate"  — highest accuracy
        """
        candidates = self._filter_candidates(
            capability_name, max_cost, max_latency_ms, min_accuracy, require_gpu
        )
        if not candidates:
            return None

        if strategy == "cheapest":
            return min(candidates, key=lambda s: s.performance.cost_per_call)
        elif strategy == "fastest":
            return min(candidates, key=lambda s: s.performance.avg_latency_ms)
        elif strategy == "accurate":
            return max(candidates, key=lambda s: s.performance.accuracy)
        else:  # "priority"
            return candidates[0]  # Already sorted by priority desc

    def _filter_candidates(
        self,
        capability_name: str,
        max_cost: float | None = None,
        max_latency_ms: int | None = None,
        min_accuracy: float | None = None,
        require_gpu: bool | None = None,
    ) -> list[CapabilitySpec]:
        """Filter providers by constraints."""
        specs = self._specs.get(capability_name, [])
        result: list[CapabilitySpec] = []
        for spec in specs:
            if not spec.executor.is_available:
                continue
            if max_cost is not None and spec.performance.cost_per_call > max_cost:
                continue
            if max_latency_ms is not None and spec.performance.avg_latency_ms > max_latency_ms:
                continue
            if min_accuracy is not None and spec.performance.accuracy < min_accuracy:
                continue
            if require_gpu is not None and spec.requirements.gpu != require_gpu:
                continue
            result.append(spec)
        return result

    def list_capabilities(self) -> list[str]:
        """Return all registered capability names."""
        return list(self._specs.keys())

    def list_providers(self, capability_name: str) -> list[CapabilitySpec]:
        """Return all providers for a capability."""
        return list(self._specs.get(capability_name, []))

    def has_capability(self, capability_name: str) -> bool:
        return bool(self._specs.get(capability_name))

    @property
    def resolver(self) -> CapabilityResolver:
        """Access the underlying resolver for direct dispatch."""
        return self._resolver

    def stats(self) -> dict[str, Any]:
        """Return registry statistics."""
        return {
            "total_capabilities": len(self._specs),
            "total_providers": sum(len(s) for s in self._specs.values()),
            "capabilities": {name: len(specs) for name, specs in self._specs.items()},
        }
