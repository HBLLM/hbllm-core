"""
Semantic Capability Registry — Intelligent capability discovery for planning.

Extends the existing ``CapabilityRegistry`` with rich semantic metadata,
enabling the planner to choose intelligently among alternative capabilities
rather than simply invoking the first matching tool.

Each registered capability exposes::

    purpose           — What it does (natural language)
    prerequisites     — What must be true before invocation
    estimated_cost    — Token/compute cost estimate
    latency           — Expected execution time
    confidence        — Self-reported reliability
    required_permissions — Sandbox scope requirements
    side_effects      — What it changes in the world
    domains           — Subject matter domains
    supported_modalities — Input/output modalities

Architecture::

    Plugin/Node registers capability with semantic metadata
        ↓
    SemanticCapabilityRegistry indexes by capability + domain
        ↓
    PlannerNode queries: "What can handle weather lookups?"
        ↓
    Registry returns ranked candidates with cost/latency/confidence
        ↓
    Planner selects optimal capability for the plan step

This registry builds on ``hbllm.brain.skills.capability_registry.CapabilityRegistry``
and is designed to eventually replace it as the primary discovery mechanism.

Usage::

    from hbllm.brain.planning.semantic_capability_registry import (
        SemanticCapabilityRegistry,
        CapabilityDescriptor,
    )

    registry = SemanticCapabilityRegistry()

    registry.register_capability(CapabilityDescriptor(
        name="weather.lookup",
        provider_id="hbllm-weather",
        purpose="Look up current weather conditions for a location",
        domains=["weather", "environment"],
        estimated_cost=0.001,
        latency_ms=500,
        confidence=0.95,
        required_permissions=["internet"],
        side_effects=[],
        supported_modalities=["text"],
    ))

    # Planner queries for weather capabilities
    candidates = registry.query("weather", min_confidence=0.8)
    best = registry.select_best("weather.lookup", strategy="balanced")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Selection Strategies
# ═══════════════════════════════════════════════════════════════════════════


class SelectionStrategy(StrEnum):
    """How the planner selects among alternative capabilities."""

    FASTEST = "fastest"  # Minimize latency
    CHEAPEST = "cheapest"  # Minimize cost
    MOST_RELIABLE = "most_reliable"  # Maximize confidence
    BALANCED = "balanced"  # Weighted combination


# ═══════════════════════════════════════════════════════════════════════════
# Capability Descriptor
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CapabilityDescriptor:
    """Rich metadata describing a single capability.

    This is the unit of registration. Each plugin or node registers
    one or more descriptors. The planner queries these to make
    informed tool selection decisions.

    Attributes:
        name: Unique capability identifier (e.g., "weather.lookup").
        provider_id: ID of the node/plugin providing this capability.
        purpose: Natural language description of what it does.
        prerequisites: Conditions that must hold before invocation.
        estimated_cost: Estimated cost per invocation (normalized, 0.0-1.0).
        latency_ms: Expected execution time in milliseconds.
        confidence: Self-reported reliability (0.0-1.0).
        required_permissions: Sandbox scopes needed (e.g., ["internet", "filesystem"]).
        side_effects: What this capability changes (e.g., ["sends_email", "writes_file"]).
        domains: Subject matter domains (e.g., ["weather", "finance"]).
        supported_modalities: Input/output types (e.g., ["text", "image"]).
        version: API version of this capability.
        enabled: Whether this capability is currently active.
        provider: Reference to the actual provider instance.
        metadata: Additional unstructured metadata.
    """

    name: str
    provider_id: str
    purpose: str = ""
    prerequisites: list[str] = field(default_factory=list)
    estimated_cost: float = 0.0  # 0.0 (free) to 1.0 (expensive)
    latency_ms: float = 1000.0  # Expected milliseconds
    confidence: float = 0.5  # 0.0 (unreliable) to 1.0 (perfect)
    required_permissions: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    supported_modalities: list[str] = field(default_factory=lambda: ["text"])
    version: str = "v1"
    enabled: bool = True
    provider: Any = None  # The actual callable / node reference
    metadata: dict[str, Any] = field(default_factory=dict)

    # Runtime stats (updated by the registry as capabilities are invoked)
    invocation_count: int = 0
    success_count: int = 0
    total_latency_ms: float = 0.0
    last_invoked_at: float = 0.0

    @property
    def actual_success_rate(self) -> float:
        """Observed success rate from invocation history."""
        if self.invocation_count == 0:
            return self.confidence  # Fall back to self-reported
        return self.success_count / self.invocation_count

    @property
    def actual_avg_latency_ms(self) -> float:
        """Observed average latency from invocation history."""
        if self.invocation_count == 0:
            return self.latency_ms  # Fall back to estimate
        return self.total_latency_ms / self.invocation_count

    @property
    def is_safe(self) -> bool:
        """True if this capability has no side effects."""
        return len(self.side_effects) == 0

    def record_invocation(self, success: bool, latency_ms: float) -> None:
        """Record the result of an invocation for runtime stats."""
        self.invocation_count += 1
        if success:
            self.success_count += 1
        self.total_latency_ms += latency_ms
        self.last_invoked_at = time.time()


# ═══════════════════════════════════════════════════════════════════════════
# Semantic Capability Registry
# ═══════════════════════════════════════════════════════════════════════════


class SemanticCapabilityRegistry:
    """Registry with rich metadata for intelligent capability selection.

    Unlike the basic ``CapabilityRegistry`` which maps string tags to
    providers, this registry stores full ``CapabilityDescriptor`` objects
    and supports semantic queries: filtering by domain, permission
    requirements, confidence thresholds, and cost budgets.

    Thread-safe for read operations. Registration should happen at
    bootstrap time or when plugins are loaded.
    """

    def __init__(self) -> None:
        # Primary index: capability name → descriptor
        self._capabilities: dict[str, CapabilityDescriptor] = {}

        # Secondary indices for fast queries
        self._by_domain: dict[str, list[str]] = {}  # domain → [capability names]
        self._by_provider: dict[str, list[str]] = {}  # provider_id → [capability names]
        self._by_permission: dict[str, list[str]] = {}  # permission → [capability names]

    # ── Registration ─────────────────────────────────────────────────────

    def register(self, descriptor: CapabilityDescriptor) -> None:
        """Register a capability with its semantic metadata.

        If a capability with the same name already exists, it is replaced.
        This allows plugins to update their descriptors at runtime.

        Args:
            descriptor: The capability descriptor to register.
        """
        name = descriptor.name

        # Remove old indices if replacing
        if name in self._capabilities:
            self._remove_from_indices(name)

        self._capabilities[name] = descriptor
        self._add_to_indices(descriptor)

        logger.debug(
            "Registered capability '%s' from provider '%s' "
            "(domains=%s, confidence=%.2f, latency=%dms)",
            name,
            descriptor.provider_id,
            descriptor.domains,
            descriptor.confidence,
            descriptor.latency_ms,
        )

    def unregister(self, name: str) -> bool:
        """Remove a capability.

        Args:
            name: Capability name to remove.

        Returns:
            True if the capability was found and removed.
        """
        if name not in self._capabilities:
            return False

        self._remove_from_indices(name)
        del self._capabilities[name]
        logger.debug("Unregistered capability '%s'", name)
        return True

    def unregister_provider(self, provider_id: str) -> int:
        """Remove all capabilities from a provider.

        Used when a plugin is unloaded.

        Args:
            provider_id: Provider to remove.

        Returns:
            Number of capabilities removed.
        """
        names = list(self._by_provider.get(provider_id, []))
        for name in names:
            self.unregister(name)
        return len(names)

    # ── Queries ──────────────────────────────────────────────────────────

    def get(self, name: str) -> CapabilityDescriptor | None:
        """Get a specific capability by name."""
        return self._capabilities.get(name)

    def has(self, name: str) -> bool:
        """Check if a capability exists and is enabled."""
        cap = self._capabilities.get(name)
        return cap is not None and cap.enabled

    def query(
        self,
        domain: str | None = None,
        *,
        min_confidence: float = 0.0,
        max_cost: float = 1.0,
        max_latency_ms: float = float("inf"),
        required_modality: str | None = None,
        exclude_permissions: list[str] | None = None,
        only_safe: bool = False,
        enabled_only: bool = True,
    ) -> list[CapabilityDescriptor]:
        """Query capabilities with semantic filters.

        Args:
            domain: Filter by domain tag. None = all domains.
            min_confidence: Minimum confidence threshold.
            max_cost: Maximum cost budget.
            max_latency_ms: Maximum acceptable latency.
            required_modality: Must support this modality.
            exclude_permissions: Exclude capabilities needing these permissions.
            only_safe: Only return capabilities with no side effects.
            enabled_only: Only return enabled capabilities.

        Returns:
            List of matching descriptors, sorted by confidence (descending).
        """
        # Start with domain filter or all
        if domain:
            names = self._by_domain.get(domain, [])
            candidates = [self._capabilities[n] for n in names if n in self._capabilities]
        else:
            candidates = list(self._capabilities.values())

        exclude_perms = set(exclude_permissions or [])

        results: list[CapabilityDescriptor] = []
        for cap in candidates:
            if enabled_only and not cap.enabled:
                continue
            if cap.confidence < min_confidence:
                continue
            if cap.estimated_cost > max_cost:
                continue
            if cap.latency_ms > max_latency_ms:
                continue
            if required_modality and required_modality not in cap.supported_modalities:
                continue
            if exclude_perms and exclude_perms.intersection(cap.required_permissions):
                continue
            if only_safe and not cap.is_safe:
                continue
            results.append(cap)

        # Sort by confidence descending, then latency ascending
        results.sort(key=lambda c: (-c.confidence, c.latency_ms))
        return results

    def select_best(
        self,
        name: str | None = None,
        domain: str | None = None,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
    ) -> CapabilityDescriptor | None:
        """Select the single best capability using a strategy.

        Args:
            name: Specific capability name (returns it directly if exists).
            domain: Domain to search within.
            strategy: Selection strategy.

        Returns:
            The best matching descriptor, or None.
        """
        if name and name in self._capabilities:
            cap = self._capabilities[name]
            return cap if cap.enabled else None

        candidates = self.query(domain=domain)
        if not candidates:
            return None

        if strategy == SelectionStrategy.FASTEST:
            return min(candidates, key=lambda c: c.actual_avg_latency_ms)
        elif strategy == SelectionStrategy.CHEAPEST:
            return min(candidates, key=lambda c: c.estimated_cost)
        elif strategy == SelectionStrategy.MOST_RELIABLE:
            return max(candidates, key=lambda c: c.actual_success_rate)
        else:  # BALANCED
            # Score = 0.5 * confidence + 0.3 * (1 - normalized_cost) + 0.2 * (1 - normalized_latency)
            max_latency = max(c.latency_ms for c in candidates) or 1.0
            return max(
                candidates,
                key=lambda c: (
                    0.5 * c.actual_success_rate
                    + 0.3 * (1.0 - c.estimated_cost)
                    + 0.2 * (1.0 - min(c.actual_avg_latency_ms / max_latency, 1.0))
                ),
            )

    def find_by_provider(self, provider_id: str) -> list[CapabilityDescriptor]:
        """Get all capabilities from a specific provider."""
        names = self._by_provider.get(provider_id, [])
        return [self._capabilities[n] for n in names if n in self._capabilities]

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def all_capabilities(self) -> list[str]:
        """List all registered capability names."""
        return list(self._capabilities.keys())

    @property
    def all_domains(self) -> list[str]:
        """List all registered domains."""
        return list(self._by_domain.keys())

    @property
    def all_providers(self) -> list[str]:
        """List all registered provider IDs."""
        return list(self._by_provider.keys())

    @property
    def all_permissions_used(self) -> set[str]:
        """Set of all permissions required across all capabilities."""
        perms: set[str] = set()
        for cap in self._capabilities.values():
            perms.update(cap.required_permissions)
        return perms

    def stats(self) -> dict[str, Any]:
        """Registry statistics."""
        return {
            "total_capabilities": len(self._capabilities),
            "enabled_capabilities": sum(1 for c in self._capabilities.values() if c.enabled),
            "domains": len(self._by_domain),
            "providers": len(self._by_provider),
            "permissions_used": sorted(self.all_permissions_used),
            "avg_confidence": (
                sum(c.confidence for c in self._capabilities.values()) / len(self._capabilities)
                if self._capabilities
                else 0.0
            ),
        }

    def describe_for_planner(self) -> list[dict[str, Any]]:
        """Generate a compact summary for injection into planner context.

        Returns a list of dicts suitable for including in an LLM prompt
        so the planner knows what capabilities are available.
        """
        return [
            {
                "name": cap.name,
                "purpose": cap.purpose,
                "domains": cap.domains,
                "confidence": round(cap.actual_success_rate, 2),
                "latency_ms": round(cap.actual_avg_latency_ms),
                "cost": round(cap.estimated_cost, 3),
                "permissions": cap.required_permissions,
                "side_effects": cap.side_effects,
                "safe": cap.is_safe,
            }
            for cap in self._capabilities.values()
            if cap.enabled
        ]

    # ── Index Management ─────────────────────────────────────────────────

    def _add_to_indices(self, descriptor: CapabilityDescriptor) -> None:
        """Add a descriptor to secondary indices."""
        name = descriptor.name

        for domain in descriptor.domains:
            self._by_domain.setdefault(domain, []).append(name)

        self._by_provider.setdefault(descriptor.provider_id, []).append(name)

        for perm in descriptor.required_permissions:
            self._by_permission.setdefault(perm, []).append(name)

    def _remove_from_indices(self, name: str) -> None:
        """Remove a capability from all secondary indices."""
        descriptor = self._capabilities.get(name)
        if not descriptor:
            return

        for domain in descriptor.domains:
            if domain in self._by_domain:
                self._by_domain[domain] = [n for n in self._by_domain[domain] if n != name]
                if not self._by_domain[domain]:
                    del self._by_domain[domain]

        pid = descriptor.provider_id
        if pid in self._by_provider:
            self._by_provider[pid] = [n for n in self._by_provider[pid] if n != name]
            if not self._by_provider[pid]:
                del self._by_provider[pid]

        for perm in descriptor.required_permissions:
            if perm in self._by_permission:
                self._by_permission[perm] = [n for n in self._by_permission[perm] if n != name]
                if not self._by_permission[perm]:
                    del self._by_permission[perm]
