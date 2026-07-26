"""
Generation Policy — dynamic, context-aware execution decisions.

Decisions depend on runtime context (not cognitive context):
    - Available VRAM / memory
    - Latency requirements
    - Provider capabilities
    - Tenant preferences
    - Cost budget
    - Battery state
    - Licensing constraints

NO domain. NO style. Those are cognitive concerns translated
to execution constraints at the GenerationNode boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Snapshot of current system resources."""

    available_vram_gb: float = 0.0
    available_ram_gb: float = 0.0
    battery_level: float = 1.0
    active_provider: str = "local"
    loaded_modifiers: list[str] = field(default_factory=list)
    network_available: bool = True
    gpu_utilization: float = 0.0
    concurrent_executions: int = 0


@dataclass
class PolicyCondition:
    """
    Evaluable condition based on EXECUTION context.

    NO domain. NO style. Those are cognitive concerns.
    """

    min_vram_gb: float | None = None
    max_latency_ms: int | None = None
    provider: str | None = None
    tenant_id: str | None = None
    battery_above: float | None = None
    max_concurrent: int | None = None
    required_capabilities: list[str] = field(default_factory=list)

    def evaluate(self, state: SystemState, tenant_id: str | None = None) -> bool:
        """Evaluate this condition against the current system state."""
        if self.min_vram_gb is not None and state.available_vram_gb < self.min_vram_gb:
            return False
        if self.battery_above is not None and state.battery_level < self.battery_above:
            return False
        if self.provider is not None and state.active_provider != self.provider:
            return False
        if self.tenant_id is not None and tenant_id != self.tenant_id:
            return False
        if self.max_concurrent is not None and state.concurrent_executions >= self.max_concurrent:
            return False
        return True


@dataclass
class PolicyRule:
    """A single policy rule: condition → modifier selection."""

    condition: PolicyCondition
    modifiers: list[str] = field(default_factory=list)
    provider_preference: str | None = None
    priority: int = 0


@dataclass
class GenerationPolicy:
    """
    Dynamic policy for execution decisions.

    Rules are evaluated in priority order (highest first).
    The first matching rule determines modifier selection.
    If no rules match, the default modifiers are used.
    """

    rules: list[PolicyRule] = field(default_factory=list)
    default_modifiers: list[str] = field(default_factory=list)
    default_provider: str | None = None

    def resolve_modifiers(
        self,
        system_state: SystemState,
        tenant_id: str | None = None,
    ) -> list[str]:
        """
        Evaluate rules against system state, return modifier names.

        Rules are checked in priority order (highest first).
        First matching rule wins.
        """
        # Sort by priority descending
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            if rule.condition.evaluate(system_state, tenant_id):
                logger.debug(
                    "Policy rule matched: modifiers=%s, provider=%s",
                    rule.modifiers,
                    rule.provider_preference,
                )
                return rule.modifiers

        return list(self.default_modifiers)

    def resolve_provider(
        self,
        system_state: SystemState,
        tenant_id: str | None = None,
    ) -> str | None:
        """Resolve preferred provider from matching policy rule."""
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            if rule.condition.evaluate(system_state, tenant_id):
                if rule.provider_preference:
                    return rule.provider_preference

        return self.default_provider

    @staticmethod
    def default() -> GenerationPolicy:
        """Create a default policy with no rules (pass-through)."""
        return GenerationPolicy(
            rules=[],
            default_modifiers=[],
            default_provider="local",
        )
