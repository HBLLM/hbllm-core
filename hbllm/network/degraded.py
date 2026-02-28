"""
Degraded Mode Manager — tracks system capabilities and informs users.

When nodes go down, the system continues operating with reduced capabilities.
This manager provides a clear picture of what's available and what's degraded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hbllm.network.fallback import FallbackManager
from hbllm.network.registry import ServiceRegistry

logger = logging.getLogger(__name__)


@dataclass
class SystemCapabilities:
    """Current state of system capabilities."""

    available: set[str]
    degraded: dict[str, str]  # capability → degradation reason
    offline: dict[str, str]  # capability → offline reason
    total_nodes: int
    healthy_nodes: int

    @property
    def is_fully_operational(self) -> bool:
        return len(self.degraded) == 0 and len(self.offline) == 0

    @property
    def operational_percentage(self) -> float:
        total = len(self.available) + len(self.degraded) + len(self.offline)
        if total == 0:
            return 100.0
        return ((len(self.available) + len(self.degraded)) / total) * 100

    def status_summary(self) -> str:
        """Human-readable status summary."""
        if self.is_fully_operational:
            return "✅ All systems operational"

        lines = []
        if self.degraded:
            lines.append("⚠️ Degraded capabilities:")
            for cap, reason in self.degraded.items():
                lines.append(f"  • {reason}")

        if self.offline:
            lines.append("❌ Offline capabilities:")
            for cap, reason in self.offline.items():
                lines.append(f"  • {reason}")

        lines.append(
            f"\n📊 System health: {self.healthy_nodes}/{self.total_nodes} nodes, "
            f"{self.operational_percentage:.0f}% operational"
        )
        return "\n".join(lines)


class DegradedModeManager:
    """
    Manages and reports on system degradation state.

    Queries the fallback manager and registry to build
    a real-time picture of system capabilities.
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        fallback_manager: FallbackManager,
    ):
        self._registry = registry
        self._fallback = fallback_manager
        # Core capabilities the system should have
        self._expected_capabilities: set[str] = set()

    def register_expected_capability(self, capability: str) -> None:
        """Register a capability the system is expected to have."""
        self._expected_capabilities.add(capability)

    async def get_system_capabilities(self) -> SystemCapabilities:
        """Get current system capabilities state."""
        all_health = await self._registry.get_all_health()
        available_caps = await self._registry.get_available_capabilities()

        total_nodes = len(all_health)
        healthy_nodes = sum(
            1 for h in all_health.values() if h.status.value in ("healthy", "degraded")
        )

        status = await self._fallback.get_system_status()

        available: set[str] = set()
        degraded: dict[str, str] = {}
        offline: dict[str, str] = {}

        for capability in self._expected_capabilities:
            cap_status = status.get(capability)
            if cap_status is None:
                if capability in available_caps:
                    available.add(capability)
                else:
                    offline[capability] = f"❌ {capability} — not registered"
            elif cap_status["status"] == "healthy":
                available.add(capability)
            elif cap_status["status"] == "degraded":
                degraded[capability] = cap_status["message"]
            else:
                offline[capability] = cap_status["message"]

        return SystemCapabilities(
            available=available,
            degraded=degraded,
            offline=offline,
            total_nodes=total_nodes,
            healthy_nodes=healthy_nodes,
        )

    async def get_response_disclaimer(self, capability: str) -> str | None:
        """
        Get a disclaimer to prepend to a response if running in degraded mode.

        Returns None if the capability is fully operational.
        """
        result = await self._fallback.resolve(capability)
        if result and result.is_fallback:
            return result.degraded_message
        if result is None:
            return f"❌ {capability} module is offline — cannot process this request"
        return None
