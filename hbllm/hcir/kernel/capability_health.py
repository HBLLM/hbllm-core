"""
Capability Health Registry — Pre-cutover readiness validation.

Tracks readiness status (status, latency_ms, success_rate, last_failure)
and enforces health checks before promoting MigrationMode to HCIR.
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CapabilityHealth(BaseModel):
    """Health status and performance metrics for a registered capability."""

    capability_name: str
    backend: str = "hcir"
    status: str = "READY"  # READY, DEGRADED, UNAVAILABLE
    latency_ms: int = 10
    success_rate: float = 1.0
    last_failure: str | None = None
    last_checked: float = Field(default_factory=time.time)


class CapabilityHealthRegistry:
    """Registry maintaining readiness health metadata for all kernel capabilities."""

    def __init__(self) -> None:
        self._health_map: dict[str, CapabilityHealth] = {}

    def register_health(
        self,
        capability_name: str,
        backend: str = "hcir",
        status: str = "READY",
        latency_ms: int = 10,
        success_rate: float = 1.0,
    ) -> CapabilityHealth:
        """Register or update health status for a capability."""
        health = CapabilityHealth(
            capability_name=capability_name,
            backend=backend,
            status=status,
            latency_ms=latency_ms,
            success_rate=success_rate,
        )
        self._health_map[capability_name] = health
        return health

    def is_all_ready(self, required_capabilities: list[str] | None = None) -> bool:
        """Verify if all required capabilities report status=READY."""
        targets = required_capabilities or list(self._health_map.keys())
        for cap in targets:
            h = self._health_map.get(cap)
            if not h or h.status != "READY":
                return False
        return True

    def get_health(self, capability_name: str) -> CapabilityHealth | None:
        return self._health_map.get(capability_name)
