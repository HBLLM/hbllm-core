"""
Capability Sandboxing — security & isolation boundaries for capability execution.

Provides permission policies, resource constraints, and trust scoring
for external tool and capability execution:

    Capability
    ├── permissions     (filesystem, network, subprocess, db_write)
    ├── resource_limits (cpu_seconds, memory_mb, max_network_calls)
    ├── isolation_mode  (in_process, subprocess, container)
    └── trust_level     (untrusted, verified, system)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


class IsolationMode(StrEnum):
    """Execution isolation strategy for capabilities."""

    IN_PROCESS = "in_process"
    SUBPROCESS = "subprocess"
    CONTAINER = "container"


class TrustLevel(StrEnum):
    """Trust score classification for capability providers."""

    UNTRUSTED = "untrusted"
    VERIFIED = "verified"
    SYSTEM = "system"


@dataclass
class CapabilityPermissions:
    """Explicit permission flags for a capability."""

    allow_filesystem: bool = False
    allow_network: bool = False
    allow_subprocess: bool = False
    allow_db_write: bool = False
    allowed_domains: list[str] = field(default_factory=list)


@dataclass
class CapabilityResourceLimits:
    """Resource constraints for sandboxed capability execution."""

    max_cpu_seconds: float = 5.0
    max_memory_mb: int = 512
    max_api_calls: int = 10
    timeout_seconds: float = 10.0


@dataclass
class SandboxedCapabilityPolicy:
    """Complete security policy bound to a capability provider."""

    capability_name: str
    provider_id: str
    trust_level: TrustLevel = TrustLevel.VERIFIED
    isolation_mode: IsolationMode = IsolationMode.IN_PROCESS
    permissions: CapabilityPermissions = field(default_factory=CapabilityPermissions)
    resource_limits: CapabilityResourceLimits = field(default_factory=CapabilityResourceLimits)

    def validate_execution(self, requested_permissions: set[str]) -> bool:
        """Validate if requested operations exceed granted permissions."""
        if "filesystem" in requested_permissions and not self.permissions.allow_filesystem:
            logger.warning("Sandbox violation: %s denied filesystem access", self.provider_id)
            return False
        if "network" in requested_permissions and not self.permissions.allow_network:
            logger.warning("Sandbox violation: %s denied network access", self.provider_id)
            return False
        if "subprocess" in requested_permissions and not self.permissions.allow_subprocess:
            logger.warning("Sandbox violation: %s denied subprocess access", self.provider_id)
            return False
        return True


class CapabilitySandboxManager:
    """Enforces sandbox policy checks before capability dispatch."""

    def __init__(self) -> None:
        self._policies: dict[str, SandboxedCapabilityPolicy] = {}

    def register_policy(self, policy: SandboxedCapabilityPolicy) -> None:
        """Register a sandbox policy for a provider."""
        key = f"{policy.capability_name}:{policy.provider_id}"
        self._policies[key] = policy

    def get_policy(
        self, capability_name: str, provider_id: str
    ) -> SandboxedCapabilityPolicy | None:
        """Retrieve policy for a capability provider."""
        key = f"{capability_name}:{provider_id}"
        return self._policies.get(key)

    def check_permission(
        self,
        capability_name: str,
        provider_id: str,
        operation: str,
    ) -> bool:
        """Check if an operation is permitted under the provider's sandbox policy."""
        key = f"{capability_name}:{provider_id}"
        policy = self._policies.get(key)
        if policy is None:
            # Default: untrusted, deny dangerous operations
            return operation not in ("filesystem", "network", "subprocess")

        return policy.validate_execution({operation})
