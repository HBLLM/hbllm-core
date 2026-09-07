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
from typing import Any

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


_FS_TERMS = {
    "file",
    "fs",
    "path",
    "disk",
    "write",
    "read",
    "save",
    "load",
    "config",
    "dir",
    "folder",
    "mkdir",
    "rmdir",
    "delete",
    "remove",
    "log",
    "store",
    "export",
    "import",
    "dump",
    "append",
    "storage",
    "io",
    "cat",
    "touch",
    "cp",
    "mv",
}
_SUBPROCESS_TERMS = {
    "exec",
    "subprocess",
    "shell",
    "python",
    "script",
    "cmd",
    "command",
    "bash",
    "sh",
    "run",
    "process",
    "spawn",
    "terminal",
    "cli",
    "system",
    "popen",
}
_NETWORK_TERMS = {
    "net",
    "http",
    "https",
    "url",
    "fetch",
    "api",
    "download",
    "upload",
    "web",
    "socket",
    "request",
    "remote",
    "curl",
    "dns",
    "ping",
    "scrape",
    "webhook",
    "connect",
    "endpoint",
    "ip",
}
_DB_TERMS = {
    "db",
    "database",
    "sql",
    "query",
    "table",
    "record",
    "insert",
    "update",
    "upsert",
    "mutate",
    "persist",
    "commit",
    "sqlite",
    "postgres",
    "mysql",
}


def infer_capability_permissions(
    capability_name: str,
    tags: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> set[str]:
    """Auto-infer required permissions from capability name, tags, and runtime parameters."""
    perms: set[str] = set()
    cap_lower = capability_name.lower().replace("-", "_")
    cap_tokens = set(cap_lower.split("_"))

    # Name-based matching (exact token or substring)
    if any(t in cap_tokens or t in cap_lower for t in _FS_TERMS):
        perms.add("filesystem")
    if any(t in cap_tokens or t in cap_lower for t in _SUBPROCESS_TERMS):
        perms.add("subprocess")
    if any(t in cap_tokens or t in cap_lower for t in _NETWORK_TERMS):
        perms.add("network")
    if any(t in cap_tokens or t in cap_lower for t in _DB_TERMS):
        perms.add("db_write")

    # Tag-based matching
    if tags:
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in ("filesystem", "fs", "io"):
                perms.add("filesystem")
            elif tag_lower in ("subprocess", "shell", "cmd"):
                perms.add("subprocess")
            elif tag_lower in ("network", "net", "web", "api"):
                perms.add("network")
            elif tag_lower in ("db", "database", "sql"):
                perms.add("db_write")

    # Parameter-based matching
    if params:
        for key in params.keys():
            k_lower = key.lower()
            if k_lower in (
                "path",
                "filepath",
                "filename",
                "file",
                "dir",
                "directory",
                "dest",
                "src",
            ):
                perms.add("filesystem")
            elif k_lower in ("url", "endpoint", "host", "domain", "web_url"):
                perms.add("network")
            elif k_lower in ("cmd", "command", "script", "code", "shell_cmd"):
                perms.add("subprocess")
            elif k_lower in ("table", "sql", "db", "database", "query"):
                perms.add("db_write")

    return perms


@dataclass
class SandboxedCapabilityPolicy:
    """Complete security policy bound to a capability provider."""

    capability_name: str
    provider_id: str
    trust_level: TrustLevel = TrustLevel.VERIFIED
    isolation_mode: IsolationMode = IsolationMode.IN_PROCESS
    permissions: CapabilityPermissions = field(default_factory=CapabilityPermissions)
    resource_limits: CapabilityResourceLimits = field(default_factory=CapabilityResourceLimits)
    required_permissions: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if isinstance(self.required_permissions, (list, tuple)):
            self.required_permissions = set(self.required_permissions)
        if not self.required_permissions:
            self.required_permissions = infer_capability_permissions(self.capability_name)

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
        if "db_write" in requested_permissions and not self.permissions.allow_db_write:
            logger.warning("Sandbox violation: %s denied db_write access", self.provider_id)
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
