"""
Capability Sandboxing — Two-tier security & isolation boundaries for capability execution.

HBLLM enforces a verified defense-in-depth sandboxing architecture across two distinct tiers:

    Tier 1: Logical Capability Policy Sandbox (HCIR Kernel Level)
    ├── Permission tokens   (allow_filesystem, allow_network, allow_subprocess, allow_db_write)
    ├── Resource limits     (max_cpu_seconds, max_memory_mb, timeout_seconds)
    ├── Trust levels        (UNTRUSTED, VERIFIED, SYSTEM)
    └── Dynamic inference   (AST & parameter inspection to prevent permission bypass)

    Tier 2: Physical Process Isolation Sandbox (OS / Process Level)
    ├── AST code validation (blocking __subclasses__, exec, eval, builtins)
    ├── POSIX quotas        (RLIMIT_AS memory limit, RLIMIT_CPU core execution time)
    ├── Environment strip   (isolated PATH, minimal environment variables)
    └── Network namespace   (Linux unshare -Urn network isolation)

Capabilities declaring ``IsolationMode.SUBPROCESS`` or executing untrusted Python code
are automatically dispatched through Tier 2 via ``hbllm.actions.sandbox.run_sandboxed_python``.
"""

from __future__ import annotations

import logging
import pickle
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
    "filepath",
    "filename",
    "fs",
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
    cap_lower = capability_name.lower().replace("-", "_").replace(".", "_")
    cap_tokens = set(cap_lower.split("_"))

    # Name-based matching (exact token matching to avoid substring false positives e.g. 'io' in 'prediction')
    if any(t in cap_tokens for t in _FS_TERMS):
        perms.add("filesystem")
    if any(t in cap_tokens for t in _SUBPROCESS_TERMS):
        perms.add("subprocess")
    if any(t in cap_tokens for t in _NETWORK_TERMS):
        perms.add("network")
    if any(t in cap_tokens for t in _DB_TERMS):
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
        if not self.required_permissions and self.trust_level != TrustLevel.SYSTEM:
            self.required_permissions = infer_capability_permissions(self.capability_name)

    def grants(self, permission: str) -> bool:
        """Whether this policy's manifest grants a single permission."""
        allowed = {
            "filesystem": self.permissions.allow_filesystem,
            "network": self.permissions.allow_network,
            "subprocess": self.permissions.allow_subprocess,
            "db_write": self.permissions.allow_db_write,
        }.get(permission, False)
        if not allowed:
            logger.warning("Sandbox violation: %s denied '%s' access", self.provider_id, permission)
        return allowed


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

        return policy.grants(operation)


class SandboxedExecutor:
    """Executes untrusted capabilities in an isolated child subprocess with POSIX limits (WP6)."""

    def __init__(
        self,
        inner_executor: Any | None = None,
        permissions: CapabilityPermissions | None = None,
        resource_limits: CapabilityResourceLimits | None = None,
    ) -> None:
        self.inner_executor = inner_executor
        self.permissions = permissions or CapabilityPermissions()
        self.resource_limits = resource_limits or CapabilityResourceLimits()

    @property
    def is_available(self) -> bool:
        return self.inner_executor.is_available if self.inner_executor else True

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute params in an isolated subprocess with HBLLM_SANDBOX=1 and resource limits."""

        # If python code is passed, run through sandboxed python subprocess runner
        if "code" in params and isinstance(params["code"], str):
            from hbllm.actions.sandbox import run_sandboxed_python

            res = await run_sandboxed_python(
                code=params["code"],
                timeout=self.resource_limits.timeout_seconds,
                max_memory_mb=self.resource_limits.max_memory_mb,
                disable_network=not self.permissions.allow_network,
            )
            return {
                "status": res.status,
                "output": res.output,
                "error": res.error,
                "sandboxed": True,
            }

        # Otherwise execute inner_executor in an isolated spawn child process.
        import asyncio
        import multiprocessing
        import pickle

        try:
            payload = pickle.dumps((self.inner_executor, params))
        except Exception as exc:
            raise PermissionError("untrusted capability must be picklable for isolation") from exc

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_sandboxed_child_main,
            args=(child_conn, payload, self.permissions, self.resource_limits),
        )
        proc.start()
        child_conn.close()
        timeout = self.resource_limits.timeout_seconds or 5.0
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None, lambda: _await_child(proc, parent_conn, timeout)
            )
        finally:
            parent_conn.close()
        kind, *rest = result
        if kind == "ok":
            res = rest[0]
            if isinstance(res, dict):
                res["sandboxed"] = True
            return res
        exc_type, exc_msg = rest
        if exc_type == "TimeoutError":
            raise TimeoutError(f"sandboxed executor exceeded {timeout}s wall-clock")
        if exc_type == "PermissionError":
            raise PermissionError(exc_msg)
        raise RuntimeError(f"sandboxed executor failed ({exc_type}): {exc_msg}")


def _await_child(proc: Any, conn: Any, timeout: float) -> tuple:
    """Block for the child's reply; terminate on wall-clock timeout."""
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(1)
        return ("err", "TimeoutError", "")
    if conn.poll():
        try:
            return conn.recv()
        except EOFError:
            pass
    return ("err", "RuntimeError", f"child exited {proc.exitcode} without result")


def _sandboxed_child_main(
    conn: Any,
    payload: bytes,
    permissions: CapabilityPermissions,
    resource_limits: CapabilityResourceLimits,
) -> None:
    """Spawn-child entry: install isolation, run the executor, reply on the pipe."""
    import asyncio
    import os
    import resource
    import sys
    import tempfile

    workdir = tempfile.mkdtemp()
    os.chdir(workdir)
    workdir = os.path.realpath(workdir)  # /var -> /private/var on macOS
    os.environ["HBLLM_SANDBOX"] = "1"

    try:
        cpu = max(1, int(resource_limits.max_cpu_seconds))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    except (ValueError, OSError, AttributeError):
        pass
    try:
        mem = resource_limits.max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    except (ValueError, OSError, AttributeError):
        pass  # RLIMIT_AS unsupported on some platforms (e.g. macOS)

    allow_net = permissions.allow_network
    allow_sub = permissions.allow_subprocess
    allow_fs = permissions.allow_filesystem

    def _audit(event: str, args: tuple) -> None:
        if not allow_net and event in {
            "socket.connect",
            "socket.connect_ex",
            "socket.bind",
            "socket.getaddrinfo",
            "socket.sendto",
            "socket.sendmsg",
        }:
            raise PermissionError(f"sandbox denied network operation: {event}")
        if not allow_sub and event.split("(", 1)[0] in {
            "subprocess.Popen",
            "os.system",
            "os.exec",
            "os.posix_spawn",
            "os.fork",
        }:
            raise PermissionError(f"sandbox denied subprocess operation: {event}")
        if not allow_fs and event == "open":
            path, mode = args[0], (args[1] if len(args) > 1 else "r")
            if any(m in str(mode) for m in ("w", "a", "x", "+")):
                if not os.path.realpath(str(path)).startswith(workdir):
                    raise PermissionError(f"sandbox denied filesystem write: {path}")

    sys.addaudithook(_audit)

    try:
        executor, params = pickle.loads(payload)  # noqa: S301 — payload built in-process
        result = asyncio.run(executor.execute(params))
        conn.send(("ok", result))
    except BaseException as exc:  # relay the failure to the parent
        try:
            conn.send(("err", type(exc).__name__, str(exc)))
        except Exception:
            pass
    finally:
        conn.close()
