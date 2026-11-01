"""Unit tests for WP6: Sandboxed capability execution and manifest least-privilege enforcement."""

from __future__ import annotations

import os

import pytest

from hbllm.hcir.kernel.capability_resolver import (
    CapabilityImplementation,
    CapabilityResolver,
)
from hbllm.hcir.kernel.capability_sandboxing import (
    CapabilityPermissions,
    CapabilityResourceLimits,
    CapabilitySandboxManager,
    SandboxedCapabilityPolicy,
    SandboxedExecutor,
    TrustLevel,
)
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.workspace import HCIRWorkspaceState


class DummyExecutor:
    """Mock executor for testing sandboxing."""

    def __init__(self) -> None:
        self.last_params: dict | None = None
        self.seen_sandbox_env: str | None = None

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict) -> dict:
        self.last_params = params
        self.seen_sandbox_env = os.environ.get("HBLLM_SANDBOX")
        return {"result": "ok", "seen_env": self.seen_sandbox_env}


@pytest.mark.asyncio
async def test_least_privilege_downgrade_in_default_services():
    """Verify built-in capabilities in KernelServices are strictly least-privilege."""
    ws = HCIRWorkspaceState()
    services = KernelServices.create_default(ws)
    sandbox_mgr = services.sandbox_manager
    assert sandbox_mgr is not None

    # world_prediction should not permit network or filesystem
    assert (
        sandbox_mgr.check_permission("world_prediction", "default_world_prediction", "network")
        is False
    )
    assert (
        sandbox_mgr.check_permission("world_prediction", "default_world_prediction", "filesystem")
        is False
    )
    assert (
        sandbox_mgr.check_permission("world_prediction", "default_world_prediction", "subprocess")
        is False
    )

    # Caller-supplied permission requests are ignored (manifest-only, WP6.2):
    # check_permission still denies network for the provider.
    assert (
        sandbox_mgr.check_permission("world_prediction", "default_world_prediction", "network")
        is False
    )


@pytest.mark.asyncio
async def test_sandboxed_executor_ambient_environment():
    """Verify SandboxedExecutor sets HBLLM_SANDBOX=1 during execution and cleans up."""
    dummy = DummyExecutor()
    executor = SandboxedExecutor(inner_executor=dummy)

    assert os.environ.get("HBLLM_SANDBOX") is None
    res = await executor.execute({"task": "compute"})
    assert res.get("status") == "SUCCESS" or res.get("sandboxed") is True
    assert res.get("seen_env") == "1"
    # Ensure environment cleaned up after execution
    assert os.environ.get("HBLLM_SANDBOX") is None


@pytest.mark.asyncio
async def test_sandboxed_executor_python_subprocess():
    """Verify SandboxedExecutor runs python code in isolated subprocess with strict limits."""
    executor = SandboxedExecutor(
        permissions=CapabilityPermissions(allow_network=False),
    )
    res = await executor.execute({"code": "x = 42\nprint('result is', x)"})
    assert res["status"] == "SUCCESS"
    assert "result is 42" in res["output"]
    assert res["sandboxed"] is True


@pytest.mark.asyncio
async def test_capability_manifest_static_permissions_enforced():
    """Verify static required_permissions on CapabilityImplementation trigger sandbox violations."""
    sandbox_mgr = CapabilitySandboxManager()
    resolver = CapabilityResolver(sandbox_manager=sandbox_mgr)

    dummy = DummyExecutor()
    # Implementation statically requires filesystem
    impl = CapabilityImplementation(
        capability_name="custom_file_writer",
        implementation_id="writer_1",
        executor=dummy,
        required_permissions={"filesystem"},
    )
    resolver.register(impl)

    # Register policy denying filesystem
    policy = SandboxedCapabilityPolicy(
        capability_name="custom_file_writer",
        provider_id="writer_1",
        trust_level=TrustLevel.UNTRUSTED,
        permissions=CapabilityPermissions(allow_filesystem=False),
        required_permissions={"filesystem"},
    )
    sandbox_mgr.register_policy(policy)

    res = await resolver.resolve_and_execute("custom_file_writer", {"path": "/tmp/test"})
    assert "error" in res
    assert "permission 'filesystem' denied" in res["error"]


# ── WP6.2: real subprocess isolation for non-SYSTEM capabilities ─────────────


class SocketOpenExecutor:
    """Opens a network socket (module-level for spawn pickling)."""

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict) -> dict:
        import socket

        socket.socket().connect(("127.0.0.1", 8080))
        return {"x": 1}


class OutsideWriteExecutor:
    """Writes a file outside its sandbox temp dir."""

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict) -> dict:
        import tempfile

        with open(f"{tempfile.gettempdir()}/sandbox_escape_{id(self)}.txt", "w") as f:
            f.write("escape")
        return {"x": 1}


class InsideWriteExecutor:
    """Writes a file inside the child's cwd (its temp dir)."""

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict) -> dict:
        with open("sandbox_local.txt", "w") as f:
            f.write("inside")
        return {"x": 1}


class SubprocessExecutor:
    """Spawns a subprocess."""

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict) -> dict:
        import subprocess

        subprocess.run(["echo", "hi"])
        return {"x": 1}


class SleepExecutor:
    """Sleeps far past the sandbox timeout."""

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict) -> dict:
        import asyncio

        await asyncio.sleep(120)
        return {"x": 1}


def _sandbox_executor(inner, perms=None, limits=None):
    return SandboxedExecutor(
        inner_executor=inner,
        permissions=perms or CapabilityPermissions(),
        resource_limits=limits or CapabilityResourceLimits(timeout_seconds=15, max_cpu_seconds=10),
    )


@pytest.mark.asyncio
async def test_sandbox_denies_network_for_untrusted():
    with pytest.raises(PermissionError):
        await _sandbox_executor(SocketOpenExecutor()).execute({})


@pytest.mark.asyncio
async def test_sandbox_denies_write_outside_tempdir():
    with pytest.raises(PermissionError):
        await _sandbox_executor(OutsideWriteExecutor()).execute({})


@pytest.mark.asyncio
async def test_sandbox_allows_write_inside_tempdir():
    res = await _sandbox_executor(InsideWriteExecutor()).execute({})
    assert res["x"] == 1 and res["sandboxed"] is True


@pytest.mark.asyncio
async def test_sandbox_denies_subprocess_spawn():
    with pytest.raises(PermissionError):
        await _sandbox_executor(SubprocessExecutor()).execute({})


@pytest.mark.asyncio
async def test_sandbox_wallclock_timeout():
    with pytest.raises(TimeoutError):
        await _sandbox_executor(
            SleepExecutor(), limits=CapabilityResourceLimits(timeout_seconds=3, max_cpu_seconds=30)
        ).execute({})


@pytest.mark.asyncio
async def test_sandbox_denies_unpicklable_executor():
    class Unpicklable:
        @property
        def is_available(self) -> bool:
            return True

        async def execute(self, params: dict) -> dict:
            return {}

    bad = Unpicklable()
    bad._fn = lambda: None  # lambdas are not picklable
    with pytest.raises(PermissionError):
        await _sandbox_executor(bad).execute({})


@pytest.mark.asyncio
async def test_register_denies_manifest_beyond_policy():
    """A manifest requiring 'network' against a policy without it fails at register."""
    from hbllm.hcir.kernel.capability_sandboxing import (
        SandboxedCapabilityPolicy,
        TrustLevel,
    )

    mgr = CapabilitySandboxManager()
    mgr.register_policy(
        SandboxedCapabilityPolicy(
            capability_name="net_capability",
            provider_id="net_impl",
            trust_level=TrustLevel.UNTRUSTED,
            permissions=CapabilityPermissions(allow_network=False),
        )
    )
    resolver = CapabilityResolver(sandbox_manager=mgr)
    impl = CapabilityImplementation(
        capability_name="net_capability",
        implementation_id="net_impl",
        executor=SocketOpenExecutor(),
        required_permissions={"network"},
    )
    with pytest.raises(PermissionError):
        resolver.register(impl)


@pytest.mark.asyncio
async def test_system_capability_runs_in_process(tmp_path, monkeypatch):
    """SYSTEM-trusted capabilities bypass subprocess isolation."""
    monkeypatch.chdir(tmp_path)  # in-process writes land in tmp_path, not repo cwd
    from hbllm.hcir.kernel.capability_sandboxing import (
        SandboxedCapabilityPolicy,
        TrustLevel,
    )

    mgr = CapabilitySandboxManager()
    resolver = CapabilityResolver(sandbox_manager=mgr)
    impl = CapabilityImplementation(
        capability_name="sys_cap",
        implementation_id="sys_impl",
        executor=InsideWriteExecutor(),
    )
    mgr.register_policy(
        SandboxedCapabilityPolicy(
            capability_name="sys_cap",
            provider_id="sys_impl",
            trust_level=TrustLevel.SYSTEM,
        )
    )
    resolver.register(impl)
    res = await resolver.resolve_and_execute("sys_cap", {})
    assert res.get("x") == 1
