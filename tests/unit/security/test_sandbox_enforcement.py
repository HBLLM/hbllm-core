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
    resolver = services.capability_resolver
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

    # Attempting to request network on world_prediction must fail
    res = await resolver.resolve_and_execute(
        "world_prediction",
        {"code": "import socket"},
        required_permissions={"network"},
    )
    assert "error" in res
    assert "Sandbox policy violation" in res["error"]


@pytest.mark.asyncio
async def test_sandboxed_executor_ambient_environment():
    """Verify SandboxedExecutor sets HBLLM_SANDBOX=1 during execution and cleans up."""
    dummy = DummyExecutor()
    executor = SandboxedExecutor(inner_executor=dummy)

    assert os.environ.get("HBLLM_SANDBOX") is None
    res = await executor.execute({"task": "compute"})
    assert res.get("status") == "SUCCESS" or res.get("sandboxed") is True
    assert dummy.seen_sandbox_env == "1"
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
