"""Unit tests for sandboxed capability dispatch in HCIR kernel."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from hbllm.hcir.kernel.capability_resolver import (
    CapabilityImplementation,
    CapabilityResolver,
    ICapabilityExecutor,
)
from hbllm.hcir.kernel.capability_sandboxing import (
    CapabilityPermissions,
    CapabilityResourceLimits,
    CapabilitySandboxManager,
    IsolationMode,
    SandboxedCapabilityPolicy,
    TrustLevel,
)
from hbllm.hcir.kernel.scheduler import KernelInstructionScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.workspace import HCIRWorkspaceState


class MockExecutor(ICapabilityExecutor):
    """Mock capability executor for testing."""

    def __init__(self, delay: float = 0.0, fail: bool = False, return_data: Any = None) -> None:
        self.delay = delay
        self.fail = fail
        self.return_data = return_data or {"status": "ok"}
        self.executed_count = 0

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        self.executed_count += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("Mock execution failure")
        return {**self.return_data, "echo": params}


@pytest.mark.asyncio
async def test_capability_resolver_without_sandbox():
    """Unsandboxed resolver dispatches directly."""
    resolver = CapabilityResolver()
    executor = MockExecutor()
    resolver.register(
        CapabilityImplementation(
            capability_name="test_cap",
            implementation_id="impl_1",
            executor=executor,
            priority=10,
        )
    )

    result = await resolver.resolve_and_execute("test_cap", {"query": "hello"})
    assert result["status"] == "ok"
    assert result["echo"]["query"] == "hello"
    assert executor.executed_count == 1


@pytest.mark.asyncio
async def test_capability_resolver_sandbox_permission_check():
    """Sandbox manager denies unauthorized operations."""
    sandbox_mgr = CapabilitySandboxManager()
    policy = SandboxedCapabilityPolicy(
        capability_name="file_reader",
        provider_id="provider_local",
        trust_level=TrustLevel.VERIFIED,
        isolation_mode=IsolationMode.IN_PROCESS,
        permissions=CapabilityPermissions(
            allow_filesystem=False,  # Disallowed
            allow_network=True,  # Allowed
        ),
        resource_limits=CapabilityResourceLimits(timeout_seconds=5.0),
    )
    sandbox_mgr.register_policy(policy)

    resolver = CapabilityResolver(sandbox_manager=sandbox_mgr)
    executor = MockExecutor()
    resolver.register(
        CapabilityImplementation(
            capability_name="file_reader",
            implementation_id="provider_local",
            executor=executor,
            priority=10,
        )
    )

    # 1. Request filesystem permission -> Blocked
    result_denied = await resolver.resolve_and_execute(
        "file_reader",
        {"path": "/etc/passwd"},
        required_permissions={"filesystem"},
    )
    assert "error" in result_denied
    assert "Sandbox policy violation" in result_denied["error"]
    assert "filesystem" in result_denied["error"]
    assert executor.executed_count == 0  # Not executed!

    # 2. Request network permission -> Permitted
    result_allowed = await resolver.resolve_and_execute(
        "file_reader",
        {"url": "https://example.com"},
        required_permissions={"network"},
    )
    assert result_allowed["status"] == "ok"
    assert executor.executed_count == 1


@pytest.mark.asyncio
async def test_capability_resolver_sandbox_timeout_enforcement():
    """Sandbox resource limits enforce execution timeout."""
    sandbox_mgr = CapabilitySandboxManager()
    policy = SandboxedCapabilityPolicy(
        capability_name="slow_task",
        provider_id="slow_impl",
        trust_level=TrustLevel.VERIFIED,
        resource_limits=CapabilityResourceLimits(timeout_seconds=0.05),  # 50ms limit
    )
    sandbox_mgr.register_policy(policy)

    resolver = CapabilityResolver(sandbox_manager=sandbox_mgr)
    slow_executor = MockExecutor(delay=0.2)  # Takes 200ms
    resolver.register(
        CapabilityImplementation(
            capability_name="slow_task",
            implementation_id="slow_impl",
            executor=slow_executor,
            priority=10,
        )
    )

    result = await resolver.resolve_and_execute("slow_task", {"task": "heavy_calc"})
    assert "error" in result
    assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_kernel_services_container_wiring():
    """KernelServices holds sandbox_manager and shares it."""
    ws = HCIRWorkspaceState()
    tx_mgr = TransactionManager(ws)
    sandbox_mgr = CapabilitySandboxManager()
    resolver = CapabilityResolver(sandbox_manager=sandbox_mgr)
    scheduler = KernelInstructionScheduler()

    services = KernelServices(
        workspace=ws,
        transaction_manager=tx_mgr,
        capability_resolver=resolver,
        scheduler=scheduler,
        sandbox_manager=sandbox_mgr,
    )

    assert services.sandbox_manager is sandbox_mgr
    assert services.capability_resolver.sandbox_manager is sandbox_mgr
