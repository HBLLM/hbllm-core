"""Tests for Bus-Level Tenant Context Propagation.

Validates:
  1. TenantInterceptor stamps ambient context onto messages
  2. restore_tenant_context restores context from message fields
  3. Bus dispatch wraps handlers in correct tenant scope
  4. TaskCapsule auto-populates tenant_id from ambient context
  5. End-to-end: publish inside TenantContext → handler sees correct tenant
"""

import asyncio

import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.security.tenant_guard import TenantContext, get_current_tenant
from hbllm.security.tenant_interceptor import TenantInterceptor, restore_tenant_context

# ── TenantInterceptor Tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_interceptor_stamps_ambient_context():
    """Message published inside TenantContext gets tenant_id stamped."""
    interceptor = TenantInterceptor()

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="test.event",
    )
    assert msg.tenant_id == "default"

    with TenantContext("acme", user_id="alice", device_id="phone_1"):
        result = await interceptor(msg)

    assert result is not None
    assert result.tenant_id == "acme"
    assert result.user_id == "alice"
    assert result.device_id == "phone_1"


@pytest.mark.asyncio
async def test_interceptor_noop_without_context():
    """Message published without TenantContext keeps original values."""
    interceptor = TenantInterceptor()

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="test.event",
    )

    result = await interceptor(msg)
    assert result is not None
    assert result.tenant_id == "default"
    assert result.user_id == "default"
    assert result.device_id == "default"


@pytest.mark.asyncio
async def test_interceptor_no_overwrite_explicit_values():
    """Message with explicit tenant_id is not overwritten by ambient context."""
    interceptor = TenantInterceptor()

    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="test.event",
        tenant_id="custom_tenant",
        user_id="custom_user",
    )

    with TenantContext("acme", user_id="alice"):
        result = await interceptor(msg)

    assert result is not None
    # Explicit values preserved
    assert result.tenant_id == "custom_tenant"
    assert result.user_id == "custom_user"


# ── restore_tenant_context Tests ────────────────────────────────────────


def test_restore_tenant_context_sets_ambient():
    """Handler running inside restore_tenant_context sees correct tenant."""
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="test.event",
        tenant_id="project_alpha",
        user_id="bob",
        device_id="desktop_1",
    )

    # Before restore — no context
    assert get_current_tenant() is None

    with restore_tenant_context(msg):
        assert get_current_tenant() == "project_alpha"

    # After restore — context cleaned up
    assert get_current_tenant() is None


def test_restore_tenant_context_noop_for_default():
    """No TenantContext is set when message has default tenant_id."""
    msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="test.event",
    )

    with restore_tenant_context(msg):
        # Should remain None since message has "default" tenant_id
        assert get_current_tenant() is None


# ── End-to-End Bus Integration Test ─────────────────────────────────────


@pytest.mark.asyncio
async def test_bus_dispatch_restores_tenant_context():
    """Handlers dispatched by the bus run inside the correct TenantContext."""
    bus = InProcessBus()
    await bus.start()

    # Wire the TenantInterceptor
    bus.add_interceptor(TenantInterceptor())

    captured_tenant = []

    async def handler(msg: Message) -> Message | None:
        captured_tenant.append(get_current_tenant())
        return None

    await bus.subscribe("test.topic", handler)

    # Publish inside a TenantContext
    with TenantContext("workspace_abc"):
        await bus.publish(
            "test.topic",
            Message(
                type=MessageType.EVENT,
                source_node_id="test",
                topic="test.topic",
            ),
        )

    # Give dispatch loop time to process
    await asyncio.sleep(0.3)
    await bus.stop()

    # Handler should have seen the stamped tenant
    assert len(captured_tenant) == 1
    assert captured_tenant[0] == "workspace_abc"


@pytest.mark.asyncio
async def test_bus_dispatch_explicit_tenant_id():
    """Messages with explicit tenant_id are dispatched to handler with correct scope."""
    bus = InProcessBus()
    await bus.start()
    bus.add_interceptor(TenantInterceptor())

    captured_tenant = []

    async def handler(msg: Message) -> Message | None:
        captured_tenant.append(get_current_tenant())
        return None

    await bus.subscribe("test.topic", handler)

    # Publish with explicit tenant_id (no ambient context)
    await bus.publish(
        "test.topic",
        Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="test.topic",
            tenant_id="explicit_tenant",
        ),
    )

    await asyncio.sleep(0.3)
    await bus.stop()

    assert len(captured_tenant) == 1
    assert captured_tenant[0] == "explicit_tenant"


# ── TaskCapsule Auto-Population Tests ───────────────────────────────────


def test_capsule_auto_populates_tenant_from_context():
    """TaskCapsule created inside TenantContext gets tenant_id automatically."""
    from hbllm.brain.mesh.capsule import TaskCapsule

    with TenantContext("ws_123"):
        capsule = TaskCapsule()

    assert capsule.tenant_id == "ws_123"


def test_capsule_default_tenant_without_context():
    """TaskCapsule created without TenantContext gets 'default' tenant_id."""
    from hbllm.brain.mesh.capsule import TaskCapsule

    capsule = TaskCapsule()
    assert capsule.tenant_id == "default"


def test_capsule_explicit_tenant_not_overwritten():
    """TaskCapsule with explicit tenant_id preserves it even inside context."""
    from hbllm.brain.mesh.capsule import TaskCapsule

    with TenantContext("ws_999"):
        capsule = TaskCapsule(tenant_id="explicit")

    assert capsule.tenant_id == "explicit"
