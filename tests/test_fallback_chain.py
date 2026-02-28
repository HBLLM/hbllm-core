"""Tests for FallbackManager — graceful degradation and chain resolution."""

import pytest

from hbllm.network.circuit_breaker import CircuitBreakerRegistry
from hbllm.network.fallback import FallbackManager, FallbackResult
from hbllm.network.node import HealthStatus, NodeHealth, NodeInfo, NodeType
from hbllm.network.registry import ServiceRegistry


@pytest.fixture
async def setup():
    """Registry with coding (primary) and general (fallback) nodes."""
    registry = ServiceRegistry()
    await registry.start()
    breakers = CircuitBreakerRegistry(failure_threshold=2)

    # Register coding specialist
    await registry.register(NodeInfo(
        node_id="coding_01",
        node_type=NodeType.DOMAIN_MODULE,
        capabilities=["coding"],
    ))
    await registry.update_health(NodeHealth(
        node_id="coding_01", status=HealthStatus.HEALTHY,
    ))

    # Register general fallback
    await registry.register(NodeInfo(
        node_id="general_01",
        node_type=NodeType.DOMAIN_MODULE,
        capabilities=["general"],
    ))
    await registry.update_health(NodeHealth(
        node_id="general_01", status=HealthStatus.HEALTHY,
    ))

    fm = FallbackManager(registry, breakers)
    fm.register_chain("coding", ["coding", "general"])

    yield fm, registry, breakers
    await registry.stop()


async def test_resolve_primary(setup):
    """Primary node selected when healthy."""
    fm, _, _ = setup
    result = await fm.resolve("coding")
    assert result is not None
    assert result.target_node_id == "coding_01"
    assert result.is_fallback is False
    assert result.degraded_message is None


async def test_resolve_fallback_on_circuit_open(setup):
    """Falls back to general when coding circuit is open."""
    fm, _, breakers = setup
    breakers.get("coding_01").record_failure()
    breakers.get("coding_01").record_failure()
    # Circuit now open for coding_01

    result = await fm.resolve("coding")
    assert result is not None
    assert result.target_node_id == "general_01"
    assert result.is_fallback is True
    assert result.degraded_message is not None


async def test_no_available_node():
    """Returns None when all nodes in chain are down."""
    registry = ServiceRegistry()
    await registry.start()
    breakers = CircuitBreakerRegistry(failure_threshold=1)

    fm = FallbackManager(registry, breakers)
    fm.register_chain("unknown", ["unknown"])

    result = await fm.resolve("unknown")
    assert result is None
    await registry.stop()


async def test_resolve_no_chain_configured(setup):
    """Direct discovery when no chain is configured for capability."""
    fm, _, _ = setup

    # "general" has no explicit chain — falls back to direct discovery
    result = await fm.resolve("general")
    assert result is not None
    assert result.target_node_id == "general_01"
    assert result.is_fallback is False


async def test_system_status_healthy(setup):
    """System status shows healthy when primary is up."""
    fm, _, _ = setup
    status = await fm.get_system_status()

    assert "coding" in status
    assert status["coding"]["status"] == "healthy"
    assert "✅" in status["coding"]["message"]


async def test_system_status_degraded(setup):
    """System status shows degraded when using fallback."""
    fm, _, breakers = setup
    breakers.get("coding_01").record_failure()
    breakers.get("coding_01").record_failure()

    status = await fm.get_system_status()
    assert status["coding"]["status"] == "degraded"
    assert "⚠️" in status["coding"]["message"]


async def test_chain_position():
    """Chain position reflects which fallback level is being used."""
    registry = ServiceRegistry()
    await registry.start()
    breakers = CircuitBreakerRegistry(failure_threshold=1)

    # Register 3 levels
    for i, cap in enumerate(["expert", "standard", "basic"]):
        await registry.register(NodeInfo(
            node_id=f"node_{cap}",
            node_type=NodeType.DOMAIN_MODULE,
            capabilities=[cap],
        ))
        await registry.update_health(NodeHealth(
            node_id=f"node_{cap}", status=HealthStatus.HEALTHY,
        ))

    fm = FallbackManager(registry, breakers)
    fm.register_chain("expert", ["expert", "standard", "basic"])

    # Expert down
    breakers.get("node_expert").record_failure()
    result = await fm.resolve("expert")
    assert result.chain_position == 1

    # Standard also down
    breakers.get("node_standard").record_failure()
    result = await fm.resolve("expert")
    assert result.chain_position == 2

    await registry.stop()


async def test_custom_degraded_messages(setup):
    """Custom degraded messages override defaults."""
    fm, _, breakers = setup
    fm.register_chain(
        "coding",
        ["coding", "general"],
        degraded_messages={"coding": "Custom fallback message"},
    )
    breakers.get("coding_01").record_failure()
    breakers.get("coding_01").record_failure()

    result = await fm.resolve("coding")
    assert result.degraded_message == "Custom fallback message"
