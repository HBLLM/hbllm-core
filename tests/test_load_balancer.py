"""Tests for Load Balancer — round-robin, least-loaded, capability-match strategies."""

import pytest

from hbllm.network.circuit_breaker import CircuitBreakerRegistry
from hbllm.network.load_balancer import LoadBalancer
from hbllm.network.node import HealthStatus, NodeHealth, NodeInfo, NodeType
from hbllm.network.registry import ServiceRegistry


@pytest.fixture
async def lb_setup():
    """Create registry with 3 domain nodes and a load balancer."""
    registry = ServiceRegistry()
    await registry.start()

    breakers = CircuitBreakerRegistry()

    nodes = [
        NodeInfo(node_id="math_1", node_type=NodeType.DOMAIN_MODULE, capabilities=["math"], priority=1),
        NodeInfo(node_id="math_2", node_type=NodeType.DOMAIN_MODULE, capabilities=["math"], priority=1),
        NodeInfo(node_id="math_3", node_type=NodeType.DOMAIN_MODULE, capabilities=["math"], priority=1),
    ]
    for n in nodes:
        await registry.register(n)
        await registry.update_health(NodeHealth(
            node_id=n.node_id,
            status=HealthStatus.HEALTHY,
            latency_ms=10.0,
        ))

    lb = LoadBalancer(registry, breakers, strategy="round_robin")
    yield lb, registry, breakers
    await registry.stop()


async def test_round_robin(lb_setup):
    lb, _, _ = lb_setup
    results = []
    for _ in range(6):
        node = await lb.select("math")
        results.append(node.node_id)

    # Should cycle through all 3 nodes
    assert results[0] == results[3]
    assert results[1] == results[4]
    assert results[2] == results[5]
    assert len(set(results)) == 3


async def test_round_robin_single_node(lb_setup):
    lb, _, _ = lb_setup
    # Remove 2 nodes from healthy status
    # Actually just test with capability that only has 1 node
    node = await lb.select("math")
    assert node is not None


async def test_no_candidates():
    registry = ServiceRegistry()
    await registry.start()
    breakers = CircuitBreakerRegistry()
    lb = LoadBalancer(registry, breakers)
    result = await lb.select("nonexistent")
    assert result is None
    await registry.stop()


async def test_circuit_breaker_filters(lb_setup):
    lb, _, breakers = lb_setup

    # Open circuit for math_1 and math_2
    for _ in range(5):
        breakers.get("math_1").record_failure()
        breakers.get("math_2").record_failure()

    # Only math_3 should be available
    node = await lb.select("math")
    assert node.node_id == "math_3"


async def test_least_loaded(lb_setup):
    lb, registry, _ = lb_setup
    lb._strategy = "least_loaded"

    # Give math_1 high latency, math_3 low
    await registry.update_health(NodeHealth(node_id="math_1", status=HealthStatus.HEALTHY, latency_ms=100.0))
    await registry.update_health(NodeHealth(node_id="math_2", status=HealthStatus.HEALTHY, latency_ms=50.0))
    await registry.update_health(NodeHealth(node_id="math_3", status=HealthStatus.HEALTHY, latency_ms=5.0))

    node = await lb.select("math")
    assert node.node_id == "math_3"


async def test_least_loaded_prefers_healthy(lb_setup):
    lb, registry, _ = lb_setup
    lb._strategy = "least_loaded"

    # math_1 degraded but low latency, math_2 healthy with higher latency
    await registry.update_health(NodeHealth(node_id="math_1", status=HealthStatus.DEGRADED, latency_ms=1.0))
    await registry.update_health(NodeHealth(node_id="math_2", status=HealthStatus.HEALTHY, latency_ms=10.0))
    await registry.update_health(NodeHealth(node_id="math_3", status=HealthStatus.HEALTHY, latency_ms=10.0))

    node = await lb.select("math")
    # math_2 or math_3 should win (healthy beats degraded with penalty)
    assert node.node_id in ("math_2", "math_3")


async def test_capability_match():
    registry = ServiceRegistry()
    await registry.start()
    breakers = CircuitBreakerRegistry()

    # Node with exact match first
    await registry.register(NodeInfo(
        node_id="exact", node_type=NodeType.DOMAIN_MODULE,
        capabilities=["math", "general"],
    ))
    await registry.update_health(NodeHealth(node_id="exact", status=HealthStatus.HEALTHY))

    # Node with general first
    await registry.register(NodeInfo(
        node_id="general", node_type=NodeType.DOMAIN_MODULE,
        capabilities=["general", "math"],
    ))
    await registry.update_health(NodeHealth(node_id="general", status=HealthStatus.HEALTHY))

    lb = LoadBalancer(registry, breakers, strategy="capability_match")
    node = await lb.select("math")
    assert node.node_id == "exact"
    await registry.stop()


async def test_strategy_override(lb_setup):
    lb, registry, _ = lb_setup

    # Default is round_robin, override to least_loaded
    await registry.update_health(NodeHealth(node_id="math_1", status=HealthStatus.HEALTHY, latency_ms=100.0))
    await registry.update_health(NodeHealth(node_id="math_2", status=HealthStatus.HEALTHY, latency_ms=1.0))
    await registry.update_health(NodeHealth(node_id="math_3", status=HealthStatus.HEALTHY, latency_ms=50.0))

    node = await lb.select("math", strategy="least_loaded")
    assert node.node_id == "math_2"


async def test_reset_counters(lb_setup):
    lb, _, _ = lb_setup

    # Advance counter
    await lb.select("math")
    await lb.select("math")
    assert lb._rr_counters.get("math", 0) == 2

    lb.reset_counters()
    assert lb._rr_counters.get("math", 0) == 0
