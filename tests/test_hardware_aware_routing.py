from unittest.mock import AsyncMock, MagicMock

import pytest

from hbllm.network.load_balancer import LoadBalancer
from hbllm.network.node import DeviceTier, NodeInfo, NodeType


@pytest.mark.asyncio
async def test_hardware_aware_selection_exact_match():
    node_mobile = NodeInfo(
        node_id="mobile_1",
        node_type=NodeType.ACTION,
        capabilities=["perception"],
        device_tier=DeviceTier.MOBILE,
    )
    node_server = NodeInfo(
        node_id="server_1",
        node_type=NodeType.ACTION,
        capabilities=["perception"],
        device_tier=DeviceTier.SERVER,
    )

    registry = MagicMock()
    registry.discover = AsyncMock(return_value=[node_mobile, node_server])
    circuit_breakers = MagicMock()
    lb = LoadBalancer(registry=registry, circuit_breakers=circuit_breakers)

    # Prefer MOBILE for perception
    winner = await lb.select(
        strategy="hardware_aware", capability="perception", preferred_tier=DeviceTier.MOBILE
    )
    assert winner.node_id == "mobile_1"


@pytest.mark.asyncio
async def test_hardware_aware_fallback_logic():
    # Only server is available for a task that prefers MOBILE
    node_server = NodeInfo(
        node_id="server_1",
        node_type=NodeType.ACTION,
        capabilities=["perception"],
        device_tier=DeviceTier.SERVER,
    )

    registry = MagicMock()
    registry.discover = AsyncMock(return_value=[node_server])
    circuit_breakers = MagicMock()
    lb = LoadBalancer(registry=registry, circuit_breakers=circuit_breakers)

    # Should fallback to SERVER if MOBILE is missing
    winner = await lb.select(
        strategy="hardware_aware", capability="perception", preferred_tier=DeviceTier.MOBILE
    )
    assert winner.node_id == "server_1"


@pytest.mark.asyncio
async def test_hardware_aware_cloud_preference():
    node_cloud = NodeInfo(
        node_id="cloud_1",
        node_type=NodeType.ACTION,
        capabilities=["web_search"],
        device_tier=DeviceTier.CLOUD,
    )
    node_server = NodeInfo(
        node_id="server_1",
        node_type=NodeType.ACTION,
        capabilities=["web_search"],
        device_tier=DeviceTier.SERVER,
    )

    registry = MagicMock()
    registry.discover = AsyncMock(return_value=[node_cloud, node_server])
    circuit_breakers = MagicMock()
    lb = LoadBalancer(registry=registry, circuit_breakers=circuit_breakers)

    # Prefer CLOUD for web_search
    winner = await lb.select(
        strategy="hardware_aware", capability="web_search", preferred_tier=DeviceTier.CLOUD
    )
    assert winner.node_id == "cloud_1"
