"""Tests for Cluster Configuration — YAML parsing, validation, env interpolation."""

import os
import pytest
import tempfile

from hbllm.network.cluster_config import (
    ClusterConfig,
    ClusterInfo,
    ServerConfig,
    NodeDefaults,
    LoadBalancingConfig,
    load_cluster_config,
    _interpolate_env,
)


# ─── Env Var Interpolation ────────────────────────────────────────────────────

def test_interpolate_env_simple(monkeypatch):
    monkeypatch.setenv("MY_VAR", "hello")
    assert _interpolate_env("${MY_VAR}") == "hello"


def test_interpolate_env_default():
    # Unset var with default
    result = _interpolate_env("${NONEXISTENT_VAR_123:fallback}")
    assert result == "fallback"


def test_interpolate_env_missing_no_default():
    result = _interpolate_env("${NONEXISTENT_VAR_456}")
    assert result == ""


def test_interpolate_env_nested_dict(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "my-redis")
    data = {"url": "redis://${REDIS_HOST}:6379", "name": "test"}
    result = _interpolate_env(data)
    assert result["url"] == "redis://my-redis:6379"
    assert result["name"] == "test"


def test_interpolate_env_list(monkeypatch):
    monkeypatch.setenv("HOST", "example.com")
    data = ["${HOST}", "other"]
    result = _interpolate_env(data)
    assert result[0] == "example.com"


# ─── ClusterConfig Model ─────────────────────────────────────────────────────

def test_default_cluster_config():
    config = ClusterConfig()
    assert config.cluster.name == "hbllm-default"
    assert config.cluster.redis_url == "redis://localhost:6379"
    assert len(config.servers) == 0


def test_server_config():
    server = ServerConfig(
        host="gpu-1.internal",
        port=8001,
        nodes=["planner", "critic"],
        description="Reasoning",
    )
    assert server.host == "gpu-1.internal"
    assert len(server.nodes) == 2


def test_get_server():
    config = ClusterConfig(
        servers={
            "gateway": ServerConfig(nodes=["router"]),
            "memory": ServerConfig(nodes=["memory"]),
        }
    )
    assert config.get_server("gateway").nodes == ["router"]


def test_get_server_not_found():
    config = ClusterConfig()
    with pytest.raises(KeyError, match="not found"):
        config.get_server("nonexistent")


def test_get_all_nodes():
    config = ClusterConfig(
        servers={
            "a": ServerConfig(nodes=["router", "workspace"]),
            "b": ServerConfig(nodes=["planner", "router"]),  # duplicate router
        }
    )
    all_nodes = config.get_all_nodes()
    assert all_nodes == ["router", "workspace", "planner"]  # no dupes


def test_get_server_for_node():
    config = ClusterConfig(
        servers={
            "gateway": ServerConfig(nodes=["router"]),
            "reasoning": ServerConfig(nodes=["planner"]),
        }
    )
    assert config.get_server_for_node("router") == "gateway"
    assert config.get_server_for_node("planner") == "reasoning"
    assert config.get_server_for_node("unknown") is None


# ─── Topology Validation ─────────────────────────────────────────────────────

def test_validate_duplicate_nodes():
    config = ClusterConfig(
        servers={
            "a": ServerConfig(nodes=["router"]),
            "b": ServerConfig(nodes=["router"]),
        }
    )
    warnings = config.validate_topology()
    assert any("router" in w for w in warnings)


def test_validate_empty_server():
    config = ClusterConfig(
        servers={"empty": ServerConfig(nodes=[])}
    )
    warnings = config.validate_topology()
    assert any("no nodes" in w for w in warnings)


def test_validate_no_redis_multi_server():
    config = ClusterConfig(
        cluster=ClusterInfo(redis_url=""),
        servers={
            "a": ServerConfig(nodes=["router"]),
            "b": ServerConfig(nodes=["planner"]),
        },
    )
    warnings = config.validate_topology()
    assert any("redis_url" in w for w in warnings)


def test_validate_clean():
    config = ClusterConfig(
        cluster=ClusterInfo(redis_url="redis://localhost:6379"),
        servers={"gateway": ServerConfig(nodes=["router"])},
    )
    warnings = config.validate_topology()
    assert warnings == []


# ─── YAML Loading ─────────────────────────────────────────────────────────────

def test_load_from_yaml():
    yaml_content = """
cluster:
  name: "test-cluster"
  redis_url: "redis://test:6379"

servers:
  gw:
    host: "0.0.0.0"
    port: 9000
    nodes:
      - router
      - workspace
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = load_cluster_config(f.name)

    os.unlink(f.name)

    assert config.cluster.name == "test-cluster"
    assert "gw" in config.servers
    assert config.servers["gw"].nodes == ["router", "workspace"]
    assert config.servers["gw"].port == 9000


def test_load_missing_file():
    config = load_cluster_config("/nonexistent/path.yaml")
    assert isinstance(config, ClusterConfig)
    assert config.cluster.name == "hbllm-default"


def test_load_with_env_interpolation(monkeypatch):
    monkeypatch.setenv("TEST_SECRET", "my-secret-key")
    yaml_content = """
cluster:
  auth_secret: "${TEST_SECRET}"
servers:
  gw:
    nodes: [router]
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = load_cluster_config(f.name)

    os.unlink(f.name)
    assert config.cluster.auth_secret == "my-secret-key"
