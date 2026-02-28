"""
Cluster Configuration — YAML-driven distributed topology.

Parses cluster.yaml to define which nodes run on which server,
Redis connection details, load balancing strategy, and node defaults.
Supports environment variable interpolation (${VAR}) in values.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _interpolate_env(value: Any) -> Any:
    """Recursively interpolate ${VAR} and ${VAR:default} in strings."""
    if isinstance(value, str):
        def _replacer(m: re.Match) -> str:
            var_name = m.group(1)
            default = m.group(2)
            return os.environ.get(var_name, default if default is not None else "")
        return _ENV_VAR_PATTERN.sub(_replacer, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class ClusterInfo(BaseModel):
    """Top-level cluster identity."""
    name: str = "hbllm-default"
    redis_url: str = "redis://localhost:6379"
    auth_secret: str = ""
    serializer: str = "json"


class NodeDefaults(BaseModel):
    """Default parameters for all nodes."""
    heartbeat_interval: float = 10.0
    circuit_breaker_threshold: int = 3
    circuit_breaker_recovery: float = 30.0
    request_timeout: float = 30.0


class LoadBalancingConfig(BaseModel):
    """Load balancing strategy configuration."""
    strategy: str = "round_robin"  # round_robin | least_loaded | capability_match
    health_weight: float = 0.7
    latency_weight: float = 0.3


class ApiConfig(BaseModel):
    """API server configuration for a server role."""
    enabled: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class ServerConfig(BaseModel):
    """Configuration for a single server role."""
    host: str = "0.0.0.0"
    port: int = 8000
    description: str = ""
    nodes: list[str] = Field(default_factory=list)
    api: ApiConfig = Field(default_factory=ApiConfig)


class ClusterConfig(BaseModel):
    """Complete cluster topology configuration."""
    cluster: ClusterInfo = Field(default_factory=ClusterInfo)
    node_defaults: NodeDefaults = Field(default_factory=NodeDefaults)
    load_balancing: LoadBalancingConfig = Field(default_factory=LoadBalancingConfig)
    servers: dict[str, ServerConfig] = Field(default_factory=dict)

    def get_server(self, server_name: str) -> ServerConfig:
        """Get a server config by name, raising if not found."""
        if server_name not in self.servers:
            available = ", ".join(self.servers.keys())
            raise KeyError(
                f"Server '{server_name}' not found in cluster config. "
                f"Available: {available}"
            )
        return self.servers[server_name]

    def get_all_nodes(self) -> list[str]:
        """Get all node IDs across all servers (for --server all mode)."""
        seen: set[str] = set()
        nodes: list[str] = []
        for server in self.servers.values():
            for node in server.nodes:
                if node not in seen:
                    seen.add(node)
                    nodes.append(node)
        return nodes

    def get_server_for_node(self, node_id: str) -> str | None:
        """Find which server role a node belongs to."""
        for server_name, server in self.servers.items():
            if node_id in server.nodes:
                return server_name
        return None

    def validate_topology(self) -> list[str]:
        """
        Validate the cluster topology for common issues.
        Returns a list of warning messages (empty = valid).
        """
        warnings: list[str] = []

        # Check for duplicate nodes across servers
        seen_nodes: dict[str, str] = {}
        for server_name, server in self.servers.items():
            for node in server.nodes:
                if node in seen_nodes:
                    warnings.append(
                        f"Node '{node}' assigned to both '{seen_nodes[node]}' "
                        f"and '{server_name}'"
                    )
                seen_nodes[node] = server_name

        # Check for empty servers
        for server_name, server in self.servers.items():
            if not server.nodes:
                warnings.append(f"Server '{server_name}' has no nodes assigned")

        # Check for missing Redis URL when multiple servers exist
        if len(self.servers) > 1 and not self.cluster.redis_url:
            warnings.append(
                "Multiple servers defined but no redis_url set — "
                "nodes won't be able to communicate cross-process"
            )

        return warnings


def load_cluster_config(
    path: str | Path | None = None,
) -> ClusterConfig:
    """
    Load cluster configuration from YAML file.

    Resolution order:
        1. Explicit path argument
        2. HBLLM_CLUSTER_CONFIG env var
        3. config/cluster.yaml (relative to CWD)
        4. Default (single-server, all nodes)
    """
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed — using default cluster config")
        return ClusterConfig()

    if path is None:
        path = os.environ.get("HBLLM_CLUSTER_CONFIG", "config/cluster.yaml")

    config_path = Path(path)
    if not config_path.exists():
        logger.info("No cluster config at %s — using defaults", config_path)
        return ClusterConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        return ClusterConfig()

    # Interpolate environment variables
    raw = _interpolate_env(raw)

    config = ClusterConfig.model_validate(raw)

    # Log validation warnings
    warnings = config.validate_topology()
    for w in warnings:
        logger.warning("Cluster config: %s", w)

    logger.info(
        "Loaded cluster config '%s' with %d servers, %d total nodes",
        config.cluster.name,
        len(config.servers),
        len(config.get_all_nodes()),
    )
    return config
