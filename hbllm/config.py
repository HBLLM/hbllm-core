"""
Unified Configuration System for HBLLM Core.

This module provides a central entry point for loading all system configurations,
including model parameters, cluster topology, and the new LoRA Adapter Registry.
"""

import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from hbllm.modules.adapter_registry import AdapterRegistryConfig
from hbllm.network.cluster_config import ClusterConfig


class HBLLMCoreConfig(BaseModel):
    """The root configuration object for the entire HBLLM system."""

    # Environment identity
    env: str = "development"

    # Nested configurations
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    adapters: AdapterRegistryConfig = Field(default_factory=AdapterRegistryConfig)

    # Global paths
    checkpoints_dir: str = "./checkpoints"
    data_dir: str = "./data"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "HBLLMCoreConfig":
        """
        Loads the full configuration from a YAML file.

        Resolution order:
            1. Explicit path
            2. HBLLM_CONFIG_PATH env var
            3. hbllm.yaml in CWD
        """
        if path is None:
            path = os.environ.get("HBLLM_CONFIG_PATH", "hbllm.yaml")

        config_path = Path(path)
        if not config_path.exists():
            # Return defaults if no file found
            return cls()

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        if not raw:
            return cls()

        return cls.model_validate(raw)

    def to_dict(self) -> dict[str, Any]:
        """Convert to simple dictionary."""
        return self.model_dump()
