"""
Unified Configuration System for HBLLM Core.

This module provides a central entry point for loading all system configurations,
including model parameters, cluster topology, the LoRA Adapter Registry, and
multi-tenant security settings.
"""

import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator, model_validator

from hbllm.modules.adapter_registry import AdapterRegistryConfig
from hbllm.network.cluster_config import ClusterConfig


class SecurityConfig(BaseModel):
    """
    Multi-tenant security configuration.

    Controls tenant isolation enforcement, audit logging, rate limiting,
    and encryption at rest.

    Example YAML::

        security:
          tenant_guard_mode: strict    # off | warn | strict
          isolation_level: namespace   # shared | namespace | dedicated
          audit_enabled: true
          audit_db_path: data/audit.db
          rate_limit_enabled: true
          rate_limit_rpm: 60
          encryption_key_path: ""
    """

    # Tenant isolation enforcement level
    tenant_guard_mode: str = "warn"  # off | warn | strict

    # Data separation strategy
    isolation_level: str = "shared"  # shared | namespace | dedicated

    # Audit logging
    audit_enabled: bool = True
    audit_db_path: str = "data/audit.db"

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_rpm: int = 60

    # Encryption at rest
    encryption_key_source: str = "auto"  # auto | env | file | vault
    encryption_key_env_var: str = "HBLLM_ENCRYPTION_KEY"
    encryption_key_path: str = "data/encryption.key"

    @field_validator("encryption_key_source")
    @classmethod
    def validate_key_source(cls, v: str) -> str:
        allowed = {"auto", "env", "file", "vault"}
        if v not in allowed:
            raise ValueError(f"encryption_key_source must be one of {allowed}")
        return v


class HBLLMCoreConfig(BaseModel):
    """The root configuration object for the entire HBLLM system."""

    # Environment identity
    env: str = "development"

    # Nested configurations
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    adapters: AdapterRegistryConfig = Field(default_factory=AdapterRegistryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # Global paths
    checkpoints_dir: str = "./checkpoints"
    data_dir: str = "./data"

    @model_validator(mode="after")
    def validate_semantics(self) -> "HBLLMCoreConfig":
        """Semantic validation of configuration dependencies."""
        if self.security.audit_enabled and not self.security.audit_db_path:
            raise ValueError("audit_db_path must be set when audit_enabled is True")

        if self.security.rate_limit_enabled and self.security.rate_limit_rpm <= 0:
            raise ValueError("rate_limit_rpm must be strictly positive")

        if self.security.encryption_key_source == "file" and not self.security.encryption_key_path:
            raise ValueError("encryption_key_path must be set when source is 'file'")

        return self

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
            env = os.environ.get("HBLLM_ENV", "development").lower()
            if env == "production":
                import logging

                logging.getLogger(__name__).warning(
                    "No config file found at '%s' in production mode. "
                    "Using strict security defaults. Create a config file for customisation.",
                    config_path,
                )
                return cls(
                    env="production",
                    security=SecurityConfig(
                        tenant_guard_mode="strict",
                        isolation_level="namespace",
                    ),
                )
            # Return permissive defaults in development
            return cls()

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        if not raw:
            return cls()

        return cls.model_validate(raw)

    def to_dict(self) -> dict[str, Any]:
        """Convert to simple dictionary."""
        return self.model_dump()
