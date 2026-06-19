"""
Pluggable Secret Provider — Enterprise Secret Management Abstraction.

Provides a unified interface for accessing secrets, with backends for:
  - Environment variables (default, zero-config)
  - HashiCorp Vault (enterprise KV-v2)
  - AWS Secrets Manager (cloud-native)

Usage::

    from hbllm.security.secrets import get_secret_provider

    secrets = get_secret_provider()
    jwt_key = secrets.get("HBLLM_JWT_SECRET")
    db_url  = secrets.get("HBLLM_DATABASE_URL")

Switch backend via env var: HBLLM_SECRET_BACKEND=vault|aws|env (default: env)
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Protocol ──────────────────────────────────────────────────────────


@runtime_checkable
class SecretProvider(Protocol):
    """Interface for secret retrieval backends."""

    def get(self, key: str, default: str | None = None) -> str | None:
        """Retrieve a secret value by key."""
        ...

    def get_required(self, key: str) -> str:
        """Retrieve a secret value, raising if not found."""
        ...


# ── Backends ──────────────────────────────────────────────────────────


class EnvSecretProvider:
    """Read secrets from environment variables (default, zero-config)."""

    def get(self, key: str, default: str | None = None) -> str | None:
        return os.environ.get(key, default)

    def get_required(self, key: str) -> str:
        value = os.environ.get(key)
        if value is None:
            raise KeyError(f"Required secret '{key}' not found in environment")
        return value


class VaultSecretProvider:
    """Read secrets from HashiCorp Vault (KV-v2 engine).

    Requires:
      - ``pip install hvac``
      - VAULT_ADDR and VAULT_TOKEN (or VAULT_ROLE_ID + VAULT_SECRET_ID)
      - HBLLM_VAULT_MOUNT (default: "secret")
      - HBLLM_VAULT_PATH (default: "hbllm")
    """

    def __init__(self) -> None:
        try:
            import hvac  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("VaultSecretProvider requires 'hvac'. Install with: pip install hvac")

        vault_addr = os.environ.get("VAULT_ADDR", "http://127.0.0.1:8200")
        vault_token = os.environ.get("VAULT_TOKEN")

        self._client = hvac.Client(url=vault_addr, token=vault_token)

        # AppRole auth fallback
        if not vault_token:
            role_id = os.environ.get("VAULT_ROLE_ID")
            secret_id = os.environ.get("VAULT_SECRET_ID")
            if role_id and secret_id:
                self._client.auth.approle.login(role_id=role_id, secret_id=secret_id)

        if not self._client.is_authenticated():
            raise RuntimeError("Vault authentication failed")

        self._mount = os.environ.get("HBLLM_VAULT_MOUNT", "secret")
        self._path = os.environ.get("HBLLM_VAULT_PATH", "hbllm")
        self._cache: dict[str, str] = {}
        self._loaded = False
        logger.info(
            "VaultSecretProvider connected to %s (mount=%s, path=%s)",
            vault_addr,
            self._mount,
            self._path,
        )

    def _load(self) -> None:
        """Lazy-load all secrets from Vault into local cache."""
        if self._loaded:
            return
        try:
            response = self._client.secrets.kv.v2.read_secret_version(
                mount_point=self._mount,
                path=self._path,
            )
            self._cache = response["data"]["data"]
            self._loaded = True
            logger.info("Loaded %d secrets from Vault", len(self._cache))
        except Exception as e:
            logger.error("Failed to read secrets from Vault: %s", e)
            raise

    def get(self, key: str, default: str | None = None) -> str | None:
        self._load()
        return self._cache.get(key, os.environ.get(key, default))

    def get_required(self, key: str) -> str:
        value = self.get(key)
        if value is None:
            raise KeyError(f"Required secret '{key}' not found in Vault or environment")
        return value


class AWSSecretsProvider:
    """Read secrets from AWS Secrets Manager.

    Requires:
      - ``pip install boto3``
      - AWS credentials (IAM role, env vars, or ~/.aws/credentials)
      - HBLLM_AWS_SECRET_NAME (default: "hbllm/secrets")
      - HBLLM_AWS_REGION (default: "us-east-1")
    """

    def __init__(self) -> None:
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "AWSSecretsProvider requires 'boto3'. Install with: pip install boto3"
            )

        region = os.environ.get("HBLLM_AWS_REGION", "us-east-1")
        self._secret_name = os.environ.get("HBLLM_AWS_SECRET_NAME", "hbllm/secrets")
        self._client = boto3.client("secretsmanager", region_name=region)
        self._cache: dict[str, str] = {}
        self._loaded = False
        logger.info(
            "AWSSecretsProvider initialized (region=%s, secret=%s)", region, self._secret_name
        )

    def _load(self) -> None:
        """Lazy-load secrets from AWS Secrets Manager."""
        if self._loaded:
            return
        import json

        try:
            response = self._client.get_secret_value(SecretId=self._secret_name)
            secret_string = response.get("SecretString", "{}")
            self._cache = json.loads(secret_string)
            self._loaded = True
            logger.info("Loaded %d secrets from AWS Secrets Manager", len(self._cache))
        except Exception as e:
            logger.error("Failed to read secrets from AWS: %s", e)
            raise

    def get(self, key: str, default: str | None = None) -> str | None:
        self._load()
        return self._cache.get(key, os.environ.get(key, default))

    def get_required(self, key: str) -> str:
        value = self.get(key)
        if value is None:
            raise KeyError(
                f"Required secret '{key}' not found in AWS Secrets Manager or environment"
            )
        return value


# ── Factory ───────────────────────────────────────────────────────────

_BACKENDS: dict[str, type] = {
    "env": EnvSecretProvider,
    "vault": VaultSecretProvider,
    "aws": AWSSecretsProvider,
}

_instance: SecretProvider | None = None


def get_secret_provider() -> SecretProvider:
    """Get or create the configured secret provider singleton.

    Backend is selected via ``HBLLM_SECRET_BACKEND`` env var.
    Defaults to ``env`` (environment variables).
    """
    global _instance
    if _instance is not None:
        return _instance

    backend_name = os.environ.get("HBLLM_SECRET_BACKEND", "env").lower()
    backend_cls = _BACKENDS.get(backend_name)
    if backend_cls is None:
        raise ValueError(
            f"Unknown secret backend '{backend_name}'. Supported: {', '.join(_BACKENDS.keys())}"
        )

    _instance = backend_cls()
    logger.info("Secret provider initialized: %s", backend_name)
    return _instance


def reset_provider() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None
