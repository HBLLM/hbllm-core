"""
Encryption at Rest for HBLLM Core.

Provides field-level encryption for sensitive tenant data using
symmetric encryption (AES-like XOR keystream + HMAC-SHA256).

Encrypts:
  - API keys (stored encrypted, decrypted on validation)
  - Tenant config (custom model endpoints, secrets)
  - Training data (PII fields, annotations)
  - Webhook secrets

Usage::

    vault = EncryptionVault()                          # auto-generates key
    vault = EncryptionVault.from_key_file("data/key")  # load from file
    encrypted = vault.encrypt("sensitive-data")
    original  = vault.decrypt(encrypted)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Iterator

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


# ── Key Derivation ──────────────────────────────────────────────────


import hashlib
import json
import logging
import secrets


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 32-byte encryption key from password + salt using PBKDF2."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)


def _generate_key() -> bytes:
    """Generate a random 32-byte encryption key (url-safe base64 for Fernet)."""
    return Fernet.generate_key()


# ── Fernet-like Encryption ──────────────────────────────────────────


class EncryptionVault:
    """
    AES-based symmetric encryption vault for field-level data protection.

    Uses Fernet (AES128 in CBC mode with SHA256 HMAC authentication).
    """

    def __init__(self, key: bytes | None = None):
        self._key = key or _generate_key()
        self._fernet = Fernet(self._key)
        self._salt: bytes | None = None

    @classmethod
    def from_env(cls, env_var: str = "HBLLM_ENCRYPTION_KEY") -> EncryptionVault:
        """Load encryption key from an environment variable."""
        key_str = os.environ.get(env_var)
        if not key_str:
            raise ValueError(f"Environment variable {env_var} is not set")
        return cls(key=key_str.encode("utf-8"))

    @classmethod
    def from_key_file(cls, path: str) -> EncryptionVault:
        """Load encryption key from file, or create if missing."""
        p = Path(path)
        if p.exists():
            key = p.read_text().strip().encode("utf-8")
            return cls(key=key)
        key = _generate_key()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(key.decode("utf-8"))
        p.chmod(0o600)
        logger.info("Generated new encryption key at %s", path)
        return cls(key=key)

    @classmethod
    def from_password(cls, password: str, salt: bytes | None = None) -> EncryptionVault:
        """Derive key from password."""
        import base64

        salt = salt or secrets.token_bytes(16)
        raw_key = _derive_key(password, salt)
        # Fernet requires url-safe base64 encoded 32-byte key
        key = base64.urlsafe_b64encode(raw_key)
        vault = cls(key=key)
        vault._salt = salt
        return vault

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string value. Returns base64-encoded ciphertext."""
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def decrypt(self, token: str) -> str:
        """Decrypt a token. Raises ValueError on tampering."""
        try:
            return self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")
        except InvalidToken:
            raise ValueError("Token authentication failed — data may be tampered or key is wrong")
        except Exception as e:
            raise ValueError(f"Invalid encryption token: {e}")

    def encrypt_dict(self, data: dict[str, Any]) -> str:
        """Encrypt a dictionary as JSON."""
        return self.encrypt(json.dumps(data, default=str))

    def decrypt_dict(self, token: str) -> dict[str, Any]:
        """Decrypt a token back to a dictionary."""
        return dict(json.loads(self.decrypt(token)))

    def rotate_key(self, new_key: bytes | None = None) -> EncryptionVault:
        """Create a new vault with a rotated key."""
        return EncryptionVault(key=new_key or _generate_key())

    def rotate_and_reencrypt(
        self, data_iterable: Iterable[str]
    ) -> tuple[EncryptionVault, Iterator[str]]:
        """
        Create a new vault and re-encrypt a stream of tokens with the new key.
        Useful for key rotation migrations over large datasets without memory exhaustion.
        """
        new_vault = self.rotate_key()

        def _generator() -> Iterator[str]:
            for token in data_iterable:
                plaintext = self.decrypt(token)
                yield new_vault.encrypt(plaintext)

        return new_vault, _generator()

    @property
    def key_fingerprint(self) -> str:
        """Return a safe fingerprint of the key (for logging)."""
        return hashlib.sha256(self._key).hexdigest()[:12]
