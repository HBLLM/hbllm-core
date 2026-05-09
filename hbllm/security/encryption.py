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

import base64
import hashlib
import hmac
import json
import logging
import secrets
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Key Derivation ──────────────────────────────────────────────────


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 32-byte encryption key from password + salt using PBKDF2."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)


def _generate_key() -> bytes:
    """Generate a random 32-byte encryption key."""
    return secrets.token_bytes(32)


# ── Fernet-like Encryption ──────────────────────────────────────────


class EncryptionVault:
    """
    AES-like symmetric encryption vault for field-level data protection.

    Uses HMAC-SHA256 for authentication and base64 encoding for storage.
    For production, swap with `cryptography.fernet.Fernet` or AWS KMS.
    """

    VERSION = b"\x80"  # version byte for future upgrades

    def __init__(self, key: bytes | None = None):
        self._key = key or _generate_key()
        self._enc_key = self._key[:16]  # first 16 bytes for "encryption"
        self._mac_key = self._key[16:]  # last 16 bytes for HMAC
        self._salt: bytes | None = None

    @classmethod
    def from_key_file(cls, path: str) -> EncryptionVault:
        """Load encryption key from file, or create if missing."""
        p = Path(path)
        if p.exists():
            key = base64.urlsafe_b64decode(p.read_text().strip())
            return cls(key=key)
        key = _generate_key()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(base64.urlsafe_b64encode(key).decode())
        p.chmod(0o600)
        logger.info("Generated new encryption key at %s", path)
        return cls(key=key)

    @classmethod
    def from_password(cls, password: str, salt: bytes | None = None) -> EncryptionVault:
        """Derive key from password."""
        salt = salt or secrets.token_bytes(16)
        key = _derive_key(password, salt)
        vault = cls(key=key)
        vault._salt = salt
        return vault

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string value. Returns base64-encoded ciphertext."""
        data = plaintext.encode("utf-8")
        # Simple XOR-based encryption with HMAC auth
        # (For production: use cryptography.fernet.Fernet)
        nonce = secrets.token_bytes(16)
        stream = self._keystream(nonce, len(data))
        ciphertext = bytes(a ^ b for a, b in zip(data, stream))

        # Authenticate: version + nonce + ciphertext
        payload = self.VERSION + nonce + ciphertext
        mac = hmac.new(self._mac_key, payload, hashlib.sha256).digest()

        return base64.urlsafe_b64encode(payload + mac).decode()

    def decrypt(self, token: str) -> str:
        """Decrypt a token. Raises ValueError on tampering."""
        try:
            raw = base64.urlsafe_b64decode(token.encode())
        except Exception:
            raise ValueError("Invalid encryption token")

        if len(raw) < 1 + 16 + 32:
            raise ValueError("Token too short")

        payload = raw[:-32]
        mac = raw[-32:]

        # Verify HMAC
        expected_mac = hmac.new(self._mac_key, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(mac, expected_mac):
            raise ValueError("Token authentication failed — data may be tampered")

        version = payload[0:1]
        if version != self.VERSION:
            raise ValueError(f"Unsupported token version: {version!r}")

        nonce = payload[1:17]
        ciphertext = payload[17:]

        stream = self._keystream(nonce, len(ciphertext))
        plaintext = bytes(a ^ b for a, b in zip(ciphertext, stream))
        return plaintext.decode("utf-8")

    def encrypt_dict(self, data: dict[str, Any]) -> str:
        """Encrypt a dictionary as JSON."""
        return self.encrypt(json.dumps(data, default=str))

    def decrypt_dict(self, token: str) -> dict[str, Any]:
        """Decrypt a token back to a dictionary."""
        return dict(json.loads(self.decrypt(token)))

    def _keystream(self, nonce: bytes, length: int) -> bytes:
        """Generate a pseudo-random keystream from nonce + key."""
        stream = b""
        counter = 0
        while len(stream) < length:
            block = hmac.new(
                self._enc_key,
                nonce + counter.to_bytes(4, "big"),
                hashlib.sha256,
            ).digest()
            stream += block
            counter += 1
        return stream[:length]

    def rotate_key(self, new_key: bytes | None = None) -> EncryptionVault:
        """Create a new vault with a rotated key."""
        return EncryptionVault(key=new_key or _generate_key())

    @property
    def key_fingerprint(self) -> str:
        """Return a safe fingerprint of the key (for logging)."""
        return hashlib.sha256(self._key).hexdigest()[:12]
