from __future__ import annotations

import base64
import logging
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

logger = logging.getLogger(__name__)


class NodeIdentity:
    """
    Manages the Ed25519 identity of an HBLLM node.
    Used for signing and verifying messages across the distributed swarm.
    """

    def __init__(self, private_key: ed25519.Ed25519PrivateKey):
        self._private_key = private_key
        self._public_key = private_key.public_key()

    @classmethod
    def generate(cls) -> NodeIdentity:
        """Create a brand new identity."""
        return cls(ed25519.Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(cls, key_path: str | Path) -> NodeIdentity:
        """Load identity from disk or create if missing."""
        p = Path(key_path)
        if p.exists():
            try:
                raw = p.read_bytes()
                key = serialization.load_pem_private_key(raw, password=None)
                if not isinstance(key, ed25519.Ed25519PrivateKey):
                    raise ValueError("Key is not an Ed25519 private key")
                return cls(key)
            except Exception as e:
                logger.error("Failed to load node identity: %s", e)
                # Fallback to new key if corrupted

        identity = cls.generate()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(
            identity._private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.OpenSSH,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        p.chmod(0o600)
        logger.info("Generated new node identity at %s", key_path)
        return identity

    @property
    def public_key_bytes(self) -> bytes:
        """Public key in Raw format."""
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @property
    def public_key_b64(self) -> str:
        """Base64 encoded public key for registration."""
        return base64.b64encode(self.public_key_bytes).decode()

    def sign(self, data: bytes) -> str:
        """Sign binary data. Returns Base64 signature."""
        sig = self._private_key.sign(data)
        return base64.b64encode(sig).decode()

    @staticmethod
    def verify(public_key_b64: str, data: bytes, signature_b64: str) -> bool:
        """Verify a signature against a Base64 public key."""
        try:
            pub_bytes = base64.b64decode(public_key_b64)
            sig_bytes = base64.b64decode(signature_b64)
            pub_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
            pub_key.verify(sig_bytes, data)
            return True
        except Exception:
            return False
