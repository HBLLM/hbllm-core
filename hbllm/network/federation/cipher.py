"""
Federation Cipher — Handles cryptographic signing, encryption, and envelope validation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
except ImportError:
    ed25519 = None  # type: ignore


class EnvelopeCipher:
    """
    Manages asymmetric key pairs and intent capsule packaging.

    Utilizes Ed25519 signatures to establish cryptographic provenance.
    If the cryptography module is missing, it falls back to SHA-256 HMAC for testing.
    """

    def __init__(self, private_key_bytes: bytes | None = None) -> None:
        self._private_key: Any = None
        self._public_key: Any = None

        if ed25519 is not None:
            if private_key_bytes:
                try:
                    self._private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
                    self._public_key = self._private_key.public_key()
                except Exception as e:
                    logger.error("Failed to load private bytes: %s. Generating fresh keypair.", e)
                    self._generate_fresh_keypair()
            else:
                self._generate_fresh_keypair()
        else:
            logger.warning("cryptography package not available. Falling back to signature mock-mode.")

    def _generate_fresh_keypair(self) -> None:
        if ed25519 is not None:
            self._private_key = ed25519.Ed25519PrivateKey.generate()
            self._public_key = self._private_key.public_key()

    @property
    def public_key_hex(self) -> str:
        """Get the base64-encoded public key representation."""
        if ed25519 is not None and self._public_key:
            return base64.b64encode(self._public_key.public_bytes_raw()).decode("utf-8")
        # Mock mode fallback
        return base64.b64encode(b"mock_public_key_bytes_128_bit_signing").decode("utf-8")

    def sign_payload(self, payload: dict[str, Any]) -> str:
        """Sign a dictionary payload and return a base64-encoded signature."""
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        if ed25519 is not None and self._private_key:
            signature = self._private_key.sign(serialized)
            return base64.b64encode(signature).decode("utf-8")
        
        # Mock mode fallback: HMAC-SHA256
        h = hashlib.sha256(serialized)
        h.update(b"mock_private_key_fallback")
        return base64.b64encode(h.digest()).decode("utf-8")

    def verify_envelope(self, sender_public_key_hex: str, payload: dict[str, Any], signature_b64: str) -> bool:
        """
        Verify that an envelope payload matches its signature and sender public key.

        Asserts:
          - Signature mathematically resolves to the public key.
          - Timestamp in payload is not expired (replay protection).
        """
        # 1. Check expiration (e.g. 5 minutes TTL)
        timestamp = payload.get("timestamp", 0.0)
        if time.time() - timestamp > 300.0:
            logger.warning("Envelope validation failed: Replay protection triggered (expired timestamp).")
            return False

        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")

        # 2. Check mathematical signature
        try:
            if ed25519 is not None:
                public_bytes = base64.b64decode(sender_public_key_hex)
                pub_key = ed25519.Ed25519PublicKey.from_public_bytes(public_bytes)
                signature = base64.b64decode(signature_b64)
                pub_key.verify(signature, serialized)
                return True
            else:
                # Mock verification fallback
                expected_hash = hashlib.sha256(serialized)
                expected_hash.update(b"mock_private_key_fallback")
                expected = base64.b64encode(expected_hash.digest()).decode("utf-8")
                return signature_b64 == expected
        except Exception as e:
            logger.warning("Cryptographic signature validation failed: %s", e)
            return False

    def pack_envelope(self, recipient_public_hex: str, topic: str, intent_data: dict[str, Any]) -> dict[str, Any]:
        """Pack intent data into a signed, timestamped envelope."""
        payload = {
            "topic": topic,
            "data": intent_data,
            "timestamp": time.time(),
            "sender": self.public_key_hex,
            "recipient": recipient_public_hex,
        }
        signature = self.sign_payload(payload)
        return {
            "envelope": payload,
            "signature": signature,
        }
