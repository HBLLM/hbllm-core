"""
Trust Chain — Cryptographic governance for HBLLM nodes.

Enables the Owner Node (Root of Trust) to sign joining node certificates,
ensuring that only authorized hardware can participate in the sovereign network.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.security.identity import NodeIdentity

logger = logging.getLogger(__name__)


class TrustChain:
    """
    Manages node certificates and Root of Trust (RoT) signatures.
    """

    def __init__(self, owner_identity: NodeIdentity | None = None):
        self._owner_identity = owner_identity

    @property
    def has_owner(self) -> bool:
        return self._owner_identity is not None

    def sign_node_registration(self, node_id: str, public_key_b64: str) -> str:
        """
        Owner Node signs a registration request from a new node.
        """
        if not self._owner_identity:
            raise RuntimeError("Cannot sign registration: Owner identity not set.")

        # Data to sign: node_id + public_key
        payload = f"register:{node_id}:{public_key_b64}"
        signature = self._owner_identity.sign(payload.encode())
        return signature

    def verify_node_registration(
        self,
        node_id: str,
        public_key_b64: str,
        owner_signature_b64: str,
        owner_public_key_b64: str,
    ) -> bool:
        """
        Verify that a node's registration was signed by the legitimate Owner.
        """
        payload = f"register:{node_id}:{public_key_b64}"
        return NodeIdentity.verify(
            public_key_b64=owner_public_key_b64,
            data=payload.encode(),
            signature_b64=owner_signature_b64,
        )
