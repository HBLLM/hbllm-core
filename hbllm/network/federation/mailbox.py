"""
Federated Mailbox — Sandbox inbox for incoming intent envelopes.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.network.federation.cipher import EnvelopeCipher
from hbllm.network.federation.firewall import FederatedFirewall, FederationSecurityError
from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


class FederatedMailbox:
    """
    Sovereign mailbox gateway that accepts, authenticates, and filters intent envelopes.

    Coordinates EnvelopeCipher verification and FederatedFirewall threat auditing before
    optionally writing events to the local internal MessageBus.
    """

    def __init__(self, bus: Any = None, cipher: EnvelopeCipher | None = None, embedder: Any = None) -> None:
        self.bus = bus
        self.cipher = cipher or EnvelopeCipher()
        self.embedder = embedder
        self.peer_public_keys: dict[str, str] = {}  # alias -> hex

        # Simple schema keys for standard task intent payloads
        self.intent_schema_keys = {"task_description", "context_query", "code_payload"}

    def register_peer(self, alias: str, public_key_hex: str) -> None:
        """Register a trusted external peer's public key (DID alias mapping)."""
        self.peer_public_keys[alias] = public_key_hex
        logger.info("Registered federated peer alias '%s' -> %s...", alias, public_key_hex[:12])

    async def receive_envelope(self, envelope_package: dict[str, Any]) -> dict[str, Any]:
        """
        Receive, decrypt, audit, and process an untrusted incoming envelope.

        Steps:
          1. Authenticate signature and timestamps.
          2. Audit all dictionary fields against prompt injection & path traversals.
          3. Audit code strings using AST sandboxing controls.
          4. Inject into the local MessageBus.
        """
        # 1. Structural parse
        envelope = envelope_package.get("envelope")
        signature = envelope_package.get("signature")

        if not envelope or not signature:
            logger.error("Envelope validation failed: Missing payload or signature envelope structure.")
            return {"status": "error", "reason": "Malformed payload packaging structure."}

        sender_key = envelope.get("sender")
        if not sender_key:
            logger.error("Envelope validation failed: Missing sender key provenance.")
            return {"status": "error", "reason": "Missing sender cryptographic identifier."}

        # 2. Cryptographic signature and replay verification
        is_valid = self.cipher.verify_envelope(sender_key, envelope, signature)
        if not is_valid:
            logger.error("Envelope validation failed: Invalid mathematical signature or timestamp expiration.")
            return {"status": "error", "reason": "Cryptographic signature validation failure."}

        # 3. Payload quarantine sanitization
        intent_data = envelope.get("data")
        if not isinstance(intent_data, dict):
            logger.error("Envelope validation failed: Quarantined intent payload must be a dictionary.")
            return {"status": "error", "reason": "Invalid payload dictionary format."}

        sanitized_data = FederatedFirewall.sanitize_payload_structure(intent_data, self.intent_schema_keys)

        # 4. Security gates audits
        try:
            # Audit standard textual variables
            for text_field in ("task_description", "context_query"):
                if text_field in sanitized_data:
                    FederatedFirewall.audit_text_field(
                        text_field, sanitized_data[text_field], embedder=self.embedder
                    )

            # Audit dynamic code execution payloads
            if "code_payload" in sanitized_data:
                FederatedFirewall.audit_python_code("code_payload", sanitized_data["code_payload"])
        except FederationSecurityError as fse:
            logger.warning("Threat Shield Triggered: Federation Security Error: %s", fse)
            return {"status": "blocked", "reason": str(fse)}


        # 5. Route to internal MessageBus (if available)
        logger.info("External payload successfully validated and passed firewall gates. Processing event...")
        if self.bus:
            try:
                internal_msg = Message(
                    type=MessageType.EVENT,
                    source_node_id="federation_mailbox",
                    topic=envelope.get("topic", "external.event"),
                    payload={
                        "sender": sender_key,
                        "data": sanitized_data,
                        "timestamp": envelope.get("timestamp"),
                    },
                )
                await self.bus.publish(envelope.get("topic", "external.event"), internal_msg)
            except Exception as e:
                logger.error("Failed to publish federated message onto local bus: %s", e)

        return {
            "status": "success",
            "message": "Payload verified, scanned, and successfully queued for internal processing.",
            "recipient": self.cipher.public_key_hex,
        }
