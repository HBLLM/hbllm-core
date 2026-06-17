from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hbllm.network.bus import MessageBus
    from hbllm.network.messages import Message
    from hbllm.network.registry import ServiceRegistry

logger = logging.getLogger(__name__)


class TrustInterceptor:
    """
    MessageBus interceptor that enforces the Trust Model.
    Verifies signatures, prevents replay attacks, checks revocation,
    and forwards malicious/invalid messages to the DLQ.
    """

    def __init__(self, registry: ServiceRegistry, bus: MessageBus | None = None):
        self.registry = registry
        self.bus = bus
        self._seen_messages: OrderedDict[str, float] = OrderedDict()
        self._max_seen = 10000

    async def __call__(self, message: Message) -> Message | None:
        """
        Intercepts message and verifies trust.
        Returns message if valid, None if invalid/untrusted.
        """
        source = message.source_node_id

        # Well-known system sources bypass all checks
        if source in ("system", "api_server", "pipeline", "studio", "audio_ws", "registry"):
            return message

        # 1. Replay Protection
        if message.id in self._seen_messages:
            logger.warning(
                "[TrustInterceptor] Replay attack detected! Dropping message %s", message.id
            )
            await self._send_to_dlq(message, "replay_attack")
            return None

        self._seen_messages[message.id] = time.time()
        if len(self._seen_messages) > self._max_seen:
            self._seen_messages.popitem(last=False)

        # 2. Revocation Check
        if await self.registry.is_revoked(source):
            logger.warning(
                "[TrustInterceptor] Message from revoked node! Dropping message %s", message.id
            )
            await self._send_to_dlq(message, "node_revoked")
            return None

        # 3. Registered internal nodes are trusted (they are part of the brain)
        #    This covers all composite sub-nodes (e.g. memory_system.sleep, audio_in, etc.)
        base_node = source.split(".")[0]  # e.g. "memory_system.sleep" → "memory_system"
        if source in self.registry._nodes or base_node in self.registry._nodes:
            return message

        # 4. Signature Verification (only for external/unknown nodes)
        is_valid = await self.registry.verify_message(message)
        if not is_valid:
            logger.warning(
                "[TrustInterceptor] Untrusted message detected! Dropping message %s from node '%s'",
                message.id,
                source,
            )
            await self._send_to_dlq(message, "invalid_signature")
            return None

        return message

    async def _send_to_dlq(self, message: Message, reason: str) -> None:
        """Forward untrusted message to the Dead Letter Queue."""
        if not self.bus:
            return

        if hasattr(self.bus, "_route_to_dlq"):
            await self.bus._route_to_dlq(message, reason)
