from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hbllm.network.messages import Message
    from hbllm.network.registry import ServiceRegistry

logger = logging.getLogger(__name__)


class TrustInterceptor:
    """
    MessageBus interceptor that enforces the Trust Model.
    Verifies signatures of all incoming messages using the ServiceRegistry.
    """

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry

    async def __call__(self, message: Message) -> Message | None:
        """
        Intercepts message and verifies signature.
        Returns message if valid, None if invalid/untrusted.
        """
        # System messages and registration might need special handling
        if message.source_node_id == "system":
            return message

        # Verify signature
        is_valid = await self.registry.verify_message(message)
        if not is_valid:
            logger.warning(
                "[TrustInterceptor] Untrusted message detected! Dropping message %s from node '%s'",
                message.id,
                message.source_node_id,
            )
            return None

        return message
