"""
Formal SDK for writing third-party HBLLM plugins.

Provides declarative decorators to simplify creating Nodes bridging
external systems (APIs, UI, Webhooks) into the cognitive MessageBus.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from hbllm.network.messages import Message
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


def subscribe(topic: str) -> Callable[[Any], Any]:
    """
    Declarative decorator for HBLLMPlugin methods.
    Automatically binds the annotated method to the given topic on the MessageBus.

    Example:
        @subscribe("perception.vision")
        async def on_image_seen(self, message: Message) -> None:
            ...
    """

    def decorator(fn: Callable[[Any, Message], Any]) -> Callable[[Any, Message], Any]:
        fn._hbllm_subscribe_topic = topic  # type: ignore
        return fn

    return decorator


class HBLLMPlugin(Node):
    """
    Base class for declarative third-party plugins.

    Subclass this and use ``@subscribe("topic")`` on methods to handle bus events.
    The PluginManager will automatically discover and load subclasses of HBLLMPlugin.
    """

    def __init__(self, node_id: str, capabilities: list[str] | None = None) -> None:
        super().__init__(node_id=node_id, node_type=NodeType.DETECTOR, capabilities=capabilities)

    async def on_start(self) -> None:
        """
        Auto-binds any method decorated with @subscribe to the MessageBus.
        """
        bound_count = 0
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "_hbllm_subscribe_topic"):
                topic = getattr(attr, "_hbllm_subscribe_topic")
                await self.bus.subscribe(topic, attr)
                bound_count += 1

        if bound_count > 0:
            logger.info("[%s] Auto-bound %d @subscribe handlers", self.node_id, bound_count)

    async def on_stop(self) -> None:
        # Bus subscriptions are typically dropped when the bus closes, but
        # cleanup code for external sockets (like Telegram) goes here.
        pass

    async def handle_message(self, message: Message) -> Message | None:
        # Since methods bind directly via @subscribe, the fallback handle_message
        # isn't strictly required unless intercepting generic wildcard traffic.
        return None
