"""
Message Pool — Object reuse container for inter-node communication.

Reduces garbage collection pressure and memory allocations by pooling
and recycling Message objects on high-throughput message bus paths.
"""

from __future__ import annotations

import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

from hbllm.network.messages import Message, MessageType, Priority


class MessagePool:
    """
    Object pool for recycling Message instances.
    """

    def __init__(self, max_size: int = 512) -> None:
        self.max_size = max_size
        self._pool: deque[Message] = deque(maxlen=max_size)
        self._acquired = 0
        self._recycled = 0

    def acquire(
        self,
        msg_type: MessageType,
        source_node_id: str,
        topic: str,
        payload: dict[str, Any] | None = None,
        target_node_id: str | None = None,
        tenant_id: str = "default",
        user_id: str = "default",
        device_id: str = "default",
        session_id: str = "default",
        priority: Priority = Priority.NORMAL,
        correlation_id: str | None = None,
    ) -> Message:
        """Acquire a Message instance, recycling from pool if available."""
        self._acquired += 1
        payload_dict = payload or {}
        now = datetime.now(timezone.utc)
        msg_id = str(uuid.uuid4())

        if self._pool:
            msg = self._pool.popleft()
            msg.id = msg_id
            msg.type = msg_type
            msg.source_node_id = source_node_id
            msg.target_node_id = target_node_id
            msg.tenant_id = tenant_id
            msg.user_id = user_id
            msg.device_id = device_id
            msg.session_id = session_id
            msg.topic = topic
            msg.payload = payload_dict
            msg.priority = priority
            msg.timestamp = now
            msg.correlation_id = correlation_id
            msg.ttl_seconds = None
            msg.is_security_cleared = False
            msg.signature = None
            msg.vector_clock = None
            return msg

        return Message.model_construct(
            id=msg_id,
            type=msg_type,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            tenant_id=tenant_id,
            user_id=user_id,
            device_id=device_id,
            session_id=session_id,
            topic=topic,
            payload=payload_dict,
            priority=priority,
            timestamp=now,
            correlation_id=correlation_id,
            ttl_seconds=None,
            is_security_cleared=False,
            signature=None,
            vector_clock=None,
        )

    def release(self, message: Message) -> None:
        """Release a Message instance back to the pool."""
        if len(self._pool) < self.max_size:
            self._recycled += 1
            self._pool.append(message)

    def stats(self) -> dict[str, int]:
        """Return pool usage statistics."""
        return {
            "pooled": len(self._pool),
            "max_size": self.max_size,
            "acquired": self._acquired,
            "recycled": self._recycled,
        }
