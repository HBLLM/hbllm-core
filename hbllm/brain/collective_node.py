"""
Collective Intelligence Network — cross-instance knowledge sharing.

Allows multiple HBLLM instances to share learned knowledge. When one
instance discovers a new domain or learns a new skill, it broadcasts
a KnowledgeDigest so other instances can hot-load the capability.

Communication happens over the RedisBus (for cross-process) or
InProcessBus (for testing).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeDigest:
    """A shareable knowledge artifact from one instance."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_instance_id: str = ""
    domain: str = ""
    capability: str = ""
    artifact_type: str = ""  # "lora_weights", "skill", "semantic_fact", "identity_update"
    artifact_data: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    timestamp: float = field(default_factory=time.time)

    def compute_checksum(self) -> str:
        """Compute a content-based checksum for deduplication."""
        content = json.dumps(self.artifact_data, sort_keys=True)
        self.checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self.checksum

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CollectiveNode(Node):
    """
    Service node for cross-instance knowledge sharing.
    
    Subscribes to:
        system.learning_update — local learning completions (LoRA, skills, etc.)
        collective.sync — incoming knowledge from peer instances
        collective.query — query the received knowledge log
    
    Publishes:
        collective.broadcast — outgoing knowledge digests to peers
    """

    def __init__(
        self,
        node_id: str,
        instance_id: str = "",
        max_received: int = 500,
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["collective_intelligence", "knowledge_sharing"],
        )
        self.instance_id = instance_id or uuid.uuid4().hex[:8]
        self.max_received = max_received
        
        # Knowledge tracking
        self.broadcast_log: list[KnowledgeDigest] = []
        self.received_log: list[KnowledgeDigest] = []
        self.seen_checksums: set[str] = set()
        
        # Stats
        self.stats = {
            "broadcasts_sent": 0,
            "digests_received": 0,
            "duplicates_filtered": 0,
        }

    async def on_start(self) -> None:
        logger.info("Starting CollectiveNode (instance=%s)", self.instance_id)
        await self.bus.subscribe("system.learning_update", self._handle_learning_update)
        await self.bus.subscribe("collective.sync", self._handle_sync)
        await self.bus.subscribe("collective.query", self._handle_query)

    async def on_stop(self) -> None:
        logger.info(
            "Stopping CollectiveNode — sent=%d, received=%d, deduped=%d",
            self.stats["broadcasts_sent"],
            self.stats["digests_received"],
            self.stats["duplicates_filtered"],
        )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def _handle_learning_update(self, message: Message) -> Message | None:
        """
        A local learning event completed — broadcast it to peers.
        Expected payload:
            domain: str
            capability: str
            artifact_type: str
            artifact_data: dict
        """
        payload = message.payload
        
        digest = KnowledgeDigest(
            source_instance_id=self.instance_id,
            domain=payload.get("domain", "general"),
            capability=payload.get("capability", ""),
            artifact_type=payload.get("artifact_type", "skill"),
            artifact_data=payload.get("artifact_data", {}),
        )
        digest.compute_checksum()
        
        # Avoid re-broadcasting our own digests
        if digest.checksum in self.seen_checksums:
            return None
        
        self.seen_checksums.add(digest.checksum)
        self.broadcast_log.append(digest)
        self.stats["broadcasts_sent"] += 1
        
        # Broadcast to peers
        await self.publish("collective.broadcast", Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            topic="collective.broadcast",
            payload=digest.to_dict(),
        ))
        
        logger.info(
            "Broadcast knowledge digest: domain=%s capability=%s checksum=%s",
            digest.domain, digest.capability, digest.checksum,
        )
        return None

    async def _handle_sync(self, message: Message) -> Message | None:
        """
        Receive a knowledge digest from a peer instance.
        Deduplicates by checksum and stores for later integration.
        """
        payload = message.payload
        
        checksum = payload.get("checksum", "")
        source = payload.get("source_instance_id", "")
        
        # Skip our own broadcasts
        if source == self.instance_id:
            return None
        
        # Deduplicate
        if checksum in self.seen_checksums:
            self.stats["duplicates_filtered"] += 1
            return None
        
        digest = KnowledgeDigest(
            id=payload.get("id", uuid.uuid4().hex[:12]),
            source_instance_id=source,
            domain=payload.get("domain", ""),
            capability=payload.get("capability", ""),
            artifact_type=payload.get("artifact_type", ""),
            artifact_data=payload.get("artifact_data", {}),
            checksum=checksum,
            timestamp=payload.get("timestamp", time.time()),
        )
        
        self.seen_checksums.add(checksum)
        self.received_log.append(digest)
        self.stats["digests_received"] += 1
        
        # Trim old entries
        if len(self.received_log) > self.max_received:
            removed = self.received_log.pop(0)
            self.seen_checksums.discard(removed.checksum)
        
        logger.info(
            "Received knowledge from instance %s: domain=%s capability=%s",
            source, digest.domain, digest.capability,
        )
        return None

    async def _handle_query(self, message: Message) -> Message | None:
        """Return collective intelligence stats and recent digests."""
        payload = message.payload
        limit = int(payload.get("limit", 10))
        
        recent_received = [
            d.to_dict() for d in self.received_log[-limit:]
        ]
        recent_broadcast = [
            d.to_dict() for d in self.broadcast_log[-limit:]
        ]
        
        return message.create_response({
            "instance_id": self.instance_id,
            "stats": self.stats,
            "recent_received": recent_received,
            "recent_broadcast": recent_broadcast,
        })
