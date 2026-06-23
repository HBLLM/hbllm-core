"""
RelationshipNode — bus-connected adapter for RelationshipMemory.

Listens for interactions and extracts person mentions to continuously
build and update the social graph.

Bus Topics:
    Subscribes:
        system.experience     → Extract person mentions from queries
        system.evaluation     → Track sentiment in responses about people
        calendar.event        → Learn relationships from attendees

    Publishes:
        relationship.updated  → Relationship quality changed
        relationship.new      → New person detected
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.relationship_memory import RelationshipMemory, extract_person_mentions
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class RelationshipNode(Node):
    """Bus-connected wrapper for RelationshipMemory.

    Passively listens for person mentions and builds the social graph.
    """

    def __init__(
        self,
        node_id: str,
        relationship_memory: RelationshipMemory | None = None,
        knowledge_graph: Any | None = None,
        data_dir: str = "data",
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["relationship_tracking", "social_graph"],
        )
        self.memory = relationship_memory or RelationshipMemory(
            knowledge_graph=knowledge_graph, data_dir=data_dir
        )
        self._new_persons_detected = 0

    async def on_start(self) -> None:
        logger.info("Starting RelationshipNode")
        await self.bus.subscribe("system.experience", self._handle_experience)
        await self.bus.subscribe("system.evaluation", self._handle_evaluation)
        await self.bus.subscribe("calendar.event", self._handle_calendar)
        await self.bus.subscribe("relationship.query", self._handle_query)

    async def on_stop(self) -> None:
        logger.info(
            "Stopping RelationshipNode — %d new persons detected",
            self._new_persons_detected,
        )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Event Handlers ───────────────────────────────────────────────

    async def _handle_experience(self, message: Message) -> None:
        """Extract person mentions from every interaction."""
        payload = message.payload
        tenant_id = message.tenant_id or "default"
        text = payload.get("text", payload.get("query", ""))
        topic = payload.get("topic", "")

        if not text:
            return

        mentions = extract_person_mentions(text)
        for name in mentions:
            existing = self.memory.get_person(name, tenant_id)
            is_new = existing is None

            person = self.memory.record_mention(
                person_name=name,
                context=text[:200],
                tenant_id=tenant_id,
                topic=topic,
            )

            if is_new:
                self._new_persons_detected += 1
                await self.publish(
                    "relationship.new",
                    Message(
                        type=MessageType.EVENT,
                        source_node_id=self.node_id,
                        topic="relationship.new",
                        tenant_id=tenant_id,
                        payload={
                            "person_id": person.person_id,
                            "person_name": person.name,
                        },
                    ),
                )

    async def _handle_evaluation(self, message: Message) -> None:
        """Track sentiment from evaluation context."""
        payload = message.payload
        tenant_id = message.tenant_id or "default"
        text = payload.get("text", payload.get("content", ""))
        score = payload.get("overall_score", 0.5)

        if not text:
            return

        # Extract mentions and infer sentiment from evaluation score
        mentions = extract_person_mentions(text)
        sentiment = (score - 0.5) * 2.0  # Map 0-1 to -1 to 1
        for name in mentions:
            self.memory.record_event(
                person_name=name,
                event_type="evaluation_mention",
                context=text[:200],
                sentiment_delta=sentiment * 0.1,  # Small incremental updates
                tenant_id=tenant_id,
            )

    async def _handle_calendar(self, message: Message) -> None:
        """Learn relationships from calendar events."""
        payload = message.payload
        tenant_id = message.tenant_id or "default"
        attendees = payload.get("attendees", [])
        organizer = payload.get("organizer", "")
        event_name = payload.get("name", payload.get("title", ""))

        if not attendees:
            return

        for attendee in attendees:
            name = attendee if isinstance(attendee, str) else attendee.get("name", "")
            if not name:
                continue

            self.memory.record_event(
                person_name=name,
                event_type="meeting",
                context=f"Calendar event: {event_name}",
                tenant_id=tenant_id,
            )

            # If organizer is known, learn relationship
            if organizer and organizer != name:
                self.memory.learn_relationship(
                    organizer,
                    name,
                    "colleague",
                    context=f"Both attended: {event_name}",
                    tenant_id=tenant_id,
                )

    async def _handle_query(self, message: Message) -> Message | None:
        """Return relationship stats."""
        return message.create_response(self.memory.stats())
