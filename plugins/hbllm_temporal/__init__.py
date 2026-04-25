"""
Temporal Reasoning Plugin for HBLLM.

Subscribes to temporal queries and actions, interfaces with the SchedulerNode
to handle future reminders, background processing, and checking past events.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.plugin.sdk import HBLLMPlugin, subscribe

logger = logging.getLogger("hbllm_temporal")

__plugin__ = {
    "name": "hbllm_temporal",
    "version": "0.1.0",
    "description": "Temporal reasoning: 'what happened before', 'remind me'.",
}


class TemporalReasoningNode(HBLLMPlugin):
    """
    Subscribes to temporal queries, parses timestamps, and queries/creates
    schedule entries via the SchedulerNode.
    """

    def __init__(self, node_id: str = "temporal_reasoner") -> None:
        super().__init__(node_id=node_id, capabilities=["temporal_planning", "scheduling"])

    @subscribe("query.temporal")
    async def on_temporal_query(self, message: Message) -> None:
        """
        Handles requests to check schedule or past events.
        """
        query_text = message.payload.get("text", "").lower()
        logger.info("[%s] Received temporal query: %s", self.node_id, query_text)

        # Mocking temporal processing logic:
        response_text = "I have checked your schedule. "
        if "remind" in query_text:
            response_text += "I'll remind you."
            # Here we would send a message to the SchedulerNode to create a job
        elif "before" in query_text or "past" in query_text:
            response_text += "Looking into episodic memory..."
            # Here we would query episodic memory
        else:
            response_text += f"The current system time is {datetime.now(timezone.utc).isoformat()}."

        # Publish result back
        reply = message.create_response(payload={"text": response_text, "temporal_data": {}})
        if self.bus:
            await self.bus.publish(reply.topic, reply)
