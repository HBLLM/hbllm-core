import asyncio
import logging
import json
import os
import uuid
from collections import defaultdict
from typing import Any

from hbllm.network.node import Node, NodeType
from hbllm.network.messages import Message, MessageType, FeedbackPayload, SystemImprovePayload

logger = logging.getLogger(__name__)

class MetaReasoningNode(Node):
    """
    AGI Layer: Meta-Reasoning Supervisor.

    Monitors system-wide user feedback. If it detects a systemic weakness
    in a specific domain (high negative feedback volume), it orchestrates a
    self-improvement offline loop by dumping failed interactions to a reflection
    dataset and signaling the admin/system bus to trigger heavy offline fine-tuning.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id=node_id, node_type=NodeType.ROUTER)
        
        # Buffer to store negative feedback by domain
        self.negative_feedback_buffer = defaultdict(list)
        
        # Threshold: if we get 3 negative feedbacks for a domain, we trigger reflection
        self.weakness_threshold = 3 
        self.reflection_dir = "workspace/reflection"
        os.makedirs(self.reflection_dir, exist_ok=True)

    async def on_start(self) -> None:
        """Subscribe to feedback broadcasts to monitor system health."""
        logger.info("Starting MetaReasoningNode '%s' Supervisor", self.node_id)
        await self.bus.subscribe("system.feedback", self.handle_message)

    async def on_stop(self) -> None:
        """Clean up."""
        logger.info("Stopping MetaReasoningNode")

    async def handle_message(self, message: Message) -> Message | None:
        """Process incoming feedback silently to monitor health."""
        if message.type != MessageType.FEEDBACK:
            return None

        try:
            payload = FeedbackPayload(**message.payload)
        except Exception as e:
            return None # Ignore invalid

        domain = payload.module_id or "general"
        rating = payload.rating

        if rating == -1:
            logger.warning("MetaReasoningNode detected negative feedback for domain '%s'", domain)
            
            # Store the interaction context if available
            if payload.prompt and payload.response:
                sample = {
                    "instruction": payload.prompt,
                    "response": payload.response,
                    "rejected": True,
                    "domain": domain
                }
                self.negative_feedback_buffer[domain].append(sample)
                
                # Check if this crosses the systemic weakness threshold
                if len(self.negative_feedback_buffer[domain]) >= self.weakness_threshold:
                    await self._trigger_reflection(domain)

        return None

    async def _trigger_reflection(self, domain: str) -> None:
        """Creates a reflection dataset and triggers the self-improvement loop."""
        logger.critical("--- SYSTEMIC WEAKNESS DETECTED IN DOMAIN '%s' ---", domain.upper())
        logger.info("MetaReasoningNode is initiating a self-improvement loop.")
        
        # 1. Dump dataset to disk
        filename = f"weakness_{domain}_{uuid.uuid4().hex[:8]}.jsonl"
        filepath = os.path.join(self.reflection_dir, filename)
        
        dataset = self.negative_feedback_buffer[domain]
        
        try:
            # Thread file IO
            def _write():
                with open(filepath, "w") as f:
                    for item in dataset:
                        # In a real system, you might generate preferred responses here 
                        # using a stronger teacher model or search. For now we just isolate the failures.
                        f.write(json.dumps(item) + "\n")
            
            await asyncio.to_thread(_write)
            logger.info("Saved reflection dataset to %s", filepath)
            
        except Exception as e:
            logger.error("Failed to dump reflection dataset: %s", e)
            return

        # 2. Fire the improvement signal over the bus
        improve_msg = Message(
            type=MessageType.SYSTEM_IMPROVE,
            source_node_id=self.node_id,
            target_node_id="", 
            topic="system.improve",
            payload=SystemImprovePayload(
                domain=domain,
                reasoning=f"Accumulated {self.weakness_threshold} negative feedback events recently.",
                dataset_path=filepath
            ).model_dump()
        )
        await self.bus.publish("system.improve", improve_msg)
        
        # 3. Clear the buffer to prevent spamming
        self.negative_feedback_buffer[domain] = []
