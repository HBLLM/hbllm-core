import asyncio
import logging
import torch
from typing import Any

from hbllm.network.node import Node, NodeType
from hbllm.network.messages import Message, MessageType, FeedbackPayload

logger = logging.getLogger(__name__)


class LearnerNode(Node):
    """
    Continuous Learning Engine.

    Listens for Feedback messages on the bus. When a user provides feedback
    (positive or negative) on a generation, it accumulates the sample.
    Once enough samples are gathered, it performs DPO (Direct Preference Optimization)
    offline in a background thread to update the active LoRA adapter weights, then
    broadcasts that the weights have been updated.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id=node_id, node_type=NodeType.DOMAIN_MODULE) # Can be typed differently if needed
        self.node_type = NodeType.MEMORY # reusing for now, or custom
        self.feedback_buffer = []  # Store FeedbackPayloads
        self.batch_size = 4        # Trigger training after 4 feedbacks
        self.training_task = None

    async def on_start(self) -> None:
        """Subscribe to feedback messages."""
        logger.info("Starting LearnerNode '%s'", self.node_id)
        await self.bus.subscribe("system.feedback", self.handle_message)

    async def on_stop(self) -> None:
        """Clean up."""
        logger.info("Stopping LearnerNode '%s'", self.node_id)
        if self.training_task and not self.training_task.done():
            self.training_task.cancel()

    async def handle_message(self, message: Message) -> Message | None:
        """Process incoming feedback."""
        if message.type != MessageType.FEEDBACK:
            return None

        try:
            payload = FeedbackPayload(**message.payload)
        except Exception as e:
            return message.create_error(f"Invalid FeedbackPayload: {e}")

        logger.info("Learner '%s' received feedback for msg %s: %d", self.node_id, payload.message_id, payload.rating)

        if payload.prompt and payload.response:
            self.feedback_buffer.append(payload)

        # Trigger background training if batch size is met
        if len(self.feedback_buffer) >= self.batch_size:
            batch = self.feedback_buffer[:self.batch_size]
            self.feedback_buffer = self.feedback_buffer[self.batch_size:]
            
            if self.training_task and not self.training_task.done():
                logger.warning("Training already in progress. Dropping batch or enqueueing...")
            else:
                self.training_task = asyncio.create_task(self._run_dpo_training(batch))

        return None

    async def _run_dpo_training(self, batch: list[FeedbackPayload]) -> None:
        """Execute DPO algorithm in a thread so as to not block the asyncio bus."""
        logger.info("LearnerNode starting background DPO training on %d samples...", len(batch))
        
        def _train():
            # In a real cluster:
            # 1. Pull current LoRA weights from ServiceRegistry or Model repo
            # 2. Tokenize prompt/response pairs
            # 3. If rating == 1, set response as chosen, query alternate for rejected, or vice versa
            # 4. Compute `compute_dpo_loss`
            # 5. Backprop and step optimizer
            # 6. Save new weights
            import time
            time.sleep(1) # simulate optimization
            pass
            
        try:
            await asyncio.to_thread(_train)
            logger.info("LearnerNode DPO training complete. Broadcasting update...")
            
            # Broadcast that new weights are available
            update_msg = Message(
                type=MessageType.LEARNING_UPDATE,
                source_node_id=self.node_id,
                target_node_id="", # broadcast
                topic="system.learning_update",
                payload={"status": "weights_updated"}
            )
            await self.bus.publish("system.learning_update", update_msg)
            
        except Exception as e:
            logger.error("LearnerNode DPO training failed: %s", e)
