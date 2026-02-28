"""
Multimodal Vision Node.

Uses a lightweight Vision Transformer to extract semantic descriptions 
from images, allowing the text-based cognitive architecture to "see".
"""

import logging
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)

class VisionNode(Node):
    """
    Multimodal sensory node that converts images to dense text captions,
    bridging the gap between the modular LLM and visual perception.
    """
    def __init__(self, node_id: str):
        super().__init__(node_id=node_id, node_type=NodeType.DOMAIN, capabilities=["multimodal_processing", "image_captioning"])
        self.topic_sub = "vision.process"
        self.pipeline = None

    async def on_start(self) -> None:
        logger.info("Starting VisionNode. Loading ViT model...")
        import asyncio
        await asyncio.to_thread(self._load_model)
        await self.bus.subscribe(self.topic_sub, self.handle_message)
        # Multi-modal workspace: participate as a competing thought source
        await self.bus.subscribe("module.evaluate", self.handle_workspace_query)
        logger.info("VisionNode Ready. Subscribed to %s + module.evaluate", self.topic_sub)

    def _load_model(self):
        try:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            if torch.backends.mps.is_available() and device == -1:
                # pipeline device=-1 means CPU. pipeline currently drops support for strings like 'mps' in some versions unless passed explicitly as a torch.device
                device = "mps" 
            self.pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=device)
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")

    async def on_stop(self) -> None:
        logger.info("Stopping VisionNode")

    async def handle_message(self, message: Message) -> Message | None:
        if message.type != MessageType.QUERY:
            return None

        image_path = message.payload.get("image_path")
        if not image_path:
            return message.create_error("No 'image_path' provided in payload.")

        if not self.pipeline:
            return message.create_error("Vision pipeline not initialized.")

        try:
            import asyncio
            caption = await asyncio.to_thread(self._process_image, image_path)
            return message.create_response({"text": caption, "domain": "vision"})
        except Exception as e:
            logger.error("Vision processing error: %s", e)
            return message.create_error(f"Vision failure: {e}")

    def _process_image(self, path: str) -> str:
        from PIL import Image
        image = Image.open(path).convert("RGB")
        res = self.pipeline(image)
        return res[0]["generated_text"]

    async def handle_workspace_query(self, message: Message) -> Message | None:
        """
        Multi-modal workspace participation: if the query references an image,
        post a vision_perception thought to the blackboard.
        """
        payload = message.payload
        image_path = payload.get("image_path")
        
        if not image_path or not self.pipeline:
            return None  # Not a vision-relevant query
        
        try:
            import asyncio
            caption = await asyncio.to_thread(self._process_image, image_path)
            
            # Post as a competing thought on the Workspace blackboard
            thought_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="workspace.thought",
                payload={
                    "type": "vision_perception",
                    "confidence": 0.85,
                    "content": f"[Visual Analysis] {caption}",
                    "modality": "image",
                },
                correlation_id=message.correlation_id,
            )
            await self.bus.publish("workspace.thought", thought_msg)
        except Exception as e:
            logger.warning("VisionNode workspace thought failed: %s", e)
        
        return None
