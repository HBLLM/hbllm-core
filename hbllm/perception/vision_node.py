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
    Multimodal sensory node that converts images to dense text captions
    and extracts text via OCR, bridging visual perception to the LLM.
    """
    def __init__(self, node_id: str):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DOMAIN,
            capabilities=["multimodal_processing", "image_captioning", "ocr"],
        )
        self.topic_sub = "vision.process"
        self.pipeline = None
        self._ocr_reader = None

    async def on_start(self) -> None:
        logger.info("Starting VisionNode. Loading ViT model...")
        import asyncio
        await asyncio.to_thread(self._load_model)
        await self.bus.subscribe(self.topic_sub, self.handle_message)
        await self.bus.subscribe("vision.ocr", self.handle_ocr)
        await self.bus.subscribe("vision.caption", self.handle_message)
        # Multi-modal workspace: participate as a competing thought source
        await self.bus.subscribe("module.evaluate", self.handle_workspace_query)
        logger.info("VisionNode Ready. Subscribed to %s + vision.ocr + module.evaluate", self.topic_sub)

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

    def _extract_text_ocr(self, path: str) -> str:
        """Extract text from image using OCR (EasyOCR with fallback)."""
        # Try EasyOCR first
        try:
            if self._ocr_reader is None:
                import easyocr
                self._ocr_reader = easyocr.Reader(['en'], gpu=False)
            
            results = self._ocr_reader.readtext(path, detail=0)
            return "\n".join(results) if results else ""
        except ImportError:
            pass
        
        # Fallback: try pytesseract
        try:
            from PIL import Image
            import pytesseract
            image = Image.open(path)
            return pytesseract.image_to_string(image).strip()
        except ImportError:
            pass
        
        logger.debug("No OCR engine available (install easyocr or pytesseract)")
        return ""

    async def handle_ocr(self, message: Message) -> Message | None:
        """
        Extract text from images via OCR.
        
        Payload expects:
            image_path: str -> path to image file
        
        Returns:
            caption: str, ocr_text: str, combined: str
        """
        image_path = message.payload.get("image_path")
        if not image_path:
            return message.create_error("No 'image_path' provided.")
        
        try:
            import asyncio
            
            # Run caption + OCR in parallel threads
            caption_task = asyncio.to_thread(self._process_image, image_path) if self.pipeline else None
            ocr_task = asyncio.to_thread(self._extract_text_ocr, image_path)
            
            tasks = [t for t in [caption_task, ocr_task] if t is not None]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            caption = ""
            ocr_text = ""
            
            if caption_task is not None:
                caption = results[0] if not isinstance(results[0], Exception) else ""
                ocr_text = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else ""
            else:
                ocr_text = results[0] if not isinstance(results[0], Exception) else ""
            
            combined = caption
            if ocr_text:
                combined = f"{caption}\n\n[Extracted Text]:\n{ocr_text}" if caption else ocr_text
            
            return message.create_response({
                "caption": caption,
                "ocr_text": ocr_text,
                "combined": combined,
                "domain": "vision",
            })
        except Exception as e:
            logger.error("OCR processing error: %s", e)
            return message.create_error(f"OCR failure: {e}")

    async def handle_workspace_query(self, message: Message) -> Message | None:
        """
        Multi-modal workspace participation: if the query references an image,
        post a vision_perception thought (with OCR if available) to the blackboard.
        """
        payload = message.payload
        image_path = payload.get("image_path")
        
        if not image_path or not self.pipeline:
            return None  # Not a vision-relevant query
        
        try:
            import asyncio
            
            # Caption + OCR
            caption = await asyncio.to_thread(self._process_image, image_path)
            ocr_text = ""
            try:
                ocr_text = await asyncio.to_thread(self._extract_text_ocr, image_path)
            except Exception:
                pass
            
            content = f"[Visual Analysis] {caption}"
            if ocr_text:
                content += f"\n[Extracted Text] {ocr_text[:500]}"
            
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
                    "content": content,
                    "modality": "image",
                },
                correlation_id=message.correlation_id,
            )
            await self.bus.publish("workspace.thought", thought_msg)
        except Exception as e:
            logger.warning("VisionNode workspace thought failed: %s", e)
        
        return None
