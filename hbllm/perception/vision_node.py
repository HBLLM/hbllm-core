"""
Multimodal Vision Node.

Uses a lightweight Vision Transformer to extract semantic descriptions
from images, allowing the text-based cognitive architecture to "see".
"""

from __future__ import annotations

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

    def __init__(self, node_id: str) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PERCEPTION,
            capabilities=["multimodal_processing", "image_captioning", "ocr"],
        )
        self.topic_sub = "vision.process"
        self.pipeline: Any = None
        self._ocr_reader: Any = None

    async def on_start(self) -> None:
        logger.info("Starting VisionNode. Loading ViT model...")
        import asyncio

        await asyncio.to_thread(self._load_model)
        await self.bus.subscribe(self.topic_sub, self.handle_message)
        await self.bus.subscribe("vision.ocr", self.handle_ocr)
        await self.bus.subscribe("vision.caption", self.handle_message)
        # Multi-modal workspace: participate as a competing thought source
        await self.bus.subscribe("module.evaluate", self.handle_workspace_query)
        logger.info(
            "VisionNode Ready. Subscribed to %s + vision.ocr + module.evaluate", self.topic_sub
        )

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import pipeline  # type: ignore

            device: int | str = 0 if torch.cuda.is_available() else -1
            if torch.backends.mps.is_available() and device == -1:
                # pipeline device=-1 means CPU. pipeline currently drops support for strings like 'mps' in some versions unless passed explicitly as a torch.device
                device = "mps"
            self.pipeline = pipeline(
                "image-to-text", model="Salesforce/blip-image-captioning-base", device=device
            )
        except Exception as e:
            logger.error("Failed to load vision model: %s", e)

    async def on_stop(self) -> None:
        logger.info("Stopping VisionNode")

    async def handle_message(self, message: Message) -> Message | None:
        if message.type != MessageType.QUERY:
            return None

        image_data = message.payload.get("image_path") or message.payload.get("image_data")
        if not image_data:
            return message.create_error("No 'image_path' or 'image_data' provided in payload.")

        if not self.pipeline:
            return message.create_error("Vision pipeline not initialized.")

        try:
            import asyncio

            caption = await asyncio.to_thread(self._process_image, str(image_data))
            return message.create_response({"text": caption, "domain": "vision"})
        except Exception as e:
            logger.error("Vision processing error: %s", e)
            return message.create_error(f"Vision failure: {e}")

    def _process_image(self, path_or_hex: str) -> str:
        import io

        from PIL import Image  # type: ignore

        # Handle hex encoded data or path
        try:
            if len(path_or_hex) > 512:  # Likely hex
                image = Image.open(io.BytesIO(bytes.fromhex(path_or_hex))).convert("RGB")
            else:
                image = Image.open(path_or_hex).convert("RGB")
        except Exception:
            # Fallback to path
            image = Image.open(path_or_hex).convert("RGB")

        res = self.pipeline(image)
        return str(res[0]["generated_text"])

    def _extract_text_ocr(self, path_or_hex: str) -> str:
        """Extract text from image using OCR (EasyOCR with fallback)."""
        # Try EasyOCR first
        try:
            if self._ocr_reader is None:
                import easyocr  # type: ignore

                self._ocr_reader = easyocr.Reader(["en"], gpu=False)

            # EasyOCR can take bytes, ndarray, or path
            data: Any = path_or_hex
            if len(path_or_hex) > 512:
                data = bytes.fromhex(path_or_hex)

            results = self._ocr_reader.readtext(data, detail=0)
            return "\n".join(results) if results else ""
        except ImportError:
            pass

        # Fallback: try pytesseract
        try:
            import io

            import pytesseract  # type: ignore
            from PIL import Image

            if len(path_or_hex) > 512:
                image = Image.open(io.BytesIO(bytes.fromhex(path_or_hex)))
            else:
                image = Image.open(path_or_hex)

            return str(pytesseract.image_to_string(image).strip())
        except ImportError:
            pass

        logger.debug("No OCR engine available (install easyocr or pytesseract)")
        return ""

    async def handle_ocr(self, message: Message) -> Message | None:
        """
        Extract text from images via OCR.

        Payload expects:
            image_path: str -> path to image file
            or
            image_data: str -> hex encoded image

        Returns:
            caption: str, ocr_text: str, combined: str
        """
        image_data = message.payload.get("image_path") or message.payload.get("image_data")
        if not image_data:
            return message.create_error("No 'image_path' or 'image_data' provided.")

        try:
            import asyncio

            data_str = str(image_data)

            # Run caption + OCR in parallel threads
            caption_task = (
                asyncio.to_thread(self._process_image, data_str) if self.pipeline else None
            )
            ocr_task = asyncio.to_thread(self._extract_text_ocr, data_str)

            tasks = [t for t in [caption_task, ocr_task] if t is not None]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            caption = ""
            ocr_text = ""

            if caption_task is not None:
                caption = str(results[0]) if not isinstance(results[0], Exception) else ""
                ocr_text = (
                    str(results[1])
                    if len(results) > 1 and not isinstance(results[1], Exception)
                    else ""
                )
            else:
                ocr_text = str(results[0]) if not isinstance(results[0], Exception) else ""

            combined = caption
            if ocr_text:
                combined = f"{caption}\n\n[Extracted Text]:\n{ocr_text}" if caption else ocr_text

            return message.create_response(
                {
                    "caption": caption,
                    "ocr_text": ocr_text,
                    "combined": combined,
                    "domain": "vision",
                }
            )
        except Exception as e:
            logger.error("OCR processing error: %s", e)
            return message.create_error(f"OCR failure: {e}")

    async def handle_workspace_query(self, message: Message) -> Message | None:
        """
        Multi-modal workspace participation: if the query references an image,
        post a vision_perception thought (with OCR if available) to the blackboard.
        """
        payload = message.payload
        image_data = payload.get("image_path") or payload.get("image_data")

        if not image_data or not self.pipeline:
            return None  # Not a vision-relevant query

        try:
            import asyncio

            data_str = str(image_data)

            # Caption + OCR
            caption = await asyncio.to_thread(self._process_image, data_str)
            ocr_text = ""
            try:
                ocr_text = await asyncio.to_thread(self._extract_text_ocr, data_str)
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
