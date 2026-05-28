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
from hbllm.perception.vector_projector import MultimodalProjector

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
        self.rust_engine: Any = None
        self.change_detector: Any = None
        self.projector = MultimodalProjector(llm_dim=4096)
        self._last_caption_cache: dict[str, str] = {}
        self._last_embedding_cache: dict[str, list[float]] = {}

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
        # Try loading Rust engine first
        try:
            import hbllm_perception_rs  # type: ignore

            self.rust_engine = hbllm_perception_rs.VisionEngine()
            self.rust_engine.load_model("mock")
            self.change_detector = hbllm_perception_rs.ChangeDetector(threshold=5)
            logger.info("Successfully initialized Rust-native ONNX perception engine (mock mode).")
            return
        except ImportError:
            logger.info("hbllm_perception_rs not found. Falling back to PyTorch/transformers.")
        except Exception as e:
            logger.error(
                "Failed to initialize hbllm_perception_rs: %s. Falling back to PyTorch.", e
            )

        try:
            import torch
            from transformers import pipeline  # type: ignore

            device: Any = 0 if torch.cuda.is_available() else -1
            if torch.backends.mps.is_available() and device == -1:
                # pipeline device=-1 means CPU. pipeline currently drops support for strings like 'mps' in some versions unless passed explicitly as a torch.device
                device = "mps"
            pipeline_fn: Any = pipeline
            self.pipeline = pipeline_fn(
                "image-to-text",
                model="Salesforce/blip-image-captioning-base",
                device=device,
            )
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Failed to load vision model: %s", e)

    async def on_stop(self) -> None:
        logger.info("Stopping VisionNode")

    async def handle_message(self, message: Message) -> Message | None:
        if message.type != MessageType.QUERY:
            return None

        image_data = message.payload.get("image_path") or message.payload.get("image_data")
        if not image_data:
            return message.create_error("No 'image_path' or 'image_data' provided in payload.")

        if not self.pipeline and not self.rust_engine:
            return message.create_error("Vision pipeline and Rust engine not initialized.")

        try:
            import asyncio

            # Check change detector
            entity_id = message.payload.get("entity_id") or "default"
            session_id = message.session_id or "default"
            cache_key = f"{entity_id}:{session_id}"

            changed = True
            image_bytes = None
            if self.change_detector:
                try:
                    image_bytes = self._get_image_bytes(str(image_data))
                    changed = self.change_detector.is_changed(image_bytes)
                except Exception as e:
                    logger.error("Change detection failed: %s", e)

            if not changed and cache_key in self._last_caption_cache:
                logger.info("Frame unchanged for %s. Returning cached result.", cache_key)
                caption = self._last_caption_cache[cache_key]
                embedding = self._last_embedding_cache.get(cache_key, [])
                return message.create_response(
                    {"text": caption, "domain": "vision", "embedding": embedding, "cached": True}
                )

            # If changed (or first run), process image and get embedding
            caption = await asyncio.to_thread(self._process_image, str(image_data))
            embedding = await asyncio.to_thread(self._embed_image, str(image_data))

            self._last_caption_cache[cache_key] = caption
            self._last_embedding_cache[cache_key] = embedding

            return message.create_response(
                {"text": caption, "domain": "vision", "embedding": embedding, "cached": False}
            )
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Vision processing error: %s", e)
            return message.create_error(f"Vision failure: {e}")

    def _get_image_bytes(self, path_or_hex: str) -> bytes:
        if len(path_or_hex) > 512:  # Likely hex
            return bytes.fromhex(path_or_hex)
        else:
            with open(path_or_hex, "rb") as f:
                return f.read()

    def _process_image(self, path_or_hex: str) -> str:
        if self.rust_engine:
            try:
                image_bytes = self._get_image_bytes(path_or_hex)
                return str(self.rust_engine.caption(image_bytes))
            except Exception as e:
                logger.error("Rust captioning failed: %s. Falling back...", e)

        import io

        from PIL import Image  # type: ignore

        # Handle hex encoded data or path
        try:
            if len(path_or_hex) > 512:  # Likely hex
                image = Image.open(io.BytesIO(bytes.fromhex(path_or_hex))).convert("RGB")
            else:
                image = Image.open(path_or_hex).convert("RGB")
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError):
            # Fallback to path
            image = Image.open(path_or_hex).convert("RGB")

        if self.pipeline:
            res = self.pipeline(image)
            return str(res[0]["generated_text"])
        raise RuntimeError("No vision model pipeline is available.")

    def _embed_image(self, path_or_hex: str) -> list[float]:
        if self.rust_engine:
            try:
                image_bytes = self._get_image_bytes(path_or_hex)
                return [float(v) for v in self.rust_engine.embed(image_bytes)]
            except Exception as e:
                logger.error("Rust embedding extraction failed: %s", e)

        # Fallback pseudo-random deterministic embedding
        import hashlib

        h = hashlib.sha256(path_or_hex.encode()).digest()
        emb = []
        for i in range(768):
            val = ((h[i % 32] + i) % 256) / 255.0 * 2.0 - 1.0
            emb.append(val)
        return emb

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
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("OCR processing error: %s", e)
            return message.create_error(f"OCR failure: {e}")

    async def handle_workspace_query(self, message: Message) -> Message | None:
        """
        Multi-modal workspace participation: if the query references an image,
        post a vision_perception thought (with OCR if available) to the blackboard.
        """
        payload = message.payload
        image_data = payload.get("image_path") or payload.get("image_data")

        if not image_data or (not self.pipeline and not self.rust_engine):
            return None  # Not a vision-relevant query

        try:
            import asyncio

            data_str = str(image_data)

            # Caption + OCR
            caption = await asyncio.to_thread(self._process_image, data_str)
            ocr_text = ""
            try:
                ocr_text = await asyncio.to_thread(self._extract_text_ocr, data_str)
            except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError):
                pass

            raw_embedding = await asyncio.to_thread(self._embed_image, data_str)
            projected_embedding = self.projector.project_vision(raw_embedding)

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
                    "embedding": projected_embedding,
                },
                correlation_id=message.correlation_id,
            )
            await self.bus.publish("workspace.thought", thought_msg)
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.warning("VisionNode workspace thought failed: %s", e)

        return None
