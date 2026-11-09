"""SigLIP Vision Provider — semantic visual embeddings.

Uses Google's SigLIP (Sigmoid Language-Image Pre-training) model for
producing semantic visual embeddings.  SigLIP has better zero-shot
performance and is more compute-efficient than CLIP.

The model is loaded lazily on first ``encode()`` call to avoid
unnecessary startup cost.

Architecture:
    Image → SigLIP → VisualEmbedding (semantic, L2-normalized)
    VisualEmbedding → VisualMemory (vector store)
    HCIR holds EmbeddingRef (lightweight reference), not the vector.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import pathlib
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np

from hbllm.perception.providers.types import VisualEmbedding

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# ── Type alias ──
_ImageInput = "Image.Image | np.ndarray | bytes | pathlib.Path"


class SigLIPVisionProvider:
    """SigLIP embedding provider — semantic visual embeddings.

    Lazy model loading: the model and processor are loaded on the first
    ``encode()`` call.  Supports CPU, CUDA, and MPS backends.

    Usage::

        provider = SigLIPVisionProvider()
        embedding = await provider.encode(some_image)
        # embedding.space_id == "google/siglip-base-patch16-224-image"

    Attributes:
        model_name: HuggingFace model identifier.
        device: Compute device (``"cpu"``, ``"cuda"``, ``"mps"``, or ``None`` for auto-detect).

    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._model = None
        self._processor = None
        self._dimensions: int | None = None

    @property
    def modality(self) -> str:
        return "vision"

    @property
    def provider_id(self) -> str:
        return f"siglip:{self.model_name}"

    async def initialize(self) -> None:
        """Pre-load the model (optional — also done lazily on first encode)."""
        await asyncio.to_thread(self._ensure_loaded)

    async def shutdown(self) -> None:
        """Release model resources."""
        self._model = None
        self._processor = None

    async def encode(self, image: _ImageInput) -> VisualEmbedding:
        """Encode a single image into a SigLIP embedding.

        Thread-safe: the actual model inference runs in a thread pool
        to avoid blocking the event loop.
        """
        pil_image = self._to_pil(image)
        image_hash = self._hash_pil(pil_image)
        vector = await asyncio.to_thread(self._encode_sync, pil_image)

        return VisualEmbedding(
            vector=vector,
            model_id=self.model_name,
            space_id=f"{self.model_name}-image",
            embedding_type="semantic",
            dimensions=len(vector),
            normalization="l2",
            source="image",
            image_hash=image_hash,
        )

    async def encode_batch(self, images: list[_ImageInput]) -> list[VisualEmbedding]:
        """Encode multiple images.

        Uses batched model inference for efficiency when available.
        """
        pil_images = [self._to_pil(img) for img in images]
        hashes = [self._hash_pil(img) for img in pil_images]
        vectors = await asyncio.to_thread(self._encode_batch_sync, pil_images)

        return [
            VisualEmbedding(
                vector=vec,
                model_id=self.model_name,
                space_id=f"{self.model_name}-image",
                embedding_type="semantic",
                dimensions=len(vec),
                normalization="l2",
                source="image",
                image_hash=h,
            )
            for vec, h in zip(vectors, hashes)
        ]

    # ══════════════════════════════════════════════════════════════════
    # Synchronous internals (run in thread pool)
    # ══════════════════════════════════════════════════════════════════

    def _ensure_loaded(self) -> None:
        """Lazily load the SigLIP model and processor."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoProcessor

        # Auto-detect device
        if self._device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

        logger.info(
            "Loading SigLIP model '%s' on device '%s'",
            self.model_name,
            self._device,
        )

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
        self._model.eval()

        # Determine embedding dimensions from model config
        if hasattr(self._model.config, "vision_config"):
            self._dimensions = self._model.config.vision_config.hidden_size
        else:
            self._dimensions = self._model.config.hidden_size

        logger.info(
            "SigLIP loaded: %d dimensions, device=%s",
            self._dimensions,
            self._device,
        )

    def _encode_sync(self, pil_image: Image.Image) -> list[float]:
        """Synchronous single-image encoding."""
        import torch

        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None

        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)

        # L2 normalize
        embedding = outputs[0].cpu().numpy()
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def _encode_batch_sync(self, pil_images: list[Image.Image]) -> list[list[float]]:
        """Synchronous batched encoding."""
        import torch

        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None

        inputs = self._processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)

        # L2 normalize each embedding
        embeddings = outputs.cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        embeddings = embeddings / norms

        return [emb.tolist() for emb in embeddings]

    # ══════════════════════════════════════════════════════════════════
    # Image normalization
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _to_pil(image: _ImageInput) -> Image.Image:
        """Convert any ImageInput to PIL Image."""
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            return PILImage.fromarray(image).convert("RGB")
        if isinstance(image, bytes):
            return PILImage.open(BytesIO(image)).convert("RGB")
        if isinstance(image, pathlib.Path):
            return PILImage.open(image).convert("RGB")
        msg = f"Unsupported image type: {type(image)}"
        raise TypeError(msg)

    @staticmethod
    def _hash_pil(image: Image.Image) -> str:
        """Compute SHA-256 hash of a PIL image."""
        buf = BytesIO()
        image.save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()
