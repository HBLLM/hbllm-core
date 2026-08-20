"""Mock Vision Provider — deterministic embeddings for testing.

Produces stable, reproducible embeddings based on image content hash.
Implements the ``VisionProvider`` protocol for unit and integration
testing without any ML model dependency.
"""

from __future__ import annotations

import hashlib
import math
import pathlib
from io import BytesIO
from typing import TYPE_CHECKING

from hbllm.perception.providers.types import VisualEmbedding

if TYPE_CHECKING:
    pass


# ── Type alias for image input ──────────────────────────────────────────
_ImageInput = "Image.Image | np.ndarray | bytes | pathlib.Path"

# Default mock embedding dimension
_DEFAULT_DIM = 384


class MockVisionProvider:
    """Deterministic mock vision provider for testing.

    Produces stable embeddings derived from image content hash,
    enabling reproducible test results without loading any ML model.

    The mock embedding is a unit vector whose direction is determined
    by the image hash — identical images always produce identical
    embeddings, and different images produce different (but consistently
    reproducible) embeddings.

    Attributes:
        model_name: Mock model identifier.
        dimensions: Embedding dimensionality.

    """

    def __init__(
        self,
        model_name: str = "mock/vision-v1",
        dimensions: int = _DEFAULT_DIM,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions

    @property
    def modality(self) -> str:
        return "vision"

    @property
    def provider_id(self) -> str:
        return f"mock:{self.model_name}"

    async def initialize(self) -> None:
        """No-op for mock provider."""

    async def shutdown(self) -> None:
        """No-op for mock provider."""

    async def encode(self, image: _ImageInput) -> VisualEmbedding:
        """Produce a deterministic embedding from image content hash."""
        image_hash = self._hash_image(image)
        vector = self._hash_to_vector(image_hash)
        return VisualEmbedding(
            vector=vector,
            model_id=self.model_name,
            space_id=f"{self.model_name}-image",
            embedding_type="semantic",
            dimensions=self.dimensions,
            normalization="l2",
            source="image",
            image_hash=image_hash,
        )

    async def encode_batch(self, images: list[_ImageInput]) -> list[VisualEmbedding]:
        """Sequential encoding — no batching optimization for mock."""
        return [await self.encode(img) for img in images]

    # ── Internals ────────────────────────────────────────────────────

    def _hash_image(self, image: _ImageInput) -> str:
        """Compute a deterministic SHA-256 hash from the image content."""
        if isinstance(image, bytes):
            return hashlib.sha256(image).hexdigest()
        if isinstance(image, pathlib.Path):
            return hashlib.sha256(str(image).encode()).hexdigest()
        if hasattr(image, "tobytes"):
            # numpy ndarray
            return hashlib.sha256(image.tobytes()).hexdigest()  # type: ignore[union-attr]
        # PIL Image — convert to bytes
        try:
            buf = BytesIO()
            image.save(buf, format="PNG")  # type: ignore[union-attr]
            return hashlib.sha256(buf.getvalue()).hexdigest()
        except Exception:
            # Fallback: hash the repr
            return hashlib.sha256(repr(image).encode()).hexdigest()

    def _hash_to_vector(self, image_hash: str) -> list[float]:
        """Convert a hex hash to a deterministic unit vector.

        Uses the hash bytes as seeds for a simple deterministic
        mapping, then L2-normalizes the result.
        """
        # Use hash bytes to seed vector components
        hash_bytes = bytes.fromhex(image_hash[:64])  # 32 bytes
        raw: list[float] = []
        for i in range(self.dimensions):
            # Deterministic pseudo-random from hash
            idx = i % len(hash_bytes)
            val = (hash_bytes[idx] + i * 17) % 256
            raw.append((val / 128.0) - 1.0)  # Map to [-1, 1]

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in raw))
        if norm > 0:
            raw = [x / norm for x in raw]

        return raw
