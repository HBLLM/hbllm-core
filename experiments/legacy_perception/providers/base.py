"""Perception Provider Protocols — HBLLM Grounded Perception §V0.

Defines the capability-separated protocols for perceptual evidence
acquisition.  Each protocol represents a single capability:

    VisionProvider   — encode images into typed embeddings
    VisionDetector   — detect objects/regions in images
    VisionOCR        — extract text from images

A concrete provider can implement multiple capabilities (e.g., a VLM
could be both ``VisionProvider`` and ``VisionOCR``).

Architecture invariant:
    ``ImageInput`` is a **type alias**, not a Pydantic model.
    Images are runtime objects, not cognitive state.
    The boundary is: ``RuntimeImage → Provider → VisualEmbedding → HCIR``.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Protocol, Union, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

    from hbllm.perception.providers.types import VisualEmbedding, VisualRegion

# ── Image input — type alias, not Pydantic model ────────────────────────
# Images are runtime objects, not cognitive state.
# HCIR stores embedding_ref (lightweight reference), not the image.
ImageInput = Union["Image.Image", "np.ndarray", bytes, pathlib.Path]


# ═══════════════════════════════════════════════════════════════════════════
# Base Protocol
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class PerceptionProvider(Protocol):
    """Base protocol for all perception providers.

    Providers acquire perceptual evidence from the physical world.
    They NEVER mutate HCIR — they produce typed evidence objects.
    """

    @property
    def modality(self) -> str:
        """Perception modality (``"vision"``, ``"audio"``, ``"sensor"``)."""
        ...

    @property
    def provider_id(self) -> str:
        """Unique provider identifier (e.g., ``"siglip:google/siglip-base-patch16-224"``)."""
        ...

    async def initialize(self) -> None:
        """Initialize the provider (lazy model loading, etc.)."""
        ...

    async def shutdown(self) -> None:
        """Release resources."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Vision Capabilities
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class VisionProvider(PerceptionProvider, Protocol):
    """Encodes images into typed embeddings.

    This is the core vision capability — mapping visual input to a
    vector in a semantic or feature embedding space.

    Does NOT detect, segment, or read text.  Those are separate
    capabilities implemented by ``VisionDetector``, ``VisionSegmenter``,
    and ``VisionOCR``.
    """

    async def encode(self, image: ImageInput) -> VisualEmbedding:
        """Encode a single image into a typed embedding."""
        ...

    async def encode_batch(self, images: list[ImageInput]) -> list[VisualEmbedding]:
        """Encode multiple images.  Default: sequential calls to ``encode()``."""
        ...


@runtime_checkable
class VisionDetector(PerceptionProvider, Protocol):
    """Detects objects/regions in images.

    Returns bounding boxes with labels and confidence scores.
    Does NOT produce embeddings — use ``VisionProvider`` for that.
    """

    async def detect(self, image: ImageInput) -> list[VisualRegion]:
        """Detect objects/regions in an image."""
        ...


@runtime_checkable
class VisionOCR(PerceptionProvider, Protocol):
    """Extracts text from images."""

    async def extract_text(self, image: ImageInput) -> str:
        """Extract text content from an image."""
        ...
