"""Visual Perception Types — HBLLM Grounded Perception §V0.

Core typed data structures for visual perception:

    ``VisualEmbedding`` — Typed embedding with space_id for cross-space safety.
    ``EmbeddingRef``    — Lightweight reference to a stored embedding.
    ``VisualRegion``    — A detected region within an image.

Design invariants:
    - Raw embeddings NEVER become HCIR graph payloads.
    - HCIR holds ``EmbeddingRef`` (lightweight reference).
    - The actual vector resides in ``VisualMemory`` (vector store).
    - ``space_id`` prevents accidentally comparing incompatible vectors.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field


class VisualEmbedding(BaseModel):
    """A typed embedding from a vision model — immutable perceptual measurement.

    The ``space_id`` field uniquely identifies the embedding space.  Two
    embeddings are only meaningful to compare if they share the same
    ``space_id``.  This prevents cross-space errors such as comparing
    SigLIP-image vs. DINO-image vs. SigLIP-text.

    Attributes:
        vector: The embedding vector.
        model_id: Model that produced this embedding
            (e.g., ``"google/siglip-base-patch16-224"``).
        space_id: Embedding space identifier
            (e.g., ``"siglip-base-patch16-224-image"``).
        embedding_type: ``"semantic"`` (CLIP/SigLIP) or ``"feature"`` (DINO).
        dimensions: Dimensionality of the embedding vector.
        normalization: Normalization applied (``"l2"``, ``"none"``, or ``None``).
        source: Input modality — ``"image"`` or ``"text"``.
        timestamp: When this embedding was computed.
        image_hash: SHA-256 of the input image for deduplication.

    """

    vector: list[float]
    model_id: str
    space_id: str
    embedding_type: str  # "semantic" | "feature"
    dimensions: int
    normalization: str | None = None
    source: str = "image"  # "image" | "text"
    timestamp: float = Field(default_factory=time.time)
    image_hash: str = ""

    def is_compatible_with(self, other: VisualEmbedding) -> bool:
        """Check whether two embeddings can be meaningfully compared.

        Embeddings from different spaces (e.g., SigLIP-image vs. DINO-image)
        should NEVER be compared by cosine similarity.
        """
        return self.space_id == other.space_id


class EmbeddingRef(BaseModel):
    """Lightweight reference to an embedding stored in VisualMemory.

    HCIR nodes hold this instead of the raw vector.  Keeps the
    cognitive graph lightweight and transactional while the actual
    vector resides in the vector store.

    Attributes:
        ref_id: Unique identifier in the vector store.
        space_id: Which embedding space this belongs to.
        model_id: Which model produced it.
        image_hash: Source image identity for deduplication.

    """

    ref_id: str
    space_id: str
    model_id: str
    image_hash: str = ""


class VisualRegion(BaseModel):
    """A detected region within an image.

    Coordinates are normalized to [0, 1] so they are resolution-independent.

    Attributes:
        bbox: Bounding box ``(x1, y1, x2, y2)`` in normalized coordinates.
        label: Detected object class label.
        confidence: Detection confidence score.
        embedding_ref: Optional reference to a stored embedding for this region.

    """

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized [0,1]
    label: str
    confidence: float
    embedding_ref: str | None = None
