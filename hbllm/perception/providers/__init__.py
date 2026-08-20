"""Perception Providers — typed protocols and adapters for grounded perception.

Provider protocols define the contracts for how HBLLM acquires perceptual
evidence from the physical world.  Each protocol represents a single
capability (encode, detect, extract_text, segment) — a provider can
implement multiple capabilities.

Architecture invariant:
    Providers NEVER mutate HCIR.
    Providers produce VisualEvidence.
    HCIR transactions own state mutation.
"""

from hbllm.perception.providers.base import (
    ImageInput,
    PerceptionProvider,
    VisionDetector,
    VisionOCR,
    VisionProvider,
)
from hbllm.perception.providers.evidence import (
    CandidateRanking,
    ConceptCandidate,
    EpistemicEvidenceProfile,
    ObservationMatch,
    VisualAssessment,
    VisualEvidence,
)
from hbllm.perception.providers.policy import RecognitionPolicy
from hbllm.perception.providers.types import (
    EmbeddingRef,
    VisualEmbedding,
    VisualRegion,
)

__all__ = [
    # Protocols
    "PerceptionProvider",
    "VisionProvider",
    "VisionDetector",
    "VisionOCR",
    # Input
    "ImageInput",
    # Types
    "VisualEmbedding",
    "VisualRegion",
    "EmbeddingRef",
    # Evidence
    "VisualEvidence",
    "VisualAssessment",
    "EpistemicEvidenceProfile",
    "ObservationMatch",
    "ConceptCandidate",
    "CandidateRanking",
    # Policy
    "RecognitionPolicy",
]
