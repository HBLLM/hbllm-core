"""Vision Perception Adapter — concrete UnifiedPerceptionProvider wrapping SigLIP.

Converts visual inputs into typed VisualAssessment observations without coupling
the vision model directly to HCIR.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.types import Provenance
from hbllm.perception.providers.evidence import (
    CandidateRanking,
    EpistemicEvidenceProfile,
    VisualAssessment,
    VisualEvidence,
)
from hbllm.perception.providers.siglip_provider import SigLIPVisionProvider
from hbllm.runtime.providers.capability import ProviderCapability

logger = logging.getLogger(__name__)


class VisionPerceptionAdapter:
    """Concrete perception adapter wrapping SigLIPVisionProvider.

    Conforms to ``UnifiedPerceptionProvider``.
    Produces ``VisualAssessment`` instances that ``EvidenceNormalizer``
    converts to canonical ``PerceptualEvidenceNode`` objects.

    Usage::

        adapter = VisionPerceptionAdapter()
        await adapter.initialize()
        assessments = await adapter.observe(image_input)
    """

    def __init__(
        self,
        provider_id: str = "siglip_vision",
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
        underlying_provider: SigLIPVisionProvider | None = None,
    ) -> None:
        self._provider_id = provider_id
        self._model_name = model_name
        self._device = device
        self._provider = underlying_provider or SigLIPVisionProvider(
            model_name=model_name,
            device=device,
        )

    @property
    def capability(self) -> ProviderCapability:
        """Declarative capability manifest for SigLIP vision perception."""
        return ProviderCapability(
            provider_id=self._provider_id,
            provider_type="perception",
            capabilities=["visual_embedding", "detect_concepts", "classify_image"],
            modalities=["visual"],
            latency_profile="low",
            quality_profile="high",
            memory_requirement_mb=400,
            hardware_requirements=["cpu", "cuda", "mps"],
            requires_network=False,
            precision="fp32",
        )

    async def initialize(self) -> None:
        """Initialize resources (lazy loading happens on first encode or explicit init)."""
        logger.info("Initializing VisionPerceptionAdapter (model=%s)", self._model_name)

    async def shutdown(self) -> None:
        """Release vision model resources."""
        self._provider._model = None
        self._provider._processor = None
        logger.info("Shutdown VisionPerceptionAdapter (%s)", self._provider_id)

    async def observe(self, input_data: Any) -> list[VisualAssessment]:
        """Produce typed visual observations from input image data.

        Args:
            input_data: Image data (PIL Image, numpy array, bytes, or file path).

        Returns:
            List containing VisualAssessment with computed visual embedding.
        """
        try:
            embedding = await self._provider.encode(input_data)
            evidence = VisualEvidence(
                embedding=embedding,
                provenance=Provenance(
                    created_by=self._provider_id,
                    engine=self._model_name,
                    source_type="observed",
                ),
                image_hash=getattr(embedding, "image_hash", ""),
            )

            assessment = VisualAssessment(
                evidence=evidence,
                candidate_concepts=[],
                ranking=CandidateRanking(
                    best_score=1.0, second_score=0.0, margin=1.0, ambiguity=0.0
                ),
                epistemic_profile=EpistemicEvidenceProfile(
                    label_provenance=0.5,
                    perceptual_similarity=0.9,
                    evidence_strength=0.85,
                    source_reliability=0.95,
                ),
            )
            return [assessment]
        except Exception as e:
            logger.error("VisionPerceptionAdapter observation failed: %s", e)
            return []
