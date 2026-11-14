"""Visual Perception Runtime — produces evidence, never mutates HCIR.

This is the perception boundary: images come in, typed evidence comes out.
The runtime NEVER touches HCIR nodes, edges, or beliefs.

Architecture:
    Image → VisionProvider.encode() → VisualEvidence
    VisualEvidence + VisualMemory.search() → VisualAssessment

    VisualAssessment is then passed to VisualPerceptionTransaction
    for HCIR state commitment.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hbllm.hcir.types import Provenance
from hbllm.perception.providers.evidence import (
    EpistemicEvidenceProfile,
    VisualAssessment,
    VisualEvidence,
)

if TYPE_CHECKING:
    from hbllm.perception.providers.base import ImageInput, VisionProvider
    from hbllm.perception.visual_memory import VisualMemory

logger = logging.getLogger(__name__)


class VisualPerceptionRuntime:
    """Produces evidence and assessment. NEVER mutates HCIR.

    The runtime is the perception boundary — it converts raw images
    into typed evidence and current interpretations. All state
    mutation is delegated to VisualPerceptionTransaction.

    Usage::

        runtime = VisualPerceptionRuntime(provider, memory)
        assessment = await runtime.perceive(image)
        # assessment.evidence = immutable measurement
        # assessment.candidate_concepts = current interpretation
    """

    def __init__(
        self,
        provider: VisionProvider,
        memory: VisualMemory,
    ) -> None:
        self.provider = provider
        self.memory = memory

    async def perceive(self, image: ImageInput) -> VisualAssessment:
        """Perceive an image — produce evidence + assessment.

        1. Encode image → VisualEmbedding
        2. Build VisualEvidence (immutable measurement)
        3. Search observations → ObservationMatch[]
        4. Derive concept candidates → ConceptCandidate[] + CandidateRanking
        5. Compute EpistemicEvidenceProfile
        6. Return VisualAssessment

        Does NOT:
        - Create HCIR nodes
        - Add edges
        - Record beliefs
        - Determine novelty (that's policy, not perception)
        """
        embedding = await self.provider.encode(image)

        evidence = VisualEvidence(
            embedding=embedding,
            provenance=Provenance(
                created_by=f"visual_perception:{self.provider.provider_id}",
                source_type="observed",
            ),
            image_hash=embedding.image_hash,
        )

        # Search observations — primary evidence retrieval
        obs_matches = await self.memory.search_observations(embedding, top_k=10)

        # Derive concept candidates — second-order derivation
        candidates, ranking = self.memory.derive_concept_candidates(obs_matches)

        # Compute epistemic profile
        profile = EpistemicEvidenceProfile(
            perceptual_similarity=ranking.best_score,
            evidence_strength=min(1.0, len(obs_matches) / 10.0) if obs_matches else 0.0,
            source_reliability=1.0,
        )

        return VisualAssessment(
            evidence=evidence,
            candidate_observations=obs_matches,
            candidate_concepts=candidates,
            ranking=ranking,
            epistemic_profile=profile,
        )

    async def perceive_with_label(
        self,
        image: ImageInput,
        label: str,
        context: str = "",
    ) -> VisualAssessment:
        """Perceive an image with a user-provided label.

        Sets label_provenance=1.0 because the label comes from a
        trusted source (the user), even if the visual evidence is new.
        """
        assessment = await self.perceive(image)
        assessment.proposed_label = label
        assessment.proposed_context = context
        assessment.epistemic_profile.label_provenance = 1.0
        return assessment
