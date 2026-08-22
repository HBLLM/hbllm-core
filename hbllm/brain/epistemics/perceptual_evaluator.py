"""Perceptual Evidence Evaluator — evaluates sensory evidence quality and reliability.

Independent of any specific candidate belief proposition:
- Assesses raw sensory signal clarity (SNR, illumination, resolution).
- Assesses model confidence and temporal stability.
- Evaluates provider provenance quality.
- Computes multidimensional uncertainty vectors and general information gain.

Architecture::

    EvidenceNode (with PerceptualEpistemicProfile & ProviderProvenance)
          │
          ▼
    PerceptualEvidenceEvaluator.evaluate(evidence)
          │
          ▼
    EvidenceAssessment (reliability, uncertainty, information_gain)
"""

from __future__ import annotations

import logging
import math
from typing import Any

from hbllm.hcir.graph import CognitiveGraph, EvidenceNode, PerceptualEvidenceNode
from hbllm.hcir.types import (
    Confidence,
    EvidenceAssessment,
    PerceptualEpistemicProfile,
    ReliabilitySource,
    UncertaintyVector,
)

logger = logging.getLogger(__name__)

# Weight multipliers for provider reputations (domain-neutral calibration)
_PROVIDER_REPUTATION_WEIGHTS: dict[str, float] = {
    "whisper": 0.90,
    "moonshine": 0.88,
    "yamnet": 0.85,
    "siglip": 0.88,
    "yolo": 0.85,
    "mock": 0.70,
}


class PerceptualEvidenceEvaluator:
    """Evaluates general evidence reliability, signal fidelity, and epistemic quality.

    Completely decoupled from specific belief claims or hypothesis propositions.
    """

    def __init__(
        self,
        graph: CognitiveGraph | None = None,
        reputation_tracker: Any | None = None,
    ) -> None:
        self._graph = graph
        self._reputation_tracker = reputation_tracker

    def evaluate(self, evidence: EvidenceNode | PerceptualEvidenceNode | Any) -> EvidenceAssessment:
        """Evaluate an EvidenceNode or PerceptualEvidenceNode and return an EvidenceAssessment.

        Args:
            evidence: The EvidenceNode or PerceptualEvidenceNode to assess.

        Returns:
            EvidenceAssessment containing reliability, uncertainty vector, and info gain.
        """
        epistemic_profile = getattr(evidence, "epistemic_profile", None)
        if epistemic_profile is None:
            # Construct a default profile from evidence strength if none attached
            strength = float(getattr(evidence, "strength", 0.8))
            epistemic_profile = PerceptualEpistemicProfile(
                sensory_clarity=strength,
                model_confidence=strength,
                temporal_stability=0.8,
            )

        # 1. Base reliability from multidimensional epistemic profile
        base_reliability = epistemic_profile.reliability

        # 2. Provenance quality weighting
        prov_quality = 0.8
        prov = getattr(evidence, "provider_provenance", None)
        if prov and isinstance(prov, dict):
            prov_name = str(prov.get("provider", "")).lower()
            prov_quality = _PROVIDER_REPUTATION_WEIGHTS.get(prov_name, 0.75)

        # 3. Combined calibrated reliability score in [0.0, 1.0]
        calibrated_reliability: Confidence = float(
            max(0.01, min(0.99, 0.6 * base_reliability + 0.4 * prov_quality))
        )

        # 4. Multi-dimensional uncertainty vector
        uncertainty = UncertaintyVector(
            confidence=calibrated_reliability,
            freshness_ms=0,
            reliability=ReliabilitySource.OBSERVED,
            volatility=float(1.0 - epistemic_profile.temporal_stability),
        )

        # 5. Shannon entropy-based information gain estimate
        # H(p) = -p*log2(p) - (1-p)*log2(1-p); Information gain = 1 - H(p)
        p = max(0.001, min(0.999, calibrated_reliability))
        entropy = -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
        info_gain = float(max(0.0, 1.0 - entropy))

        assessment = EvidenceAssessment(
            evidence_id=evidence.id,
            reliability=calibrated_reliability,
            uncertainty=uncertainty,
            epistemic_profile=epistemic_profile,
            provenance_quality=prov_quality,
            information_gain=info_gain,
        )

        logger.debug(
            "Evaluated evidence %s: reliability=%.3f, info_gain=%.3f",
            evidence.id,
            calibrated_reliability,
            info_gain,
        )

        return assessment
