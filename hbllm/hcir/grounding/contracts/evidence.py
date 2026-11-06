from __future__ import annotations

import time
from dataclasses import dataclass

from hbllm.hcir.grounding.contracts.enums import ModalitySource


@dataclass(frozen=True)
class EvidenceRecord:
    """Normalized evidence produced by a perception adapter or model.

    Separates perceptual scores, calibrated likelihoods, and intrinsic sensor
    precision. Explicitly groups models sharing identical input data or encoder
    backbones into `dependency_group` to prevent artificial Bayesian inflation.
    """

    evidence_id: str
    observation_id: str
    source_channel_id: str
    modality: ModalitySource
    model_name: str
    model_version: str
    dependency_group: str
    perceptual_score: float
    calibrated_likelihood: float
    evidence_strength: float
    provenance_hash: str
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("evidence_id must be non-empty")
        if not self.observation_id:
            raise ValueError("observation_id must be non-empty")
        if not self.dependency_group:
            raise ValueError("dependency_group must be non-empty")

        # Validate score bounds [0.0, 1.0]
        for name, val in [
            ("perceptual_score", self.perceptual_score),
            ("calibrated_likelihood", self.calibrated_likelihood),
            ("evidence_strength", self.evidence_strength),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in range [0.0, 1.0], got {val}")

        if self.timestamp <= 0.0:
            object.__setattr__(self, "timestamp", time.time())
