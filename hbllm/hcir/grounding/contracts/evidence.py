from __future__ import annotations

import time
from dataclasses import dataclass

from hbllm.hcir.grounding.contracts.enums import EvidenceRelation, ModalitySource


@dataclass(frozen=True)
class EvidenceRecord:
    """Normalized evidence produced by a perception adapter or model.

    Separates perceptual scores, calibrated likelihoods / posteriors, and intrinsic sensor
    precision. Explicitly groups models sharing identical input data or encoder
    backbones into `dependency_group` to prevent artificial Bayesian inflation.
    Tracks explicit polarity via `EvidenceRelation` (SUPPORTS, REFUTES, NEUTRAL).
    """

    evidence_id: str
    observation_id: str
    source_channel_id: str
    modality: ModalitySource
    model_name: str
    model_version: str
    dependency_group: str
    perceptual_score: float
    calibrated_likelihood: float = 0.5
    evidence_strength: float = 1.0
    provenance_hash: str = ""
    timestamp: float = 0.0
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS
    calibrated_posterior: float | None = None
    likelihood_ratio: float | None = None
    calibration_profile: str | None = None

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("evidence_id must be non-empty")
        if not self.observation_id:
            raise ValueError("observation_id must be non-empty")
        if not self.dependency_group:
            raise ValueError("dependency_group must be non-empty")

        # Synchronize calibrated_posterior and calibrated_likelihood
        eff_posterior = (
            self.calibrated_posterior
            if self.calibrated_posterior is not None
            else self.calibrated_likelihood
        )
        object.__setattr__(self, "calibrated_posterior", eff_posterior)
        object.__setattr__(self, "calibrated_likelihood", eff_posterior)

        # Validate score bounds [0.0, 1.0]
        for name, val in [
            ("perceptual_score", self.perceptual_score),
            ("calibrated_likelihood", self.calibrated_likelihood),
            ("evidence_strength", self.evidence_strength),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in range [0.0, 1.0], got {val}")

        # Compute likelihood ratio if not provided
        if self.likelihood_ratio is None:
            p_clamped = max(0.001, min(0.999, eff_posterior))
            base_lr = p_clamped / (1.0 - p_clamped)
            if self.relation == EvidenceRelation.REFUTES:
                eff_lr = 1.0 / base_lr
            elif self.relation == EvidenceRelation.NEUTRAL:
                eff_lr = 1.0
            else:
                eff_lr = base_lr
            object.__setattr__(self, "likelihood_ratio", eff_lr)

        if self.timestamp <= 0.0:
            object.__setattr__(self, "timestamp", time.time())
