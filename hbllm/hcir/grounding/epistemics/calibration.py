from __future__ import annotations

import math
from dataclasses import dataclass

from hbllm.hcir.grounding.contracts.enums import EvidenceRelation
from hbllm.hcir.grounding.contracts.evidence import EvidenceRecord


@dataclass(frozen=True)
class PlattProfile:
    """Platt scaling parameters: P(E|H) = 1 / (1 + exp(-(slope * s + intercept)))."""

    slope: float
    intercept: float


class EvidenceCalibrator:
    """Transforms raw detector confidence scores into calibrated likelihoods P(E|H).

    Prevents raw uncalibrated softmax scores from masquerading as true probabilities.
    Maintains model-specific calibration curves based on empirical reliability.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, PlattProfile] = {
            # Standard default profile (mild regression toward mean)
            "default": PlattProfile(slope=4.0, intercept=-2.0),
            # Overconfident neural detectors (strong compression of extreme scores)
            "overconfident_vision": PlattProfile(slope=3.0, intercept=-1.8),
            # Conservative acoustic classifiers
            "conservative_audio": PlattProfile(slope=5.0, intercept=-2.2),
        }

    def register_profile(self, model_name: str, slope: float, intercept: float) -> None:
        """Registers a calibration profile for a specific perception model."""
        self._profiles[model_name] = PlattProfile(slope=slope, intercept=intercept)

    def get_profile(self, model_name: str) -> PlattProfile:
        return self._profiles.get(model_name, self._profiles["default"])

    def calibrate(self, raw_score: float, model_name: str = "default") -> float:
        """Applies Platt scaling to map a raw score [0.0, 1.0] to a calibrated likelihood."""
        clamped_score = max(0.0, min(1.0, raw_score))
        profile = self.get_profile(model_name)

        logit = profile.slope * clamped_score + profile.intercept
        # Sigmoid with numerical stability clipping
        clipped_logit = max(-15.0, min(15.0, logit))
        calibrated = 1.0 / (1.0 + math.exp(-clipped_logit))

        # Clamp to [0.001, 0.999] to prevent zero or infinite odds in Bayesian updates
        return max(0.001, min(0.999, calibrated))

    def calibrate_record(self, record: EvidenceRecord) -> EvidenceRecord:
        """Returns an EvidenceRecord with calibrated posterior and likelihood ratio."""
        calibrated_p = self.calibrate(record.perceptual_score, model_name=record.model_name)
        profile_name = record.model_name if record.model_name in self._profiles else "default"

        clamped_p = max(0.001, min(0.999, calibrated_p))
        base_lr = clamped_p / (1.0 - clamped_p)
        if record.relation == EvidenceRelation.REFUTES:
            lr = 1.0 / base_lr
        elif record.relation == EvidenceRelation.NEUTRAL:
            lr = 1.0
        else:
            lr = base_lr

        return EvidenceRecord(
            evidence_id=record.evidence_id,
            observation_id=record.observation_id,
            source_channel_id=record.source_channel_id,
            modality=record.modality,
            model_name=record.model_name,
            model_version=record.model_version,
            dependency_group=record.dependency_group,
            perceptual_score=record.perceptual_score,
            calibrated_likelihood=calibrated_p,
            calibrated_posterior=calibrated_p,
            likelihood_ratio=lr,
            evidence_strength=record.evidence_strength,
            relation=record.relation,
            calibration_profile=profile_name,
            provenance_hash=record.provenance_hash,
            timestamp=record.timestamp,
        )
