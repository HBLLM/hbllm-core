from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingQuality:
    """Multi-dimensional grounding quality assessment for latent concepts and entities.

    Avoids collapsing distinct epistemic dimensions into a single scalar confidence.
    Permits modality-tailored lifecycle transitions (e.g., visual shapes need stability,
    physical tools need interaction support).
    """

    perceptual_stability: float
    temporal_persistence: float
    relational_consistency: float
    cross_modal_support: float
    predictive_utility: float
    interaction_support: float
    boundary_stability: float

    def __post_init__(self) -> None:
        dims = {
            "perceptual_stability": self.perceptual_stability,
            "temporal_persistence": self.temporal_persistence,
            "relational_consistency": self.relational_consistency,
            "cross_modal_support": self.cross_modal_support,
            "predictive_utility": self.predictive_utility,
            "interaction_support": self.interaction_support,
            "boundary_stability": self.boundary_stability,
        }
        for name, val in dims.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"GroundingQuality dimension {name} must be in [0.0, 1.0], got {val}"
                )

    @classmethod
    def zero(cls) -> GroundingQuality:
        """Returns a baseline zero-quality assessment."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def balanced(cls, score: float) -> GroundingQuality:
        """Returns a uniform grounding quality across all dimensions."""
        return cls(score, score, score, score, score, score, score)
