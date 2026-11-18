from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QualityMeasurement:
    """Explicit measurement record for an epistemic grounding dimension.

    Tracks score value, sample size, confidence interval, and whether the
    dimension has been actively tested vs remains untested.
    Prevents untested affordances from being treated as demonstrated failures.
    """

    value: float | None = None
    sample_count: int = 0
    confidence_interval: tuple[float, float] | None = None
    measurement_status: str = "UNTESTED"  # "UNTESTED" | "MEASURED" | "ESTIMATED"

    @property
    def is_tested(self) -> bool:
        return self.measurement_status == "MEASURED" and self.sample_count > 0

    @classmethod
    def untested(cls) -> QualityMeasurement:
        """Returns an explicitly untested dimension assessment."""
        return cls(
            value=None,
            sample_count=0,
            confidence_interval=None,
            measurement_status="UNTESTED",
        )

    @classmethod
    def measured(
        cls,
        value: float,
        sample_count: int = 1,
        confidence_interval: tuple[float, float] | None = None,
    ) -> QualityMeasurement:
        """Returns an empirically measured dimension assessment."""
        return cls(
            value=value,
            sample_count=sample_count,
            confidence_interval=confidence_interval,
            measurement_status="MEASURED",
        )


@dataclass(frozen=True)
class GroundingQuality:
    """Multi-dimensional grounding quality assessment for latent concepts and entities.

    Avoids collapsing distinct epistemic dimensions into a single scalar confidence.
    Permits modality-tailored lifecycle transitions (e.g., visual shapes need stability,
    physical tools need interaction support).
    Includes rich QualityMeasurement profiles per dimension.
    """

    perceptual_stability: float
    temporal_persistence: float
    relational_consistency: float
    cross_modal_support: float
    predictive_utility: float  # Accuracy delta over base rate (can be in [-1.0, 1.0])
    interaction_support: float
    boundary_stability: float
    measurements: dict[str, QualityMeasurement] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dims_standard = {
            "perceptual_stability": self.perceptual_stability,
            "temporal_persistence": self.temporal_persistence,
            "relational_consistency": self.relational_consistency,
            "cross_modal_support": self.cross_modal_support,
            "interaction_support": self.interaction_support,
            "boundary_stability": self.boundary_stability,
        }
        for name, val in dims_standard.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"GroundingQuality dimension {name} must be in [0.0, 1.0], got {val}"
                )

        # predictive_utility is an accuracy delta over base rate; can be negative
        if not (-1.0 <= self.predictive_utility <= 1.0):
            raise ValueError(
                f"predictive_utility must be in delta range [-1.0, 1.0], got {self.predictive_utility}"
            )

        # Populate measurements dictionary if empty
        if not self.measurements:
            meas = {dim: QualityMeasurement.measured(val) for dim, val in dims_standard.items()}
            meas["predictive_utility"] = QualityMeasurement.measured(self.predictive_utility)
            object.__setattr__(self, "measurements", meas)

    def get_measurement(self, dimension_name: str) -> QualityMeasurement:
        """Retrieves structured measurement metadata for a given dimension."""
        return self.measurements.get(dimension_name, QualityMeasurement.untested())

    @classmethod
    def zero(cls) -> GroundingQuality:
        """Returns a baseline zero-quality assessment."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def balanced(cls, score: float) -> GroundingQuality:
        """Returns a uniform grounding quality across all dimensions."""
        return cls(score, score, score, score, score, score, score)

    @classmethod
    def with_untested(
        cls,
        perceptual_stability: float = 0.5,
        temporal_persistence: float = 0.5,
        relational_consistency: float = 0.5,
        cross_modal_support: float = 0.0,
        predictive_utility: float = 0.0,
        interaction_support: float = 0.0,
        boundary_stability: float = 0.5,
        untested_dimensions: set[str] | None = None,
    ) -> GroundingQuality:
        """Constructs GroundingQuality with explicit untested dimensions."""
        untested = untested_dimensions or set()
        meas: dict[str, QualityMeasurement] = {}
        all_vals = {
            "perceptual_stability": perceptual_stability,
            "temporal_persistence": temporal_persistence,
            "relational_consistency": relational_consistency,
            "cross_modal_support": cross_modal_support,
            "predictive_utility": predictive_utility,
            "interaction_support": interaction_support,
            "boundary_stability": boundary_stability,
        }
        for k, v in all_vals.items():
            if k in untested:
                meas[k] = QualityMeasurement.untested()
            else:
                meas[k] = QualityMeasurement.measured(v)

        return cls(
            perceptual_stability=perceptual_stability,
            temporal_persistence=temporal_persistence,
            relational_consistency=relational_consistency,
            cross_modal_support=cross_modal_support,
            predictive_utility=predictive_utility,
            interaction_support=interaction_support,
            boundary_stability=boundary_stability,
            measurements=meas,
        )
