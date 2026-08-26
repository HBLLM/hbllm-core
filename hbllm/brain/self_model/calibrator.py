"""Epistemic Calibration and Multi-Signal Confidence Scorer for A21.

Computes Brier Score and Expected Calibration Error (ECE) across confidence bins.
Enforces that calibration measures alignment between confidence and reality,
independent of task competence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CalibrationBin:
    """A confidence bin for computing Expected Calibration Error (ECE)."""

    bin_lower: float
    bin_upper: float
    predictions: list[float] = field(default_factory=list)
    actuals: list[bool] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.predictions)

    @property
    def avg_confidence(self) -> float:
        return sum(self.predictions) / self.count if self.count > 0 else 0.0

    @property
    def accuracy(self) -> float:
        if self.count == 0:
            return 0.0
        return sum(1.0 for a in self.actuals if a) / self.count

    @property
    def calibration_error(self) -> float:
        return abs(self.accuracy - self.avg_confidence)


@dataclass
class CalibrationReport:
    """Detailed calibration analysis across empirical attempt history."""

    domain: str
    sample_size: int
    brier_score: float
    expected_calibration_error: float  # ECE
    overconfidence_count: int
    underconfidence_count: int
    is_well_calibrated: bool


class EpistemicCalibrator:
    """Computes statistical calibration metrics (Brier, ECE) over empirical prediction records."""

    def __init__(self, num_bins: int = 5) -> None:
        self.num_bins = num_bins

    def compute_brier_score(self, predictions: list[float], actuals: list[bool]) -> float:
        """Compute mean squared error: (1/N) * sum((f_i - o_i)^2)."""
        if not predictions or len(predictions) != len(actuals):
            return 0.25
        n = len(predictions)
        sq_errs = [(predictions[i] - (1.0 if actuals[i] else 0.0)) ** 2 for i in range(n)]
        return round(sum(sq_errs) / float(n), 4)

    def compute_expected_calibration_error(
        self,
        predictions: list[float],
        actuals: list[bool],
    ) -> tuple[float, list[CalibrationBin]]:
        """Compute Expected Calibration Error (ECE) across confidence bins."""
        if not predictions or len(predictions) != len(actuals):
            return 0.0, []

        bin_width = 1.0 / self.num_bins
        bins = [
            CalibrationBin(bin_lower=i * bin_width, bin_upper=(i + 1) * bin_width)
            for i in range(self.num_bins)
        ]

        for pred, act in zip(predictions, actuals):
            # Clamp into appropriate bin
            idx = min(int(pred / bin_width), self.num_bins - 1)
            bins[idx].predictions.append(pred)
            bins[idx].actuals.append(act)

        total_n = len(predictions)
        ece = 0.0
        for b in bins:
            if b.count > 0:
                ece += (b.count / float(total_n)) * b.calibration_error

        return round(ece, 4), bins

    def evaluate_calibration(
        self,
        domain: str,
        predictions: list[float],
        actuals: list[bool],
    ) -> CalibrationReport:
        """Produce a comprehensive calibration report for a domain."""
        n = len(predictions)
        if n == 0:
            return CalibrationReport(
                domain=domain,
                sample_size=0,
                brier_score=0.25,
                expected_calibration_error=0.0,
                overconfidence_count=0,
                underconfidence_count=0,
                is_well_calibrated=True,
            )

        brier = self.compute_brier_score(predictions, actuals)
        ece, bins = self.compute_expected_calibration_error(predictions, actuals)

        overconfident = sum(1 for p, a in zip(predictions, actuals) if p >= 0.75 and not a)
        underconfident = sum(1 for p, a in zip(predictions, actuals) if p <= 0.35 and a)

        # Well-calibrated if ECE <= 0.18 and Brier <= 0.26 (covers unskewed 50/50 base rates)
        is_calibrated = ece <= 0.18 and brier <= 0.26

        return CalibrationReport(
            domain=domain,
            sample_size=n,
            brier_score=brier,
            expected_calibration_error=ece,
            overconfidence_count=overconfident,
            underconfidence_count=underconfident,
            is_well_calibrated=is_calibrated,
        )
