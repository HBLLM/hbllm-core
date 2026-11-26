"""HCIR Autonomy Tracker & Graduation Telemetry.

Tracks Pure-HCIR parsing, planning, and realization autonomy rates
across dialogue and interaction turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AutonomyTracker:
    """Tracks Pure-HCIR vs LLM-scaffold autonomy rates across dialogue turns."""

    parse_attempts: int = 0
    parse_hcir_success: int = 0
    plan_attempts: int = 0
    plan_hcir_success: int = 0
    realize_attempts: int = 0
    realize_hcir_success: int = 0
    shadow_invocations: int = 0
    distilled_lessons: int = 0
    distilled_entities: int = 0
    distilled_lexicon_entries: int = 0
    rejected_lessons: int = 0
    retirement_threshold: float = 0.98
    min_turns_for_retirement: int = 50

    @property
    def hcir_parse_autonomy_rate(self) -> float:
        return self.parse_hcir_success / self.parse_attempts if self.parse_attempts else 1.0

    @property
    def hcir_plan_autonomy_rate(self) -> float:
        return self.plan_hcir_success / self.plan_attempts if self.plan_attempts else 1.0

    @property
    def hcir_realize_autonomy_rate(self) -> float:
        return self.realize_hcir_success / self.realize_attempts if self.realize_attempts else 1.0

    @property
    def overall_autonomy_rate(self) -> float:
        total = self.parse_attempts + self.plan_attempts + self.realize_attempts
        if total == 0:
            return 1.0
        ok = self.parse_hcir_success + self.plan_hcir_success + self.realize_hcir_success
        return ok / total

    @property
    def can_retire_llm(self) -> bool:
        return (
            self.parse_attempts >= self.min_turns_for_retirement
            and self.hcir_parse_autonomy_rate >= self.retirement_threshold
            and self.hcir_realize_autonomy_rate >= self.retirement_threshold
        )

    def record_parse(self, hcir_ok: bool) -> None:
        self.parse_attempts += 1
        if hcir_ok:
            self.parse_hcir_success += 1

    def record_plan(self, hcir_ok: bool) -> None:
        self.plan_attempts += 1
        if hcir_ok:
            self.plan_hcir_success += 1

    def record_realize(self, hcir_ok: bool) -> None:
        self.realize_attempts += 1
        if hcir_ok:
            self.realize_hcir_success += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "parse_attempts": self.parse_attempts,
            "hcir_parse_autonomy_rate": round(self.hcir_parse_autonomy_rate, 4),
            "hcir_plan_autonomy_rate": round(self.hcir_plan_autonomy_rate, 4),
            "hcir_realize_autonomy_rate": round(self.hcir_realize_autonomy_rate, 4),
            "overall_autonomy_rate": round(self.overall_autonomy_rate, 4),
            "shadow_invocations": self.shadow_invocations,
            "distilled_lessons": self.distilled_lessons,
            "distilled_entities": self.distilled_entities,
            "distilled_lexicon_entries": self.distilled_lexicon_entries,
            "rejected_lessons": self.rejected_lessons,
            "can_retire_llm": self.can_retire_llm,
        }


@dataclass(frozen=True)
class CalibrationReport:
    """Empirical confidence calibration metrics over binned predictions."""

    domain: str
    expected_calibration_error: float
    brier_score: float
    overconfidence_count: int
    underconfidence_count: int
    sample_count: int

    @property
    def is_well_calibrated(self) -> bool:
        return self.expected_calibration_error < 0.05


class EpistemicCalibrator:
    """Computes Expected Calibration Error (ECE) and Brier score across predictions."""

    def __init__(self, num_bins: int = 10) -> None:
        self.num_bins = max(1, num_bins)

    def evaluate_calibration(
        self,
        domain: str,
        predicted_confidences: list[float],
        actual_outcomes: list[bool],
    ) -> CalibrationReport:
        n = min(len(predicted_confidences), len(actual_outcomes))
        if n == 0:
            return CalibrationReport(
                domain=domain,
                expected_calibration_error=0.0,
                brier_score=0.0,
                overconfidence_count=0,
                underconfidence_count=0,
                sample_count=0,
            )

        pairs = list(zip(predicted_confidences[:n], actual_outcomes[:n]))
        brier = round(
            sum((float(p) - (1.0 if a else 0.0)) ** 2 for p, a in pairs) / n,
            4,
        )

        ece = 0.0
        for b in range(self.num_bins):
            b_min = b / self.num_bins
            b_max = (b + 1) / self.num_bins
            if b == self.num_bins - 1:
                bin_items = [(float(p), a) for p, a in pairs if b_min <= float(p) <= b_max]
            else:
                bin_items = [(float(p), a) for p, a in pairs if b_min <= float(p) < b_max]
            if not bin_items:
                continue
            bin_size = len(bin_items)
            avg_conf = sum(p for p, _ in bin_items) / bin_size
            avg_acc = sum(1.0 if a else 0.0 for _, a in bin_items) / bin_size
            ece += (bin_size / n) * abs(avg_acc - avg_conf)

        overconfidence = sum(1 for p, a in pairs if float(p) >= 0.80 and not a)
        underconfidence = sum(1 for p, a in pairs if float(p) <= 0.20 and a)
        return CalibrationReport(
            domain=domain,
            expected_calibration_error=round(ece, 4),
            brier_score=brier,
            overconfidence_count=overconfidence,
            underconfidence_count=underconfidence,
            sample_count=n,
        )
