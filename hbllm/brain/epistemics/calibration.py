"""Epistemic Calibration Engine — meta-epistemic self-assessment.

Answers: "How good am I at knowing things?"

This is different from ``self_model/confidence_estimator.py`` which
estimates per-task confidence.  This is *system-wide epistemic
calibration* across all domains and reasoning history.

Key questions this engine answers::

    Do I overestimate confidence?     → Calibration curve analysis
    Am I biased toward evidence types? → Evidence type → accuracy correlation
    Do I generate too many hypotheses? → Hypothesis survival rate
    Do I falsify enough?              → Falsification rate vs confirmation
    Which domains am I worst at?      → Per-domain calibration

Architecture::

    EpistemicCalibrationEngine
        ├── Uses EpistemicMemory for historical data
        ├── calibrate()                → CalibrationReport
        ├── compute_calibration_curve() → [(predicted, actual), ...]
        ├── detect_epistemic_biases()  → ["overconfidence", ...]
        └── recommend_strategy_adjustment() → ResearchStrategyType | None

Usage::

    calibrator = EpistemicCalibrationEngine(memory=memory)
    report = await calibrator.calibrate()
    if report.overconfidence_bias > 0.1:
        print(f"System is overconfident by {report.overconfidence_bias:.2f}")
    for rec in report.recommendations:
        print(f"  → {rec}")
"""

from __future__ import annotations

import logging

from hbllm.brain.epistemics.epistemic_memory import EpistemicMemory
from hbllm.brain.epistemics.interfaces import CalibrationReport

logger = logging.getLogger(__name__)

#: Minimum data points required for meaningful calibration.
_MIN_CALIBRATION_POINTS = 5


class EpistemicCalibrationEngine:
    """Meta-epistemic self-calibration engine.

    Implements the ``IEpistemicCalibrator`` protocol.

    Uses ``EpistemicMemory`` to compute calibration metrics over
    the full history of predictions, hypotheses, and beliefs.

    The engine is domain-neutral — it analyzes calibration quality
    across any domain, identifying where the system is well-calibrated
    and where it's systematically biased.
    """

    def __init__(
        self,
        memory: EpistemicMemory,
    ) -> None:
        """Initialize the calibration engine.

        Args:
            memory: The EpistemicMemory to analyze.
        """
        self._memory = memory

    async def calibrate(self) -> CalibrationReport:
        """Run full calibration analysis.

        Produces a ``CalibrationReport`` that describes the system's
        epistemic performance across all dimensions.

        Returns:
            A comprehensive CalibrationReport.
        """
        # Gather data
        prediction_accuracy = await self._memory.get_prediction_accuracy()
        survival_rate = await self._memory.get_hypothesis_survival_rate()
        falsification_rate = await self._memory.get_falsification_rate()
        total_counts = await self._memory.get_total_counts()

        # Compute calibration curve
        calibration_curve = await self.compute_calibration_curve()

        # Compute overall calibration error
        overall_calibration = self._compute_calibration_error(calibration_curve)

        # Compute overconfidence bias
        overconfidence_bias = self._compute_overconfidence_bias(calibration_curve)

        # Domain calibration
        domain_calibration = await self._compute_domain_calibration()

        # Evidence bias analysis
        evidence_bias = await self._compute_evidence_bias()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_calibration=overall_calibration,
            overconfidence_bias=overconfidence_bias,
            survival_rate=survival_rate,
            falsification_rate=falsification_rate,
            prediction_accuracy=prediction_accuracy,
        )

        report = CalibrationReport(
            overall_calibration=overall_calibration,
            overconfidence_bias=overconfidence_bias,
            hypothesis_survival_rate=survival_rate,
            falsification_rate=falsification_rate,
            prediction_accuracy=prediction_accuracy,
            total_predictions=total_counts.get("prediction_history", 0),
            total_hypotheses=total_counts.get("hypothesis_history", 0),
            domain_calibration=domain_calibration,
            evidence_bias=evidence_bias,
            recommendations=recommendations,
        )

        logger.info(
            "Calibration complete: cal=%.3f, bias=%.3f, pred_acc=%.3f, "
            "hyp_survival=%.3f, falsification=%.3f, %d recs",
            overall_calibration,
            overconfidence_bias,
            prediction_accuracy,
            survival_rate,
            falsification_rate,
            len(recommendations),
        )

        return report

    async def compute_calibration_curve(
        self,
        n_bins: int = 10,
    ) -> list[tuple[float, float, int]]:
        """Compute the calibration curve.

        Bins predictions by predicted confidence and compares with
        actual accuracy in each bin.

        A perfectly calibrated system has predicted confidence ≈ actual
        accuracy in every bin.

        Args:
            n_bins: Number of confidence bins.

        Returns:
            List of (bin_center, actual_accuracy, count) tuples.
        """
        calib_data = await self._memory.get_calibration_data()

        if len(calib_data) < _MIN_CALIBRATION_POINTS:
            return []

        # Initialize bins
        bin_width = 1.0 / n_bins
        bins: list[tuple[float, list[int]]] = [
            (i * bin_width + bin_width / 2, []) for i in range(n_bins)
        ]

        # Assign data points to bins
        for point in calib_data:
            predicted = point.get("predicted_confidence", 0.5)
            actual = point.get("actual_outcome", 0)
            bin_idx = min(n_bins - 1, int(predicted * n_bins))
            bins[bin_idx][1].append(actual)

        # Compute accuracy per bin
        curve: list[tuple[float, float, int]] = []
        for bin_center, outcomes in bins:
            count = len(outcomes)
            if count > 0:
                accuracy = sum(outcomes) / count
                curve.append((bin_center, accuracy, count))

        return curve

    async def detect_epistemic_biases(self) -> list[str]:
        """Identify systematic biases in the system's reasoning.

        Returns:
            List of bias descriptions.
        """
        biases: list[str] = []

        # Check overconfidence
        curve = await self.compute_calibration_curve()
        overconfidence = self._compute_overconfidence_bias(curve)
        if overconfidence > 0.1:
            biases.append(
                f"Overconfidence bias: predicted confidence exceeds actual "
                f"accuracy by {overconfidence:.2f} on average"
            )
        elif overconfidence < -0.1:
            biases.append(
                f"Underconfidence bias: actual accuracy exceeds predicted "
                f"confidence by {abs(overconfidence):.2f} on average"
            )

        # Check hypothesis generation vs testing ratio
        survival_rate = await self._memory.get_hypothesis_survival_rate()
        falsification_rate = await self._memory.get_falsification_rate()

        if survival_rate > 0.8:
            biases.append(
                f"Confirmation bias: {survival_rate:.0%} hypothesis survival rate "
                f"suggests insufficient falsification testing"
            )
        elif survival_rate < 0.1:
            biases.append(
                f"Overly strict hypothesis filtering: only {survival_rate:.0%} "
                f"of hypotheses survive. Consider relaxing validation criteria."
            )

        if falsification_rate < 0.1:
            biases.append(
                f"Low falsification rate ({falsification_rate:.0%}): the system "
                f"rarely actively disproves hypotheses. Consider COUNTEREXAMPLE "
                f"strategy."
            )

        # Check domain-specific biases
        domain_cal = await self._compute_domain_calibration()
        for domain, error in domain_cal.items():
            if error > 0.2:
                biases.append(
                    f"Domain '{domain}' is poorly calibrated (error={error:.2f}). "
                    f"Consider more cautious confidence in this domain."
                )

        return biases

    async def recommend_strategy_adjustment(self) -> str | None:
        """Recommend a ResearchStrategyType change based on calibration.

        Returns:
            A ResearchStrategyType value string, or None if no change needed.
        """
        report = await self.calibrate()

        # If overconfident → switch to counterexample search
        if report.overconfidence_bias > 0.15:
            return "counterexample_search"

        # If too few falsifications → switch to verification
        if report.falsification_rate < 0.1 and report.total_hypotheses > 5:
            return "verification"

        # If prediction accuracy is low → switch to systematic
        if report.prediction_accuracy < 0.4 and report.total_predictions > 10:
            return "systematic"

        # If survival rate is very high → switch to counterexample
        if report.hypothesis_survival_rate > 0.9 and report.total_hypotheses > 5:
            return "counterexample_search"

        # If well-calibrated and productive → continue
        return None

    # ── Internal Methods ───────────────────────────────────────────────

    def _compute_calibration_error(
        self,
        curve: list[tuple[float, float, int]],
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        ECE = Σ (|accuracy_i - confidence_i| × n_i) / N

        Returns:
            ECE score [0.0=perfect, 1.0=maximally miscalibrated].
        """
        if not curve:
            return 0.0

        total_count = sum(count for _, _, count in curve)
        if total_count == 0:
            return 0.0

        ece = (
            sum(abs(accuracy - confidence) * count for confidence, accuracy, count in curve)
            / total_count
        )

        return min(1.0, ece)

    def _compute_overconfidence_bias(
        self,
        curve: list[tuple[float, float, int]],
    ) -> float:
        """Compute the directional calibration bias.

        Positive = overconfident (predicted > actual).
        Negative = underconfident (actual > predicted).

        Returns:
            Bias score [-1.0, 1.0].
        """
        if not curve:
            return 0.0

        total_count = sum(count for _, _, count in curve)
        if total_count == 0:
            return 0.0

        bias = (
            sum((confidence - accuracy) * count for confidence, accuracy, count in curve)
            / total_count
        )

        return max(-1.0, min(1.0, bias))

    async def _compute_domain_calibration(self) -> dict[str, float]:
        """Compute calibration error per domain."""
        calib_data = await self._memory.get_calibration_data()

        # Group by domain
        domain_data: dict[str, list[tuple[float, int]]] = {}
        for point in calib_data:
            domain = point.get("domain", "")
            if not domain:
                continue
            predicted = point.get("predicted_confidence", 0.5)
            actual = point.get("actual_outcome", 0)
            domain_data.setdefault(domain, []).append((predicted, actual))

        # Compute per-domain error
        result: dict[str, float] = {}
        for domain, points in domain_data.items():
            if len(points) < 3:
                continue
            error = sum(abs(p - a) for p, a in points) / len(points)
            result[domain] = error

        return result

    async def _compute_evidence_bias(self) -> dict[str, float]:
        """Analyze evidence type biases.

        Checks if certain evidence types lead to better/worse predictions.
        """
        # This requires cross-referencing evidence types with prediction
        # outcomes. For now, return empty — will be enriched when evidence
        # types are tracked in prediction_history.
        return {}

    def _generate_recommendations(
        self,
        overall_calibration: float,
        overconfidence_bias: float,
        survival_rate: float,
        falsification_rate: float,
        prediction_accuracy: float,
    ) -> list[str]:
        """Generate human-readable recommendations."""
        recs: list[str] = []

        if overall_calibration > 0.2:
            recs.append(
                "Calibration is poor. Consider using more conservative confidence estimates."
            )

        if overconfidence_bias > 0.1:
            recs.append(
                "System is overconfident. Reduce initial confidence or "
                "increase falsification testing."
            )
        elif overconfidence_bias < -0.1:
            recs.append(
                "System is underconfident. Evidence quality may be "
                "undervalued — check evidence evaluation weights."
            )

        if survival_rate > 0.8:
            recs.append(
                "Too many hypotheses survive. The filter may be too "
                "permissive, or falsification is insufficient."
            )

        if falsification_rate < 0.1:
            recs.append(
                "Falsification rate is very low. Switch to "
                "COUNTEREXAMPLE_SEARCH strategy periodically."
            )

        if prediction_accuracy < 0.4:
            recs.append(
                "Prediction accuracy is low. Hypotheses may be poorly "
                "formed or evidence is insufficient."
            )
        elif prediction_accuracy > 0.9:
            recs.append(
                "Prediction accuracy is suspiciously high. May indicate "
                "trivial predictions or confirmation bias."
            )

        if not recs:
            recs.append("Epistemic performance is within acceptable ranges.")

        return recs
