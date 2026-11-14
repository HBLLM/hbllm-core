"""Statistical Aggregation and Confidence Intervals for the Scientific Comparison.

Aggregates multi-seed experimental runs into mean, median, standard deviation,
and 95% confidence intervals to ensure statistically sound comparisons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MetricSummary:
    """Summary statistics for a single metric across experimental seeds."""

    name: str
    mean: float
    median: float
    std: float
    ci_95_low: float
    ci_95_high: float
    n_samples: int


@dataclass
class ProportionSummary:
    """Summary statistics for a binomial proportion (e.g. success rate) with Wilson score CI."""

    name: str
    successes: int
    n_samples: int
    rate: float
    ci_95_low: float
    ci_95_high: float


class ExperimentStatistics:
    """Aggregates values across random seeds into formal statistical summaries."""

    @staticmethod
    def wilson_score_interval(
        successes: int,
        n_samples: int,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Calculate the Wilson score confidence interval for a binomial proportion.

        Provides superior coverage over standard normal approximation,
        especially when proportion is near 0.0 or 1.0, or n is small.
        """
        if n_samples == 0:
            return (0.0, 0.0)

        # Standard normal quantiles: 0.90 -> 1.645, 0.95 -> 1.96, 0.99 -> 2.576
        z = 1.95996 if abs(confidence - 0.95) < 1e-3 else 1.96
        p_hat = float(successes) / float(n_samples)
        denominator = 1.0 + (z**2) / float(n_samples)
        center_adj = p_hat + (z**2) / (2.0 * float(n_samples))
        spread = z * math.sqrt(
            (p_hat * (1.0 - p_hat) / float(n_samples)) + ((z**2) / (4.0 * (float(n_samples) ** 2)))
        )

        ci_low = max(0.0, (center_adj - spread) / denominator)
        ci_high = min(1.0, (center_adj + spread) / denominator)
        return (round(ci_low, 4), round(ci_high, 4))

    @classmethod
    def summarize_proportion(
        cls,
        metric_name: str,
        successes: int,
        n_samples: int,
        confidence: float = 0.95,
    ) -> ProportionSummary:
        """Calculate empirical rate and 95% Wilson score CI for a binomial outcome."""
        rate = float(successes) / float(n_samples) if n_samples > 0 else 0.0
        ci_low, ci_high = cls.wilson_score_interval(successes, n_samples, confidence)
        return ProportionSummary(
            name=metric_name,
            successes=successes,
            n_samples=n_samples,
            rate=round(rate, 4),
            ci_95_low=ci_low,
            ci_95_high=ci_high,
        )

    @staticmethod
    def summarize(metric_name: str, values: list[float]) -> MetricSummary:
        """Calculate mean, median, std, and 95% CI for a collection of seed measurements."""
        if not values:
            return MetricSummary(metric_name, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        n = len(values)
        sorted_vals = sorted(values)
        mean_val = sum(values) / float(n)

        # Median
        if n % 2 == 1:
            median_val = sorted_vals[n // 2]
        else:
            median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

        # Sample Standard Deviation
        if n > 1:
            variance = sum((x - mean_val) ** 2 for x in values) / float(n - 1)
            std_val = math.sqrt(variance)
        else:
            std_val = 0.0

        # 95% Confidence Interval (z = 1.96)
        margin = (1.96 * (std_val / math.sqrt(n))) if n > 1 else 0.0

        return MetricSummary(
            name=metric_name,
            mean=round(mean_val, 4),
            median=round(median_val, 4),
            std=round(std_val, 4),
            ci_95_low=round(mean_val - margin, 4),
            ci_95_high=round(mean_val + margin, 4),
            n_samples=n,
        )
