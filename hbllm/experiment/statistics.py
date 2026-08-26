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


class ExperimentStatistics:
    """Aggregates values across random seeds into formal statistical summaries."""

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
