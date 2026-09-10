"""Developmental Metrics Registry and Statistical Aggregator.

Computes discrete distributions, medians, IQRs, 95% Bootstrap Confidence Intervals,
and belief transition event summaries across cohorts for Milestone A23.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from dataclasses import dataclass, field

from .cohorts import CohortDiscoveryResult


@dataclass
class CohortStatisticalSummary:
    """Statistical aggregation across randomized experimental seeds for a cohort."""

    cohort_id: str
    trials_count: int
    discovery_rate: float
    median_n_tau: float
    mean_n_tau: float
    iqr_n_tau: tuple[float, float]
    ci_95_n_tau: tuple[float, float]
    discrete_n_tau_distribution: dict[int, int]
    failures_count: int
    mean_wasted_interventions: float
    mean_false_hypotheses: float
    level1_train_accuracy: float
    level2_unseen_entities_accuracy: float
    level3_unseen_world_accuracy: float
    mean_brier_score: float
    all_n_tau: list[int] = field(default_factory=list)


class DevelopmentalMetricsTracker:
    """Collects individual trial results and computes publication-grade statistics."""

    def __init__(self) -> None:
        self.trial_records: list[CohortDiscoveryResult] = []

    def record_result(self, result: CohortDiscoveryResult) -> None:
        self.trial_records.append(result)

    def aggregate_by_cohort(self) -> dict[str, CohortStatisticalSummary]:
        """Aggregate statistical distribution for each cohort."""
        cohort_groups: dict[str, list[CohortDiscoveryResult]] = {}
        for r in self.trial_records:
            cohort_groups.setdefault(r.cohort_id, []).append(r)

        summaries: dict[str, CohortStatisticalSummary] = {}

        for cid, records in cohort_groups.items():
            n = len(records)
            discoveries = sum(1 for r in records if r.identified_causal_rule)
            discovery_rate = discoveries / n if n > 0 else 0.0
            failures = n - discoveries

            n_taus = [r.interventions_to_discovery for r in records]
            median_n_tau = float(statistics.median(n_taus)) if n_taus else 0.0
            mean_n_tau = float(statistics.mean(n_taus)) if n_taus else 0.0

            # Discrete distribution counts
            dist_counts = dict(Counter(n_taus))

            # IQR calculation
            sorted_taus = sorted(n_taus)
            if len(sorted_taus) >= 4:
                q1 = statistics.median(sorted_taus[: len(sorted_taus) // 2])
                q3 = statistics.median(sorted_taus[(len(sorted_taus) + 1) // 2 :])
                iqr = (float(q1), float(q3))
            else:
                iqr = (median_n_tau, median_n_tau)

            # 95% Bootstrap Confidence Interval
            ci_low, ci_high = self._compute_bootstrap_ci(n_taus, n_resamples=1000)

            mean_wasted = statistics.mean([r.interventions_wasted for r in records])
            mean_false_hyps = statistics.mean([r.false_hypotheses_generated for r in records])
            mean_l1 = statistics.mean([r.level1_train_accuracy for r in records])
            mean_l2 = statistics.mean([r.level2_unseen_entities_accuracy for r in records])
            mean_l3 = statistics.mean([r.level3_unseen_world_accuracy for r in records])
            mean_brier = statistics.mean([r.brier_score for r in records])

            summaries[cid] = CohortStatisticalSummary(
                cohort_id=cid,
                trials_count=n,
                discovery_rate=round(discovery_rate, 4),
                median_n_tau=round(median_n_tau, 2),
                mean_n_tau=round(mean_n_tau, 2),
                iqr_n_tau=(round(iqr[0], 2), round(iqr[1], 2)),
                ci_95_n_tau=(round(ci_low, 2), round(ci_high, 2)),
                discrete_n_tau_distribution=dist_counts,
                failures_count=failures,
                mean_wasted_interventions=round(mean_wasted, 2),
                mean_false_hypotheses=round(mean_false_hyps, 2),
                level1_train_accuracy=round(mean_l1, 4),
                level2_unseen_entities_accuracy=round(mean_l2, 4),
                level3_unseen_world_accuracy=round(mean_l3, 4),
                mean_brier_score=round(mean_brier, 4),
                all_n_tau=n_taus,
            )

        return summaries

    @staticmethod
    def _compute_bootstrap_ci(
        data: list[int],
        n_resamples: int = 1000,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Compute non-parametric bootstrap confidence interval for the mean."""
        if not data:
            return 0.0, 0.0
        if len(data) == 1 or len(set(data)) == 1:
            return float(data[0]), float(data[0])

        rng = random.Random(42)
        n = len(data)
        means: list[float] = []

        for _ in range(n_resamples):
            sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
            means.append(statistics.mean(sample))

        means.sort()
        low_idx = int((alpha / 2) * n_resamples)
        high_idx = int((1 - alpha / 2) * n_resamples)
        return float(means[low_idx]), float(means[high_idx])

    def format_comparison_table(self) -> str:
        """Format an ASCII comparison table across all five cohorts."""
        summaries = self.aggregate_by_cohort()
        lines = [
            "+------------------------------------+------------+--------------+------------+------------+------------+-------------+",
            "| Cohort                             | Causal Disc| Median N_tau | BootstrapCI| Wasted Int | L2 Gen Acc | L3 World Acc|",
            "+------------------------------------+------------+--------------+------------+------------+------------+-------------+",
        ]
        for cid, s in summaries.items():
            disc_str = f"{s.discovery_rate * 100:.1f}%"
            ci_str = f"[{s.ci_95_n_tau[0]:.1f}, {s.ci_95_n_tau[1]:.1f}]"
            l2_str = f"{s.level2_unseen_entities_accuracy * 100:.1f}%"
            l3_str = f"{s.level3_unseen_world_accuracy * 100:.1f}%"
            name_abbr = cid.replace("Cohort_", "")
            lines.append(
                f"| {name_abbr:<34} | {disc_str:<10} | {s.median_n_tau:<12} | {ci_str:<10} | {s.mean_wasted_interventions:<10} | {l2_str:<10} | {l3_str:<11} |"
            )
        lines.append(
            "+------------------------------------+------------+--------------+------------+------------+------------+-------------+"
        )
        return "\n".join(lines)

    def format_discrete_distribution_table(self) -> str:
        """Format the complete discrete empirical distribution breakdown."""
        summaries = self.aggregate_by_cohort()
        blocks = [
            "=====================================================================================",
            "COMPLETE DISCRETE EMPIRICAL DISTRIBUTIONS (N_tau Intervention Distribution)",
            "=====================================================================================",
        ]

        for cid, s in summaries.items():
            name = cid.replace("Cohort_", "")
            blocks.append(f"\nCohort: {name} (N = {s.trials_count} trials)")
            blocks.append("  N_tau Discrete Breakdown:")
            for tau_val in sorted(s.discrete_n_tau_distribution.keys()):
                count = s.discrete_n_tau_distribution[tau_val]
                pct = (count / s.trials_count) * 100.0
                tag = " (Failures/MaxBudget)" if tau_val >= 20 else ""
                blocks.append(
                    f"    {tau_val} interventions{tag:<23}: {count:>2} trials ({pct:>5.1f}%)"
                )
            blocks.append(
                f"  Summary Statistics: Median={s.median_n_tau} | IQR=[{s.iqr_n_tau[0]}, {s.iqr_n_tau[1]}] | 95% Bootstrap CI=[{s.ci_95_n_tau[0]}, {s.ci_95_n_tau[1]}]"
            )
            blocks.append(
                f"  Efficiency: Wasted Probes={s.mean_wasted_interventions} | False Hypotheses={s.mean_false_hypotheses}"
            )
            blocks.append(
                f"  Transfer: L2 Generalization={s.level2_unseen_entities_accuracy * 100:.1f}% | L3 World={s.level3_unseen_world_accuracy * 100:.1f}%\n"
            )

        return "\n".join(blocks)
