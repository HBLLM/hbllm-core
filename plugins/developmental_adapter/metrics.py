"""Developmental Metrics Registry and Statistical Aggregator.

Computes distributions, medians, 95% Confidence Intervals, and belief transition
event summaries across cohorts for Milestone A23.
"""

from __future__ import annotations

import math
import statistics
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
    ci_95_n_tau: tuple[float, float]
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

            n_taus = [r.interventions_to_discovery for r in records]
            median_n_tau = float(statistics.median(n_taus)) if n_taus else 0.0
            mean_n_tau = float(statistics.mean(n_taus)) if n_taus else 0.0

            # 95% Confidence Interval (Normal approx / t-distribution or bootstrap)
            if n > 1:
                stdev = statistics.stdev(n_taus)
                se = stdev / math.sqrt(n)
                ci_low = max(1.0, mean_n_tau - 1.96 * se)
                ci_high = mean_n_tau + 1.96 * se
            else:
                ci_low = mean_n_tau
                ci_high = mean_n_tau

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
                ci_95_n_tau=(round(ci_low, 2), round(ci_high, 2)),
                mean_wasted_interventions=round(mean_wasted, 2),
                mean_false_hypotheses=round(mean_false_hyps, 2),
                level1_train_accuracy=round(mean_l1, 4),
                level2_unseen_entities_accuracy=round(mean_l2, 4),
                level3_unseen_world_accuracy=round(mean_l3, 4),
                mean_brier_score=round(mean_brier, 4),
                all_n_tau=n_taus,
            )

        return summaries

    def format_comparison_table(self) -> str:
        """Format an ASCII comparison table across all five cohorts."""
        summaries = self.aggregate_by_cohort()
        lines = [
            "+------------------------------------+------------+--------------+------------+------------+------------+-------------+",
            "| Cohort                             | Causal Disc| Median N_tau | 95% CI     | Wasted Int | L2 Gen Acc | L3 World Acc|",
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
