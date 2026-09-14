"""Developmental Metrics Registry and Statistical Aggregator.

Computes discrete distributions, medians, IQRs, 95% Bootstrap Confidence Intervals,
and belief transition event summaries across cohorts for Milestone A23.
"""

from __future__ import annotations

import contextlib
import random
import statistics
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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


# ═══════════════════════════════════════════════════════════════════════════
# Real-Time Developmental Telemetry Emitter & Observability Bridge
# ═══════════════════════════════════════════════════════════════════════════


class DevelopmentalTelemetryEmitter:
    """Thread-safe telemetry emitter and observability bridge for developmental learning.

    Maintains in-memory counts, distributions, and recent events, and automatically
    bridges metrics to hbllm.network.metrics.MetricsCollector when available.
    """

    _instance: DevelopmentalTelemetryEmitter | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = defaultdict(float)
        self._latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=500))
        self._events: deque[dict[str, Any]] = deque(maxlen=200)

    @classmethod
    def get_instance(cls) -> DevelopmentalTelemetryEmitter:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def _get_core_collector(self) -> Any:
        try:
            from hbllm.network.metrics import MetricsCollector

            return MetricsCollector.get_instance()
        except Exception:
            return None

    def record_intervention(self, action_type: str, result: str = "success") -> None:
        """Record an active intervention trial."""
        with self._lock:
            self._counters[f"intervention:{action_type}:{result}"] += 1
            self._events.append(
                {
                    "type": "intervention",
                    "action": action_type,
                    "result": result,
                    "timestamp": time.time(),
                }
            )
        collector = self._get_core_collector()
        if collector and hasattr(collector, "record_developmental_intervention"):
            collector.record_developmental_intervention(action_type=action_type, result=result)

    def record_hypothesis_event(self, outcome: str) -> None:
        """Record hypothesis generation, falsification, or confirmation."""
        with self._lock:
            self._counters[f"hypothesis:{outcome}"] += 1
            self._events.append(
                {
                    "type": "hypothesis",
                    "outcome": outcome,
                    "timestamp": time.time(),
                }
            )
        collector = self._get_core_collector()
        if collector and hasattr(collector, "record_developmental_hypothesis"):
            collector.record_developmental_hypothesis(outcome=outcome)

    def record_concept_acquired(self, domain: str) -> None:
        """Record schema/concept acquisition."""
        with self._lock:
            self._counters[f"concept:{domain}"] += 1
            self._events.append(
                {
                    "type": "concept",
                    "domain": domain,
                    "timestamp": time.time(),
                }
            )
        collector = self._get_core_collector()
        if collector and hasattr(collector, "record_developmental_concept"):
            collector.record_developmental_concept(domain=domain)

    def record_entropy(self, system: str, entropy_val: float) -> None:
        """Record belief or affordance entropy level."""
        with self._lock:
            self._gauges[f"entropy:{system}"] = float(entropy_val)
        collector = self._get_core_collector()
        if collector and hasattr(collector, "set_developmental_entropy"):
            collector.set_developmental_entropy(system=system, entropy_val=entropy_val)

    def record_latency(self, stage: str, duration_seconds: float) -> None:
        """Record execution duration of a developmental phase."""
        with self._lock:
            self._latencies[stage].append(duration_seconds)
        collector = self._get_core_collector()
        if collector and hasattr(collector, "observe_developmental_duration"):
            collector.observe_developmental_duration(stage=stage, duration_seconds=duration_seconds)

    @contextlib.contextmanager
    def measure_latency(self, stage: str) -> Generator[None, None, None]:
        """Context manager to measure and record developmental latency."""
        start = time.monotonic()
        try:
            yield
        finally:
            duration = time.monotonic() - start
            self.record_latency(stage, duration)

    def get_telemetry_snapshot(self) -> dict[str, Any]:
        """Return a structured dictionary snapshot of current telemetry."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "latencies": {
                    stage: {
                        "count": len(samples),
                        "avg": round(sum(samples) / len(samples), 6) if samples else 0.0,
                        "min": round(min(samples), 6) if samples else 0.0,
                        "max": round(max(samples), 6) if samples else 0.0,
                    }
                    for stage, samples in self._latencies.items()
                },
                "recent_events_count": len(self._events),
            }
