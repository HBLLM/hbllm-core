"""Milestone M4.4 Scaled Evaluation Suite: 100-Task Randomized Procedural Benchmark.

Executes a 100-task randomized procedural benchmark across 4 balanced families:
1. Geometric / Affine (N=25): Non-square grids, asymmetric morphologies, reflections, rotations.
2. Morphological / Relational (N=25): Connected components, bbox extraction, relative coordinate shifts.
3. Attribute / Recolor (N=25): Color permutations, multi-color swaps, background variations.
4. Compositional Depth-2 & 3 (N=25): Multi-stage pipelines f_2(f_1(x)) and f_3(f_2(f_1(x))).

Tracks:
- Exact match % and 95% Wilson confidence intervals per family and overall.
- Timing distributions: mean, median, and p95 tail latency across search, verification, ranking, and execution.
- Causal ablation comparison: Full HCIR vs. Atomic Only vs. No MDL vs. No Relational on identical instances.
- Hypothesis search traces: proposed, refuted on demo 0, refuted on later demos, and surviving.
- Epistemic uncertainty: detecting when multiple distinct hypotheses produce conflicting query predictions.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from experiments.benchmarks.manifests.transformation_task_manifest import ManifestTask
from experiments.benchmarks.task_generators.procedural_task_generator import ProceduralTaskGenerator
from hbllm.hcir.world.grid_operator import TransformationProgramSearch

logger = logging.getLogger(__name__)


@dataclass
class ScaledTaskTrace:
    """Detailed trace of hypothesis proposal, refutation, ranking, and execution for one task."""

    task_id: str
    family: str
    depth: int
    exact_match: bool
    pixel_accuracy: float
    candidates_proposed: int
    candidates_refuted_demo0: int
    candidates_refuted_later: int
    survivors_count: int
    winning_description: str | None
    winning_complexity: float | None
    is_minimal: bool
    is_ambiguous_on_test: bool
    epistemic_uncertainty: float
    candidate_gen_ms: float
    verification_ms: float
    ranking_selection_ms: float
    search_duration_ms: float
    execution_ms: float
    total_wall_clock_ms: float


@dataclass
class FamilyScorecard:
    """Evaluation summary for a specific transformation family."""

    family: str
    total_tasks: int
    solved_tasks: int
    exact_match_pct: float
    confidence_interval_95: tuple[float, float]
    mean_search_ms: float
    mean_execution_ms: float
    mean_total_ms: float


@dataclass
class ScaledBenchmarkReport:
    """Comprehensive benchmark report across all 100 procedural tasks."""

    total_tasks: int
    solved_tasks: int
    overall_exact_match_pct: float
    overall_confidence_interval_95: tuple[float, float]
    family_scorecards: dict[str, FamilyScorecard]
    latency_profile: dict[str, dict[str, float]]  # mean, median, p95 for each timer
    traces: list[ScaledTaskTrace] = field(default_factory=list)


class ScaledGeneralizationBenchmark:
    """Executes the 100-task procedural evaluation and causal ablation analysis."""

    @staticmethod
    def compute_wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
        """Compute Wilson score 95% confidence interval."""
        if total == 0:
            return 0.0, 0.0
        p_hat = successes / total
        denom = 1.0 + (z**2) / total
        centre = (p_hat + (z**2) / (2 * total)) / denom
        diff = z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * total)) / total) / denom
        lower = max(0.0, float(centre - diff))
        upper = min(1.0, float(centre + diff))
        return round(lower * 100, 2), round(upper * 100, 2)

    @classmethod
    def generate_100_task_suite(cls, seed: int = 2026) -> list[ManifestTask]:
        """Procedurally generate 100 tasks (25 Geometric, 25 Morphological, 25 Recolor, 25 Compositional)."""
        generator = ProceduralTaskGenerator(seed=seed)
        tasks: list[ManifestTask] = []

        # 1. Geometric / Affine Family (N=25)
        for i in range(25):
            t = generator.generate_random_task(
                task_id=f"geom_task_{i:02d}", depth=1, num_demos=2, family="geometric_affine"
            )
            tasks.append(t)

        # 2. Morphological / Relational Family (N=25)
        for i in range(25):
            t = generator.generate_random_task(
                task_id=f"morph_task_{i:02d}",
                depth=1,
                num_demos=2,
                family="morphological_relational",
            )
            tasks.append(t)

        # 3. Attribute / Recolor Family (N=25)
        for i in range(25):
            t = generator.generate_random_task(
                task_id=f"recolor_task_{i:02d}", depth=1, num_demos=2, family="attribute_recolor"
            )
            tasks.append(t)

        # 4. Compositional Depth-2 & Depth-3 Family (N=25)
        for i in range(25):
            d = 2 if i < 15 else 3
            t = generator.generate_random_task(
                task_id=f"comp_task_{i:02d}", depth=d, num_demos=2, family="compositional_deep"
            )
            tasks.append(t)

        return tasks

    @classmethod
    def evaluate_task(
        cls,
        task: ManifestTask,
        searcher: TransformationProgramSearch,
    ) -> ScaledTaskTrace:
        """Run single task through searcher and capture fine-grained trace."""
        t_start = time.perf_counter()
        pred, winning_b, meta = searcher.solve(list(task.train_pairs), task.test_input)
        tot_ms = (time.perf_counter() - t_start) * 1000

        exact = bool(pred is not None and np.array_equal(pred, task.test_output))
        acc = 1.0 if exact else 0.0
        if pred is not None and pred.shape == task.test_output.shape and not exact:
            diff = np.sum(pred != task.test_output)
            acc = float(1.0 - (diff / max(task.test_output.size, 1)))

        total_cand = meta.get("total_candidates", 0)
        ref_count = meta.get("refuted_count", 0)
        ref_later = meta.get("spurious_rejected_on_later_demos", 0)
        ref_demo0 = max(0, ref_count - ref_later)
        surv_count = meta.get("survivor_count", 0)

        win_desc = winning_b.description if winning_b else None
        win_comp = winning_b.complexity if winning_b else None
        is_min = bool(surv_count > 0 and winning_b and win_comp is not None and win_comp <= 3.0)

        return ScaledTaskTrace(
            task_id=task.task_id,
            family=task.family,
            depth=task.depth,
            exact_match=exact,
            pixel_accuracy=acc,
            candidates_proposed=total_cand,
            candidates_refuted_demo0=ref_demo0,
            candidates_refuted_later=ref_later,
            survivors_count=surv_count,
            winning_description=win_desc,
            winning_complexity=win_comp,
            is_minimal=is_min,
            is_ambiguous_on_test=meta.get("is_ambiguous_on_test", False),
            epistemic_uncertainty=meta.get("epistemic_uncertainty", 0.0),
            candidate_gen_ms=meta.get("candidate_gen_ms", 0.0),
            verification_ms=meta.get("verification_ms", 0.0),
            ranking_selection_ms=meta.get("ranking_selection_ms", 0.0),
            search_duration_ms=meta.get("search_duration_ms", 0.0),
            execution_ms=meta.get("execution_ms", 0.0),
            total_wall_clock_ms=meta.get("total_wall_clock_ms", tot_ms),
        )

    @classmethod
    def run_benchmark(
        cls,
        tasks: list[ManifestTask] | None = None,
        max_depth: int = 3,
        use_mdl: bool = True,
    ) -> ScaledBenchmarkReport:
        """Run full 100-task suite and produce comprehensive scorecard and latency distributions."""
        suite = tasks or cls.generate_100_task_suite()
        searcher = TransformationProgramSearch(max_depth=max_depth, use_mdl=use_mdl)
        traces: list[ScaledTaskTrace] = []

        for task in suite:
            tr = cls.evaluate_task(task, searcher)
            traces.append(tr)

        total = len(traces)
        solved = sum(1 for t in traces if t.exact_match)
        overall_pct = round((solved / max(total, 1)) * 100, 2)
        overall_ci = cls.compute_wilson_interval(solved, total)

        # Family scorecards
        families = sorted(list({t.family for t in traces}))
        family_scorecards: dict[str, FamilyScorecard] = {}
        for f in families:
            f_traces = [t for t in traces if t.family == f]
            f_total = len(f_traces)
            f_solved = sum(1 for t in f_traces if t.exact_match)
            f_pct = round((f_solved / max(f_total, 1)) * 100, 2)
            f_ci = cls.compute_wilson_interval(f_solved, f_total)
            family_scorecards[f] = FamilyScorecard(
                family=f,
                total_tasks=f_total,
                solved_tasks=f_solved,
                exact_match_pct=f_pct,
                confidence_interval_95=f_ci,
                mean_search_ms=round(float(np.mean([t.search_duration_ms for t in f_traces])), 2),
                mean_execution_ms=round(float(np.mean([t.execution_ms for t in f_traces])), 2),
                mean_total_ms=round(float(np.mean([t.total_wall_clock_ms for t in f_traces])), 2),
            )

        # Latency profile (mean, median, p95)
        def _get_stats(vals: list[float]) -> dict[str, float]:
            if not vals:
                return {"mean": 0.0, "median": 0.0, "p95": 0.0}
            return {
                "mean": round(float(np.mean(vals)), 2),
                "median": round(float(np.median(vals)), 2),
                "p95": round(float(np.percentile(vals, 95)), 2),
            }

        latency_profile = {
            "candidate_generation": _get_stats([t.candidate_gen_ms for t in traces]),
            "verification": _get_stats([t.verification_ms for t in traces]),
            "ranking_selection": _get_stats([t.ranking_selection_ms for t in traces]),
            "search_duration": _get_stats([t.search_duration_ms for t in traces]),
            "execution": _get_stats([t.execution_ms for t in traces]),
            "total_wall_clock": _get_stats([t.total_wall_clock_ms for t in traces]),
        }

        return ScaledBenchmarkReport(
            total_tasks=total,
            solved_tasks=solved,
            overall_exact_match_pct=overall_pct,
            overall_confidence_interval_95=overall_ci,
            family_scorecards=family_scorecards,
            latency_profile=latency_profile,
            traces=traces,
        )
