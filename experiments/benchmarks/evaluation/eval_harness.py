"""Domain-Agnostic 2D Discrete Inductive Evaluation Harness (W083 Milestone).

Provides a reproducible, principled evaluation framework for grid world modeling:
1. Records training demonstrations and held-out test input/output.
2. Tracks candidate transformation rules, complexity parameters, and refutations.
3. Generates predictions for every training example and the final test grid.
4. Distinguishes training consistency (fitting) from held-out test accuracy (generalization).
5. Categorizes failures under a standardized failure taxonomy.
6. Reports solve rates, pixel accuracy, compute latency, and 95% confidence intervals.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.world.predictors.whole_grid import WholeGridPredictor
from hbllm.hcir.world.rule_induction import RuleInductionEngine

logger = logging.getLogger(__name__)


class FailureCategory(StrEnum):
    """Standardized failure taxonomy for inductive demonstration rule search."""

    NO_CANDIDATES = "NO_CANDIDATES"  # Generator produced no candidate rules
    FITTING_FAILED = "FITTING_FAILED"  # No candidate rule satisfied all training demonstrations
    OVERFITTING_FALSE_SELECTION = (
        "OVERFITTING_FALSE_SELECTION"  # Rule satisfied demos but failed on held-out test
    )
    AMBIGUOUS_SURVIVORS = (
        "AMBIGUOUS_SURVIVORS"  # Multiple rules satisfied demos, selected rule failed
    )
    EXECUTION_CRASH = "EXECUTION_CRASH"  # Rule raised unhandled exception during prediction
    NO_DIMENSION_RULE = "NO_DIMENSION_RULE"  # Target canvas shape could not be determined


@dataclass
class DemonstrationTask:
    """Domain-agnostic task with training demonstration pairs and held-out test input/output."""

    task_id: str
    train_pairs: list[tuple[np.ndarray, np.ndarray]]
    test_input: np.ndarray
    test_output: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], task_id: str = "unnamed_task") -> DemonstrationTask:
        """Instantiate from paired demonstration dictionary format."""
        train_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for pair in data.get("train", []):
            x = np.asarray(pair["input"], dtype=int)
            y = np.asarray(pair["output"], dtype=int)
            train_pairs.append((x, y))

        test_list = data.get("test", [])
        if not test_list:
            raise ValueError(f"Task {task_id} contains no test pairs")

        first_test = test_list[0]
        test_in = np.asarray(first_test["input"], dtype=int)
        test_out = np.asarray(first_test["output"], dtype=int) if "output" in first_test else None

        return cls(
            task_id=task_id,
            train_pairs=train_pairs,
            test_input=test_in,
            test_output=test_out,
            metadata=data.get("metadata", {}),
        )


@dataclass
class TaskEvaluationRecord:
    """Per-task evaluation record capturing predictions, metrics, and failure category."""

    task_id: str
    train_fit_rate: float
    test_exact_match: bool
    test_pixel_accuracy: float
    surviving_rules_count: int
    selected_rule_id: str | None
    selected_rule_complexity: float | None
    hypotheses_evaluated: int
    hypotheses_refuted: int
    duration_ms: float
    failure_category: FailureCategory | None
    predicted_train_grids: list[list[list[int]]] = field(default_factory=list)
    predicted_test_grid: list[list[int]] | None = None
    ground_truth_test_grid: list[list[int]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation record to JSON-serializable dictionary."""
        d = asdict(self)
        if self.failure_category is not None:
            d["failure_category"] = self.failure_category.value
        return d


@dataclass
class EvaluationSummary:
    """Aggregate benchmark report across multiple ARC tasks."""

    total_tasks: int
    train_perfect_fit_count: int
    train_perfect_fit_rate: float
    test_exact_solve_count: int
    test_exact_solve_rate: float
    mean_pixel_accuracy: float
    mean_duration_ms: float
    failure_distribution: dict[str, int]
    confidence_interval_95: tuple[float, float]
    task_records: list[TaskEvaluationRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert aggregate summary to JSON-serializable dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "train_perfect_fit_count": self.train_perfect_fit_count,
            "train_perfect_fit_rate": round(self.train_perfect_fit_rate, 4),
            "test_exact_solve_count": self.test_exact_solve_count,
            "test_exact_solve_rate": round(self.test_exact_solve_rate, 4),
            "mean_pixel_accuracy": round(self.mean_pixel_accuracy, 4),
            "mean_duration_ms": round(self.mean_duration_ms, 2),
            "failure_distribution": self.failure_distribution,
            "confidence_interval_95": [
                round(self.confidence_interval_95[0], 4),
                round(self.confidence_interval_95[1], 4),
            ],
            "task_records": [r.to_dict() for r in self.task_records],
        }


class InductiveEvaluationHarness:
    """Domain-Agnostic Inductive Evaluation Harness centered around Whole-Grid Prediction (W083)."""

    def __init__(self, max_candidates: int = 100) -> None:
        self.max_candidates = max_candidates
        self.predictor = WholeGridPredictor()

    @staticmethod
    def compute_wilson_score_interval(
        successes: int, total: int, z: float = 1.96
    ) -> tuple[float, float]:
        """Calculate 95% Wilson score confidence interval for binomial solve rate."""
        if total == 0:
            return (0.0, 0.0)
        p = successes / total
        denominator = 1.0 + (z**2) / total
        centre_adjusted_probability = p + (z**2) / (2 * total)
        adjusted_standard_deviation = math.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total)
        lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
        upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
        return (max(0.0, lower_bound), min(1.0, upper_bound))

    def evaluate_task(self, task: DemonstrationTask) -> TaskEvaluationRecord:
        """Run inductive rule search, Popperian refutation, and held-out prediction on a task."""
        start_time = time.perf_counter()
        engine = RuleInductionEngine(max_candidates=self.max_candidates)

        selected_rule, survivors, meta = engine.induce_rule(task.train_pairs)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        hypotheses_evaluated = meta["candidates_generated"]
        hypotheses_refuted = meta["refuted_count"]
        surviving_count = meta["survivors_count"]

        # 1. No candidates generated
        if hypotheses_evaluated == 0:
            return TaskEvaluationRecord(
                task_id=task.task_id,
                train_fit_rate=0.0,
                test_exact_match=False,
                test_pixel_accuracy=0.0,
                surviving_rules_count=0,
                selected_rule_id=None,
                selected_rule_complexity=None,
                hypotheses_evaluated=0,
                hypotheses_refuted=0,
                duration_ms=duration_ms,
                failure_category=FailureCategory.NO_CANDIDATES,
            )

        # 2. No candidate satisfied all training demonstrations
        if selected_rule is None or surviving_count == 0:
            return TaskEvaluationRecord(
                task_id=task.task_id,
                train_fit_rate=0.0,
                test_exact_match=False,
                test_pixel_accuracy=0.0,
                surviving_rules_count=0,
                selected_rule_id=None,
                selected_rule_complexity=None,
                hypotheses_evaluated=hypotheses_evaluated,
                hypotheses_refuted=hypotheses_refuted,
                duration_ms=duration_ms,
                failure_category=FailureCategory.FITTING_FAILED,
            )

        # 3. Predict and record training demonstrations
        train_matches = 0
        predicted_trains: list[list[list[int]]] = []
        for x, y in task.train_pairs:
            bg = self.predictor.estimate_background_color(x)
            y_shape: tuple[int, int] = (int(y.shape[0]), int(y.shape[1]))
            pred_y = self.predictor.render_prediction(
                selected_rule, x, target_shape=y_shape, default_bg=bg
            )
            predicted_trains.append(pred_y.tolist())
            is_exact, _, _ = self.predictor.compute_grid_metrics(pred_y, y)
            if is_exact:
                train_matches += 1

        train_fit_rate = train_matches / len(task.train_pairs) if task.train_pairs else 0.0

        # 4. Predict held-out test grid
        test_bg = self.predictor.estimate_background_color(task.test_input)
        dim_rule = self.predictor.infer_dimension_rule(task.train_pairs)
        test_in_shape: tuple[int, int] = (
            int(task.test_input.shape[0]),
            int(task.test_input.shape[1]),
        )
        target_shape = dim_rule.compute_output_shape(test_in_shape)

        try:
            pred_test = self.predictor.render_prediction(
                selected_rule,
                task.test_input,
                target_shape=target_shape,
                default_bg=test_bg,
            )
        except Exception:
            return TaskEvaluationRecord(
                task_id=task.task_id,
                train_fit_rate=train_fit_rate,
                test_exact_match=False,
                test_pixel_accuracy=0.0,
                surviving_rules_count=surviving_count,
                selected_rule_id=selected_rule.rule_id,
                selected_rule_complexity=selected_rule.complexity,
                hypotheses_evaluated=hypotheses_evaluated,
                hypotheses_refuted=hypotheses_refuted,
                duration_ms=duration_ms,
                failure_category=FailureCategory.EXECUTION_CRASH,
                predicted_train_grids=predicted_trains,
            )

        # 5. Evaluate against held-out ground truth
        test_exact_match = False
        test_pixel_accuracy = 0.0
        failure_category: FailureCategory | None = None

        if task.test_output is not None:
            test_exact_match, test_pixel_accuracy, _ = self.predictor.compute_grid_metrics(
                pred_test, task.test_output
            )
            if not test_exact_match:
                if surviving_count > 1:
                    failure_category = FailureCategory.AMBIGUOUS_SURVIVORS
                else:
                    failure_category = FailureCategory.OVERFITTING_FALSE_SELECTION

        return TaskEvaluationRecord(
            task_id=task.task_id,
            train_fit_rate=train_fit_rate,
            test_exact_match=test_exact_match,
            test_pixel_accuracy=test_pixel_accuracy,
            surviving_rules_count=surviving_count,
            selected_rule_id=selected_rule.rule_id,
            selected_rule_complexity=selected_rule.complexity,
            hypotheses_evaluated=hypotheses_evaluated,
            hypotheses_refuted=hypotheses_refuted,
            duration_ms=duration_ms,
            failure_category=failure_category,
            predicted_train_grids=predicted_trains,
            predicted_test_grid=pred_test.tolist(),
            ground_truth_test_grid=(
                task.test_output.tolist() if task.test_output is not None else None
            ),
        )

    def evaluate_suite(self, tasks: list[DemonstrationTask]) -> EvaluationSummary:
        """Evaluate a benchmark suite of demonstration tasks and produce aggregate metrics."""
        records: list[TaskEvaluationRecord] = []
        train_perfect_count = 0
        test_solve_count = 0
        total_pixel_acc = 0.0
        total_duration = 0.0
        failure_dist: dict[str, int] = {}

        for task in tasks:
            record = self.evaluate_task(task)
            records.append(record)

            if record.train_fit_rate == 1.0:
                train_perfect_count += 1
            if record.test_exact_match:
                test_solve_count += 1
            total_pixel_acc += record.test_pixel_accuracy
            total_duration += record.duration_ms

            if record.failure_category is not None:
                cat_name = record.failure_category.value
                failure_dist[cat_name] = failure_dist.get(cat_name, 0) + 1

        total_tasks = len(tasks)
        train_perfect_rate = train_perfect_count / total_tasks if total_tasks > 0 else 0.0
        test_solve_rate = test_solve_count / total_tasks if total_tasks > 0 else 0.0
        mean_pixel_acc = total_pixel_acc / total_tasks if total_tasks > 0 else 0.0
        mean_duration = total_duration / total_tasks if total_tasks > 0 else 0.0

        ci_95 = self.compute_wilson_score_interval(test_solve_count, total_tasks)

        return EvaluationSummary(
            total_tasks=total_tasks,
            train_perfect_fit_count=train_perfect_count,
            train_perfect_fit_rate=train_perfect_rate,
            test_exact_solve_count=test_solve_count,
            test_exact_solve_rate=test_solve_rate,
            mean_pixel_accuracy=mean_pixel_acc,
            mean_duration_ms=mean_duration,
            failure_distribution=failure_dist,
            confidence_interval_95=ci_95,
            task_records=records,
        )


# ── Backward Compatibility Aliases ──────────────────────────────────────────
ARCTask = DemonstrationTask
ARCEvaluationHarness = InductiveEvaluationHarness
WorldModelEvaluationHarness = InductiveEvaluationHarness
