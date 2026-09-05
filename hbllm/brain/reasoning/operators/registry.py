"""
A12 — Operator Registry: discovery, scoring, and selection.

The registry discovers all reasoning operators, scores them against a
ReasoningProblem using multi-dimensional selection, and returns a
ranked list.  This is the control surface that A19 will eventually
learn to optimize.

Design:
    - Operators are registered explicitly, not auto-discovered.
      This keeps the system deterministic and auditable.
    - Selection uses a multiplicative composite score across
      applicability, reliability, info gain, utility, budget, and
      prerequisites.
    - The registry never executes operators — that is the runtime's job.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    OperatorSelectionScore,
    ReasoningBudget,
    ReasoningOperator,
    ReasoningProblem,
)

logger = logging.getLogger(__name__)


class OperatorRegistry:
    """Registry of all available reasoning operators.

    Responsibilities:
        1. Register / unregister operators.
        2. Given a problem + context, score all operators.
        3. Return a ranked selection list.

    The registry does NOT execute operators.

    Usage::

        registry = OperatorRegistry()
        registry.register(DeductionOperator())
        registry.register(InductionOperator())

        scores = registry.select(problem, context)
        for score in scores:
            print(f"{score.operator_id}: {score.composite_score:.4f}")
    """

    def __init__(self) -> None:
        self._operators: dict[str, ReasoningOperator] = {}
        # Historical reliability tracking (operator_id → success rate)
        self._reliability_history: dict[str, float] = {}

    # ── Registration ─────────────────────────────────────────────────

    def register(self, operator: ReasoningOperator) -> None:
        """Register a reasoning operator.

        Raises:
            ValueError: If an operator with the same ID is already registered.
        """
        oid = operator.operator_id
        if oid in self._operators:
            raise ValueError(
                f"Operator '{oid}' is already registered. "
                f"Unregister it first if you want to replace it."
            )
        self._operators[oid] = operator
        if oid not in self._reliability_history:
            self._reliability_history[oid] = 0.5  # Neutral prior
        logger.info("Registered operator '%s' (%s)", oid, operator.operator_name)

    def unregister(self, operator_id: str) -> ReasoningOperator | None:
        """Remove an operator from the registry."""
        op = self._operators.pop(operator_id, None)
        if op is not None:
            logger.info("Unregistered operator '%s'", operator_id)
        return op

    def get(self, operator_id: str) -> ReasoningOperator | None:
        """Retrieve an operator by ID."""
        return self._operators.get(operator_id)

    @property
    def operator_ids(self) -> frozenset[str]:
        """All registered operator IDs."""
        return frozenset(self._operators.keys())

    @property
    def operator_count(self) -> int:
        return len(self._operators)

    # ── Reliability Tracking ─────────────────────────────────────────

    def record_outcome(
        self,
        operator_id: str,
        success: bool,
        decay: float = 0.05,
    ) -> None:
        """Update reliability history for an operator.

        Uses exponential moving average:
            reliability = (1 - decay) * old + decay * outcome

        Args:
            operator_id: The operator.
            success: Whether the execution was successful.
            decay: Learning rate for the moving average.
        """
        current = self._reliability_history.get(operator_id, 0.5)
        outcome = 1.0 if success else 0.0
        updated = (1.0 - decay) * current + decay * outcome
        self._reliability_history[operator_id] = updated

    def get_reliability(self, operator_id: str) -> float:
        """Get the current reliability score for an operator."""
        return self._reliability_history.get(operator_id, 0.5)

    # ── Selection ────────────────────────────────────────────────────

    def select(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
        min_applicability: float = 0.01,
    ) -> list[OperatorSelectionScore]:
        """Score all operators and return a ranked selection list.

        Args:
            problem: The reasoning problem.
            context: Immutable cognitive context.
            min_applicability: Minimum applicability score to include
                an operator in the result.

        Returns:
            Ranked list of OperatorSelectionScore (highest first).
            Only includes operators with applicability >= min_applicability.
        """
        scores: list[OperatorSelectionScore] = []

        for oid, operator in self._operators.items():
            # 1. Applicability — does the operator think it can handle this?
            try:
                applicability = operator.can_handle(problem, context)
            except Exception:
                logger.warning(
                    "Operator '%s' raised during can_handle; skipping",
                    oid,
                    exc_info=True,
                )
                applicability = 0.0

            if applicability < min_applicability:
                continue

            # 2. Prerequisites — are they satisfied?
            prereqs = operator.prerequisites
            prereq_satisfaction = 1.0
            if prereqs:
                # In a pipeline, prerequisites are tracked by the runtime.
                # Here we just check if the prereq operators exist.
                missing = [p for p in prereqs if p not in self._operators]
                if missing:
                    prereq_satisfaction = 0.0
                    logger.debug(
                        "Operator '%s' has missing prerequisites: %s",
                        oid,
                        missing,
                    )

            # 3. Estimated cost vs budget
            budget = context.budget
            try:
                est_cost = operator.estimated_cost(problem, context)
                budget_fit = self._compute_budget_fit(est_cost, budget)
            except Exception:
                logger.warning(
                    "Operator '%s' raised during estimated_cost; assuming default budget fit",
                    oid,
                    exc_info=True,
                )
                budget_fit = 0.5

            # 4. Reliability — historical success rate
            reliability = self._reliability_history.get(oid, 0.5)

            # 5. Expected information gain — placeholder heuristic
            #    A19 will eventually learn to estimate this.
            expected_info_gain = self._estimate_info_gain(operator, problem, context)

            # 6. Expected utility — placeholder
            expected_utility = applicability * reliability

            # Build score
            score = OperatorSelectionScore(
                operator_id=oid,
                applicability=applicability,
                reliability=reliability,
                expected_info_gain=expected_info_gain,
                expected_utility=expected_utility,
                budget_fit=budget_fit,
                prerequisite_satisfaction=prereq_satisfaction,
            )
            score.compute_composite()
            scores.append(score)

        # Sort by composite score, descending
        scores.sort(key=lambda s: s.composite_score, reverse=True)

        if scores:
            logger.info(
                "Selected %d operators for problem '%s'; top: '%s' (score=%.4f)",
                len(scores),
                problem.problem_id,
                scores[0].operator_id,
                scores[0].composite_score,
            )
        else:
            logger.warning(
                "No applicable operators found for problem '%s' (%s)",
                problem.problem_id,
                problem.problem_type,
            )

        return scores

    # ── Internal Helpers ─────────────────────────────────────────────

    @staticmethod
    def _compute_budget_fit(
        est_cost: Any,  # ResourceCost
        budget: ReasoningBudget,
    ) -> float:
        """Compute how well an operator's estimated cost fits the budget.

        Returns 1.0 if comfortably within budget, 0.0 if clearly exceeds.
        """
        fit = 1.0

        if budget.compute_ms > 0 and est_cost.wall_clock_ms > 0:
            ratio = est_cost.wall_clock_ms / budget.compute_ms
            if ratio > 1.0:
                fit *= max(0.0, 1.0 - (ratio - 1.0))
            else:
                fit *= 1.0

        if budget.simulation_steps > 0 and est_cost.simulation_steps_used > 0:
            ratio = est_cost.simulation_steps_used / budget.simulation_steps
            if ratio > 1.0:
                fit *= max(0.0, 1.0 - (ratio - 1.0))

        return max(0.0, min(1.0, fit))

    @staticmethod
    def _estimate_info_gain(
        operator: ReasoningOperator,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float:
        """Heuristic estimate of expected information gain.

        This is a placeholder.  A19 will learn to predict information
        gain from problem structure + context features.

        Current heuristic:
        - Explanation/diagnosis problems → higher info gain
        - More evidence nodes → lower marginal gain
        - Higher uncertainty tolerance → higher info gain (exploring)
        """
        from hbllm.brain.reasoning.operators.base import ProblemType

        base = 0.5

        # Problem type adjustments
        high_gain_types = {
            ProblemType.EXPLANATION,
            ProblemType.DIAGNOSIS,
            ProblemType.CONTRADICTION,
            ProblemType.COUNTERFACTUAL,
        }
        if problem.problem_type in high_gain_types:
            base += 0.15

        # More evidence → diminishing returns
        n_evidence = len(problem.evidence_node_ids)
        if n_evidence > 5:
            base -= 0.05 * min(5, n_evidence - 5)

        # Higher uncertainty tolerance → more exploratory
        base += 0.1 * (context.budget.uncertainty_tolerance - 0.5)

        return max(0.0, min(1.0, base))


def create_default_operator_registry() -> OperatorRegistry:
    """Create and return an OperatorRegistry pre-loaded with standard reasoning operators."""
    from hbllm.brain.reasoning.operators.abduction import AbductionOperator
    from hbllm.brain.reasoning.operators.active_inference import ActiveInferenceOperator
    from hbllm.brain.reasoning.operators.analogy import AnalogyOperator
    from hbllm.brain.reasoning.operators.causal import CausalOperator
    from hbllm.brain.reasoning.operators.contradiction import ContradictionOperator
    from hbllm.brain.reasoning.operators.counterfactual import CounterfactualOperator
    from hbllm.brain.reasoning.operators.deduction import DeductionOperator
    from hbllm.brain.reasoning.operators.induction import InductionOperator
    from hbllm.brain.reasoning.operators.prediction import PredictionOperator
    from hbllm.brain.reasoning.operators.simulation import SimulationOperator
    from hbllm.brain.reasoning.operators.spatial import SpatialOperator
    from hbllm.brain.reasoning.operators.temporal import TemporalOperator

    registry = OperatorRegistry()
    registry.register(DeductionOperator())
    registry.register(InductionOperator())
    registry.register(AbductionOperator())
    registry.register(SpatialOperator())
    registry.register(TemporalOperator())
    registry.register(CausalOperator())
    registry.register(AnalogyOperator())
    registry.register(CounterfactualOperator())
    registry.register(ContradictionOperator())
    registry.register(PredictionOperator())
    registry.register(SimulationOperator())
    registry.register(ActiveInferenceOperator())
    return registry
