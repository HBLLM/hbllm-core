"""
Simulation Engine — Mental rehearsal and candidate evaluation.

Enables HBLLM to *think before acting* by simulating candidate
responses internally before committing to execution.

Cognitive loop evolution::

    Before M4 (reactive):
        Planner → Decision → Action

    After M4 (deliberative):
        Planner → Simulation → Critic → Decision → Action

Process:
    1. Planner proposes candidate action/response
    2. SimulationEngine imagines executing it:
       - Predicts likely consequences (via CognitivePredictors)
       - Estimates user reaction (via emotion predictor)
       - Checks goal alignment (via GoalMemory)
       - Evaluates belief consistency (via BeliefGraph)
    3. Critic scores the simulated outcome
    4. If score < threshold → reject, simulate alternative
    5. If score ≥ threshold → approve for execution

Implements ``ISimulator`` from cognitive_interfaces.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from hbllm.brain.cognitive_interfaces import IGoalProvider, ISimulator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Deliberation Budget — adaptive computation
# ═══════════════════════════════════════════════════════════════════════════


class DeliberationLevel:
    """Deliberation levels based on budget calculation."""

    SKIP = "skip"  # confidence > 0.9 — no simulation needed
    SINGLE = "single"  # moderate uncertainty — one simulation
    MULTIPLE = "multiple"  # high uncertainty — multiple candidates
    BEAM = "beam"  # very high uncertainty — beam search


@dataclass
class DeliberationBudget:
    """Adaptive computation budget for simulation.

    Brains don't deliberate over every action. This calculates
    how much deliberation is warranted based on:

        budget = uncertainty × importance × novelty × goal_priority

    Attributes:
        uncertainty: How uncertain the system is [0, 1].
        importance: How important this decision is [0, 1].
        novelty: How novel/unfamiliar this situation is [0, 1].
        goal_priority: Priority of the relevant goal [0, 1].
    """

    uncertainty: float = 0.5
    importance: float = 0.5
    novelty: float = 0.5
    goal_priority: float = 0.5

    @property
    def score(self) -> float:
        """Composite budget score [0, 1]."""
        return self.uncertainty * self.importance * self.novelty * self.goal_priority

    @property
    def level(self) -> str:
        """Recommended deliberation level."""
        s = self.score
        if s < 0.05:
            return DeliberationLevel.SKIP
        elif s < 0.15:
            return DeliberationLevel.SINGLE
        elif s < 0.4:
            return DeliberationLevel.MULTIPLE
        else:
            return DeliberationLevel.BEAM

    @property
    def recommended_candidates(self) -> int:
        """Recommended number of candidates to simulate."""
        level = self.level
        if level == DeliberationLevel.SKIP:
            return 0
        elif level == DeliberationLevel.SINGLE:
            return 1
        elif level == DeliberationLevel.MULTIPLE:
            return 3
        else:  # BEAM
            return 5


# ═══════════════════════════════════════════════════════════════════════════
# Critic Protocol
# ═══════════════════════════════════════════════════════════════════════════


class ICritic(Protocol):
    """Protocol for scoring simulation results."""

    async def score(
        self,
        candidate: str,
        consequences: list[str],
        goal_alignment: float,
        belief_consistency: float,
    ) -> float:
        """Score a candidate action [0.0, 1.0].

        Higher scores indicate better candidates.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Simulation Result
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SimulationResult:
    """Outcome of simulating a candidate action.

    Attributes:
        candidate: The candidate action/response that was simulated.
        predicted_consequences: Predicted downstream effects.
        predicted_user_reaction: Estimated user valence [-1, 1].
        goal_alignment: How well this serves active goals [0, 1].
        belief_consistency: Consistency with known beliefs [0, 1].
        critic_score: Overall quality score from critic [0, 1].
        approved: Whether this candidate passed the threshold.
        rejection_reason: Explanation if rejected.
        simulation_time: Time spent on this simulation (seconds).
    """

    candidate: str
    predicted_consequences: list[str] = field(default_factory=list)
    predicted_user_reaction: float = 0.0
    goal_alignment: float = 0.0
    belief_consistency: float = 1.0
    critic_score: float = 0.0
    approved: bool = False
    rejection_reason: str | None = None
    simulation_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate[:100],
            "consequences": self.predicted_consequences[:5],
            "user_reaction": round(self.predicted_user_reaction, 2),
            "goal_alignment": round(self.goal_alignment, 3),
            "belief_consistency": round(self.belief_consistency, 3),
            "critic_score": round(self.critic_score, 3),
            "approved": self.approved,
            "rejection_reason": self.rejection_reason,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Default Critic — heuristic scorer
# ═══════════════════════════════════════════════════════════════════════════


class HeuristicCritic:
    """Default critic using weighted heuristic scoring.

    Combines goal alignment, belief consistency, and user reaction
    into a composite score.

    Args:
        goal_weight: Weight for goal alignment in scoring.
        belief_weight: Weight for belief consistency.
        reaction_weight: Weight for predicted user reaction.
    """

    def __init__(
        self,
        goal_weight: float = 0.4,
        belief_weight: float = 0.3,
        reaction_weight: float = 0.3,
    ) -> None:
        self._goal_weight = goal_weight
        self._belief_weight = belief_weight
        self._reaction_weight = reaction_weight

    async def score(
        self,
        candidate: str,
        consequences: list[str],
        goal_alignment: float,
        belief_consistency: float,
        user_reaction: float = 0.0,
    ) -> float:
        """Score a candidate action.

        Args:
            candidate: The action being evaluated.
            consequences: Predicted downstream effects.
            goal_alignment: Goal alignment [0, 1].
            belief_consistency: Belief consistency [0, 1].
            user_reaction: Predicted user valence [-1, 1].

        Returns:
            Composite score [0, 1].
        """
        # Normalize user reaction from [-1, 1] to [0, 1]
        reaction_normalized = (user_reaction + 1.0) / 2.0

        # Penalty for excessive consequences (high risk)
        risk_penalty = min(0.2, len(consequences) * 0.02)

        score = (
            self._goal_weight * goal_alignment
            + self._belief_weight * belief_consistency
            + self._reaction_weight * reaction_normalized
            - risk_penalty
        )
        return max(0.0, min(1.0, score))


# ═══════════════════════════════════════════════════════════════════════════
# SimulationEngine
# ═══════════════════════════════════════════════════════════════════════════


class SimulationEngine(ISimulator):
    """Internal simulation between planning and execution.

    Evaluates candidate actions by predicting their consequences,
    checking goal alignment and belief consistency, and scoring
    via a critic function.

    Args:
        goal_provider: Source of active goals for alignment checking.
        belief_graph: Belief provenance graph for consistency checks.
        predictors: Cognitive prediction engine (optional).
        critic: Scoring function (defaults to HeuristicCritic).
        approval_threshold: Minimum critic score to approve a candidate.
    """

    def __init__(
        self,
        goal_provider: IGoalProvider | None = None,
        belief_graph: Any = None,
        predictors: Any = None,
        critic: HeuristicCritic | None = None,
        approval_threshold: float = 0.5,
    ) -> None:
        self._goal_provider = goal_provider
        self._belief_graph = belief_graph
        self._predictors = predictors
        self._critic = critic or HeuristicCritic()
        self._approval_threshold = approval_threshold

        # Statistics
        self._simulations_run: int = 0
        self._approvals: int = 0
        self._rejections: int = 0

    # ── ISimulator interface ─────────────────────────────────────────

    async def simulate(
        self,
        candidate_action: str,
        context: Any = None,
        tenant_id: str = "default",
    ) -> SimulationResult:
        """Simulate a candidate action and evaluate its quality.

        Args:
            candidate_action: The action/response to evaluate.
            context: Current cognitive state (optional).
            tenant_id: Multi-tenant isolation key.

        Returns:
            SimulationResult with scores and approval decision.
        """
        start_time = time.time()
        self._simulations_run += 1

        # Step 1: Predict consequences
        consequences = await self._predict_consequences(candidate_action, context)

        # Step 2: Estimate user reaction
        user_reaction = await self._estimate_user_reaction(candidate_action, context)

        # Step 3: Check goal alignment
        goal_alignment = await self._check_goal_alignment(candidate_action, tenant_id)

        # Step 4: Check belief consistency
        belief_consistency = await self._check_belief_consistency(candidate_action)

        # Step 5: Critic scoring
        critic_score = await self._critic.score(
            candidate=candidate_action,
            consequences=consequences,
            goal_alignment=goal_alignment,
            belief_consistency=belief_consistency,
            user_reaction=user_reaction,
        )

        # Step 6: Approval decision
        approved = critic_score >= self._approval_threshold
        rejection_reason = None
        if not approved:
            rejection_reason = self._explain_rejection(
                critic_score, goal_alignment, belief_consistency, user_reaction
            )
            self._rejections += 1
        else:
            self._approvals += 1

        elapsed = time.time() - start_time

        result = SimulationResult(
            candidate=candidate_action,
            predicted_consequences=consequences,
            predicted_user_reaction=user_reaction,
            goal_alignment=goal_alignment,
            belief_consistency=belief_consistency,
            critic_score=critic_score,
            approved=approved,
            rejection_reason=rejection_reason,
            simulation_time=elapsed,
        )

        logger.debug(
            "Simulation: candidate=%s score=%.2f approved=%s",
            candidate_action[:50],
            critic_score,
            approved,
        )
        return result

    async def compare_candidates(
        self,
        candidates: list[str],
        context: Any = None,
        tenant_id: str = "default",
    ) -> list[SimulationResult]:
        """Simulate multiple candidates and rank by score.

        Args:
            candidates: List of candidate actions to evaluate.
            context: Current cognitive state (optional).
            tenant_id: Multi-tenant isolation key.

        Returns:
            Sorted list of SimulationResults (best first).
        """
        results: list[SimulationResult] = []
        for candidate in candidates:
            result = await self.simulate(candidate, context, tenant_id)
            results.append(result)

        # Sort by critic score (descending)
        results.sort(key=lambda r: r.critic_score, reverse=True)
        return results

    # ── Internal evaluation methods ──────────────────────────────────

    async def _predict_consequences(self, candidate: str, context: Any) -> list[str]:
        """Predict downstream consequences of an action.

        Uses CognitivePredictors if available, otherwise returns
        heuristic consequence list.
        """
        consequences: list[str] = []

        if self._predictors:
            # Use action predictor to anticipate next states
            try:
                dist = self._predictors.action.predict()
                for state, prob in sorted(dist.items(), key=lambda x: -x[1])[:3]:
                    if prob > 0.1:
                        consequences.append(f"Likely follow-up: {state} (p={prob:.2f})")
            except Exception:
                pass

        return consequences

    async def _estimate_user_reaction(self, candidate: str, context: Any) -> float:
        """Estimate predicted user reaction valence [-1, 1].

        Uses emotion predictor if available. Default: neutral (0.0).
        """
        if self._predictors:
            try:
                dist = self._predictors.emotion.predict()
                # Map emotional states to valence
                valence_map = {
                    "positive": 0.8,
                    "neutral": 0.0,
                    "negative": -0.6,
                    "frustrated": -0.8,
                    "satisfied": 0.7,
                    "curious": 0.3,
                    "confused": -0.3,
                }
                weighted_valence = sum(
                    dist.get(state, 0.0) * valence for state, valence in valence_map.items()
                )
                return max(-1.0, min(1.0, weighted_valence))
            except Exception:
                pass
        return 0.0

    async def _check_goal_alignment(self, candidate: str, tenant_id: str) -> float:
        """Check how well the candidate aligns with active goals.

        Returns alignment score [0, 1].
        """
        if not self._goal_provider:
            return 0.5  # Neutral if no goal provider

        try:
            active_goals = await self._goal_provider.get_active_goals(tenant_id)
            if not active_goals:
                return 0.5

            # Simple keyword matching heuristic
            candidate_lower = candidate.lower()
            match_count = 0
            for goal in active_goals:
                desc = getattr(goal, "description", str(goal)).lower()
                # Check if any significant words from the goal appear
                words = [w for w in desc.split() if len(w) > 3]
                if any(w in candidate_lower for w in words):
                    match_count += 1

            if active_goals:
                return min(1.0, match_count / max(1, len(active_goals)) + 0.3)
            return 0.5
        except Exception:
            return 0.5

    async def _check_belief_consistency(self, candidate: str) -> float:
        """Check if the candidate contradicts known beliefs.

        Returns consistency score [0, 1].
        """
        if not self._belief_graph:
            return 1.0  # Consistent by default if no belief graph

        try:
            contested = await self._belief_graph.get_contested_beliefs()
            # If there are many contested beliefs, overall consistency is lower
            if contested:
                avg_confidence = sum(b.confidence for b in contested) / len(contested)
                return avg_confidence
            return 1.0
        except Exception:
            return 1.0

    def _explain_rejection(
        self,
        score: float,
        goal_alignment: float,
        belief_consistency: float,
        user_reaction: float,
    ) -> str:
        """Generate human-readable rejection explanation."""
        reasons: list[str] = []

        if goal_alignment < 0.3:
            reasons.append("Low goal alignment")
        if belief_consistency < 0.5:
            reasons.append("Belief inconsistency detected")
        if user_reaction < -0.3:
            reasons.append("Negative predicted user reaction")
        if score < self._approval_threshold:
            reasons.append(
                f"Critic score {score:.2f} below threshold {self._approval_threshold:.2f}"
            )

        return "; ".join(reasons) if reasons else "Score below threshold"

    def stats(self) -> dict[str, Any]:
        """Simulation engine statistics."""
        total = self._approvals + self._rejections
        return {
            "simulations_run": self._simulations_run,
            "approvals": self._approvals,
            "rejections": self._rejections,
            "approval_rate": round(self._approvals / max(1, total), 3),
        }
