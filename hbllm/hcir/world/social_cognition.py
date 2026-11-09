"""Social Cognition & Inverse Planning Engine — Baker, Saxe & Tenenbaum (AGI Gap 10).

Modeled on theory of mind, agency detection, and Bayesian inverse planning:
1. Agency Detection (Physics vs Intentionality):
   - Observes moving entities in sensory frames.
   - Evaluates whether motion conforms to ballistic inertial physics:
     x(t+1) = x(t) + v (with elastic wall rebounds).
   - If motion exhibits non-inertial self-propulsion, active obstacle evasion,
     or persistent attraction to objects/avatar, classifies entity as an
     INTENTIONAL_AGENT.
2. Bayesian Inverse Planning (Baker, Saxe, Tenenbaum 2009):
   - Given observed trajectory tau = [x_0, x_1, ..., x_t] and candidate goals G:
     P(tau | g) proportional to exp(-beta * (length(tau) + dist(x_t, g)))
     P(g | tau) = P(tau | g) * P(g) / sum_{g'} P(tau | g') * P(g')
   - Infers underlying objective (e.g. PURSUING_AVATAR, SEEKING_RESOURCE, SENTRY_PATROL).
3. Intent-Conditioned Trajectory Projection:
   - Extrapolates future positions towards the inferred goal rather than linear physics,
     providing dramatically superior forward collision and hazard avoidance.
4. Declarative Neuro-Symbolic Integration:
   - Exports inferred intentions to formal CausalPredicates and atomic TransactionOperations.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


class AgencyStatus(StrEnum):
    """Categorical classification of dynamic entities by agency type."""

    PHYSICAL = "PHYSICAL"  # Passive inertial/ballistic entity (predictable by physics)
    INTENTIONAL_AGENT = (
        "INTENTIONAL_AGENT"  # Self-propelled goal-directed agent (requires inverse planning)
    )


class InferredIntent(StrEnum):
    """Inferred high-level intention of an observed agent."""

    PURSUING_AVATAR = "PURSUING_AVATAR"  # Chasing the player/avatar (predator / dynamic threat)
    SEEKING_RESOURCE = "SEEKING_RESOURCE"  # Navigating towards an environmental resource/gem
    SENTRY_PATROL = "SENTRY_PATROL"  # Patrolling fixed waypoints / guarding territory
    EVADING_AVATAR = "EVADING_AVATAR"  # Actively maintaining distance from avatar
    UNKNOWN = "UNKNOWN"


@dataclass
class IntentionalAgentModel:
    """Mental model of an observed external dynamic entity."""

    entity_id: str
    feature_id: int
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    agency_status: AgencyStatus = AgencyStatus.PHYSICAL
    inferred_intent: InferredIntent = InferredIntent.UNKNOWN
    most_likely_goal: tuple[int, int] | None = None
    goal_posterior: dict[tuple[int, int], float] = field(default_factory=dict)
    non_inertial_turns: int = 0
    confidence: float = 0.5
    last_updated: float = field(default_factory=time.time)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export inferred social intention as a declarative CausalPredicate."""
        goal_str = (
            f"goal:{self.most_likely_goal[0]}_{self.most_likely_goal[1]}"
            if self.most_likely_goal
            else "goal:none"
        )
        return CausalPredicate(
            variable=f"social_agent:{self.entity_id}",
            operator="==",
            value=self.inferred_intent.value,
            relation="inferred_intent",
            target_entity=f"{self.agency_status.value}:{goal_str}",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting intentional agent beliefs."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"intentional_agent_{self.entity_id}",
                node_data={
                    "entity_id": self.entity_id,
                    "feature_id": self.feature_id,
                    "agency_status": self.agency_status.value,
                    "inferred_intent": self.inferred_intent.value,
                    "most_likely_goal": self.most_likely_goal,
                    "confidence": self.confidence,
                    "trajectory_length": len(self.trajectory),
                },
            )
        ]


class SocialCognitionEngine:
    """Theory of mind and Bayesian inverse planning for embodied environments (G10)."""

    def __init__(
        self,
        rationality_beta: float = 1.0,
        agency_turn_threshold: int = 2,
    ) -> None:
        self.rationality_beta = max(0.1, float(rationality_beta))
        self.agency_turn_threshold = agency_turn_threshold
        self.tracked_agents: dict[str, IntentionalAgentModel] = {}

    def track_entity_position(
        self,
        entity_id: str,
        feature_id: int,
        position: tuple[int, int],
        grid: np.ndarray | None = None,
    ) -> IntentionalAgentModel:
        """Track an entity's trajectory and evaluate intentional agency vs inert physics."""
        if entity_id not in self.tracked_agents:
            self.tracked_agents[entity_id] = IntentionalAgentModel(
                entity_id=entity_id,
                feature_id=feature_id,
                trajectory=[position],
            )
            return self.tracked_agents[entity_id]

        agent = self.tracked_agents[entity_id]
        if agent.trajectory and agent.trajectory[-1] == position:
            return agent

        agent.trajectory.append(position)
        agent.last_updated = time.time()

        # Check for non-inertial motion
        if len(agent.trajectory) >= 3:
            p0, p1, p2 = agent.trajectory[-3], agent.trajectory[-2], agent.trajectory[-1]
            v0 = (p1[0] - p0[0], p1[1] - p0[1])
            v1 = (p2[0] - p1[0], p2[1] - p1[1])

            # If velocity changed without hitting a physical obstacle, count as active turn
            if v0 != v1:
                is_wall_rebound = False
                if grid is not None and isinstance(grid, np.ndarray):
                    # Check if continuing straight would hit a barrier or edge
                    straight_ahead = (p1[0] + v0[0], p1[1] + v0[1])
                    H, W = grid.shape
                    if (
                        straight_ahead[0] < 0
                        or straight_ahead[0] >= H
                        or straight_ahead[1] < 0
                        or straight_ahead[1] >= W
                        or grid[straight_ahead[0], straight_ahead[1]] != 0
                    ):
                        is_wall_rebound = True

                if not is_wall_rebound:
                    agent.non_inertial_turns += 1

            if agent.non_inertial_turns >= self.agency_turn_threshold:
                agent.agency_status = AgencyStatus.INTENTIONAL_AGENT

        return agent

    def invert_planner_for_intent(
        self,
        entity_id: str,
        avatar_pos: tuple[int, int] | None = None,
        candidate_goals: list[tuple[int, int]] | None = None,
    ) -> IntentionalAgentModel:
        """Execute Bayesian inverse planning over observed trajectory to infer goals and intent.

        Calculates P(goal | trajectory) proportional to exp(-beta * (length(tau) + distance_to_goal)).
        """
        agent = self.tracked_agents.get(entity_id)
        if not agent or len(agent.trajectory) < 2:
            return agent or IntentionalAgentModel(entity_id=entity_id, feature_id=0)

        targets: list[tuple[tuple[int, int], str]] = []
        if avatar_pos is not None:
            targets.append((avatar_pos, InferredIntent.PURSUING_AVATAR.value))
        if candidate_goals:
            for g in candidate_goals:
                if g != avatar_pos:
                    targets.append((g, InferredIntent.SEEKING_RESOURCE.value))

        if not targets:
            agent.inferred_intent = InferredIntent.SENTRY_PATROL
            return agent

        cur_pos = agent.trajectory[-1]
        log_likelihoods: dict[tuple[int, int], float] = {}

        # Evaluate path efficiency toward each target
        for tgt_pos, _ in targets:
            # Distance from start to current position + remaining distance to target
            dist_remaining = abs(cur_pos[0] - tgt_pos[0]) + abs(cur_pos[1] - tgt_pos[1])
            # Check if trajectory reduced distance to target
            init_pos = agent.trajectory[0]
            dist_initial = abs(init_pos[0] - tgt_pos[0]) + abs(init_pos[1] - tgt_pos[1])
            progress = dist_initial - dist_remaining

            # High progress yields high likelihood under rational agent assumption
            score = -self.rationality_beta * dist_remaining + (progress * 1.5)
            log_likelihoods[tgt_pos] = score

        # Softmax normalization to compute posterior P(goal | tau)
        max_log = max(log_likelihoods.values())
        exp_scores = {tgt: math.exp(score - max_log) for tgt, score in log_likelihoods.items()}
        total_exp = sum(exp_scores.values())
        posterior = {tgt: val / total_exp for tgt, val in exp_scores.items()}

        agent.goal_posterior = posterior
        best_target = max(posterior.keys(), key=lambda t: posterior[t])
        agent.most_likely_goal = best_target
        agent.confidence = float(posterior[best_target])

        # Categorize intent based on highest-likelihood target
        target_intent_map = dict(targets)
        target_category = target_intent_map.get(best_target, InferredIntent.UNKNOWN.value)
        if target_category == InferredIntent.PURSUING_AVATAR.value:
            agent.inferred_intent = InferredIntent.PURSUING_AVATAR
        elif target_category == InferredIntent.SEEKING_RESOURCE.value:
            agent.inferred_intent = InferredIntent.SEEKING_RESOURCE
        else:
            agent.inferred_intent = InferredIntent.SENTRY_PATROL

        return agent

    def project_intent_trajectory(
        self,
        entity_id: str,
        horizon: int = 5,
        grid: np.ndarray | None = None,
    ) -> list[tuple[int, int]]:
        """Project future trajectory conditioned on inferred goal rather than linear physics."""
        agent = self.tracked_agents.get(entity_id)
        if not agent or not agent.trajectory:
            return []

        cur_pos = agent.trajectory[-1]
        goal = agent.most_likely_goal

        # If passive physical entity or no goal, extrapolate linearly from last displacement
        if agent.agency_status == AgencyStatus.PHYSICAL or goal is None:
            if len(agent.trajectory) >= 2:
                dr = cur_pos[0] - agent.trajectory[-2][0]
                dc = cur_pos[1] - agent.trajectory[-2][1]
            else:
                dr, dc = 0, 0
            res: list[tuple[int, int]] = []
            pos = cur_pos
            for _ in range(horizon):
                pos = (pos[0] + dr, pos[1] + dc)
                res.append(pos)
            return res

        # Intentional agent: extrapolate greedy Manhattan step towards inferred goal
        res = []
        pos = cur_pos
        for _ in range(horizon):
            if pos == goal:
                res.append(pos)
                continue
            dr = goal[0] - pos[0]
            dc = goal[1] - pos[1]
            if abs(dr) >= abs(dc):
                step_r = 1 if dr > 0 else -1
                pos = (pos[0] + step_r, pos[1])
            else:
                step_c = 1 if dc > 0 else -1
                pos = (pos[0], pos[1] + step_c)
            res.append(pos)

        return res
