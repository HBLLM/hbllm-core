"""Working Memory Models, Schemas, and Attractor Slots."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.graph import ConceptNode
from hbllm.hcir.skills.declarative import (
    SpatialActionIntent,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemorySalienceState:
    """Snapshot of working memory salient counts to detect phase shift events."""

    calibrated_actions: int = 0
    goal_positions: int = 0
    barriers: int = 0
    trajectories: int = 0
    mobile_threats: int = 0


@dataclass
class HeldItem:
    """An entity retained in latent working memory as carried / possessed by the agent."""

    feature_id: int
    role: str
    acquired_step: int
    original_position: tuple[int, int]
    properties: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export held item state as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="held_item",
            operator="==",
            value=self.feature_id,
            relation=self.role,
            target_entity="avatar",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR state updates reflecting held item permanence."""
        node_id = f"held_item_{self.feature_id}"
        node = ConceptNode(
            id=node_id,
            label=f"held_item_{self.feature_id}",
            definition=f"Possessed tool or entity with feature {self.feature_id}",
            domain="body_schema",
            properties={
                "feature_id": self.feature_id,
                "role": self.role,
                "acquired_step": self.acquired_step,
                "original_position": list(self.original_position),
                "properties": dict(self.properties),
            },
        )
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node_id,
                node_data=node.model_dump(),
            )
        ]


@dataclass
class HomeostaticState:
    """Homeostatic state regulating intrinsic motivation and goal arbitration (Gap 5, WP10).

    Biologically grounded in allostatic regulation and active inference:
    - energy_budget: Available computational / physical energy (0.0 to 100.0).
      Depletion forces low-cost, safety-first policies.
    - safety_margin: Minimum acceptable safety margin / clearance distance (0.0 to 1.0).
    - competence_progress: Rate of prediction error reduction / learning progress d/dt(Error).
    - social_instructions: Explicit externally-commanded goals or directives that preempt open-ended curiosity.
    - fatigue: Accumulated cognitive fatigue decaying after rest or idle consolidation.
    """

    energy_budget: float = 100.0
    safety_margin: float = 1.0
    competence_progress: float = 0.0
    social_instructions: list[dict[str, Any]] = field(default_factory=list)
    fatigue: float = 0.0

    def consume_energy(self, amount: float) -> float:
        """Consume energy budget and accumulate fatigue."""
        self.energy_budget = max(0.0, float(self.energy_budget - amount))
        self.fatigue = min(100.0, float(self.fatigue + amount * 0.5))
        return self.energy_budget

    def replenish_energy(self, amount: float) -> float:
        """Replenish energy budget and alleviate fatigue."""
        self.energy_budget = min(100.0, float(self.energy_budget + amount))
        self.fatigue = max(0.0, float(self.fatigue - amount))
        return self.energy_budget

    def record_progress(self, progress: float) -> None:
        """Record competence progress signal (d/dt of prediction accuracy or error)."""
        self.competence_progress = float(progress)

    def add_instruction(self, instruction: dict[str, Any]) -> None:
        """Add an explicit social or environmental instruction."""
        self.social_instructions.append(dict(instruction))

    def clear_instructions(self) -> None:
        """Clear all social instructions upon fulfillment or cancellation."""
        self.social_instructions.clear()

    @property
    def is_critical_energy(self) -> bool:
        """True when energy is severely depleted (<= 25.0), requiring immediate risk-averse preservation."""
        return self.energy_budget <= 25.0

    def to_causal_predicates(self) -> list[CausalPredicate]:
        """Export homeostatic variables as declarative CausalPredicates."""
        return [
            CausalPredicate(
                variable="homeostatic:energy_budget",
                operator="==",
                value=round(self.energy_budget, 2),
                relation="allostatic_regulation",
                target_entity="prefrontal_working_memory",
            ),
            CausalPredicate(
                variable="homeostatic:competence_progress",
                operator="==",
                value=round(self.competence_progress, 3),
                relation="learning_progress",
                target_entity="prefrontal_working_memory",
            ),
            CausalPredicate(
                variable="homeostatic:social_instructions_count",
                operator="==",
                value=len(self.social_instructions),
                relation="instructed_goals",
                target_entity="prefrontal_working_memory",
            ),
        ]

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR updates reflecting homeostatic state."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="homeostatic_state",
                node_data={
                    "energy_budget": float(self.energy_budget),
                    "safety_margin": float(self.safety_margin),
                    "competence_progress": float(self.competence_progress),
                    "social_instructions_count": len(self.social_instructions),
                    "fatigue": float(self.fatigue),
                    "is_critical_energy": self.is_critical_energy,
                },
            )
        ]


@dataclass
class DomainCompetenceProfile:
    """Metacognitive competence profile per domain (G9).

    Tracks calibration, win rate, and learning progress to answer:
    'What am I bad at, and is practicing it worth it?'
    """

    domain_id: str
    total_episodes: int = 0
    successful_episodes: int = 0
    running_win_rate: float = 0.0
    mean_prediction_error: float = 0.50
    error_history: list[float] = field(default_factory=list)
    calibration_score: float = 0.50
    learning_progress_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def record_episode(
        self,
        success: bool,
        mean_error: float = 0.20,
        predicted_confidence: float = 0.80,
    ) -> None:
        self.total_episodes += 1
        if success:
            self.successful_episodes += 1
        self.running_win_rate = self.successful_episodes / self.total_episodes

        clamped_err = max(0.0, min(1.0, float(mean_error)))
        self.error_history.append(clamped_err)
        if len(self.error_history) > 30:
            self.error_history.pop(0)

        alpha = 0.25
        self.mean_prediction_error = (
            1.0 - alpha
        ) * self.mean_prediction_error + alpha * clamped_err

        if len(self.error_history) >= 4:
            split = len(self.error_history) // 2
            older_mean = float(np.mean(self.error_history[:split]))
            recent_mean = float(np.mean(self.error_history[split:]))
            self.learning_progress_rate = max(-1.0, min(1.0, older_mean - recent_mean))

        outcome_val = 1.0 if success else 0.0
        brier = (float(predicted_confidence) - outcome_val) ** 2
        self.calibration_score = max(0.0, min(1.0, 1.0 - brier))
        self.last_updated = time.time()

    @property
    def practice_priority(self) -> float:
        """Computes Zone of Proximal Development practice priority in [0.0, 1.0].

        High priority when:
        1. Not yet fully mastered (1.0 - win_rate is high)
        2. Capable of learning (learning_progress_rate is positive or non-negative)
        3. Well-calibrated (not delusionally overconfident on impossible tasks)
        """
        if self.running_win_rate >= 0.95 and self.total_episodes >= 5:
            return 0.05

        unmastered_factor = 1.0 - self.running_win_rate
        learnability = max(0.10, min(1.0, 0.40 + self.learning_progress_rate * 1.5))
        priority = unmastered_factor * learnability * max(0.2, self.calibration_score)
        return float(max(0.0, min(1.0, priority)))

    def to_causal_predicates(self) -> list[CausalPredicate]:
        return [
            CausalPredicate(
                variable=f"domain_competence:{self.domain_id}",
                operator="==",
                value=f"win_rate:{self.running_win_rate:.2f}:priority:{self.practice_priority:.2f}",
                relation="domain_metacognition",
                target_entity=self.domain_id,
            )
        ]

    def to_transaction_operations(self) -> list[TransactionOperation]:
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"domain_competence_{self.domain_id}",
                node_data={
                    "domain_id": self.domain_id,
                    "total_episodes": self.total_episodes,
                    "successful_episodes": self.successful_episodes,
                    "running_win_rate": self.running_win_rate,
                    "practice_priority": self.practice_priority,
                    "learning_progress_rate": self.learning_progress_rate,
                    "calibration_score": self.calibration_score,
                },
            )
        ]


@dataclass
class SubgoalStage:
    """A discrete stage in a hierarchical prefrontal behavioral schema."""

    stage_id: str
    target_role: str
    target_positions: set[tuple[int, int]]
    required_feature: int | None = None
    action_on_arrival: int | None = None  # None = CONTACT, or explicit action like 5
    is_completed: bool = False

    def to_symbolic_subgoal(self) -> SymbolicSubgoal:
        """Lift this stage to a declarative SymbolicSubgoal."""
        intent = (
            SpatialActionIntent.INTERACT
            if self.stage_id == "UNLOCK_BARRIER"
            else SpatialActionIntent.NAVIGATE
        )
        return SymbolicSubgoal(
            intent=intent,
            target_query={
                "stage_id": self.stage_id,
                "role": self.target_role,
                "target_positions": list(self.target_positions),
                "required_feature": self.required_feature,
            },
        )


@dataclass
class SubgoalSchema:
    """An active hierarchical plan composed of sequential stages."""

    schema_name: str
    stages: list[SubgoalStage]
    current_stage_idx: int = 0
    failure_count: int = 0
    confidence: float = 1.0

    @property
    def current_stage(self) -> SubgoalStage | None:
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    def advance(self) -> SubgoalStage | None:
        if self.current_stage:
            self.current_stage.is_completed = True
        self.current_stage_idx += 1
        return self.current_stage

    @property
    def is_finished(self) -> bool:
        return self.current_stage_idx >= len(self.stages)

    def to_subgoal_sequence(self) -> SubgoalSequence:
        """Convert all stages into a declarative HCIR SubgoalSequence."""
        return SubgoalSequence(*[stage.to_symbolic_subgoal() for stage in self.stages])

    def is_current_stage_satisfied(
        self,
        avatar_pos: tuple[int, int] | None,
        held_items: Sequence[HeldItem] | None = None,
        unlocked_barriers: set[tuple[int, int]] | None = None,
    ) -> bool:
        """Check if current stage target is satisfied by current epistemic state."""
        st = self.current_stage
        if st is None:
            return True
        if avatar_pos is not None and avatar_pos in st.target_positions:
            return True
        if st.stage_id == "ACQUIRE_TOOL" and held_items:
            if st.required_feature is None or any(
                h.feature_id == st.required_feature for h in held_items
            ):
                return True
        if st.stage_id == "UNLOCK_BARRIER" and unlocked_barriers:
            if any(p in unlocked_barriers for p in st.target_positions):
                return True
        return False
