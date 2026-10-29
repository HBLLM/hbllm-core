from __future__ import annotations

"""Anterior Insular Cortex (AIC) Interoceptive Allostasis & Energy Budgeting.

Biologically grounded in:
1. A.D. Craig (2002) - Interoception and the feeling of what happens.
2. P. Sterling (2012) - Allostasis: A model of predictive regulation.
3. L.F. Barrett (2017) - Interoceptive predictive coding and autonomic budgeting.

The Anterior Insula continuously monitors internal energetic resources:
- Tracks visceral metabolic reserves E(t) ∈ [0, 100], compute load, and allostatic debt.
- Regulates cognitive mode:
  * High Energy (E >= 60): Deliberative, deep multi-step counterfactual rollouts (OFC/MCTS).
  * Moderate Energy (30 <= E < 60): Pragmatic, shallow planning and heuristic execution.
  * Low Energy (E < 30): Shift to low-cost habitual skills, procedural macros, or quiescence.
- Imposes allostatic costs on candidate actions to prevent computational and behavioral burnout.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate


class AllostaticRegime(StrEnum):
    """Metabolic operational regimes determined by interoceptive allostasis."""

    OPTIMAL = "OPTIMAL"  # E >= 60: Deep mental simulation permitted
    FATIGUED = "FATIGUED"  # 30 <= E < 60: Shallow planning, energy-aware pruning
    EXHAUSTED = "EXHAUSTED"  # E < 30: Deliberation inhibited; habit/macro only
    RECHARGING = "RECHARGING"  # Quiescent restoration phase


@dataclass
class InteroceptiveState:
    """Snapshot of visceral energetic reserves and allostatic burden."""

    energy_level: float = 100.0
    max_energy: float = 100.0
    allostatic_load: float = 0.0
    fatigue_index: float = 0.0
    regime: AllostaticRegime = AllostaticRegime.OPTIMAL
    deliberation_multiplier: float = 1.0
    is_exhausted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export interoceptive state as a declarative causal predicate."""
        return CausalPredicate(
            variable="allostasis:interoceptive_state",
            operator="==",
            value=self.regime.value,
            relation=f"energy:{self.energy_level:.1f}:load:{self.allostatic_load:.2f}:exhausted:{self.is_exhausted}",
            target_entity="anterior_insula",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording allostatic interoceptive state."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="interoceptive_state",
                node_data={
                    "energy_level": float(self.energy_level),
                    "max_energy": float(self.max_energy),
                    "allostatic_load": float(self.allostatic_load),
                    "fatigue_index": float(self.fatigue_index),
                    "regime": self.regime.value,
                    "deliberation_multiplier": float(self.deliberation_multiplier),
                    "is_exhausted": self.is_exhausted,
                },
            )
        ]


class InteroceptiveAllostasisCortex:
    """Anterior Insular Cortex regulating energy reserves, cognitive fatigue, and allostatic shifts."""

    # Standard energetic costs per cognitive activity
    ACTIVITY_COSTS = {
        "motor_step": 1.0,
        "perception_cycle": 0.5,
        "mental_simulation": 5.0,
        "counterfactual_rollout": 4.0,
        "skill_induction": 8.0,
        "epistemic_probe": 2.5,
        "macro_execution": 0.5,  # Cached procedural habits are metabolically cheap
    }

    def __init__(
        self,
        max_energy: float = 100.0,
        initial_energy: float = 100.0,
        basal_metabolism: float = 0.1,
        exhaustion_threshold: float = 30.0,
        fatigue_recovery_rate: float = 2.0,
    ) -> None:
        self.max_energy = max_energy
        self.current_energy = min(max_energy, max(0.0, initial_energy))
        self.basal_metabolism = basal_metabolism
        self.exhaustion_threshold = exhaustion_threshold
        self.fatigue_recovery_rate = fatigue_recovery_rate
        self.allostatic_load = 0.0

    def get_state(self) -> InteroceptiveState:
        """Compute the current interoceptive state snapshot."""
        fatigue = 1.0 - (self.current_energy / self.max_energy)
        is_exhausted = self.current_energy < self.exhaustion_threshold

        if is_exhausted:
            regime = AllostaticRegime.EXHAUSTED
            multiplier = 0.2  # Sharply penalize expensive deliberation
        elif self.current_energy < 60.0:
            regime = AllostaticRegime.FATIGUED
            multiplier = 0.6
        else:
            regime = AllostaticRegime.OPTIMAL
            multiplier = 1.0

        return InteroceptiveState(
            energy_level=self.current_energy,
            max_energy=self.max_energy,
            allostatic_load=self.allostatic_load,
            fatigue_index=fatigue,
            regime=regime,
            deliberation_multiplier=multiplier,
            is_exhausted=is_exhausted,
        )

    def consume_energy(self, activity_type: str, scale: float = 1.0) -> InteroceptiveState:
        """Deduct metabolic energy based on the cognitive activity performed."""
        base_cost = self.ACTIVITY_COSTS.get(activity_type, 1.0)
        cost = (base_cost * scale) + self.basal_metabolism

        self.current_energy = max(0.0, self.current_energy - cost)
        # Allostatic debt accumulates faster under heavy fatigue
        if self.current_energy < self.exhaustion_threshold:
            self.allostatic_load += cost * 0.5
        else:
            self.allostatic_load = max(0.0, self.allostatic_load - 0.05)

        return self.get_state()

    def replenish_energy(self, amount: float) -> InteroceptiveState:
        """Replenish energy through quiescent resting, goal achievement, or level advance."""
        self.current_energy = min(self.max_energy, self.current_energy + amount)
        self.allostatic_load = max(0.0, self.allostatic_load - amount * 0.25)
        return self.get_state()

    def should_permit_deliberation(self, estimated_cost: float = 5.0) -> tuple[bool, str]:
        """Check whether interoceptive state allows expensive forward mental simulation."""
        if self.current_energy < self.exhaustion_threshold:
            return (
                False,
                f"Insular inhibition: energy {self.current_energy:.1f} < threshold {self.exhaustion_threshold:.1f}. Use habit/macro.",
            )
        if self.current_energy - estimated_cost < 10.0:
            return (
                False,
                f"Insular caution: simulation cost {estimated_cost:.1f} threatens critical exhaustion.",
            )
        return True, "Deliberation permitted by interoceptive allostasis."

    def apply_allostatic_cost_penalty(
        self,
        candidate_utilities: dict[Any, float],
        is_deliberative_action: bool = True,
    ) -> dict[Any, float]:
        """Scale action utilities by metabolic affordance."""
        state = self.get_state()
        adjusted: dict[Any, float] = {}
        for action, utility in candidate_utilities.items():
            if is_deliberative_action and state.is_exhausted:
                # Deliberative paths penalize utility under exhaustion
                adjusted[action] = utility * state.deliberation_multiplier
            else:
                adjusted[action] = utility
        return adjusted
