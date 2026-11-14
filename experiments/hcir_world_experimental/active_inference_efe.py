from __future__ import annotations

"""Active Inference Expected Free Energy (EFE) Policy Evaluator.

Biologically grounded in Karl Friston's Free Energy Principle (FEP):
Evaluates candidate actions/policies by decomposing Expected Free Energy G(π) into:
1. Pragmatic Value (Extrinsic Preference):
   - Maximizes alignment with desired goal distributions (e.g. goal proximity, positive reward).
2. Epistemic Value (Intrinsic Information Gain):
   - Maximizes resolution of ambiguity and Shannon surprise reduction:
     - Novelty: inverse visitation density 1 / (1 + count(s'))
     - Frontier Gain: unmapped lattice contours within sensory horizon
     - Ambiguity: unconfirmed causal hypothesis entropy
3. Dynamic Precision Schedule β(t):
   - Dynamically balances exploration vs. exploitation based on world theory uncertainty:
     High uncertainty / unmapped goals -> β(t) >> 1 (epistemic curiosity dominates).
     Low uncertainty / confirmed goal -> β(t) -> 0.05 (pragmatic goal convergence dominates).
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate


@dataclass
class ExpectedFreeEnergyScore:
    """Decomposition of Expected Free Energy for a candidate action."""

    action: Any
    target_pos: tuple[int, int]
    pragmatic_value: float
    epistemic_value: float
    beta_curiosity: float
    total_expected_value: float
    expected_free_energy: float
    information_gain_components: dict[str, float] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export EFE score as a declarative causal predicate."""
        return CausalPredicate(
            variable="active_inference:efe",
            operator="==",
            value=f"efe:{self.expected_free_energy:.3f}",
            relation=f"pragmatic:{self.pragmatic_value:.3f}:epistemic:{self.epistemic_value:.3f}:beta:{self.beta_curiosity:.2f}",
            target_entity=str(self.action),
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording EFE evaluation."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"efe_eval_{self.action}",
                node_data={
                    "action": str(self.action),
                    "target_pos": list(self.target_pos),
                    "pragmatic_value": float(self.pragmatic_value),
                    "epistemic_value": float(self.epistemic_value),
                    "beta_curiosity": float(self.beta_curiosity),
                    "total_expected_value": float(self.total_expected_value),
                    "expected_free_energy": float(self.expected_free_energy),
                    "information_gain_components": self.information_gain_components,
                },
            )
        ]


class ExpectedFreeEnergyEvaluator:
    """Evaluates candidate actions using Karl Friston's Expected Free Energy decomposition."""

    def __init__(
        self,
        beta_base: float = 1.0,
        novelty_weight: float = 0.5,
        frontier_weight: float = 0.35,
        ambiguity_weight: float = 0.15,
    ) -> None:
        self.beta_base = beta_base
        self.novelty_weight = novelty_weight
        self.frontier_weight = frontier_weight
        self.ambiguity_weight = ambiguity_weight

    def compute_dynamic_beta(
        self,
        world_theory_uncertainty: float = 0.5,
        goal_confidence: float = 0.0,
    ) -> float:
        """Compute active inference dynamic precision schedule β(t).

        High uncertainty / unmapped goals -> high β (epistemic exploration).
        Low uncertainty / confirmed goal -> low β (pragmatic exploitation).
        """
        uncertainty = float(np.clip(world_theory_uncertainty, 0.0, 1.0))
        confidence = float(np.clip(goal_confidence, 0.0, 1.0))
        numerator = 1.0 + uncertainty
        denominator = 1.0 + 3.0 * confidence
        return float(self.beta_base * (numerator / denominator))

    def evaluate_action(
        self,
        action: Any,
        current_pos: tuple[int, int] | None,
        target_pos: tuple[int, int],
        grid: np.ndarray,
        goal_positions: set[tuple[int, int]] | None = None,
        position_visit_counts: dict[tuple[int, int], int] | None = None,
        unconfirmed_hypotheses_count: int = 0,
        goal_confidence: float = 0.0,
        world_theory_uncertainty: float = 0.5,
    ) -> ExpectedFreeEnergyScore:
        """Compute full pragmatic and epistemic value components for a candidate transition."""
        H, W = grid.shape
        max_diag = math.sqrt(H * H + W * W) or 1.0

        # 1. Pragmatic Value: alignment with desired goal states
        pragmatic_val = 0.0
        goals = goal_positions or set()
        if goals and current_pos is not None:
            min_target_dist = min(
                math.sqrt((target_pos[0] - g[0]) ** 2 + (target_pos[1] - g[1]) ** 2) for g in goals
            )
            min_curr_dist = min(
                math.sqrt((current_pos[0] - g[0]) ** 2 + (current_pos[1] - g[1]) ** 2)
                for g in goals
            )
            # Distance reduction + normalized proximity
            progress = max(-1.0, min(1.0, min_curr_dist - min_target_dist))
            proximity = 1.0 - (min_target_dist / max_diag)
            pragmatic_val = 0.6 * proximity + 0.4 * progress
        elif goals:
            min_dist = min(
                math.sqrt((target_pos[0] - g[0]) ** 2 + (target_pos[1] - g[1]) ** 2) for g in goals
            )
            pragmatic_val = max(0.0, 1.0 - (min_dist / max_diag))

        # 2. Epistemic Value: Information Gain / Ambiguity Reduction
        visits = (position_visit_counts or {}).get(target_pos, 0)
        novelty = 1.0 / (1.0 + math.log1p(float(visits)))

        # Frontier Gain: count unvisited/unexplored immediate neighbors around target
        frontier_cells = 0
        r, c = target_pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if (position_visit_counts or {}).get((nr, nc), 0) == 0:
                    frontier_cells += 1
        frontier_gain = frontier_cells / 4.0

        # Ambiguity reduction from unconfirmed causal hypotheses or world theory uncertainty
        ambiguity_val = max(
            min(1.0, unconfirmed_hypotheses_count / 10.0),
            float(world_theory_uncertainty),
        )

        epistemic_val = (
            self.novelty_weight * novelty
            + self.frontier_weight * frontier_gain
            + self.ambiguity_weight * ambiguity_val
        )

        beta = self.compute_dynamic_beta(
            world_theory_uncertainty=world_theory_uncertainty,
            goal_confidence=goal_confidence,
        )

        total_val = pragmatic_val + beta * epistemic_val
        expected_free_energy = -total_val

        components = {
            "novelty": novelty,
            "frontier_gain": frontier_gain,
            "ambiguity": ambiguity_val,
        }

        return ExpectedFreeEnergyScore(
            action=action,
            target_pos=target_pos,
            pragmatic_value=pragmatic_val,
            epistemic_value=epistemic_val,
            beta_curiosity=beta,
            total_expected_value=total_val,
            expected_free_energy=expected_free_energy,
            information_gain_components=components,
        )

    def rank_candidate_actions(
        self,
        candidate_transitions: dict[Any, tuple[int, int]],
        current_pos: tuple[int, int] | None,
        grid: np.ndarray,
        goal_positions: set[tuple[int, int]] | None = None,
        position_visit_counts: dict[tuple[int, int], int] | None = None,
        unconfirmed_hypotheses_count: int = 0,
        goal_confidence: float = 0.0,
        world_theory_uncertainty: float = 0.5,
    ) -> list[ExpectedFreeEnergyScore]:
        """Rank all candidate actions by Expected Free Energy (highest total value first)."""
        scores: list[ExpectedFreeEnergyScore] = []
        for action, target_pos in candidate_transitions.items():
            score = self.evaluate_action(
                action=action,
                current_pos=current_pos,
                target_pos=target_pos,
                grid=grid,
                goal_positions=goal_positions,
                position_visit_counts=position_visit_counts,
                unconfirmed_hypotheses_count=unconfirmed_hypotheses_count,
                goal_confidence=goal_confidence,
                world_theory_uncertainty=world_theory_uncertainty,
            )
            scores.append(score)

        scores.sort(key=lambda s: s.total_expected_value, reverse=True)
        return scores
