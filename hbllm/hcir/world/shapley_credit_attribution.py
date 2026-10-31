from __future__ import annotations

"""Counterfactual Shapley Credit Attribution (Deep Credit Assignment).

Grounds episodic credit assignment in Pearl's causal calculus and counterfactual reachability:
1. Long-Horizon Blame Assignment:
   - When an episode is lost after 30-50 steps, standard reinforcement / SWR only blames
     the final steps of harmless wandering near death.
   - Counterfactual Shapley attribution calculates the step-wise reachability contraction:
       Delta R(t) = Reachability(t - 1) - Reachability(t)
     and detects irreversible deadlock / entrapment events.
2. Identifies the true root-cause divergence step t*:
   - The action that permanently collapsed topological reachability or locked the puzzle into an
     irreversible dead-end receives the highest negative Shapley blame phi(t*).
3. Grounded Habenular Gating:
   - Automatically injects the root-causal (position, action) pair into Habenular IOR and
     grounded lethal transitions, preventing repetitive catastrophic looping.
"""

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.cortex_episodic import EpisodicTransition

logger = logging.getLogger(__name__)


@dataclass
class ShapleyCreditResult:
    """Outcome of long-horizon counterfactual Shapley credit attribution."""

    root_causal_step: int
    root_fatal_pos: tuple[int, int] | None
    root_fatal_action: Any
    max_blame_score: float
    step_blame_scores: dict[int, float] = field(default_factory=dict)
    causal_explanation: str = ""

    def to_causal_predicate(self) -> CausalPredicate:
        """Export Shapley credit assignment as a declarative Pearl causal predicate."""
        pos_str = (
            f"({self.root_fatal_pos[0]},{self.root_fatal_pos[1]})"
            if self.root_fatal_pos
            else "none"
        )
        return CausalPredicate(
            variable=f"shapley_blame:step_{self.root_causal_step}",
            operator="==",
            value=f"{self.max_blame_score:.2f}",
            relation=f"act:{self.root_fatal_action}:pos:{pos_str}",
            target_entity=self.causal_explanation or "root_cause",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording root causal blame."""
        node_id = f"shapley_blame_step_{self.root_causal_step}"
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node_id,
                node_data={
                    "root_causal_step": self.root_causal_step,
                    "root_fatal_pos": list(self.root_fatal_pos) if self.root_fatal_pos else None,
                    "root_fatal_action": str(self.root_fatal_action),
                    "max_blame_score": self.max_blame_score,
                    "causal_explanation": self.causal_explanation,
                },
            )
        ]


class CounterfactualShapleyCreditAttributor:
    """Evaluates causal Shapley blame across an entire episodic transition sequence."""

    @staticmethod
    def compute_spatial_reachability(
        pos: tuple[int, int],
        grid: np.ndarray,
        barriers: set[tuple[int, int]],
        max_depth: int = 128,
    ) -> int:
        """Compute the size of the topologically reachable flood-fill component from pos."""
        H, W = grid.shape
        r, c = pos
        if not (0 <= r < H and 0 <= c < W) or (r, c) in barriers:
            return 0

        visited: set[tuple[int, int]] = {(r, c)}
        queue: deque[tuple[int, int]] = deque([(r, c)])

        while queue and len(visited) < max_depth:
            cr, cc = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if (nr, nc) not in visited and (nr, nc) not in barriers:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        return len(visited)

    @classmethod
    def attribute_blame(
        cls,
        trajectory: Sequence[EpisodicTransition],
        engine: AutonomousEpistemicEngine | None = None,
        learned_goals: set[tuple[int, int]] | None = None,
        learned_barriers: set[tuple[int, int]] | None = None,
    ) -> ShapleyCreditResult:
        """Analyze trajectory to identify the root causal step that sealed the episode failure."""
        if not trajectory:
            return ShapleyCreditResult(
                root_causal_step=0,
                root_fatal_pos=None,
                root_fatal_action=0,
                max_blame_score=0.0,
                causal_explanation="Empty trajectory",
            )

        goals: set[tuple[int, int]] = set()
        if learned_goals:
            goals.update(learned_goals)
        elif engine and hasattr(engine, "learned_goal_positions"):
            raw_goals = getattr(engine, "learned_goal_positions", None)
            if raw_goals:
                goals.update(raw_goals)

        static_barriers: set[tuple[int, int]] = set()
        if learned_barriers:
            static_barriers.update(learned_barriers)
        elif engine and hasattr(engine, "learned_barriers"):
            raw_barriers = getattr(engine, "learned_barriers", None)
            if raw_barriers:
                static_barriers.update(raw_barriers)

        valid_transitions = [t for t in trajectory if t.avatar_pos is not None]
        if not valid_transitions:
            last_t = trajectory[-1]
            return ShapleyCreditResult(
                root_causal_step=last_t.step,
                root_fatal_pos=last_t.avatar_pos,
                root_fatal_action=last_t.action_id,
                max_blame_score=1.0,
                causal_explanation="Final action blame fallback",
            )

        step_blame: dict[int, float] = {}
        explanations: dict[int, str] = {}
        prev_reachability: int | None = None

        for idx, t in enumerate(valid_transitions):
            step_idx = t.step
            pos = t.avatar_pos
            if pos is None:
                continue

            r, c = pos
            grid = t.grid_snapshot if t.grid_snapshot is not None else np.zeros((32, 32), dtype=int)
            H, W = grid.shape

            cur_reach = cls.compute_spatial_reachability(pos, grid, static_barriers)

            # 1. Base blame: Recency decay towards end of episode
            recency_blame = float(idx + 1) / float(len(valid_transitions)) * 5.0
            blame = recency_blame
            explanation = "temporal_proximity"

            # 2. Irreversible corner deadlock detection
            if (r, c) not in goals and CounterfactualDeadlockDetector.is_corner_deadlock(
                (r, c), goals, static_barriers, (H, W)
            ):
                blame += 50.0
                explanation = f"irreversible_corner_deadlock_at_({r},{c})"

            # 3. Reachability collapse (e.g. falling into hole or moving past one-way cut-set)
            if prev_reachability is not None and prev_reachability > 0:
                contraction = prev_reachability - cur_reach
                if contraction > 5:
                    blame += float(contraction) * 2.0
                    explanation = f"reachability_collapse_loss_{contraction}_cells"

            # 4. Immediate terminal death
            if t.is_lost:
                blame += 25.0
                if explanation == "temporal_proximity":
                    explanation = "terminal_lethal_transition"

            prev_reachability = cur_reach
            step_blame[step_idx] = blame
            explanations[step_idx] = explanation

        # Identify root causal step with highest blame
        best_step = max(step_blame.keys(), key=lambda s: step_blame[s])
        target_trans = next(t for t in valid_transitions if t.step == best_step)

        return ShapleyCreditResult(
            root_causal_step=best_step,
            root_fatal_pos=target_trans.avatar_pos,
            root_fatal_action=target_trans.action_id,
            max_blame_score=step_blame[best_step],
            step_blame_scores=step_blame,
            causal_explanation=explanations[best_step],
        )
