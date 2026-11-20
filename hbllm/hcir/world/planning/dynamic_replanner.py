"""Dynamic replanning on unexpected observation divergences (W140)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

if TYPE_CHECKING:
    from hbllm.hcir.world.planning.mental_simulation import MentalSimulationStep


@dataclass
class ReplanResult:
    """W140: Outcome of unexpected state divergence replanning."""

    replanned: bool
    new_plan: list[MentalSimulationStep]
    divergence_reason: str
    replan_count: int

    def to_causal_predicate(self) -> CausalPredicate:
        """Export replan result as a declarative causal predicate."""
        return CausalPredicate(
            variable="replan_result",
            operator="==",
            value=self.replanned,
            relation=f"count:{self.replan_count}",
            target_entity=self.divergence_reason,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording replan result."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"replan_{self.replan_count}_{self.replanned}",
                node_data={
                    "replanned": self.replanned,
                    "plan_len": len(self.new_plan),
                    "divergence_reason": self.divergence_reason,
                    "replan_count": self.replan_count,
                },
            )
        ]


class DynamicReplanner:
    """W140: Triggers dynamic plan repair and topological rerouting upon unexpected observation divergences."""

    @staticmethod
    def replan_on_divergence(
        current_actual_pos: tuple[int, int],
        expected_pos: tuple[int, int],
        remaining_plan: list[MentalSimulationStep],
        goal_positions: set[tuple[int, int]],
        barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
    ) -> ReplanResult:
        """Repairs or generates a new plan from actual position to goal."""
        from hbllm.hcir.world.planning.mental_simulation import MentalSimulationStep

        if current_actual_pos == expected_pos and remaining_plan:
            return ReplanResult(
                replanned=False,
                new_plan=remaining_plan,
                divergence_reason="nominal",
                replan_count=0,
            )

        target = list(goal_positions)[0] if goal_positions else expected_pos
        H, W = grid_shape
        queue = [(current_actual_pos, [])]
        visited = {current_actual_pos}
        found_path: list[tuple[tuple[int, int], int]] | None = None

        while queue:
            curr, path = queue.pop(0)
            if curr == target or curr in goal_positions:
                found_path = path
                break
            for dr, dc, act in [(-1, 0, 1), (1, 0, 2), (0, -1, 3), (0, 1, 4)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                nxt = (nr, nc)
                if 0 <= nr < H and 0 <= nc < W and nxt not in barriers and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [(nxt, act)]))

        if found_path is not None:
            new_steps = [
                MentalSimulationStep(action=act, predicted_avatar_pos=pos)
                for pos, act in found_path
            ]
            return ReplanResult(
                replanned=True,
                new_plan=new_steps,
                divergence_reason=f"diverged_from_{expected_pos}_to_{current_actual_pos}",
                replan_count=1,
            )

        return ReplanResult(
            replanned=False,
            new_plan=[],
            divergence_reason="unreachable_after_divergence",
            replan_count=1,
        )

    @classmethod
    def to_causal_predicate(cls, view: Any = None) -> CausalPredicate:
        """Export dynamic replanner capability and runtime divergence count as a declarative causal predicate."""
        val = "ACTIVE"
        rel = "divergence_rerouting"
        if view is not None:
            fails = getattr(view, "consecutive_plan_failures", 0)
            rel = f"divergence_rerouting:plan_failures:{fails}"
        return CausalPredicate(
            variable="dynamic_replanner",
            operator="==",
            value=val,
            relation=rel,
            target_entity="cortex_planner",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering dynamic replanner."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="dynamic_replanner",
                node_data={
                    "faculty": "dynamic_replanner",
                    "status": "ACTIVE",
                },
            )
        ]
