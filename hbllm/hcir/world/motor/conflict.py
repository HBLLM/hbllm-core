"""Anterior Cingulate Cortex (ACC) Conflict Monitoring & Limit-Cycle Breaker."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class ACCConflictRecord:
    """Record of cognitive conflict or motor deadlock detected by the ACC."""

    step: int
    cycle_period: int
    cyclic_actions: list[Any]
    suppressed_actions: list[Any] = field(default_factory=list)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export ACC conflict record as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"acc_conflict:{self.step}",
            operator="==",
            value=self.cycle_period,
            relation=f"cyclic:{len(self.cyclic_actions)}",
            target_entity=f"suppressed:{len(self.suppressed_actions)}",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording ACC conflict."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"acc_conflict_{self.step}",
                node_data={
                    "step": self.step,
                    "cycle_period": self.cycle_period,
                    "cyclic_actions": [str(a) for a in self.cyclic_actions],
                    "suppressed_actions": [str(a) for a in self.suppressed_actions],
                },
            )
        ]


class ACCConflictMonitor:
    """Anterior Cingulate Cortex conflict detector and limit-cycle oscillation breaker."""

    def __init__(self, history_window: int = 16, oscillation_threshold: int = 3) -> None:
        self.history_window = history_window
        self.oscillation_threshold = oscillation_threshold
        self.action_history: deque[Any] = deque(maxlen=history_window)
        self.position_history: deque[Any] = deque(maxlen=history_window)
        self.conflict_records: list[ACCConflictRecord] = []

    def record_step(
        self, action: Any, avatar_pos: tuple[float, float] | tuple[int, int] | None
    ) -> None:
        """Record an executed action and resulting position."""
        self.action_history.append(action)
        if avatar_pos is not None:
            self.position_history.append(avatar_pos)

    @staticmethod
    def get_orthogonal_actions(
        available_actions: Sequence[Any],
        cyclic_actions: Sequence[Any],
        action_dynamics: dict[Any, Any],
    ) -> list[Any]:
        """Compute actions orthogonal or complementary to the oscillating action axis.

        If cycling on vertical displacement (e.g. up/down, dr!=0, dc==0),
        prioritizes horizontal displacements (dr==0, dc!=0).
        If cycling on horizontal displacement (dr==0, dc!=0),
        prioritizes vertical displacements (dr!=0, dc==0).
        Falls back to any available non-cyclic actions.
        """
        cyclic_set = set(cyclic_actions)
        non_cyclic = [a for a in available_actions if a not in cyclic_set]
        if not non_cyclic:
            return list(available_actions)

        # Compute aggregate direction vector of cyclic actions
        total_abs_dr = 0
        total_abs_dc = 0
        for a in cyclic_actions:
            dyn = action_dynamics.get(a)
            if dyn is not None:
                dr, dc = dyn.get_displacement()
                total_abs_dr += abs(dr)
                total_abs_dc += abs(dc)

        orthogonal_candidates: list[Any] = []
        other_displacement_candidates: list[Any] = []
        other_candidates: list[Any] = []

        for a in non_cyclic:
            dyn = action_dynamics.get(a)
            if dyn is not None and dyn.is_displacement_action():
                dr, dc = dyn.get_displacement()
                if total_abs_dr > total_abs_dc:
                    # Cyclic axis is primarily vertical -> orthogonal is horizontal
                    if abs(dc) > abs(dr):
                        orthogonal_candidates.append(a)
                    else:
                        other_displacement_candidates.append(a)
                elif total_abs_dc > total_abs_dr:
                    # Cyclic axis is primarily horizontal -> orthogonal is vertical
                    if abs(dr) > abs(dc):
                        orthogonal_candidates.append(a)
                    else:
                        other_displacement_candidates.append(a)
                else:
                    # Balanced or diagonal -> any displacement candidate
                    other_displacement_candidates.append(a)
            else:
                other_candidates.append(a)

        return orthogonal_candidates + other_displacement_candidates + other_candidates

    def detect_oscillation(self) -> tuple[bool, list[Any]]:
        """Detect 2-cycle or 3-cycle limit cycles in recent actions or positions.

        Returns:
            (is_oscillating, cyclic_actions_to_suppress)
        """
        acts = list(self.action_history)
        cyclic_acts: list[Any] = []
        # Check 1-cycle repetition (single action stalled against barrier/wall)
        if len(acts) >= 3 and len(set(acts[-3:])) == 1:
            pos_list = list(self.position_history)
            if len(pos_list) >= 3 and pos_list[-1] == pos_list[-3]:
                cyclic_acts = [acts[-1]]
        # Check 2-cycle: e.g. [A, B, A, B]
        elif len(acts) >= 4 and acts[-1] == acts[-3] and acts[-2] == acts[-4]:
            cyclic_acts = [acts[-1], acts[-2]]
        # Check 3-cycle: e.g. [A, B, C, A, B, C]
        elif (
            len(acts) >= 6
            and acts[-1] == acts[-4]
            and acts[-2] == acts[-5]
            and acts[-3] == acts[-6]
        ):
            cyclic_acts = [acts[-1], acts[-2], acts[-3]]
        # Check 4-cycle: e.g. [A, B, C, D, A, B, C, D]
        elif (
            len(acts) >= 8
            and acts[-1] == acts[-5]
            and acts[-2] == acts[-6]
            and acts[-3] == acts[-7]
            and acts[-4] == acts[-8]
        ):
            cyclic_acts = [acts[-1], acts[-2], acts[-3], acts[-4]]

        if cyclic_acts:
            pos_list = list(self.position_history)
            cycle_len = len(cyclic_acts)
            if len(pos_list) >= 2 * cycle_len:
                # In embodied/spatial navigation, an action repetition is only an oscillation
                # if the organism returns to the same location (zero net displacement).
                # If there is net spatial progress, it is corridor/staircase traversal, not a limit cycle.
                if pos_list[-1] != pos_list[-1 - cycle_len]:
                    cyclic_acts = []

        # Check position bounce or stagnation
        if not cyclic_acts:
            pos_list = list(self.position_history)
            if len(pos_list) >= 3 and len(set(pos_list[-3:])) == 1:
                cyclic_acts = list(set(acts[-2:])) if acts else []
            elif (
                len(pos_list) >= 4 and pos_list[-1] == pos_list[-3] and pos_list[-2] == pos_list[-4]
            ):
                cyclic_acts = list(set(acts[-2:])) if acts else []

        if cyclic_acts:
            rec = ACCConflictRecord(
                step=len(self.action_history),
                cycle_period=len(cyclic_acts),
                cyclic_actions=cyclic_acts,
                suppressed_actions=cyclic_acts,
            )
            self.conflict_records.append(rec)
            return True, cyclic_acts

        return False, []

    def reset(self) -> None:
        """Clear conflict history."""
        self.action_history.clear()
        self.position_history.clear()
        self.conflict_records.clear()

    def to_causal_predicate(self) -> CausalPredicate:
        """Export ACC conflict monitor state as a declarative causal predicate."""
        return CausalPredicate(
            variable="acc_conflict_monitor",
            operator="==",
            value=len(self.conflict_records),
            relation=f"window:{self.history_window}",
            target_entity="anterior_cingulate",
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all ACC conflict records as declarative causal predicates."""
        preds = [self.to_causal_predicate()]
        for rec in self.conflict_records:
            preds.append(rec.to_causal_predicate())
        return preds

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting ACC conflict monitor state."""
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="acc_conflict_summary",
                node_data={
                    "conflict_count": len(self.conflict_records),
                    "history_window": self.history_window,
                    "oscillation_threshold": self.oscillation_threshold,
                },
            )
        ]
        for rec in self.conflict_records:
            ops.extend(rec.to_transaction_operations())
        return ops
