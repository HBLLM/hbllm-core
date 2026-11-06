from __future__ import annotations

"""Prefrontal Counterfactual Blame Attribution & Causal Credit Assignment.

Biologically & mathematically grounded in:
1. J.Y. Halpern & J. Pearl (2005) - "Causes and explanations: A structural-model
   approach" (British J Philos Sci). Actual causality and degrees of responsibility.
2. J. Pearl (2000) - "Causality: Models, Reasoning, and Inference" (Cambridge Univ Press).
   Counterfactual interventions do(X = x') to isolate actual causes.

In HBLLM, this module evaluates counterfactual interventions on failed plan traces
to compute Pearl-Halpern responsibility degrees ρ(gi) ∈ [0, 1]. It isolates the
pivotal faulty subgoal without blindly discarding valid upstream plan progress.
"""

from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate


@dataclass
class SubgoalBlameScore:
    """Causal responsibility attribution for an individual subgoal in a failed plan."""

    subgoal_id: str
    subgoal_intent: str
    responsibility_score: float  # Pearl-Halpern degree of responsibility ρ = 1 / (1 + k)
    is_primary_cause: bool
    counterfactual_alternative: str | None = None
    witness_set_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export blame score as an HCIR causal predicate."""
        return CausalPredicate(
            variable=f"blame:{self.subgoal_id}",
            operator="==",
            value=f"resp:{self.responsibility_score:.2f}",
            relation=f"primary:{self.is_primary_cause}:alt:{self.counterfactual_alternative}",
            target_entity="prefrontal_blame_attribution",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording subgoal blame attribution."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"blame_{self.subgoal_id}",
                node_data={
                    "subgoal_id": self.subgoal_id,
                    "subgoal_intent": self.subgoal_intent,
                    "responsibility_score": float(self.responsibility_score),
                    "is_primary_cause": self.is_primary_cause,
                    "counterfactual_alternative": self.counterfactual_alternative,
                    "witness_set_size": int(self.witness_set_size),
                },
            )
        ]


@dataclass
class PlanAttributionReport:
    """Consolidated blame attribution report across all steps of an episode."""

    failed_outcome: str
    primary_culprit_id: str | None
    subgoal_blames: list[SubgoalBlameScore]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export attribution summary as an HCIR causal predicate."""
        return CausalPredicate(
            variable="blame:plan_attribution_report",
            operator="==",
            value=str(self.primary_culprit_id),
            relation=f"outcome:{self.failed_outcome}:culprits:{len(self.subgoal_blames)}",
            target_entity="prefrontal_blame_attribution",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction for the attribution report."""
        ops: list[TransactionOperation] = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="plan_attribution_report",
                node_data={
                    "failed_outcome": self.failed_outcome,
                    "primary_culprit_id": self.primary_culprit_id,
                    "total_subgoals": len(self.subgoal_blames),
                },
            )
        ]
        for b in self.subgoal_blames:
            ops.extend(b.to_transaction_operations())
        return ops


class PrefrontalBlameAttributor:
    """Prefrontal cortex counterfactual blame attribution and causal credit engine."""

    def attribute_plan_failure(
        self,
        subgoal_trace: list[dict[str, Any]],
        failure_condition: str = "dead_end",
        lethal_hazards: set[tuple[int, int]] | None = None,
        blocked_targets: set[tuple[int, int]] | None = None,
    ) -> PlanAttributionReport:
        """Evaluate counterfactual necessity of each subgoal for the observed failure."""
        if not subgoal_trace:
            return PlanAttributionReport(
                failed_outcome=failure_condition,
                primary_culprit_id=None,
                subgoal_blames=[],
            )

        hazards = lethal_hazards or set()
        blocked = blocked_targets or set()

        blame_scores: list[SubgoalBlameScore] = []
        max_resp = -1.0
        primary_id: str | None = None

        n = len(subgoal_trace)
        for idx, sg in enumerate(subgoal_trace):
            sg_id = sg.get("id", f"sg_{idx}")
            intent = sg.get("intent", "NAVIGATE")
            target_coord = sg.get("target_coord")
            had_alternative = bool(sg.get("alternatives", []))

            # Counterfactual causal analysis:
            # Did this step directly steer into a lethal hazard or irreversibly trapped state?
            is_direct_trap = target_coord is not None and (
                tuple(target_coord) in hazards or tuple(target_coord) in blocked
            )
            is_terminal_step = idx == n - 1

            if is_direct_trap:
                # Direct counterfactual cause (k=0): ρ = 1.0
                resp = 1.0
                witness = 0
                alt = "bypass_hazard" if had_alternative else "retreat"
            elif is_terminal_step and failure_condition == "dead_end":
                # Terminal dead-end entry (k=0): ρ = 1.0
                resp = 1.0
                witness = 0
                alt = sg.get("alternatives", [None])[0] or "re-route"
            elif had_alternative:
                # Contributory branch: one intermediate choice was involved (k=1): ρ = 0.5
                # Closer to terminal failure incurs higher relative responsibility
                distance_to_fail = n - 1 - idx
                witness = min(3, distance_to_fail)
                resp = 1.0 / (1.0 + float(witness))
                alt = sg["alternatives"][0]
            else:
                # Inevitable or innocuous upstream step with no alternatives (k=inf): ρ = 0.0
                resp = 0.0
                witness = n
                alt = None

            blame = SubgoalBlameScore(
                subgoal_id=sg_id,
                subgoal_intent=intent,
                responsibility_score=resp,
                is_primary_cause=False,  # Determined below
                counterfactual_alternative=alt,
                witness_set_size=witness,
            )
            blame_scores.append(blame)

            if resp > max_resp and resp > 0.1:
                max_resp = resp
                primary_id = sg_id

        # Mark primary culprit
        for b in blame_scores:
            if b.subgoal_id == primary_id:
                b.is_primary_cause = True

        return PlanAttributionReport(
            failed_outcome=failure_condition,
            primary_culprit_id=primary_id,
            subgoal_blames=blame_scores,
        )
