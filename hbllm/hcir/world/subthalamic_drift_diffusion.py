from __future__ import annotations

"""Subthalamic Nucleus (STN) Decision Threshold Modulation via Drift-Diffusion Dynamics.

Biologically grounded in:
1. R. Ratcliff & G. McKoon (2008) - "The diffusion decision model: Theory and data
   for two-choice decision tasks" (Neural Comput). Evidence accumulation to bound.
2. M.J. Frank et al. (2007) - "Hold your horses: Impulsivity, sluggishness, and the
   subthalamic nucleus" (Nature). The STN dynamically modulates decision thresholds:
   under high conflict, the hyperdirect STN pathway elevates boundaries to delay
   premature motor commitment, whereas deadline urgency collapses the threshold.
3. B.U. Forstmann et al. (2008) - Striatum and STN involvement in speed-accuracy tradeoff.

In HBLLM, this module provides bounded evidence accumulation with STN-regulated
dynamic boundaries a(t) = a0 * (1 + β * conflict) * exp(-λ * t).
"""

import math
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate


@dataclass
class DriftDiffusionDecision:
    """Outcome of drift-diffusion evidence accumulation at a decision step."""

    chosen_action: Any | None
    evidence_level: float
    decision_threshold: float
    decision_time_steps: int
    is_threshold_reached: bool
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export decision state as an HCIR causal predicate."""
        action_str = str(self.chosen_action) if self.chosen_action is not None else "undecided"
        return CausalPredicate(
            variable="stn:drift_diffusion_decision",
            operator="==",
            value=action_str,
            relation=f"evidence:{self.evidence_level:.2f}:thresh:{self.decision_threshold:.2f}:steps:{self.decision_time_steps}:reached:{self.is_threshold_reached}",
            target_entity="subthalamic_drift_diffusion",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording decision state."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="stn_decision_state",
                node_data={
                    "chosen_action": str(self.chosen_action)
                    if self.chosen_action is not None
                    else None,
                    "evidence_level": float(self.evidence_level),
                    "decision_threshold": float(self.decision_threshold),
                    "decision_time_steps": int(self.decision_time_steps),
                    "is_threshold_reached": self.is_threshold_reached,
                    "confidence": float(self.confidence),
                },
            )
        ]


class SubthalamicDriftDiffusionEngine:
    """Subthalamic Nucleus (STN) evidence accumulation engine with conflict-modulated thresholds."""

    def __init__(
        self,
        base_threshold: float = 1.0,
        conflict_gain: float = 1.5,
        urgency_decay_rate: float = 0.05,
        noise_std: float = 0.0,
        max_steps: int = 50,
    ) -> None:
        self.base_threshold = base_threshold
        self.conflict_gain = conflict_gain
        self.urgency_decay = urgency_decay_rate
        self.noise_std = noise_std
        self.max_steps = max_steps
        self.accumulated_evidence: dict[Any, float] = {}
        self.step_counter: int = 0

    def reset(self) -> None:
        """Reset internal accumulator and step clock."""
        self.accumulated_evidence.clear()
        self.step_counter = 0

    def compute_threshold(
        self, conflict_level: float = 0.0, elapsed_steps: int | None = None
    ) -> float:
        """Compute instantaneous STN decision boundary a(t).

        Elevated by ACC conflict; collapsed over time by urgency.
        """
        t = elapsed_steps if elapsed_steps is not None else self.step_counter
        # STN expands boundary under high conflict ("hold your horses")
        elevated = self.base_threshold * (1.0 + (self.conflict_gain * max(0.0, conflict_level)))
        # Urgency collapses threshold over time
        decay = math.exp(-self.urgency_decay * float(t))
        # Ensure positive non-zero lower bound
        return max(0.15, elevated * decay)

    def step_accumulation(
        self,
        evidence_rates: dict[Any, float],
        conflict_level: float = 0.0,
        dt: float = 1.0,
    ) -> DriftDiffusionDecision:
        """Step drift-diffusion evidence accumulation across competing action alternatives."""
        self.step_counter += 1
        threshold = self.compute_threshold(
            conflict_level=conflict_level, elapsed_steps=self.step_counter
        )

        # Update evidence for each candidate action
        for action, drift_rate in evidence_rates.items():
            prev = self.accumulated_evidence.get(action, 0.0)
            # Drift-diffusion update
            self.accumulated_evidence[action] = prev + (drift_rate * dt)

        # Check for boundary crossing
        best_action: Any | None = None
        max_evidence = -1e9
        second_max = -1e9

        for action, ev in self.accumulated_evidence.items():
            if ev > max_evidence:
                second_max = max_evidence
                max_evidence = ev
                best_action = action
            elif ev > second_max:
                second_max = ev

        threshold_reached = max_evidence >= threshold
        forced_decision = self.step_counter >= self.max_steps

        # Compute choice confidence: margin between top two alternatives
        margin = max(0.0, max_evidence - max(0.0, second_max))
        confidence = min(1.0, margin / max(0.1, threshold))

        chosen: Any | None = None
        if threshold_reached or forced_decision:
            chosen = best_action

        return DriftDiffusionDecision(
            chosen_action=chosen,
            evidence_level=max(0.0, max_evidence),
            decision_threshold=threshold,
            decision_time_steps=self.step_counter,
            is_threshold_reached=threshold_reached or forced_decision,
            confidence=confidence,
            metadata={
                "conflict_level": float(conflict_level),
                "forced_decision": forced_decision,
                "accumulated_evidence": {
                    str(k): float(v) for k, v in self.accumulated_evidence.items()
                },
            },
        )
