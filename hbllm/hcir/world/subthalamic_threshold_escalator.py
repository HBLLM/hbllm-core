"""Subthalamic Nucleus (STN) Decision Threshold Escalation & Conflict Braking.

Biological Foundations:
- Frank, M. J. (2006): "Hold your horses: A dynamic computational role for the
  subthalamic nucleus in decision making" (Neural Networks 19(8): 1120-1136).
- Cavanagh, J. F. et al. (2011): "Subthalamic nucleus stimulation reverses normal
  conflict-induced slowing" (Journal of Neuroscience 31(47): 16985-16995).
- Wiecki, T. V., & Frank, M. J. (2013): "A computational model of inhibitory control
  in frontal-basal ganglia loops" (NeuroImage 64: 488-499).

Key Cognitive Capabilities:
1. Conflict-Induced Decision Boundary Escalation:
   - Dynamically elevates the Drift-Diffusion Model (DDM) decision threshold:
     a(t) = a_0 * (1 + kappa * conflict), buying crucial deliberation time when
     competing motor policies exhibit high mutual uncertainty or interference.
2. Hyperdirect "Hold-Your-Horses" Braking:
   - Halts premature subcortical motor dispatch until accumulated evidential margin
     clears the elevated STN conflict threshold.
3. Decaying Urgency Gating:
   - Integrates temporal collapse / urgency signals to prevent infinite decision paralysis
     under sustained high-conflict conditions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from hbllm.hcir.world.causal_discovery import CausalPredicate


class STNBrakingStatus(str, Enum):
    """Execution status emitted by the subthalamic threshold escalator."""

    BRAKE_ACTIVE = "brake_active"  # Hold your horses: conflict exceeds evidential margin
    DISPATCH_PERMITTED = "dispatch_permitted"  # Evidence clears threshold; execute action
    EMERGENCY_OVERRIDE = "emergency_override"  # Collapsed urgency forces commitment


@dataclass(frozen=True)
class ThresholdModulationDecision:
    """Decision output from subthalamic threshold modulation."""

    status: STNBrakingStatus
    baseline_threshold: float
    effective_threshold: float
    accumulated_evidence: float
    detected_conflict: float
    urgency_level: float
    recommended_deliberation_cycles: int

    def to_causal_predicate(self) -> CausalPredicate:
        """Convert STN threshold modulation decision to HCIR CausalPredicate."""
        return CausalPredicate(
            variable="stn_conflict_threshold",
            operator="==",
            value=f"status:{self.status.value}:eff_a:{self.effective_threshold:.3f}:evid:{self.accumulated_evidence:.3f}:conf:{self.detected_conflict:.3f}",
            relation="subthalamic_boundary_modulation",
            target_entity="basal_ganglia_stn",
        )


class SubthalamicThresholdEscalator:
    """Orchestrator for subthalamic conflict-induced threshold escalation and motor braking."""

    def __init__(
        self,
        baseline_threshold: float = 1.0,
        conflict_escalation_gain: float = 1.5,
        urgency_decay_rate: float = 0.08,
        max_escalated_threshold: float = 3.5,
    ) -> None:
        """Initialize SubthalamicThresholdEscalator.

        Args:
            baseline_threshold: Base decision threshold a_0 under zero conflict.
            conflict_escalation_gain: Sensitivity factor kappa to conflict.
            urgency_decay_rate: Rate at which urgency lowers threshold over elapsed time t.
            max_escalated_threshold: Ceiling for the elevated threshold.
        """
        self.baseline_threshold: float = max(0.1, float(baseline_threshold))
        self.conflict_gain: float = max(0.0, float(conflict_escalation_gain))
        self.urgency_rate: float = max(0.0, float(urgency_decay_rate))
        self.max_threshold: float = max(self.baseline_threshold, float(max_escalated_threshold))

    def evaluate_decision_threshold(
        self,
        candidate_q_values: tuple[float, ...] | list[float],
        elapsed_deliberation_steps: int = 0,
        external_conflict_signal: float | None = None,
    ) -> ThresholdModulationDecision:
        """Evaluate evidence accumulation against dynamic conflict-escalated decision threshold.

        Args:
            candidate_q_values: Expected utilities or Q-values of top motor options.
            elapsed_deliberation_steps: Number of deliberation ticks spent on this decision.
            external_conflict_signal: Optional conflict score from ACC (Anterior Cingulate Cortex).

        Returns:
            ThresholdModulationDecision specifying whether motor dispatch is permitted or braked.
        """
        if not candidate_q_values:
            return ThresholdModulationDecision(
                status=STNBrakingStatus.DISPATCH_PERMITTED,
                baseline_threshold=self.baseline_threshold,
                effective_threshold=self.baseline_threshold,
                accumulated_evidence=0.0,
                detected_conflict=0.0,
                urgency_level=0.0,
                recommended_deliberation_cycles=0,
            )

        sorted_vals = sorted([float(v) for v in candidate_q_values], reverse=True)
        top_val = sorted_vals[0]
        second_val = sorted_vals[1] if len(sorted_vals) > 1 else 0.0

        # Evidential margin: difference between top choice and best alternative
        evidence_margin = max(0.0, top_val - second_val)

        # Conflict estimation: if external ACC signal provided, use it; else compute entropy/closeness
        if external_conflict_signal is not None:
            conflict = max(0.0, min(1.0, float(external_conflict_signal)))
        else:
            # High conflict when top two values are nearly identical
            denom = abs(top_val) + abs(second_val) + 1e-5
            conflict = max(0.0, min(1.0, 1.0 - (evidence_margin / denom)))

        # Urgency function: increases with elapsed deliberation time
        t = max(0, int(elapsed_deliberation_steps))
        urgency = 1.0 - math.exp(-self.urgency_rate * t)

        # Dynamic DDM threshold: a(t) = a_0 * (1 + kappa * conflict) * (1 - 0.70 * urgency)
        raw_escalated = self.baseline_threshold * (1.0 + self.conflict_gain * conflict)
        effective_threshold = max(
            0.20 * self.baseline_threshold,
            min(self.max_threshold, raw_escalated * (1.0 - 0.75 * urgency)),
        )

        # Decision rule:
        # 1. High urgency force-exit
        if urgency > 0.85 and evidence_margin > 0.05:
            status = STNBrakingStatus.EMERGENCY_OVERRIDE
            recommended_cycles = 0
        # 2. Evidence clears elevated threshold
        elif evidence_margin >= effective_threshold:
            status = STNBrakingStatus.DISPATCH_PERMITTED
            recommended_cycles = 0
        # 3. Brake: hold your horses
        else:
            status = STNBrakingStatus.BRAKE_ACTIVE
            shortfall = effective_threshold - evidence_margin
            # Scale extra cycles needed proportional to shortfall
            recommended_cycles = max(1, min(10, int(math.ceil(shortfall * 3.0))))

        return ThresholdModulationDecision(
            status=status,
            baseline_threshold=self.baseline_threshold,
            effective_threshold=effective_threshold,
            accumulated_evidence=evidence_margin,
            detected_conflict=conflict,
            urgency_level=urgency,
            recommended_deliberation_cycles=recommended_cycles,
        )
