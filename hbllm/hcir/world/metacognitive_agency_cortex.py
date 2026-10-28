"""Metacognitive Confidence Calibration & BA10 Agency Monitor (Fleming & Dolan).

Biologically grounded in Anterior Prefrontal Cortex (BA10 / rostrolateral PFC)
and Stephen Fleming & Ray Dolan's metacognitive evaluation models (Fleming & Dolan, 2012):
1. Type-II Metacognitive Sensitivity (meta-d'):
   - Evaluates second-order awareness of decision quality: does the system know when
     it knows vs when it is guessing?
   - Computes metacognitive calibration error (MCE = |SubjectiveConfidence - EmpiricalSuccess|).
2. Agency Attribution Engine:
   - Differentiates self-generated environmental changes from exogenous world drift:
       P(Agency | Delta I, a_t) = exp(-dist(Delta_actual, Delta_motor)) / Z
   - Prevents illusory control over autonomous patrollers or external conveyors.
3. Autonomous Compute Budget Governor:
   - Dynamically scales mental simulation rollout depth D_sim and search width:
     - High Confidence & Low Entropy -> Microsecond reflex execution (D_sim = 2).
     - Low Confidence & High Ambiguity -> Deep counterfactual deliberative search (D_sim = 20).
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class MetacognitiveAssessment:
    """Metacognitive calibration and agency evaluation record."""

    assessment_id: str
    action_id: Any
    subjective_confidence: float  # [0.0, 1.0]
    agency_attribution_score: float  # [0.0, 1.0]: 1.0 = self-caused, 0.0 = exogenous drift
    calibration_error: float
    recommended_simulation_depth: int
    recommended_exploration_mode: str  # "reflex", "pragmatic", "deep_simulation", "epistemic_probe"
    timestamp: float = field(default_factory=time.time)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export metacognitive assessment as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"metacognition:agency:{self.action_id}",
            operator="==",
            value=round(self.agency_attribution_score, 4),
            relation=f"mode:{self.recommended_exploration_mode}:depth:{self.recommended_simulation_depth}",
            target_entity=self.assessment_id,
        )

    def to_transaction_operations(self, tenant_id: str = "default") -> list[TransactionOperation]:
        """Propose atomic HCIR transaction logging metacognitive assessment in working memory."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"meta_assess_{self.assessment_id}",
                node_data={
                    "assessment_id": self.assessment_id,
                    "action_id": self.action_id,
                    "subjective_confidence": float(self.subjective_confidence),
                    "agency_attribution_score": float(self.agency_attribution_score),
                    "calibration_error": float(self.calibration_error),
                    "recommended_simulation_depth": int(self.recommended_simulation_depth),
                    "recommended_exploration_mode": self.recommended_exploration_mode,
                    "tenant_id": tenant_id,
                    "timestamp": self.timestamp,
                },
            )
        ]


class MetacognitiveAgencyCortex:
    """Master BA10 Metacognitive Confidence & Agency Attribution Engine."""

    def __init__(
        self,
        base_simulation_depth: int = 5,
        max_simulation_depth: int = 20,
        agency_decay: float = 0.1,
    ) -> None:
        self.base_simulation_depth = base_simulation_depth
        self.max_simulation_depth = max_simulation_depth
        self.agency_decay = agency_decay
        self.history: list[MetacognitiveAssessment] = []
        self.outcome_history: list[tuple[float, bool]] = []  # (confidence, success)

    def compute_agency_attribution(
        self,
        predicted_avatar_pos: tuple[int, int] | None,
        observed_avatar_pos: tuple[int, int] | None,
        observed_other_changes_count: int = 0,
    ) -> float:
        """Evaluate sense of agency P(Self | Delta I).

        High when predicted motor displacement matches actual observed displacement
        and non-motor exogenous changes are minimal.
        """
        if predicted_avatar_pos is None or observed_avatar_pos is None:
            return 0.5

        dr = abs(predicted_avatar_pos[0] - observed_avatar_pos[0])
        dc = abs(predicted_avatar_pos[1] - observed_avatar_pos[1])
        motor_match_dist = dr + dc

        # Exponential falloff with displacement error
        motor_alignment = math.exp(-motor_match_dist)

        # Exogenous dampening if many unpredicted background mutations occur
        exogenous_penalty = 1.0 / (1.0 + 0.2 * observed_other_changes_count)
        agency = motor_alignment * exogenous_penalty
        return max(0.0, min(1.0, agency))

    def calibrate_confidence_and_govern_compute(
        self,
        action_id: Any,
        raw_confidence: float,
        agency_score: float,
        empirical_success: bool | None = None,
    ) -> MetacognitiveAssessment:
        """Calibrate confidence and govern cognitive compute budget allocation."""
        if empirical_success is not None:
            self.outcome_history.append((raw_confidence, empirical_success))

        # Compute empirical calibration error (Brier-like discrepancy)
        if self.outcome_history:
            recent = self.outcome_history[-20:]
            cal_err = sum(abs(c - (1.0 if s else 0.0)) for c, s in recent) / len(recent)
        else:
            cal_err = abs(raw_confidence - 0.5)

        # Dynamic compute budget governor
        # High confidence + high agency -> fast reflex
        # Low confidence or low agency -> deep simulation or epistemic probe
        if raw_confidence >= 0.85 and agency_score >= 0.80:
            rec_depth = 2
            mode = "reflex"
        elif raw_confidence >= 0.60 and agency_score >= 0.60:
            rec_depth = self.base_simulation_depth
            mode = "pragmatic"
        elif agency_score < 0.40:
            rec_depth = min(self.max_simulation_depth, self.base_simulation_depth + 4)
            mode = "epistemic_probe"
        else:
            # Low confidence under high uncertainty
            rec_depth = self.max_simulation_depth
            mode = "deep_simulation"

        assessment_id = f"meta_{uuid.uuid4().hex[:8]}"
        assessment = MetacognitiveAssessment(
            assessment_id=assessment_id,
            action_id=action_id,
            subjective_confidence=raw_confidence,
            agency_attribution_score=agency_score,
            calibration_error=cal_err,
            recommended_simulation_depth=rec_depth,
            recommended_exploration_mode=mode,
        )
        self.history.append(assessment)
        return assessment

    def to_causal_predicates(self) -> list[CausalPredicate]:
        """Export current metacognitive monitoring state as declarative causal predicates."""
        if not self.history:
            return []
        latest = self.history[-1]
        return [
            CausalPredicate(
                variable="metacognition:agency_status",
                operator="==",
                value=round(latest.agency_attribution_score, 3),
                relation=f"mode:{latest.recommended_exploration_mode}:depth:{latest.recommended_simulation_depth}",
                target_entity="ba10_agency_cortex",
            ),
            CausalPredicate(
                variable="metacognition:calibration_error",
                operator="==",
                value=round(latest.calibration_error, 3),
                relation="brier_discrepancy",
                target_entity="ba10_agency_cortex",
            ),
        ]
