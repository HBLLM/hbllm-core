"""Global Neuronal Workspace (GNW) Ignition & Broadcast Engine.

Biologically modeled on Bernard Baars and Stanislas Dehaene's Global Neuronal
Workspace Theory (Dehaene et al., 2001, 2011; Baars, 1988):
1. Sub-Threshold Specialized Cortex Competition:
   - Specialized sensory, causal, planning, and affective cortical modules
     continuously propose candidate cognitive representations.
   - Candidates include sensory anomalies, predictive conflicts, affordance discoveries,
     counterfactual plan completions, and looming hazard detections.
2. Non-Linear Dynamic Ignition Threshold:
   - Evaluates modular salience via sigmoidal non-linear threshold dynamics:
       IgnitionScore = 1 / (1 + exp(-gain * (salience - dynamic_threshold)))
   - Sub-threshold fluctuations remain local to the initiating cortex.
   - Suprathreshold signals ignite into a self-sustaining recurrent state.
3. Conscious Global Broadcast Cycle:
   - The ignited representation is broadcast simultaneously across all cortices,
     binding multi-modal cognitive faculties and updating working memory atomically.
4. Refractory Inhibition & Adaptation:
   - Following ignition, the threshold transiently elevates to prevent runaway
     hyper-synchrony or cognitive perseveration, decaying back to baseline.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceCandidate:
    """A candidate cognitive representation competing for conscious ignition."""

    candidate_id: str
    source_cortex: str
    salience: float
    content: dict[str, Any]
    epistemic_uncertainty: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export workspace candidate as a formal causal predicate."""
        return CausalPredicate(
            variable=f"gnw_candidate:{self.source_cortex}",
            operator="==",
            value=round(self.salience, 4),
            relation=f"uncertainty:{self.epistemic_uncertainty:.3f}",
            target_entity=self.candidate_id,
        )


@dataclass
class IgnitionEvent:
    """An all-or-none ignition event broadcasting winning cognitive content."""

    event_id: str
    winner: WorkspaceCandidate
    competing_candidates_count: int
    ignition_score: float
    threshold: float
    timestamp: float = field(default_factory=time.time)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export ignition event as an authoritative declarative causal predicate."""
        return CausalPredicate(
            variable="gnw_conscious_ignition",
            operator="==",
            value=self.winner.source_cortex,
            relation=f"score:{self.ignition_score:.3f}:salience:{self.winner.salience:.3f}",
            target_entity=self.event_id,
        )

    def to_transaction_operations(self, tenant_id: str = "default") -> list[TransactionOperation]:
        """Propose atomic HCIR transaction logging the conscious broadcast in working memory."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"gnw_broadcast_{self.event_id}",
                node_data={
                    "event_id": self.event_id,
                    "source_cortex": self.winner.source_cortex,
                    "salience": float(self.winner.salience),
                    "ignition_score": float(self.ignition_score),
                    "threshold": float(self.threshold),
                    "content": self.winner.content,
                    "tenant_id": tenant_id,
                    "timestamp": self.timestamp,
                },
            )
        ]


class GlobalNeuronalWorkspace:
    """Master Global Neuronal Workspace orchestrating modular competition and broadcast."""

    def __init__(
        self,
        base_threshold: float = 0.60,
        gain: float = 8.0,
        refractory_elevation: float = 0.30,
        threshold_decay_rate: float = 0.15,
    ) -> None:
        self.base_threshold = base_threshold
        self.gain = gain
        self.refractory_elevation = refractory_elevation
        self.threshold_decay_rate = threshold_decay_rate

        self.current_threshold: float = base_threshold
        self.pending_candidates: list[WorkspaceCandidate] = []
        self.listeners: dict[str, Callable[[IgnitionEvent], None]] = {}
        self.ignition_history: list[IgnitionEvent] = []
        self.last_ignited_content: WorkspaceCandidate | None = None

    def register_listener(
        self,
        cortex_name: str,
        callback: Callable[[IgnitionEvent], None],
    ) -> None:
        """Register a cortical module listener to receive conscious global broadcasts."""
        self.listeners[cortex_name] = callback

    def unregister_listener(self, cortex_name: str) -> None:
        """Unregister a cortical module listener."""
        self.listeners.pop(cortex_name, None)

    def submit_candidate(
        self,
        source_cortex: str,
        salience: float,
        content: dict[str, Any],
        epistemic_uncertainty: float = 0.0,
    ) -> WorkspaceCandidate:
        """Submit a candidate representation from a specialized cortical module."""
        cand_id = f"cand_{uuid.uuid4().hex[:8]}"
        candidate = WorkspaceCandidate(
            candidate_id=cand_id,
            source_cortex=source_cortex,
            salience=max(0.0, min(1.0, float(salience))),
            content=dict(content),
            epistemic_uncertainty=float(epistemic_uncertainty),
        )
        self.pending_candidates.append(candidate)
        return candidate

    def compute_ignition_score(self, salience: float) -> float:
        """Compute non-linear sigmoidal ignition activation score."""
        diff = salience - self.current_threshold
        exponent = -self.gain * diff
        exponent = max(-50.0, min(50.0, exponent))
        return 1.0 / (1.0 + math.exp(exponent))

    def compete_and_ignite(self) -> IgnitionEvent | None:
        """Execute competitive arbitration among candidates and trigger global broadcast if ignited."""
        if not self.pending_candidates:
            self.step_decay()
            return None

        # Sort by salience descending
        self.pending_candidates.sort(key=lambda c: c.salience, reverse=True)
        winner = self.pending_candidates[0]
        competing_count = len(self.pending_candidates)

        score = self.compute_ignition_score(winner.salience)

        # Non-linear all-or-none ignition condition
        if score >= 0.50:
            event_id = f"ign_{uuid.uuid4().hex[:8]}"
            event = IgnitionEvent(
                event_id=event_id,
                winner=winner,
                competing_candidates_count=competing_count,
                ignition_score=score,
                threshold=self.current_threshold,
            )
            self.last_ignited_content = winner
            self.ignition_history.append(event)

            # Transient refractory elevation to prevent cognitive perseveration
            self.current_threshold = min(1.0, self.current_threshold + self.refractory_elevation)

            # Clear consumed candidates
            self.pending_candidates.clear()

            # Broadcast to all registered cortical listeners
            for listener in list(self.listeners.values()):
                try:
                    listener(event)
                except Exception as exc:
                    logger.warning("GNW broadcast error to listener: %s", exc)

            logger.info(
                "GNW: Conscious ignition from [%s] (salience=%.3f, score=%.3f, threshold=%.3f)",
                winner.source_cortex,
                winner.salience,
                score,
                event.threshold,
            )
            return event
        else:
            # Sub-threshold: clear candidates and decay threshold toward baseline
            self.pending_candidates.clear()
            self.step_decay()
            return None

    def step_decay(self) -> None:
        """Decay elevated refractory threshold toward base resting threshold."""
        if self.current_threshold > self.base_threshold:
            self.current_threshold = max(
                self.base_threshold,
                self.current_threshold - self.threshold_decay_rate,
            )

    def to_causal_predicates(self) -> list[CausalPredicate]:
        """Export current GNW state and recent ignitions as causal predicates."""
        predicates = [
            CausalPredicate(
                variable="gnw_state",
                operator="==",
                value=f"threshold:{self.current_threshold:.3f}:ignitions:{len(self.ignition_history)}",
                relation="global_workspace",
                target_entity="gnw",
            )
        ]
        if self.last_ignited_content is not None:
            predicates.append(
                CausalPredicate(
                    variable="gnw_active_consciousness",
                    operator="==",
                    value=self.last_ignited_content.source_cortex,
                    relation=f"salience:{self.last_ignited_content.salience:.3f}",
                    target_entity=self.last_ignited_content.candidate_id,
                )
            )
        return predicates
