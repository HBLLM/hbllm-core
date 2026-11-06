from __future__ import annotations

import time
from dataclasses import dataclass, field

from hbllm.hcir.grounding.contracts.enums import DecisionOutcome


@dataclass(frozen=True)
class GroundingDecision:
    """The epistemic disposition and proposed transaction for a GroundingClaim."""

    claim_id: str
    outcome: DecisionOutcome
    posterior_belief: float
    calibrated_evidence_strength: float
    contradictions: tuple[str, ...] = field(default_factory=tuple)
    supporting_evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    transaction_proposal_id: str | None = None
    decision_rationale: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("claim_id must be non-empty")
        if not isinstance(self.outcome, DecisionOutcome):
            raise TypeError(f"outcome must be DecisionOutcome, got {type(self.outcome)}")
        if not (0.0 <= self.posterior_belief <= 1.0):
            raise ValueError(f"posterior_belief must be in [0.0, 1.0], got {self.posterior_belief}")
        if not (0.0 <= self.calibrated_evidence_strength <= 1.0):
            raise ValueError(
                f"calibrated_evidence_strength must be in [0.0, 1.0], got {self.calibrated_evidence_strength}"
            )
        if not isinstance(self.contradictions, tuple):
            object.__setattr__(self, "contradictions", tuple(self.contradictions))
        if not isinstance(self.supporting_evidence_ids, tuple):
            object.__setattr__(self, "supporting_evidence_ids", tuple(self.supporting_evidence_ids))
