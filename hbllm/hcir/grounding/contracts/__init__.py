from __future__ import annotations

from hbllm.hcir.grounding.contracts.actions import (
    ActionExecutionRecord,
    ActionValidationResult,
    RenderedActionPayload,
    SymbolicActionIntent,
)
from hbllm.hcir.grounding.contracts.beliefs import BeliefState
from hbllm.hcir.grounding.contracts.claims import GroundingClaim
from hbllm.hcir.grounding.contracts.decisions import GroundingDecision
from hbllm.hcir.grounding.contracts.dialogue import (
    ActionIntentProposal,
    DialogueInterpretationResult,
    DialogueUtterance,
    GoalProposal,
    QueryIntent,
)
from hbllm.hcir.grounding.contracts.enums import (
    ActionLifecycleState,
    CausalStatus,
    ClaimOrigin,
    ConceptLifecycle,
    DecisionOutcome,
    DialogueAct,
    ModalitySource,
    OverflowPolicy,
    Polarity,
    TrafficClass,
)
from hbllm.hcir.grounding.contracts.evidence import EvidenceRecord
from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope
from hbllm.hcir.grounding.contracts.quality import GroundingQuality
from hbllm.hcir.grounding.contracts.references import (
    EntityCandidateRef,
    GroundedConceptRef,
    GroundedEntityRef,
    GroundedRef,
    LiteralValueRef,
    TrackletRef,
    UnresolvedRef,
)

__all__ = [
    # Enums
    "ModalitySource",
    "ClaimOrigin",
    "DialogueAct",
    "Polarity",
    "TrafficClass",
    "OverflowPolicy",
    "ConceptLifecycle",
    "CausalStatus",
    "DecisionOutcome",
    "ActionLifecycleState",
    # References
    "GroundedRef",
    "TrackletRef",
    "EntityCandidateRef",
    "GroundedEntityRef",
    "GroundedConceptRef",
    "LiteralValueRef",
    "UnresolvedRef",
    # Observations & Evidence
    "ObservationEnvelope",
    "EvidenceRecord",
    # Claims & Beliefs
    "GroundingClaim",
    "BeliefState",
    "GroundingDecision",
    "GroundingQuality",
    # Dialogue
    "DialogueUtterance",
    "QueryIntent",
    "GoalProposal",
    "ActionIntentProposal",
    "DialogueInterpretationResult",
    # Actions & Realization
    "SymbolicActionIntent",
    "RenderedActionPayload",
    "ActionValidationResult",
    "ActionExecutionRecord",
]
