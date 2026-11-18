from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.grounding.contracts.claims import GroundingClaim
from hbllm.hcir.grounding.contracts.enums import DialogueAct
from hbllm.hcir.grounding.contracts.references import GroundedRef, UnresolvedRef


@dataclass(frozen=True)
class DialogueUtterance:
    """Raw natural language utterance received by the system."""

    utterance_id: str
    speaker_id: str
    text: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.utterance_id:
            raise ValueError("utterance_id must be non-empty")
        if not self.text:
            raise ValueError("text must be non-empty")


@dataclass(frozen=True)
class QueryIntent:
    """A question directed to the system requiring graph retrieval or reasoning."""

    query_id: str
    target_variable: str
    constraints: tuple[dict[str, Any], ...]
    temporal_context: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GoalProposal:
    """A request proposing an objective for the executive/planner."""

    proposal_id: str
    goal_intent: str
    target_entity: GroundedRef | UnresolvedRef | None
    parameters: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5


@dataclass(frozen=True)
class ActionIntentProposal:
    """A direct command requesting motor, API, or UI execution."""

    intent_id: str
    action_type: str
    target_ref: GroundedRef | UnresolvedRef | None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DialogueInterpretationResult:
    """Structured semantic output from natural language dialogue interpretation.

    Separates communicative dialogue acts from factual claims, queries, goals, and commands.
    """

    utterance_id: str
    dialogue_act: DialogueAct
    claims: tuple[GroundingClaim, ...] = field(default_factory=tuple)
    queries: tuple[QueryIntent, ...] = field(default_factory=tuple)
    goal_proposals: tuple[GoalProposal, ...] = field(default_factory=tuple)
    action_proposals: tuple[ActionIntentProposal, ...] = field(default_factory=tuple)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not isinstance(self.claims, tuple):
            object.__setattr__(self, "claims", tuple(self.claims))
        if not isinstance(self.queries, tuple):
            object.__setattr__(self, "queries", tuple(self.queries))
        if not isinstance(self.goal_proposals, tuple):
            object.__setattr__(self, "goal_proposals", tuple(self.goal_proposals))
        if not isinstance(self.action_proposals, tuple):
            object.__setattr__(self, "action_proposals", tuple(self.action_proposals))
