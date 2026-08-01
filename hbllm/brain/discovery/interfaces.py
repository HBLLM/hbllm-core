"""Discovery Interfaces — protocol definitions for all discovery components.

These protocols define the contracts that discovery components must satisfy.
They enable clean dependency injection, testing, and future extension
without tight coupling between subsystems.

Each protocol represents a distinct cognitive capability within the
discovery lifecycle::

    IBeliefReviser         — update beliefs from evidence
    IHypothesisGenerator   — generate hypotheses from knowledge gaps
    IPredictionTracker     — track predictions against observations
    IExperimentDesigner    — design discriminative experiments
    IContradictionSeeker   — actively search for contradictions
    ISourceReputationTracker — track source reliability over time

Usage::

    class MyBeliefReviser(IBeliefReviser):
        async def revise_belief(self, belief_id, evidence_id) -> BeliefRevision:
            ...
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ═══════════════════════════════════════════════════════════════════════════
# Shared Data Types
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BeliefRevision:
    """Record of a belief confidence change."""

    belief_id: str = ""
    old_confidence: float = 0.0
    new_confidence: float = 0.0
    reason: str = ""
    evidence_id: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class HypothesisCandidate:
    """A generated hypothesis candidate before it becomes an HCIR node."""

    claim: str = ""
    novelty: float = 0.5
    plausibility: float = 0.5
    predicted_impact: float = 0.5
    testability: float = 0.5
    supporting_evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    origin: str = ""  # "contradiction", "analogy", "literature", "gap", "serendipity"
    reasoning: str = ""  # Why this hypothesis was generated


@dataclass
class ExperimentDesign:
    """A proposed experiment design before it becomes an HCIR node."""

    hypothesis_ids: list[str] = field(default_factory=list)
    design: str = ""
    variables: dict[str, Any] = field(default_factory=dict)
    controls: list[str] = field(default_factory=list)
    expected_outcomes: dict[str, str] = field(default_factory=dict)
    discriminating_power: float = 0.5
    estimated_cost: float = 0.0
    reality_level: str = "simulation"
    reasoning: str = ""  # Why this experiment was designed


@dataclass
class PredictionOutcome:
    """Result of checking a prediction against observation."""

    prediction_id: str = ""
    hypothesis_id: str = ""
    predicted: str = ""
    observed: str = ""
    correct: bool | None = None
    confidence_delta: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ContradictionReport:
    """A discovered contradiction between claims or evidence."""

    claim_a_id: str = ""
    claim_b_id: str = ""
    contradiction_type: str = ""
    possible_explanations: list[str] = field(default_factory=list)
    investigation_priority: float = 0.5
    context: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Protocol Definitions
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class IBeliefReviser(Protocol):
    """Updates beliefs based on new evidence.

    Wraps the existing ``BeliefStore`` to add discovery-aware lifecycle:
    belief creation → evidence addition → prediction generation →
    prediction testing → confidence update → belief revision.
    """

    async def revise_belief(
        self,
        belief_id: str,
        evidence_id: str,
        direction: str,  # "supporting" | "contradicting"
    ) -> BeliefRevision:
        """Update a belief's confidence based on new evidence."""
        ...

    async def revise_from_prediction(
        self,
        belief_id: str,
        prediction_outcome: PredictionOutcome,
    ) -> BeliefRevision:
        """Update a belief based on a prediction outcome (Bayesian update)."""
        ...

    async def get_falsification_candidates(
        self,
        domain: str = "",
        min_confidence: float = 0.3,
        max_confidence: float = 0.9,
    ) -> list[str]:
        """Return belief IDs that are good candidates for falsification testing."""
        ...


@runtime_checkable
class IHypothesisGenerator(Protocol):
    """Generates hypotheses from knowledge gaps, contradictions, and analogies.

    Inputs: knowledge graph, unknowns, contradictions, failed predictions.
    Output: ranked hypothesis candidates.
    """

    async def generate_from_unknown(
        self,
        unknown_id: str,
        context: dict[str, Any] | None = None,
    ) -> list[HypothesisCandidate]:
        """Generate hypotheses to explain a knowledge gap."""
        ...

    async def generate_from_contradiction(
        self,
        contradiction_id: str,
    ) -> list[HypothesisCandidate]:
        """Generate hypotheses to resolve a contradiction."""
        ...

    async def generate_from_analogy(
        self,
        source_domain: str,
        target_domain: str,
        structural_pattern: str,
    ) -> list[HypothesisCandidate]:
        """Generate hypotheses by cross-domain analogical reasoning."""
        ...


@runtime_checkable
class IPredictionTracker(Protocol):
    """Tracks predictions against observations (Popperian science).

    Every hypothesis must produce testable predictions.
    This tracker monitors prediction outcomes and triggers
    belief/hypothesis updates.
    """

    async def register_prediction(
        self,
        hypothesis_id: str,
        prediction_claim: str,
        predicted_outcome: str,
        time_horizon_ms: int = 0,
    ) -> str:
        """Register a new prediction. Returns prediction node ID."""
        ...

    async def check_prediction(
        self,
        prediction_id: str,
        observed_outcome: str,
    ) -> PredictionOutcome:
        """Check a prediction against an observed outcome."""
        ...

    async def get_pending_predictions(
        self,
        hypothesis_id: str = "",
    ) -> list[str]:
        """Return IDs of predictions that haven't been verified yet."""
        ...


@runtime_checkable
class IExperimentDesigner(Protocol):
    """Designs experiments that discriminate between competing hypotheses.

    The key capability: given hypotheses A and B, design an experiment
    whose outcome **distinguishes** between them.
    """

    async def design_discriminative_experiment(
        self,
        hypothesis_ids: list[str],
        budget: float = 0.0,
        max_reality_level: str = "simulation",
    ) -> ExperimentDesign:
        """Design an experiment to distinguish between hypotheses."""
        ...

    async def rank_experiments(
        self,
        designs: list[ExperimentDesign],
    ) -> list[ExperimentDesign]:
        """Rank experiments by information gain / cost ratio."""
        ...


@runtime_checkable
class IContradictionSeeker(Protocol):
    """Actively searches for contradictions in the knowledge base.

    Instead of just detecting contradictions when they happen,
    this protocol defines proactive contradiction hunting.
    """

    async def scan_for_contradictions(
        self,
        domain: str = "",
        scope: str = "",
    ) -> list[ContradictionReport]:
        """Systematically scan for contradictions across evidence."""
        ...

    async def analyze_contradiction(
        self,
        contradiction_id: str,
    ) -> dict[str, Any]:
        """Deep analysis of a contradiction — identify hidden variables."""
        ...


@runtime_checkable
class ISourceReputationTracker(Protocol):
    """Tracks the reliability of knowledge sources over time.

    Learns which sources are reliable, which reasoning patterns work,
    and which hypothesis origins historically succeed.
    """

    async def record_outcome(
        self,
        source_id: str,
        claim_id: str,
        confirmed: bool,
    ) -> None:
        """Record whether a claim from a source was confirmed."""
        ...

    async def get_reputation(
        self,
        source_id: str,
    ) -> float:
        """Get the reliability score for a source [0.0, 1.0]."""
        ...

    async def get_top_sources(
        self,
        domain: str = "",
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the most reliable sources for a domain."""
        ...
