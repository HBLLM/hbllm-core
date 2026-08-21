"""
Formal Cognitive Type System — HCIR §4.

All attributes in the HCIR graph are strongly typed using semantic units
rather than generic primitive classes.  This enables compile-time
validation, optimizer analysis, and IDE-level safety.

Types defined here are the atomic building blocks used by every other
HCIR module (graph nodes, transactions, bytecodes).

Design invariant:
    Every numeric field in HCIR must use one of these types, never a bare
    ``float`` or ``int``.  This ensures range validation at construction
    and semantic meaning at every callsite.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════
# Constrained Scalar Types
# ═══════════════════════════════════════════════════════════════════════════

#: A priority weight in [0.0, 1.0].  Higher = more important.
Priority = Annotated[float, Field(ge=0.0, le=1.0, description="Priority weight [0.0, 1.0]")]

#: A confidence / probability score in [0.0, 1.0].
Confidence = Annotated[float, Field(ge=0.0, le=1.0, description="Confidence score [0.0, 1.0]")]

#: A cost measured in tokens (non-negative integer).
CostMetric = Annotated[int, Field(ge=0, description="Cost in tokens")]

#: A duration measured in milliseconds (non-negative integer).
TimeDuration = Annotated[int, Field(ge=0, description="Duration in milliseconds")]

#: A monotonic-clock timestamp (epoch seconds as float).
Timestamp = Annotated[float, Field(ge=0.0, description="Monotonic clock timestamp (epoch seconds)")]


# ═══════════════════════════════════════════════════════════════════════════
# Reliability Source Enum
# ═══════════════════════════════════════════════════════════════════════════


class ReliabilitySource(StrEnum):
    """How a piece of knowledge was obtained."""

    OBSERVED = "observed"  # Direct sensor or user input
    INFERRED = "inferred"  # Derived by reasoning engine
    REPORTED = "reported"  # Asserted by external source


class BranchMode(StrEnum):
    """Execution mode isolation for workspace branches."""

    LIVE = "live"
    SIMULATION = "simulation"
    REPLAY = "replay"
    TRAINING = "training"


# ═══════════════════════════════════════════════════════════════════════════
# Uncertainty Vector
# ═══════════════════════════════════════════════════════════════════════════


class UncertaintyVector(BaseModel):
    """Multi-dimensional uncertainty representation.

    Instead of a single confidence score, beliefs track uncertainty
    across multiple cognitive vectors.  This enables smarter memory
    decay, planning risk assessment, and belief revision.

    Attributes:
        confidence: Statistical probability [0.0, 1.0].
        freshness_ms: Age since last direct verification (milliseconds).
        reliability: Source type (observed, inferred, reported).
        volatility: How likely this state is to change rapidly [0.0, 1.0].
    """

    confidence: Confidence = 0.5
    freshness_ms: TimeDuration = 0
    reliability: ReliabilitySource = ReliabilitySource.INFERRED
    volatility: Confidence = 0.0  # Reuses the [0, 1] range


# ═══════════════════════════════════════════════════════════════════════════
# Attention Parameters
# ═══════════════════════════════════════════════════════════════════════════


class DecayStrategy(StrEnum):
    """Temporal decay functions for attention."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"
    NONE = "none"


class Attention(BaseModel):
    """First-class attention representation.

    Visible to planners and schedulers as part of HCIR runtime state.

    Attributes:
        salience: Interest weight [0.0, 1.0].
        activation: Working memory activation level [0.0, 1.0].
        decay_rate: Temporal decay coefficient [0.0, 1.0].
        decay_strategy: Decay function type.
        focus_area: Current cognitive focus target (e.g., "planning").
    """

    salience: Confidence = 0.5
    activation: Confidence = 0.5
    decay_rate: Confidence = 0.05
    decay_strategy: DecayStrategy = DecayStrategy.EXPONENTIAL
    focus_area: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Provenance Metadata
# ═══════════════════════════════════════════════════════════════════════════


class Provenance(BaseModel):
    """Origin metadata for any HCIR object.

    Enables explainable introspection without using an LLM.

    Attributes:
        created_by: Node ID or subsystem that produced this object.
        timestamp: Monotonic creation timestamp.
        engine: Model or solver that generated this (e.g., ``"gemini-3.5-flash"``).
        prompt_hash: SHA-256 hash of the originating prompt (if LLM-generated).
        reasoning_step: Step index in the reasoning chain.
        simulation_branch: Branch ID if created during simulation.
        session_id: Conversation/session trace ID.
        goal_id: Parent goal that caused this object.
        parent_goal_id: Causal parent goal.
        trace_id: End-to-end request trace ID.
        model_used: LLM model that generated this (e.g., ``"gemini-2.5-flash"``).
        reason: Human-readable justification for this object's existence.
        source_node: Originating node ID.
        source_type: How this was produced (``"observed"``, ``"inferred"``, ``"reported"``).
        logical_time: Monotonically increasing logical clock (Lamport-style).
        generation: Reasoning generation / depth.
        attention_epoch: Attention recomputation epoch index.
        reflection_cycle: Reflection cycle index.
    """

    created_by: str = ""
    timestamp: Timestamp = Field(default_factory=time.time)
    engine: str = ""
    prompt_hash: str = ""
    reasoning_step: int = 0
    simulation_branch: str = "main"

    # ── Traceability ─────────────────────────────────────────────────
    session_id: str = ""
    goal_id: str = ""
    parent_goal_id: str = ""
    trace_id: str = ""
    model_used: str = ""
    reason: str = ""
    source_node: str = ""
    source_type: str = ""  # "observed", "inferred", "reported"

    # ── Cognitive Time ───────────────────────────────────────────────
    logical_time: int = 0
    generation: int = 0
    attention_epoch: int = 0
    reflection_cycle: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Scope / Identity Context
# ═══════════════════════════════════════════════════════════════════════════


class SecurityLevel(StrEnum):
    """Security clearance levels for cognitive objects."""

    PUBLIC = "public"
    TENANT = "tenant"
    USER = "user"
    DEVICE = "device"
    SYSTEM = "system"


class Scope(BaseModel):
    """Multi-dimensional isolation boundary for every HCIR object.

    Ensures absolute scope isolation (Kernel Invariant #4).

    Attributes:
        tenant_id: Tenant isolation boundary.
        workspace_id: Workspace/domain context within tenant.
        user_id: User within tenant.
        device_id: Device within user scope.
        cluster_id: Distributed cluster identifier.
        simulation_id: Simulation branch scope (empty = real world).
        security_level: Access control classification.
    """

    tenant_id: str = "default"
    workspace_id: str = "default"
    user_id: str = "default"
    device_id: str = "default"
    cluster_id: str = "local"
    simulation_id: str = ""
    security_level: SecurityLevel = SecurityLevel.TENANT


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Mode — runtime reasoning context
# ═══════════════════════════════════════════════════════════════════════════


class CognitiveMode(StrEnum):
    """Runtime reasoning mode that changes how components behave.

    The same planner, memory, and critic operate differently depending
    on the active cognitive mode.  Discovery mode optimizes for
    uncertainty reduction rather than answer production.

    Examples::

        STANDARD:   "What is X?"  → Goal: produce answer
        DISCOVERY:  "Why does X happen?"  → Goal: reduce uncertainty
        DIAGNOSTIC: "What went wrong?"  → Goal: isolate root cause
    """

    STANDARD = "standard"  # Normal reasoning — produce answers
    DISCOVERY = "discovery"  # Scientific cognition — reduce uncertainty
    DIAGNOSTIC = "diagnostic"  # Root-cause analysis — isolate failures
    CREATIVE = "creative"  # Divergent thinking — maximize novelty
    CRITICAL = "critical"  # Adversarial review — maximize rigor


# ═══════════════════════════════════════════════════════════════════════════
# Epistemic Types — scientific reasoning primitives
# ═══════════════════════════════════════════════════════════════════════════


class BeliefConfidence(BaseModel):
    """Multi-dimensional belief confidence decomposition.

    Instead of ``Confidence = 0.84`` (a single opaque number), decompose
    confidence into independent, auditable dimensions.  The derived score
    is a weighted combination, but each dimension can be inspected and
    updated independently.

    This prevents pathological behaviors such as:
    - A single weak source inflating overall confidence
    - High prediction accuracy masking low reproducibility
    - Publication bias in evidence quality

    Dimensions::

        evidence_quality     — Methodology strength of supporting evidence
        evidence_quantity    — Amount of supporting evidence (normalized)
        reproducibility      — Has the evidence been independently reproduced?
        prediction_accuracy  — Track record of predictions from this belief
        model_agreement      — Agreement with the system's causal models
        source_trust         — Weighted reputation of evidence sources

    Usage::

        bc = BeliefConfidence(evidence_quality=0.9, reproducibility=0.8)
        bc.derived_confidence  # → weighted combination
    """

    evidence_quality: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Methodology strength of supporting evidence",
    )
    evidence_quantity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Amount of supporting evidence (normalized 0=none, 1=extensive)",
    )
    reproducibility: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Has the evidence been independently reproduced?",
    )
    prediction_accuracy: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Track record of predictions derived from this belief",
    )
    model_agreement: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agreement with the system's causal models",
    )
    source_trust: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weighted reputation of evidence sources",
    )

    # Configurable weights for the derived score
    _weights: tuple[float, ...] = (0.25, 0.15, 0.20, 0.20, 0.10, 0.10)

    @property
    def derived_confidence(self) -> float:
        """Weighted combination of all dimensions → single confidence score.

        Default weights prioritize evidence quality and reproducibility,
        reflecting the epistemic hierarchy.
        """
        components = (
            self.evidence_quality,
            self.evidence_quantity,
            self.reproducibility,
            self.prediction_accuracy,
            self.model_agreement,
            self.source_trust,
        )
        return min(1.0, max(0.0, sum(w * c for w, c in zip(self._weights, components))))

    def to_dict(self) -> dict[str, float]:
        """Return all dimensions plus derived score as a dict."""
        return {
            "evidence_quality": self.evidence_quality,
            "evidence_quantity": self.evidence_quantity,
            "reproducibility": self.reproducibility,
            "prediction_accuracy": self.prediction_accuracy,
            "model_agreement": self.model_agreement,
            "source_trust": self.source_trust,
            "derived_confidence": self.derived_confidence,
        }


class KnowledgeValue(BaseModel):
    """Multi-dimensional value assessment for epistemic objects.

    Every belief, hypothesis, and unknown should answer: *why do I care?*
    Curiosity uses uncertainty, but sometimes low-uncertainty + huge-impact
    should still be investigated.

    This model separates "how uncertain is this?" (handled by
    ``BeliefConfidence``) from "how valuable is knowing this?"
    (handled here).

    Dimensions::

        novelty              — How new is this knowledge?
        impact               — Potential impact if confirmed/falsified
        risk                 — Risk of NOT knowing this
        cost                 — Cost to investigate further (lower = cheaper)
        urgency              — Time sensitivity
        strategic_relevance  — Alignment with active goals/programs

    Usage::

        kv = KnowledgeValue(impact=0.9, urgency=0.8, cost=0.2)
        kv.derived_value  # → weighted combination (higher = more valuable)
    """

    novelty: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How new or surprising is this knowledge?",
    )
    impact: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Potential impact if confirmed or falsified",
    )
    risk: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Risk of NOT knowing this (higher = riskier to ignore)",
    )
    cost: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cost to investigate further (0=free, 1=prohibitive)",
    )
    urgency: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Time sensitivity (0=no deadline, 1=critical)",
    )
    strategic_relevance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Alignment with active goals and research programs",
    )

    @property
    def derived_value(self) -> float:
        """Weighted value score optimized for investigation prioritization.

        High value = high benefit, low cost.
        The formula reflects discovery economics:
        ``(impact × novelty × strategic_relevance) / (1 + cost)``,
        boosted by urgency and risk.
        """
        benefit = (
            0.35 * self.impact
            + 0.25 * self.novelty
            + 0.20 * self.strategic_relevance
            + 0.10 * self.risk
            + 0.10 * self.urgency
        )
        cost_penalty = 1.0 + self.cost  # range [1.0, 2.0]
        return min(1.0, max(0.0, benefit / cost_penalty))

    def to_dict(self) -> dict[str, float]:
        """Return all dimensions plus derived value as a dict."""
        return {
            "novelty": self.novelty,
            "impact": self.impact,
            "risk": self.risk,
            "cost": self.cost,
            "urgency": self.urgency,
            "strategic_relevance": self.strategic_relevance,
            "derived_value": self.derived_value,
        }


class FalsificationStatus(StrEnum):
    """Popperian falsification status for beliefs and hypotheses.

    Every belief and hypothesis must be falsifiable.  This enum tracks
    where each stands in the scientific lifecycle::

        UNTESTED → CORROBORATED → (WEAKENED → FALSIFIED | SUPERSEDED)

    A belief that cannot be falsified is not scientific — it's dogma.
    """

    UNTESTED = "untested"  # No prediction has been tested
    CORROBORATED = "corroborated"  # Predictions confirmed, not yet falsified
    WEAKENED = "weakened"  # Some predictions failed
    FALSIFIED = "falsified"  # Critical prediction failed
    SUPERSEDED = "superseded"  # Replaced by a better hypothesis


class EvidenceStrength(StrEnum):
    """Qualitative evidence strength classification.

    Ordered from weakest to strongest.  This is the epistemic
    hierarchy — not all evidence is created equal::

        ANECDOTAL < OBSERVATIONAL < CORRELATIONAL <
        EXPERIMENTAL < META_ANALYTIC < REPLICATED

    The discovery engine uses this to weight evidence during
    belief revision and hypothesis evaluation.
    """

    ANECDOTAL = "anecdotal"  # Single observation, no controls
    OBSERVATIONAL = "observational"  # Systematic observation, no intervention
    CORRELATIONAL = "correlational"  # Statistical relationship identified
    EXPERIMENTAL = "experimental"  # Controlled experiment
    META_ANALYTIC = "meta_analytic"  # Aggregation across multiple studies
    REPLICATED = "replicated"  # Independently reproduced results


class ExperimentStatus(StrEnum):
    """Lifecycle of an experiment from design to completion.

    Tracks the full experiment lifecycle::

        DESIGNED → APPROVED → RUNNING → COMPLETED | FAILED | CANCELLED
    """

    DESIGNED = "designed"  # Experiment plan created
    APPROVED = "approved"  # Safety/governance review passed
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Finished with results
    FAILED = "failed"  # Execution failed (not the same as negative result)
    CANCELLED = "cancelled"  # Abandoned before completion


class ExperimentRealityLevel(StrEnum):
    """Reality level of an experiment — safety boundary.

    Determines the fidelity and risk of an experiment.  Higher levels
    require more approval and carry more evidential weight::

        SIMULATION → DIGITAL → OBSERVATIONAL → CONTROLLED → PHYSICAL

    Each level has different confidence weights (set by ExperimentEngine):
        SIMULATION:     0.2  (pure logic/heuristic)
        DIGITAL:        0.4  (software-based experiment)
        OBSERVATIONAL:  0.6  (passive real-world observation)
        CONTROLLED:     0.8  (controlled real-world experiment)
        PHYSICAL:       1.0  (direct physical manipulation)

    A robot might simulate motor control before trying it physically.
    A medical researcher might run computational models before clinical trials.
    """

    SIMULATION = "simulation"  # Pure computational simulation
    DIGITAL = "digital"  # Software-based experiment (A/B test, etc.)
    OBSERVATIONAL = "observational"  # Passive real-world observation
    CONTROLLED = "controlled"  # Controlled real-world experiment
    PHYSICAL = "physical"  # Direct physical manipulation


class EpistemicLifecycle(StrEnum):
    """Formal state machine for the epistemic lifecycle.

    Every hypothesis (and the unknowns that spawn them) follows this
    lifecycle.  The states enforce a disciplined progression from
    gap identification through to established knowledge or falsification::

        UNKNOWN → QUESTIONING → HYPOTHESIZED → UNDER_REVIEW →
        PREDICTING → EXPERIMENTING → SUPPORTED → REPLICATING →
        ESTABLISHED | FALSIFIED | ARCHIVED

    Unlike ``HypothesisLifecycle`` (which tracks only hypothesis states),
    this enum covers the **full epistemic journey** from gap to knowledge.
    """

    UNKNOWN = "unknown"  # Knowledge gap identified
    QUESTIONING = "questioning"  # Formalized as a research question
    HYPOTHESIZED = "hypothesized"  # Hypothesis generated for this gap
    UNDER_REVIEW = "under_review"  # Plausibility and novelty being assessed
    PREDICTING = "predicting"  # Testable predictions registered
    EXPERIMENTING = "experimenting"  # Experiments in progress
    SUPPORTED = "supported"  # Evidence supports, not yet conclusive
    REPLICATING = "replicating"  # Independent replication underway
    ESTABLISHED = "established"  # Multiple independent lines of evidence confirm
    FALSIFIED = "falsified"  # Critical prediction failed
    ARCHIVED = "archived"  # No longer actively investigated


class ResearchStrategyType(StrEnum):
    """Pluggable research strategies that control how the epistemic
    loop behaves.

    The same runtime can pursue different cognitive strategies depending
    on the research program's current needs::

        EXPLORATION:           Maximize coverage of unknowns
        VERIFICATION:          Focus on testing existing hypotheses
        REPLICATION:           Independent reproduction of findings
        OPTIMIZATION:          Improve known solutions
        COUNTEREXAMPLE_SEARCH: Actively try to falsify the dominant hypothesis
        LITERATURE_REVIEW:     Survey existing knowledge
        BENCHMARKING:          Comparative evaluation against baselines
    """

    EXPLORATION = "exploration"
    VERIFICATION = "verification"
    REPLICATION = "replication"
    OPTIMIZATION = "optimization"
    COUNTEREXAMPLE_SEARCH = "counterexample_search"
    LITERATURE_REVIEW = "literature_review"
    BENCHMARKING = "benchmarking"
    SYNTHESIS = "synthesis"
    ABDUCTIVE = "abductive"
    SYSTEMATIC = "systematic"


class DiscoveryTrigger(StrEnum):
    """Sources of epistemic investigation.

    Many breakthroughs begin with anomalies, not contradictions.
    Discovery triggers classify **why** an investigation was started,
    enabling the curiosity engine to learn which trigger types yield
    the highest-value discoveries::

        CONTRADICTION:       Two claims conflict
        ANOMALY:             Observation doesn't fit existing models
        NOVEL_OBSERVATION:   Something genuinely new was observed
        KNOWLEDGE_GAP:       Known unknown — we know what we don't know
        UNEXPECTED_SUCCESS:  Something worked better than predicted
        UNEXPECTED_FAILURE:  Something failed that should have succeeded
        ANALOGY:             Cross-domain structural similarity detected
        CURIOSITY:           Self-directed investigation (no external trigger)
        PERCEPTUAL_ANOMALY:   Sensory observation conflicts with expectation
        PERCEPTUAL_AMBIGUITY: Competing high-confidence sensory classifications
    """

    CONTRADICTION = "contradiction"
    ANOMALY = "anomaly"
    NOVEL_OBSERVATION = "novel_observation"
    KNOWLEDGE_GAP = "knowledge_gap"
    UNEXPECTED_SUCCESS = "unexpected_success"
    UNEXPECTED_FAILURE = "unexpected_failure"
    ANALOGY = "analogy"
    CURIOSITY = "curiosity"
    PERCEPTUAL_ANOMALY = "perceptual_anomaly"
    PERCEPTUAL_AMBIGUITY = "perceptual_ambiguity"


# ═══════════════════════════════════════════════════════════════════════════
# Perceptual & Multimodal Epistemic Types
# ═══════════════════════════════════════════════════════════════════════════


class PerceptualModality(StrEnum):
    """Sensory modalities supported by the perceptual subsystem."""

    AUDIO = "audio"
    VISUAL = "visual"
    TEXT = "text"
    IOT = "iot"
    MULTIMODAL = "multimodal"


class PerceptualEpistemicProfile(BaseModel):
    """Multidimensional epistemic profile for sensory evidence.

    Preserves underlying evidence dimensions (clarity, model confidence,
    temporal stability) without compressing into a lossy canonical scalar.
    """

    sensory_clarity: Confidence = Field(
        default=0.8,
        description="Signal-to-noise ratio, illumination, acoustic clarity [0.0, 1.0]",
    )
    model_confidence: Confidence = Field(
        default=0.8,
        description="Confidence emitted by the perception model [0.0, 1.0]",
    )
    temporal_stability: Confidence = Field(
        default=0.8,
        description="Consistency of detection across consecutive temporal windows [0.0, 1.0]",
    )

    @property
    def reliability(self) -> float:
        """Derived composite reliability score preserving multidimensional state."""
        return float(
            0.3 * self.sensory_clarity + 0.4 * self.model_confidence + 0.3 * self.temporal_stability
        )


class CorrelationCandidate(BaseModel):
    """A candidate geometric / temporal cross-modal correlation.

    Represents a measurable, epistemically neutral relationship between
    two observations without prematurely asserting causality.
    """

    source_obs_id: str
    target_obs_id: str
    source_modality: PerceptualModality
    target_modality: PerceptualModality
    temporal_overlap: float = Field(ge=0.0, le=1.0, default=0.0)
    spatial_overlap: float | None = Field(default=None)
    delta_time_ms: float = 0.0
    confidence: Confidence = 0.5
    rationale: str = ""


class PerceptualContradictionLevel(StrEnum):
    """Three-tiered hierarchy of perceptual and epistemic contradictions."""

    LEVEL_1_CLASSIFIER_DISAGREEMENT = "level_1_classifier_disagreement"
    LEVEL_2_CROSS_MODAL_CONFLICT = "level_2_cross_modal_conflict"
    LEVEL_3_BELIEF_CONFLICT = "level_3_belief_conflict"


class IncorporationStatus(StrEnum):
    """Lifecycle status of evidence incorporation into belief revisions."""

    PENDING = "pending"
    INCORPORATED = "incorporated"
    STALE = "stale"
    REDUNDANT = "redundant"


class EvidenceTemporalPattern(StrEnum):
    """Classification of temporal evidence patterns.

    - PERSISTENT: Same state repeated consistently (>3 consecutive similar).
    - TRANSITION: Significant state change from previous observation.
    - TRANSIENT: Single instantaneous event (knock, flash) — always high novelty.
    - PERIODIC: Recurring pattern with detectable periodicity.
    - UNKNOWN: Insufficient history to classify.
    """

    PERSISTENT = "persistent"
    TRANSITION = "transition"
    TRANSIENT = "transient"
    PERIODIC = "periodic"
    UNKNOWN = "unknown"


class OutcomeType(StrEnum):
    """Types of external ground truth for provider reputation validation.

    Cross-modal consensus is explicitly excluded — it updates
    cross_modal_concordance only, never empirical_accuracy.
    """

    EXPERIMENT = "experiment"
    USER_CONFIRMATION = "user_confirmation"
    TOOL_EXECUTION = "tool_execution"
    EXTERNAL_VERIFICATION = "external_verification"


class NoveltyPolicy(BaseModel):
    """Configurable weights for multidimensional novelty computation.

    Default policy: state transitions override temporal/semantic decay.
    Different modalities (audio vs vision) may use different policies.
    """

    temporal_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    semantic_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    state_change_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    state_change_override: bool = True
    half_life_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description="Half-life T½ for temporal novelty decay: n_t = 1 − 2^(−Δt/T½)",
    )
    novelty_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Below this composite novelty, evidence is treated as redundant",
    )

    def compute_temporal_novelty(self, delta_t_seconds: float) -> float:
        """Compute temporal novelty using proper half-life decay.

        n_t = 1 − 2^(−Δt / T½)
        """
        if delta_t_seconds <= 0.0:
            return 0.0
        return 1.0 - math.pow(2.0, -delta_t_seconds / self.half_life_seconds)


class EpistemicRuntimeConfig(BaseModel):
    """Immutable configuration snapshot for deterministic replay.

    Captured at session start and stored as the first journal entry.
    Replay must verify config_hash matches before reconstructing state.
    """

    novelty_policy: NoveltyPolicy = Field(default_factory=NoveltyPolicy)
    bayesian_epsilon: float = 1e-5
    falsification_threshold: float = 0.1
    corroboration_threshold: float = 0.7
    max_support_delta: float = 0.15
    max_contradict_delta: float = 0.20
    algorithm_version: str = "a11.0"
    config_hash: str = ""

    def model_post_init(self, __context: object) -> None:
        """Compute config_hash after initialization."""
        if not self.config_hash:
            # Exclude config_hash itself from the hash computation
            data = self.model_dump(exclude={"config_hash"})
            raw = json.dumps(data, sort_keys=True, default=str)
            self.config_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]


class EvidenceAssessment(BaseModel):
    """General evaluation of evidence quality and reliability.

    Produced by PerceptualEvidenceEvaluator independently of any
    specific candidate belief proposition.
    """

    evidence_id: str
    reliability: Confidence = 0.5
    uncertainty: UncertaintyVector = Field(default_factory=UncertaintyVector)
    epistemic_profile: PerceptualEpistemicProfile | None = None
    provenance_quality: Confidence = 0.8
    information_gain: float = 0.0
    incorporation_status: IncorporationStatus = IncorporationStatus.PENDING
    novelty_score: float = Field(default=1.0, ge=0.0, le=1.0)
    temporal_pattern: EvidenceTemporalPattern = EvidenceTemporalPattern.UNKNOWN
    temporal_delta_seconds: float = 0.0
    semantic_delta: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Jaccard distance from prior incorporated evidence tags",
    )


class PropositionLikelihood(BaseModel):
    """Proposition-specific likelihood evaluation for a hypothesis/belief.

    Produced by EpistemicLikelihoodEvaluator to evaluate how well evidence E
    supports hypothesis H versus its negation ¬H.

    Contains both raw LR (before dependence correction) and effective LR
    (after novelty-based dependence correction: LR_effective = LR^novelty).
    """

    belief_id: str
    evidence_id: str
    p_e_given_h: float = Field(ge=0.0, le=1.0, default=0.5, description="P(E | H)")
    p_e_given_not_h: float = Field(ge=0.0, le=1.0, default=0.5, description="P(E | ¬H)")
    likelihood_ratio: float = Field(default=1.0, description="LR = P(E|H) / P(E|¬H) (raw)")
    raw_likelihood_ratio: float = Field(
        default=1.0, description="Raw LR before dependence correction"
    )
    effective_likelihood_ratio: float = Field(
        default=1.0,
        description="LR^novelty — the value actually used for Bayesian update",
    )
    novelty_discount: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Novelty exponent applied: LR_eff = LR^novelty_discount",
    )
    status: str = Field(
        default="informative",
        description="insufficient | redundant | informative | contradictory",
    )

    def model_post_init(self, __context: object) -> None:
        """Ensure backward compatibility when only likelihood_ratio is provided."""
        if self.raw_likelihood_ratio == 1.0 and self.likelihood_ratio != 1.0:
            self.raw_likelihood_ratio = self.likelihood_ratio
        if (
            self.effective_likelihood_ratio == 1.0
            and self.likelihood_ratio != 1.0
            and self.novelty_discount == 1.0
        ):
            self.effective_likelihood_ratio = self.likelihood_ratio


class BeliefTransition(BaseModel):
    """Immutable audit record of a Bayesian belief state transition.

    Stored in HCIR event journal and as event-sourced history nodes.
    """

    transition_id: str
    belief_id: str
    prior_confidence: Confidence
    posterior_confidence: Confidence
    delta: float
    prior_revision: int = 0
    posterior_revision: int = 1
    likelihood_ratio: float = 1.0
    effective_likelihood_ratio: float = 1.0
    novelty_score: float = 1.0
    source_evidence_id: str = ""
    source_event_ids: list[str] = Field(default_factory=list)
    timestamp: Timestamp = Field(default_factory=time.time)
    rationale: str = ""
