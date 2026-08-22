"""
Typed Cognitive Hypergraph — HCIR §3.

The core data structure of the HCIR workspace.  A single graph instance
holds all cognitive entities (nodes) and semantic relationships (edges).
Different subsystems access it through **views** (filtered projections)
rather than separate graph objects:

    Graph
      ├── Knowledge View   (Facts, Beliefs, Concepts, Procedures)
      ├── Execution View   (Goals, Actions, active threads)
      ├── Simulation View  (Forked branches)
      └── Memory View      (Episodes, Skills, Values)

Graph nodes are **typed subclasses**, not untyped dictionaries.
Each subclass carries typed payload fields validated at construction.

Design invariants:
    - Node IDs are globally unique within a graph instance.
    - Edge IDs are globally unique within a graph instance.
    - Edges are *hyper*-edges: they can link multiple source/target nodes.
    - The graph never owns persistence.  Storage is delegated to ``IGraphStore``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from hbllm.hcir.proposition import (
    Proposition,
    SpatialContext,
    TemporalValidity,
)
from hbllm.hcir.types import (
    Attention,
    BeliefConfidence,
    CognitiveMode,
    Confidence,
    CostMetric,
    DiscoveryTrigger,
    EpistemicLifecycle,
    EvidenceStrength,
    ExperimentRealityLevel,
    ExperimentStatus,
    FalsificationStatus,
    KnowledgeValue,
    PerceptualContradictionLevel,
    PerceptualEpistemicProfile,
    Priority,
    Provenance,
    ResearchStrategyType,
    Scope,
    TimeDuration,
    UncertaintyVector,
)

# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════


class HCIRNodeType(StrEnum):
    """All valid cognitive node types in HCIR."""

    # --- Directives ---
    GOAL = "goal"
    CONSTRAINT = "constraint"
    INTENT = "intent"

    # --- Epistemology hierarchy ---
    OBSERVATION = "observation"
    FACT = "fact"
    BELIEF = "belief"
    BELIEF_TRANSITION = "belief_transition"
    HYPOTHESIS = "hypothesis"
    PREDICTION = "prediction"
    PREDICTION_ERROR = "prediction_error"

    # --- Discovery & Epistemic ---
    EVIDENCE = "evidence"  # Structured evidence unit with provenance
    CLAIM = "claim"  # A testable assertion from any source
    EXPERIMENT = "experiment"  # A designed experiment to test hypotheses
    CONTRADICTION = "contradiction"  # An identified contradiction between claims
    UNKNOWN = "unknown"  # A knowledge gap — discovery starts here
    RESEARCH_PROGRAM = "research_program"  # A long-lived research program
    RESEARCH_OBJECTIVE = "research_objective"  # A measurable objective under a program

    # --- Execution ---
    ACTION = "action"
    EVENT = "event"
    RESOURCE = "resource"
    CAPABILITY = "capability"

    # --- Memory classes ---
    EPISODE = "episode"
    CONCEPT = "concept"
    SKILL = "skill"
    PROCEDURE = "procedure"
    VALUE = "value"
    EXTERNAL_KNOWLEDGE = "external_knowledge"

    # --- Grounded Perception ---
    VISUAL_OBSERVATION = "visual_observation"
    VISUAL_CONCEPT = "visual_concept"
    AUDIO_OBSERVATION = "audio_observation"
    ACOUSTIC_CONCEPT = "acoustic_concept"
    PERCEPTUAL_EVIDENCE = "perceptual_evidence"

    # --- World Model & Predictive Cognitive Runtime ---
    WORLD_VARIABLE = "world_variable"
    PHYSICAL_ENTITY = "physical_entity"
    ENVIRONMENT_STATE = "environment_state"


class HCIREdgeType(StrEnum):
    """All valid typed edge relationships in HCIR."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"
    REQUIRES = "requires"
    CAUSES = "causes"
    TENANT_SCOPE = "tenant_scope"

    # Temporal relations
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    EXPIRES = "expires"
    VALID_UNTIL = "valid_until"

    # Structural
    DERIVED_FROM = "derived_from"
    SIMILAR_TO = "similar_to"
    CREATED_BY = "created_by"
    OWNED_BY = "owned_by"
    PART_OF = "part_of"

    # Epistemic relations — scientific reasoning edges
    FALSIFIES = "falsifies"  # Evidence that disproves a hypothesis
    PREDICTS = "predicts"  # Hypothesis predicts an outcome
    TESTS = "tests"  # Experiment tests a hypothesis
    REPLICATES = "replicates"  # Experiment reproduces another
    ANALOGOUS_TO = "analogous_to"  # Structural similarity across domains
    STRENGTHENS = "strengthens"  # Evidence that increases confidence
    WEAKENS = "weakens"  # Evidence that decreases confidence
    REFINES = "refines"  # Hypothesis refines another

    # Cross-modal perception edges
    CORRELATES_WITH = "correlates_with"  # Temporal/spatial association only (NOT causal)
    OBSERVED_AS = "observed_as"  # Classifier relationship (observation → classification)


class CognitiveCategory(StrEnum):
    """High-level cognitive function classification.

    Enables schedulers and attention engines to prioritize by cognitive
    function rather than concrete node class.
    """

    PERCEPTION = "perception"
    MEMORY = "memory"
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTION = "execution"
    REFLECTION = "reflection"
    VALUE = "value"
    COMMUNICATION = "communication"
    DISCOVERY = "discovery"  # Scientific cognition & hypothesis testing


class NodeLifecycle(StrEnum):
    """Generic lifecycle states for any cognitive node."""

    CREATED = "created"
    OBSERVED = "observed"
    VALIDATED = "validated"
    ACTIVE = "active"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"


class GoalLifecycle(StrEnum):
    """Specialized lifecycle states for goal nodes."""

    CREATED = "created"
    PLANNED = "planned"
    EXECUTING = "executing"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CONSOLIDATED = "consolidated"


class HypothesisLifecycle(StrEnum):
    """Lifecycle of a scientific hypothesis.

    Tracks how hypotheses evolve through the discovery process::

        GENERATED → EVALUATED → TESTED → SUPPORTED | FALSIFIED → ARCHIVED
                                       → STRENGTHENED ↗
    """

    GENERATED = "generated"  # Just created, not yet evaluated
    EVALUATED = "evaluated"  # Assessed for plausibility and testability
    TESTED = "tested"  # At least one experiment has been run
    SUPPORTED = "supported"  # Evidence supports, not yet conclusive
    STRENGTHENED = "strengthened"  # Multiple lines of evidence confirm
    WEAKENED = "weakened"  # Some counter-evidence found
    FALSIFIED = "falsified"  # Critical prediction failed
    SUPERSEDED = "superseded"  # Replaced by a better hypothesis
    ARCHIVED = "archived"  # No longer actively investigated


# ═══════════════════════════════════════════════════════════════════════════
# Node Base & Typed Subclasses
# ═══════════════════════════════════════════════════════════════════════════


def _new_id(prefix: str = "n") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class HCIRNode(BaseModel):
    """Base class for all typed cognitive graph nodes.

    Subclasses carry domain-specific typed payload fields.
    The base provides identity, provenance, uncertainty, attention,
    scope, and lifecycle — shared by every cognitive entity.
    """

    id: str = Field(default_factory=lambda: _new_id("n"))
    node_type: HCIRNodeType
    category: CognitiveCategory
    lifecycle: NodeLifecycle = NodeLifecycle.CREATED
    provenance: Provenance = Field(default_factory=Provenance)
    uncertainty: UncertaintyVector = Field(default_factory=UncertaintyVector)
    attention: Attention = Field(default_factory=Attention)
    scope: Scope = Field(default_factory=Scope)
    tags: list[str] = Field(default_factory=list)


# ── Directive Nodes ──────────────────────────────────────────────────────


class IntentNode(HCIRNode):
    """A user's high-level intent.  Survives even if plans fail."""

    node_type: HCIRNodeType = HCIRNodeType.INTENT
    category: CognitiveCategory = CognitiveCategory.PLANNING
    description: str = ""
    resolved: bool = False


class GoalNode(HCIRNode):
    """An active objective or target state to achieve."""

    node_type: HCIRNodeType = HCIRNodeType.GOAL
    category: CognitiveCategory = CognitiveCategory.PLANNING
    goal_lifecycle: GoalLifecycle = GoalLifecycle.CREATED
    description: str = ""
    priority: Priority = 0.5
    resolved: bool = False


class ConstraintNode(HCIRNode):
    """A rule, boundary, or policy restriction."""

    node_type: HCIRNodeType = HCIRNodeType.CONSTRAINT
    category: CognitiveCategory = CognitiveCategory.REASONING
    name: str = ""
    expression: str = ""
    enforcement: str = "HARD"  # "HARD" | "SOFT"


# ── Epistemology Hierarchy ───────────────────────────────────────────────


class ObservationNode(HCIRNode):
    """Raw grounded telemetry from environment, sensors, or user."""

    node_type: HCIRNodeType = HCIRNodeType.OBSERVATION
    category: CognitiveCategory = CognitiveCategory.PERCEPTION
    payload: dict[str, Any] = Field(default_factory=dict)
    sensor_source: str = ""
    modality: str = ""  # "audio", "visual", "iot", etc.
    temporal_span: dict[str, float] = Field(
        default_factory=dict
    )  # {"start_time": ..., "end_time": ...}
    spatial_span: dict[str, float] | None = (
        None  # {"azimuth": ..., "elevation": ..., "distance": ...}
    )
    provider_provenance: dict[str, Any] | None = (
        None  # {"provider": ..., "model": ..., "version": ...}
    )
    raw_reference: str = ""


class VisualObservationNode(ObservationNode):
    """A visual observation with embedding reference and provenance.

    The actual embedding vector resides in VisualMemory (vector store).
    HCIR holds the reference, not the vector — keeping the graph
    lightweight and transactional.

    Architecture::

        HCIR (cognitive state)
          └── VisualObservationNode
                 └── embedding_ref ──────→ VisualMemory (vector store)
    """

    node_type: HCIRNodeType = HCIRNodeType.VISUAL_OBSERVATION
    modality: str = "visual"
    embedding_ref: str = ""  # ID in vector store
    embedding_space: str = ""  # e.g. "siglip-base-patch16-224-image"
    embedding_model: str = ""  # e.g. "google/siglip-base-patch16-224"
    image_hash: str = ""  # SHA-256 for dedup
    caption: str = ""  # Optional text description
    regions: list[dict[str, Any]] = Field(default_factory=list)


class FactNode(HCIRNode):
    """A sensor-verified, grounded observation."""

    node_type: HCIRNodeType = HCIRNodeType.FACT
    category: CognitiveCategory = CognitiveCategory.REASONING
    claim: str = ""


class BeliefNode(HCIRNode):
    """An integrated assertion held by the system.

    Beliefs are the atomic units of knowledge in HBLLM's epistemic
    architecture. Unlike raw facts, every belief carries confidence,
    evidence provenance, falsification status, and revision tracking.

    In the epistemic OS, there are no absolute truths — only beliefs
    with varying confidence. A ``FactNode`` is simply a ``BeliefNode``
    with confidence >= threshold and strong evidence.
    """

    node_type: HCIRNodeType = HCIRNodeType.BELIEF
    category: CognitiveCategory = CognitiveCategory.REASONING
    claim: str = ""
    belief_type: str = "factual"  # factual, causal, procedural, strategic
    evidence_sources: list[str] = Field(default_factory=list)

    # ── Epistemic extensions ────────────────────────────────────────
    counter_evidence: list[str] = Field(default_factory=list)
    falsification_status: FalsificationStatus = FalsificationStatus.UNTESTED
    prediction_score: float = 0.0  # Track record of predictions made from this belief
    current_revision: int = 1
    latest_transition_id: str = ""
    revision_history: list[dict[str, Any]] = Field(default_factory=list)
    # Lightweight summary entries; detailed transition objects live in BeliefTransitionNode / event log

    # ── Multi-dimensional confidence & value ────────────────────────
    belief_confidence: BeliefConfidence = Field(default_factory=BeliefConfidence)
    knowledge_value: KnowledgeValue = Field(default_factory=KnowledgeValue)


class BeliefTransitionNode(HCIRNode):
    """An event-sourced record of a Bayesian belief state transition.

    Stored in HCIR as an immutable audit node to preserve historical
    transparency without ballooning the active BeliefNode payload.
    """

    node_type: HCIRNodeType = HCIRNodeType.BELIEF_TRANSITION
    category: CognitiveCategory = CognitiveCategory.REASONING
    belief_id: str = ""
    prior_confidence: float = 0.5
    posterior_confidence: float = 0.5
    delta: float = 0.0
    prior_revision: int = 0
    posterior_revision: int = 1
    likelihood_ratio: float = 1.0
    effective_likelihood_ratio: float = 1.0
    novelty_score: float = 1.0
    source_evidence_id: str = ""
    source_event_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class HypothesisNode(HCIRNode):
    """A candidate claim under evaluation — first-class cognitive entity.

    Hypotheses are the central objects of scientific reasoning.
    They generate predictions, are tested by experiments, and evolve
    through a full lifecycle from generation to falsification.

    Lifecycle::

        GENERATED → EVALUATED → TESTED → SUPPORTED → STRENGTHENED
                                       → WEAKENED  → FALSIFIED
                                                   → SUPERSEDED
                                                   → ARCHIVED
    """

    node_type: HCIRNodeType = HCIRNodeType.HYPOTHESIS
    category: CognitiveCategory = CognitiveCategory.REASONING
    claim: str = ""
    supporting_evidence: list[str] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)

    # ── Scientific lifecycle ────────────────────────────────────────
    hypothesis_lifecycle: HypothesisLifecycle = HypothesisLifecycle.GENERATED
    epistemic_lifecycle: EpistemicLifecycle = EpistemicLifecycle.HYPOTHESIZED
    falsification_status: FalsificationStatus = FalsificationStatus.UNTESTED
    novelty: Confidence = 0.5  # How novel is this hypothesis?
    plausibility: Confidence = 0.5  # How plausible given current evidence?
    predicted_impact: Confidence = 0.5  # Expected impact if confirmed
    testability: Confidence = 0.5  # How easily can this be tested?
    linked_predictions: list[str] = Field(default_factory=list)  # PredictionNode IDs
    linked_experiments: list[str] = Field(default_factory=list)  # ExperimentNode IDs
    origin: str = (
        ""  # How was this hypothesis generated? (e.g., "analogy", "contradiction", "literature")
    )
    trigger: DiscoveryTrigger = DiscoveryTrigger.KNOWLEDGE_GAP  # What triggered this investigation?
    research_program_id: str = ""  # Parent research program, if any
    knowledge_value: KnowledgeValue = Field(default_factory=KnowledgeValue)


class PredictionNode(HCIRNode):
    """A testable prediction derived from a hypothesis.

    Every hypothesis must generate at least one prediction.
    Predictions are tracked against observations.  Success strengthens
    the parent hypothesis; failure weakens or falsifies it.

    This is Popperian science: hypotheses that don't predict are unfalsifiable.
    """

    node_type: HCIRNodeType = HCIRNodeType.PREDICTION
    category: CognitiveCategory = CognitiveCategory.REASONING
    claim: str = ""
    predicted_outcome: str = ""
    time_horizon_ms: TimeDuration = 0

    # ── Verification ────────────────────────────────────────────────
    hypothesis_id: str = ""  # Which hypothesis generated this prediction?
    observed_outcome: str = ""  # What actually happened?
    verified: bool = False  # Has this prediction been checked?
    verification_timestamp: float = 0.0
    prediction_correct: bool | None = None  # None = unverified, True/False = result


class PredictionErrorNode(HCIRNode):
    """Measures the discrepancy between expected and observed outcomes."""

    node_type: HCIRNodeType = HCIRNodeType.PREDICTION_ERROR
    category: CognitiveCategory = CognitiveCategory.REASONING
    prediction_id: str = ""
    predicted_value: Any = None
    observed_value: Any = None
    delta: float = 0.0
    error_magnitude: float = 0.0
    suspected_cause: str = ""


# ── Execution Nodes ──────────────────────────────────────────────────────


class ActionNode(HCIRNode):
    """A declarative action, independent of specific tools/plugins.

    Actions declare *intent* and *requirements*.  The CapabilityResolver
    binds them to concrete implementations at runtime.
    """

    node_type: HCIRNodeType = HCIRNodeType.ACTION
    category: CognitiveCategory = CognitiveCategory.EXECUTION
    intent: str = ""
    requirements: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    estimated_cost: CostMetric = 0
    permissions: list[str] = Field(default_factory=list)


class EventNode(HCIRNode):
    """A first-class chronological event tick.

    Examples: UserSpoke, ToolFailed, GoalCompleted, SimulationFinished.
    """

    node_type: HCIRNodeType = HCIRNodeType.EVENT
    category: CognitiveCategory = CognitiveCategory.PERCEPTION
    event_kind: str = ""
    event_data: dict[str, Any] = Field(default_factory=dict)
    event_timestamp: float = Field(default_factory=time.time)


class ResourceNode(HCIRNode):
    """A budget or resource allocation node (cpu, tokens, time, etc.)."""

    node_type: HCIRNodeType = HCIRNodeType.RESOURCE
    category: CognitiveCategory = CognitiveCategory.EXECUTION
    resource_type: str = ""  # "cpu", "tokens", "time", "battery", "attention"
    allocated: float = 0.0
    consumed: float = 0.0
    limit: float = 0.0
    is_hard: bool = True  # Hard = physical limit; Soft = cognitive budget


class CapabilityNode(HCIRNode):
    """Declares what the system or a plugin can do.

    Declarative metadata enables market-based capability selection:
    the resolver picks the cheapest available implementation that
    meets the required constraints.
    """

    node_type: HCIRNodeType = HCIRNodeType.CAPABILITY
    category: CognitiveCategory = CognitiveCategory.EXECUTION
    capability_name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)

    # ── Declarative metadata for market-based selection ──
    estimated_cost: int = 0  # Token cost estimate
    latency_ms: int = 0  # Expected latency in ms
    cooldown_seconds: float = 0.0  # Minimum interval between invocations
    requires_approval: bool = False  # Needs governance approval
    max_concurrent: int = 0  # 0 = unlimited
    provider: str = ""  # e.g., "local", "mcp", "api", "docker"
    version: str = "1.0.0"


# ── Memory Class Nodes ───────────────────────────────────────────────────


class EpisodeNode(HCIRNode):
    """A recalled episode from conversation or experience history."""

    node_type: HCIRNodeType = HCIRNodeType.EPISODE
    category: CognitiveCategory = CognitiveCategory.MEMORY
    summary: str = ""
    outcome: str = ""
    reward: float = 0.0


class ConceptNode(HCIRNode):
    """A long-term factual concept in semantic memory."""

    node_type: HCIRNodeType = HCIRNodeType.CONCEPT
    category: CognitiveCategory = CognitiveCategory.MEMORY
    label: str = ""
    definition: str = ""
    domain: str = ""


class VisualConceptNode(ConceptNode):
    """A learned visual concept — extends ConceptNode, single hierarchy.

    Role separation::

        prototype_ref  = retrieval accelerator (centroid in vector store)
        exemplar_refs  = perceptual evidence (actual observation embeddings)
        HCIR node      = semantic identity
        BeliefRecord   = epistemic commitment

    The canonical concept identity is this HCIR node.
    VisualMemory indexes the prototype and exemplar vectors.
    """

    node_type: HCIRNodeType = HCIRNodeType.VISUAL_CONCEPT
    category: CognitiveCategory = CognitiveCategory.MEMORY
    prototype_ref: str = ""  # EmbeddingRef ID for centroid
    embedding_space: str = ""  # e.g. "siglip-base-patch16-224-image"
    embedding_model: str = ""  # e.g. "google/siglip-base-patch16-224"
    observation_count: int = 0
    exemplar_refs: list[str] = Field(default_factory=list)  # EmbeddingRef IDs
    aliases: list[str] = Field(default_factory=list)
    last_seen: float = 0.0
    contexts: list[str] = Field(default_factory=list)


class AudioObservationNode(ObservationNode):
    """An audio observation in HCIR — "the microphone received this."

    Stores a reference to the audio embedding, not the vector itself.
    The vector lives in AudioMemory.

    Architecture::

        HCIR (cognitive state)
          └── AudioObservationNode
                 └── embedding_ref ──────→ AudioMemory (vector store)

    """

    node_type: HCIRNodeType = HCIRNodeType.AUDIO_OBSERVATION
    modality: str = "audio"
    label: str = ""  # Human-readable summary
    embedding_ref: str = ""  # ID in audio vector store
    embedding_space: str = ""  # e.g. "yamnet-v1"
    embedding_model: str = ""  # e.g. "mock-audio-v1"
    audio_hash: str = ""  # SHA-256 for dedup
    event_type: str = ""  # e.g. "speech", "doorbell", "silence"
    event_id: str = ""  # Groups related observations
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    transcript: str = ""  # If speech
    speaker_ref: str = ""  # Speaker embedding ref


class AcousticConceptNode(ConceptNode):
    """A learned acoustic concept — extends ConceptNode.

    Represents a learned sound pattern (e.g., "my doorbell",
    "washing machine cycle"). Created by commit_learning(),
    which creates a cognitive artifact, NOT model training.

    Architecture::

        HCIR (cognitive state)
          └── AcousticConceptNode
                 ├── prototype_ref ─────→ AudioMemory (centroid vector)
                 └── exemplar_refs ─────→ AudioMemory (cluster vectors)

    """

    node_type: HCIRNodeType = HCIRNodeType.ACOUSTIC_CONCEPT
    category: CognitiveCategory = CognitiveCategory.MEMORY
    prototype_ref: str = ""  # EmbeddingRef ID for centroid
    embedding_space: str = ""
    embedding_model: str = ""
    observation_count: int = 0
    exemplar_refs: list[str] = Field(default_factory=list)
    last_heard: float = 0.0
    sound_type: str = ""  # e.g. "event", "ambient", "speech_pattern"
    contexts: list[str] = Field(default_factory=list)


class SkillNode(HCIRNode):
    """A reusable procedural skill discovered through experience."""

    node_type: HCIRNodeType = HCIRNodeType.SKILL
    category: CognitiveCategory = CognitiveCategory.MEMORY
    name: str = ""
    skill_name: str = ""
    description: str = ""
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    success_rate: Confidence = 0.5

    def model_post_init(self, __context: Any) -> None:
        if not self.skill_name and self.name:
            self.skill_name = self.name
        elif not self.name and self.skill_name:
            self.name = self.skill_name


class ProcedureNode(HCIRNode):
    """A multi-step executable recipe."""

    node_type: HCIRNodeType = HCIRNodeType.PROCEDURE
    category: CognitiveCategory = CognitiveCategory.MEMORY
    steps: list[str] = Field(default_factory=list)
    is_atomic: bool = False


class ValueNode(HCIRNode):
    """A preference or value axiom used in decision ranking."""

    node_type: HCIRNodeType = HCIRNodeType.VALUE
    category: CognitiveCategory = CognitiveCategory.VALUE
    dimension: str = ""  # e.g., "brevity", "accuracy", "safety"
    weight: Priority = 0.5


class ExternalKnowledgeNode(HCIRNode):
    """A reference to knowledge from an external source."""

    node_type: HCIRNodeType = HCIRNodeType.EXTERNAL_KNOWLEDGE
    category: CognitiveCategory = CognitiveCategory.MEMORY
    source_uri: str = ""
    content_hash: str = ""
    summary: str = ""


# ── Evidence Integration Mixin ───────────────────────────────────────────


class EvidenceIntegrationMixin:
    """Shared epistemic integration fields for all evidence types.

    This mixin provides the fields that the belief revision pipeline
    (BeliefManager, TemporalEvidenceModel, Replay) uses to track
    evidence incorporation.  Both discovery evidence and perceptual
    evidence participate in the same Bayesian update pipeline via
    these shared fields.

    This is deliberately NOT an HCIRNode subclass.  The semantic
    hierarchy remains flat::

        HCIRNode
        ├── EvidenceNode              (discovery)
        └── PerceptualEvidenceNode    (perception/embodiment)

    while the epistemic capability is orthogonal::

                 EvidenceIntegrationMixin
                    /              \\
        EvidenceNode     PerceptualEvidenceNode

    Usage::

        # Check if a node participates in evidence integration
        if has_evidence_integration(node):
            node.incorporation_status = "incorporated"
    """

    # Fields declared here are picked up by Pydantic when combined
    # with HCIRNode via multiple inheritance.  Defaults ensure all
    # fields are optional.
    evidence_type: EvidenceStrength = EvidenceStrength.OBSERVATIONAL
    strength: Confidence = 0.5  # Quantitative strength [0.0, 1.0]
    modality: str = ""  # "audio", "visual", "multimodal", "sensor", etc.
    epistemic_profile: PerceptualEpistemicProfile | None = None
    provider_provenance: dict[str, Any] | None = (
        None  # {"provider": ..., "model": ..., "version": ...}
    )
    candidates: list[dict[str, Any]] = Field(
        default_factory=list
    )  # Ranked classification candidates
    incorporation_status: str = "pending"  # pending | incorporated | stale | redundant
    incorporated_transitions: dict[str, str] = Field(
        default_factory=dict,
        description="Idempotency map: {proposition_id: transition_id}. "
        "Same evidence may legitimately affect multiple propositions.",
    )
    novelty_score: float = 1.0  # Composite multidimensional novelty [0.0, 1.0]
    temporal_pattern: str = "unknown"  # persistent | transition | transient | periodic | unknown
    last_incorporated_at: float = 0.0  # Timestamp of most recent incorporation


def has_evidence_integration(node: Any) -> bool:
    """Check if a node has evidence integration fields.

    Use this instead of ``isinstance(node, EvidenceNode)`` when the
    code needs to work with both discovery and perceptual evidence.
    """
    return hasattr(node, "incorporation_status") and hasattr(node, "incorporated_transitions")


# ── Discovery & Epistemic Nodes ──────────────────────────────────────────


class EvidenceNode(EvidenceIntegrationMixin, HCIRNode):
    """A structured unit of evidence with full provenance.

    Discovery-specific evidence — papers, experiments, claims.
    Carries methodology, sample size, and reproducibility metadata
    used by the discovery engine for evidence quality scoring.

    Shares epistemic integration fields (incorporation_status,
    incorporated_transitions, novelty_score, temporal_pattern) with
    ``PerceptualEvidenceNode`` via ``EvidenceIntegrationMixin``.
    """

    node_type: HCIRNodeType = HCIRNodeType.EVIDENCE
    category: CognitiveCategory = CognitiveCategory.DISCOVERY
    claim_id: str = ""  # Which claim does this evidence support/refute?
    source_uri: str = ""  # Paper DOI, dataset URL, experiment ID, etc.
    methodology: str = ""  # How was this evidence produced?
    limitations: list[str] = Field(default_factory=list)
    dataset_refs: list[str] = Field(default_factory=list)
    reproducible: bool = False  # Has this been independently reproduced?
    sample_size: int | None = None  # Statistical sample size, if applicable
    effect_size: float | None = None  # Measured effect size, if applicable


class PerceptualEvidenceNode(EvidenceIntegrationMixin, HCIRNode):
    """Modality-neutral evidence from any perception provider.

    The UNIVERSAL CONTRACT between raw world sensing and HCIR.
    Whisper, YOLO, an IMU, and a LiDAR all produce these.
    HCIR does not care which provider created it.

    Shares epistemic integration fields (incorporation_status,
    incorporated_transitions, novelty_score, temporal_pattern) with
    ``EvidenceNode`` via ``EvidenceIntegrationMixin``, enabling both
    discovery and perceptual evidence to participate in the same
    Bayesian belief revision pipeline.

    Invariant:
        Providers emit ``PerceptualObservation``.
        The ``EvidenceNormalizer`` converts to ``PerceptualEvidenceNode``.
        HCIR owns state transitions from evidence → belief.

    Examples::

        # Vision (YOLO)
        PerceptualEvidenceNode(
            proposition=Proposition(
                subject="person_17", predicate="located_at",
                object_value="kitchen"),
            spatial=SpatialContext(
                frame_id="camera_0",
                bounding_box=BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8)),
        )

        # Audio (Whisper)
        PerceptualEvidenceNode(
            proposition=Proposition(
                subject="utterance_42", predicate="transcribed_as",
                object_value="turn on the lights"),
        )

        # IMU
        PerceptualEvidenceNode(
            proposition=Proposition(
                subject="robot", predicate="acceleration",
                object_value=[0.1, -0.3, 9.8], object_type="vector3"),
        )
    """

    node_type: HCIRNodeType = HCIRNodeType.PERCEPTUAL_EVIDENCE
    category: CognitiveCategory = CognitiveCategory.PERCEPTION

    # What was observed (modality-neutral semantic triple)
    proposition: Proposition = Field(default_factory=Proposition)

    # Where in the world
    spatial: SpatialContext | None = None

    # When — with observation/ingestion distinction
    temporal_validity: TemporalValidity = Field(default_factory=TemporalValidity)

    # Modality-specific opaque payload (schema varies by provider)
    payload: dict[str, Any] = Field(default_factory=dict)


class ClaimNode(HCIRNode):
    """A testable assertion extracted from any source.

    Claims are the bridge between unstructured knowledge (papers,
    observations, conversations) and structured epistemic reasoning.
    Every claim can be supported or contradicted by evidence.
    """

    node_type: HCIRNodeType = HCIRNodeType.CLAIM
    category: CognitiveCategory = CognitiveCategory.DISCOVERY
    statement: str = ""  # The claim text
    source_uri: str = ""  # Where this claim came from
    source_type: str = ""  # "paper", "observation", "simulation", "expert", "inference"
    extracted_at: float = 0.0
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    domain: str = ""  # Knowledge domain


class ExperimentNode(HCIRNode):
    """A designed experiment to test one or more hypotheses.

    Experiments are the mechanism through which hypotheses are tested
    and beliefs are revised.  Each experiment has a reality level
    (from simulation to physical) that determines its evidential weight
    and safety requirements.

    The key capability is **discriminative power**: an experiment should
    ideally distinguish between competing hypotheses, not just confirm one.
    """

    node_type: HCIRNodeType = HCIRNodeType.EXPERIMENT
    category: CognitiveCategory = CognitiveCategory.DISCOVERY
    hypothesis_ids: list[str] = Field(default_factory=list)  # Hypotheses being tested
    design: str = ""  # Experiment protocol description
    variables: dict[str, Any] = Field(default_factory=dict)  # Independent/dependent vars
    controls: list[str] = Field(default_factory=list)  # Control conditions
    expected_outcomes: dict[str, str] = Field(
        default_factory=dict
    )  # hypothesis_id → predicted outcome
    actual_outcome: str = ""  # What actually happened
    reality_level: ExperimentRealityLevel = ExperimentRealityLevel.SIMULATION
    experiment_status: ExperimentStatus = ExperimentStatus.DESIGNED
    discriminating_power: Confidence = 0.5  # How well does this distinguish hypotheses?
    resource_cost: float = 0.0  # Estimated resource cost
    research_program_id: str = ""  # Parent research program


class ContradictionNode(HCIRNode):
    """An identified contradiction between claims, evidence, observations, or beliefs.

    Contradictions are not errors — they are **opportunities**.
    Every contradiction potentially reveals a hidden variable,
    a context dependency, or a flawed assumption.

    The contradiction engine actively searches for these rather
    than treating them as problems to resolve.
    """

    node_type: HCIRNodeType = HCIRNodeType.CONTRADICTION
    category: CognitiveCategory = CognitiveCategory.DISCOVERY
    claim_a_id: str = ""  # First conflicting claim/evidence/observation
    claim_b_id: str = ""  # Second conflicting claim/evidence/observation
    contradiction_type: str = ""  # "direct", "statistical", "methodological", "contextual"
    contradiction_level: PerceptualContradictionLevel = (
        PerceptualContradictionLevel.LEVEL_1_CLASSIFIER_DISAGREEMENT
    )
    observation_ids: list[str] = Field(default_factory=list)
    conflicting_belief_ids: list[str] = Field(default_factory=list)
    possible_explanations: list[str] = Field(default_factory=list)
    resolution_status: str = "unresolved"  # "unresolved", "explained", "resolved", "accepted"
    investigation_priority: Confidence = 0.5
    discovered_variables: list[str] = Field(default_factory=list)  # Hidden variables found


class UnknownNode(HCIRNode):
    """A knowledge gap — discovery starts here.

    Discovery does not start from facts.  It starts from gaps.
    Unknowns represent what the system knows it doesn't know.
    They drive the research agenda by generating hypotheses
    and prioritizing experiments.

    Example::

        Observation: Patients with X recover faster.
        Known: Age, Medication, Genetics
        Unknown: Why? Missing mechanism.

    The discovery loop::

        Observation → Unknown → Hypothesis → Prediction →
        Experiment → Knowledge
    """

    node_type: HCIRNodeType = HCIRNodeType.UNKNOWN
    category: CognitiveCategory = CognitiveCategory.DISCOVERY
    question: str = ""  # What don't we know?
    context: str = ""  # What do we know around this gap?
    domain: str = ""  # Knowledge domain
    trigger: DiscoveryTrigger = DiscoveryTrigger.KNOWLEDGE_GAP  # What triggered this unknown?
    related_observations: list[str] = Field(default_factory=list)  # ObservationNode IDs
    related_hypotheses: list[str] = Field(default_factory=list)  # HypothesisNode IDs
    importance: Confidence = 0.5  # How important is filling this gap?
    estimated_difficulty: Confidence = 0.5  # How hard is this to resolve?
    research_program_id: str = ""  # Parent research program
    objective_id: str = ""  # Parent research objective, if any
    knowledge_value: KnowledgeValue = Field(default_factory=KnowledgeValue)


class ResearchProgramNode(HCIRNode):
    """A long-lived research program — the cognitive object for sustained inquiry.

    Unlike a TaskFrame (ephemeral, per-goal), a ResearchProgram may persist
    for months or years.  It owns the full research lifecycle for a single
    research question and tracks all hypotheses, evidence, experiments,
    and findings.

    Example::

        ResearchProgram: "Understanding Alzheimer's progression"
        Contains:
            Questions (UnknownNodes)
            Hypotheses (HypothesisNodes)
            Evidence (EvidenceNodes)
            Experiments (ExperimentNodes)
            Findings (BeliefNodes)
    """

    node_type: HCIRNodeType = HCIRNodeType.RESEARCH_PROGRAM
    category: CognitiveCategory = CognitiveCategory.DISCOVERY
    title: str = ""
    research_question: str = ""  # The central question
    description: str = ""
    status: str = "active"  # "active", "paused", "completed", "abandoned"
    hypothesis_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)
    unknown_ids: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)  # Concluded BeliefNode IDs
    contradiction_ids: list[str] = Field(default_factory=list)
    objective_ids: list[str] = Field(default_factory=list)  # Research objectives
    overall_confidence: Confidence = 0.0  # How confident are we in conclusions?
    current_strategy: ResearchStrategyType = ResearchStrategyType.EXPLORATION
    started_at: float = Field(default_factory=time.time)
    cognitive_mode: CognitiveMode = CognitiveMode.DISCOVERY


class ResearchObjectiveNode(HCIRNode):
    """A measurable research objective under a research program.

    Research programs decompose into objectives, which decompose into
    questions, which produce unknowns.  This hierarchy scales much
    better than flat research questions::

        ResearchProgram: "Reduce battery degradation"
            │
            ├── Objective: "Find dominant degradation mechanism"
            │       ├── Question: "Why does degradation accelerate after 800 cycles?"
            │       │       └── Unknown: "Electrolyte reaction?"
            │       └── Question: "What role does temperature play?"
            │               └── Unknown: "Non-linear thermal effects?"
            │
            └── Objective: "Develop mitigation strategy"
                    └── Question: "Can charging profiles reduce degradation?"
    """

    node_type: HCIRNodeType = HCIRNodeType.RESEARCH_OBJECTIVE
    category: CognitiveCategory = CognitiveCategory.DISCOVERY
    objective: str = ""  # The measurable objective statement
    program_id: str = ""  # Parent research program
    question_ids: list[str] = Field(default_factory=list)  # Questions under this objective
    success_criteria: str = ""  # How to determine if objective is met
    progress: Confidence = 0.0  # [0.0, 1.0] progress toward success criteria
    priority: Priority = 0.5  # Relative priority within program


# ── World Model & Predictive Nodes ───────────────────────────────────────


class WorldVariableNode(HCIRNode):
    """An environmental parameter in the world model (e.g. temperature, humidity, market demand)."""

    node_type: HCIRNodeType = HCIRNodeType.WORLD_VARIABLE
    category: CognitiveCategory = CognitiveCategory.PERCEPTION
    variable_name: str = ""
    value: Any = None
    unit: str = ""
    min_value: float | None = None
    max_value: float | None = None


class PhysicalEntityNode(HCIRNode):
    """A physical asset, component, or system in the physical world."""

    node_type: HCIRNodeType = HCIRNodeType.PHYSICAL_ENTITY
    category: CognitiveCategory = CognitiveCategory.PERCEPTION
    entity_name: str = ""
    entity_type: str = ""
    status: str = "operational"
    properties: dict[str, Any] = Field(default_factory=dict)


class EnvironmentStateNode(HCIRNode):
    """A macro snapshot of environmental state."""

    node_type: HCIRNodeType = HCIRNodeType.ENVIRONMENT_STATE
    category: CognitiveCategory = CognitiveCategory.PERCEPTION
    environment_name: str = ""
    active_variables: list[str] = Field(default_factory=list)
    overall_status: str = "nominal"


# ═══════════════════════════════════════════════════════════════════════════
# Node Type Registry — for deserialization & validation
# ═══════════════════════════════════════════════════════════════════════════

#: Maps ``HCIRNodeType`` enum values to their typed subclass.
NODE_TYPE_REGISTRY: dict[HCIRNodeType, type[HCIRNode]] = {
    # --- Directives ---
    HCIRNodeType.INTENT: IntentNode,
    HCIRNodeType.GOAL: GoalNode,
    HCIRNodeType.CONSTRAINT: ConstraintNode,
    # --- Epistemology hierarchy ---
    HCIRNodeType.OBSERVATION: ObservationNode,
    HCIRNodeType.FACT: FactNode,
    HCIRNodeType.BELIEF: BeliefNode,
    HCIRNodeType.BELIEF_TRANSITION: BeliefTransitionNode,
    HCIRNodeType.HYPOTHESIS: HypothesisNode,
    HCIRNodeType.PREDICTION: PredictionNode,
    HCIRNodeType.PREDICTION_ERROR: PredictionErrorNode,
    # --- Discovery & Epistemic ---
    HCIRNodeType.EVIDENCE: EvidenceNode,
    HCIRNodeType.CLAIM: ClaimNode,
    HCIRNodeType.EXPERIMENT: ExperimentNode,
    HCIRNodeType.CONTRADICTION: ContradictionNode,
    HCIRNodeType.UNKNOWN: UnknownNode,
    HCIRNodeType.RESEARCH_PROGRAM: ResearchProgramNode,
    HCIRNodeType.RESEARCH_OBJECTIVE: ResearchObjectiveNode,
    # --- Execution ---
    HCIRNodeType.ACTION: ActionNode,
    HCIRNodeType.EVENT: EventNode,
    HCIRNodeType.RESOURCE: ResourceNode,
    HCIRNodeType.CAPABILITY: CapabilityNode,
    # --- Memory classes ---
    HCIRNodeType.EPISODE: EpisodeNode,
    HCIRNodeType.CONCEPT: ConceptNode,
    HCIRNodeType.SKILL: SkillNode,
    HCIRNodeType.PROCEDURE: ProcedureNode,
    HCIRNodeType.VALUE: ValueNode,
    HCIRNodeType.EXTERNAL_KNOWLEDGE: ExternalKnowledgeNode,
    # --- Grounded Perception ---
    HCIRNodeType.VISUAL_OBSERVATION: VisualObservationNode,
    HCIRNodeType.VISUAL_CONCEPT: VisualConceptNode,
    HCIRNodeType.AUDIO_OBSERVATION: AudioObservationNode,
    HCIRNodeType.ACOUSTIC_CONCEPT: AcousticConceptNode,
    # --- World Model ---
    HCIRNodeType.WORLD_VARIABLE: WorldVariableNode,
    HCIRNodeType.PHYSICAL_ENTITY: PhysicalEntityNode,
    HCIRNodeType.ENVIRONMENT_STATE: EnvironmentStateNode,
}


# ═══════════════════════════════════════════════════════════════════════════
# Hyperedge
# ═══════════════════════════════════════════════════════════════════════════


class HCIREdge(BaseModel):
    """A typed hyperedge connecting multiple source and target nodes.

    Hyperedges allow representations like "Goal G1 depends on
    Constraint C1, Resource R1, and Capability Cap1" as a single
    relationship.
    """

    id: str = Field(default_factory=lambda: _new_id("e"))
    edge_type: HCIREdgeType
    sources: list[str]  # Node IDs
    targets: list[str]  # Node IDs
    properties: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)
    weight: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Graph — single graph with views
# ═══════════════════════════════════════════════════════════════════════════


class CognitiveGraph:
    """The unified typed cognitive hypergraph.

    A single graph instance holds all cognitive entities.  Different
    subsystems access filtered *views* (knowledge, execution, memory,
    simulation) rather than separate graph objects.

    Internally maintains hash-map indexes for O(1) lookups by ID,
    and secondary indexes by type, category, lifecycle, scope, and
    capability for efficient querying.
    """

    def __init__(self) -> None:
        # ── Primary storage ──
        self._nodes: dict[str, HCIRNode] = {}
        self._edges: dict[str, HCIREdge] = {}

        # ── Secondary indexes (type → set of node IDs) ──
        self._idx_by_type: dict[HCIRNodeType, set[str]] = {t: set() for t in HCIRNodeType}
        self._idx_by_category: dict[CognitiveCategory, set[str]] = {
            c: set() for c in CognitiveCategory
        }
        self._idx_by_lifecycle: dict[NodeLifecycle, set[str]] = {l: set() for l in NodeLifecycle}
        self._idx_by_scope: dict[str, set[str]] = {}  # tenant_id → node IDs
        self._idx_by_tag: dict[str, set[str]] = {}  # tag → node IDs

        # ── Edge indexes ──
        self._edges_by_source: dict[str, set[str]] = {}  # node_id → edge IDs
        self._edges_by_target: dict[str, set[str]] = {}  # node_id → edge IDs

    # ── Node Operations ──────────────────────────────────────────────

    def add_node(self, node: HCIRNode) -> None:
        """Add a node to the graph.  Raises ValueError on duplicate ID."""
        if node.id in self._nodes:
            raise ValueError(f"Duplicate node ID: {node.id}")
        self._nodes[node.id] = node
        self._index_node(node)

    def upsert_node(self, node: HCIRNode) -> None:
        """Add or replace a node in the graph."""
        old = self._nodes.get(node.id)
        if old is not None:
            self._deindex_node(old)
        self._nodes[node.id] = node
        self._index_node(node)

    def remove_node(self, node_id: str) -> HCIRNode | None:
        """Remove a node and all its connected edges.  Returns the removed node."""
        node = self._nodes.pop(node_id, None)
        if node is None:
            return None
        self._deindex_node(node)
        # Remove edges connected to this node
        connected_edge_ids = set()
        connected_edge_ids.update(self._edges_by_source.pop(node_id, set()))
        connected_edge_ids.update(self._edges_by_target.pop(node_id, set()))
        for eid in connected_edge_ids:
            edge = self._edges.pop(eid, None)
            if edge:
                # Clean up reverse indexes for other endpoints
                for src in edge.sources:
                    if src != node_id:
                        self._edges_by_source.get(src, set()).discard(eid)
                for tgt in edge.targets:
                    if tgt != node_id:
                        self._edges_by_target.get(tgt, set()).discard(eid)
        return node

    def get_node(self, node_id: str) -> HCIRNode | None:
        """Retrieve a node by ID.  O(1)."""
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def all_nodes(self) -> Iterator[HCIRNode]:
        """Iterate over all nodes."""
        yield from self._nodes.values()

    # ── Edge Operations ──────────────────────────────────────────────

    def add_edge(self, edge: HCIREdge) -> None:
        """Add an edge.  Raises ValueError on duplicate ID or dangling refs."""
        if edge.id in self._edges:
            raise ValueError(f"Duplicate edge ID: {edge.id}")
        for nid in edge.sources + edge.targets:
            if nid not in self._nodes:
                raise ValueError(f"Dangling edge reference: node '{nid}' not in graph")
        self._edges[edge.id] = edge
        for src in edge.sources:
            self._edges_by_source.setdefault(src, set()).add(edge.id)
        for tgt in edge.targets:
            self._edges_by_target.setdefault(tgt, set()).add(edge.id)

    def remove_edge(self, edge_id: str) -> HCIREdge | None:
        """Remove an edge by ID."""
        edge = self._edges.pop(edge_id, None)
        if edge is None:
            return None
        for src in edge.sources:
            self._edges_by_source.get(src, set()).discard(edge_id)
        for tgt in edge.targets:
            self._edges_by_target.get(tgt, set()).discard(edge_id)
        return edge

    def get_edge(self, edge_id: str) -> HCIREdge | None:
        return self._edges.get(edge_id)

    def has_edge(self, edge_id: str) -> bool:
        return edge_id in self._edges

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def all_edges(self) -> Iterator[HCIREdge]:
        yield from self._edges.values()

    def edges_from(self, node_id: str) -> list[HCIREdge]:
        """All edges where ``node_id`` is a source."""
        return [
            self._edges[eid]
            for eid in self._edges_by_source.get(node_id, set())
            if eid in self._edges
        ]

    def edges_to(self, node_id: str) -> list[HCIREdge]:
        """All edges where ``node_id`` is a target."""
        return [
            self._edges[eid]
            for eid in self._edges_by_target.get(node_id, set())
            if eid in self._edges
        ]

    # ── Indexed Queries ──────────────────────────────────────────────

    def nodes_by_type(self, node_type: HCIRNodeType) -> list[HCIRNode]:
        """O(k) where k is the number of nodes of that type."""
        return [
            self._nodes[nid]
            for nid in self._idx_by_type.get(node_type, set())
            if nid in self._nodes
        ]

    def nodes_by_category(self, category: CognitiveCategory) -> list[HCIRNode]:
        return [
            self._nodes[nid]
            for nid in self._idx_by_category.get(category, set())
            if nid in self._nodes
        ]

    def nodes_by_lifecycle(self, lifecycle: NodeLifecycle) -> list[HCIRNode]:
        return [
            self._nodes[nid]
            for nid in self._idx_by_lifecycle.get(lifecycle, set())
            if nid in self._nodes
        ]

    def nodes_by_scope(self, tenant_id: str) -> list[HCIRNode]:
        return [
            self._nodes[nid]
            for nid in self._idx_by_scope.get(tenant_id, set())
            if nid in self._nodes
        ]

    def nodes_by_tag(self, tag: str) -> list[HCIRNode]:
        return [self._nodes[nid] for nid in self._idx_by_tag.get(tag, set()) if nid in self._nodes]

    # ── Views ────────────────────────────────────────────────────────

    def knowledge_view(self) -> list[HCIRNode]:
        """Facts, Beliefs, Concepts, Procedures — slowly changing knowledge."""
        knowledge_types = {
            HCIRNodeType.FACT,
            HCIRNodeType.BELIEF,
            HCIRNodeType.CONCEPT,
            HCIRNodeType.PROCEDURE,
            HCIRNodeType.EXTERNAL_KNOWLEDGE,
        }
        result: list[HCIRNode] = []
        for t in knowledge_types:
            result.extend(self.nodes_by_type(t))
        return result

    def execution_view(self) -> list[HCIRNode]:
        """Goals, Actions, Resources — rapidly changing execution state."""
        exec_types = {
            HCIRNodeType.GOAL,
            HCIRNodeType.INTENT,
            HCIRNodeType.ACTION,
            HCIRNodeType.RESOURCE,
            HCIRNodeType.CONSTRAINT,
        }
        result: list[HCIRNode] = []
        for t in exec_types:
            result.extend(self.nodes_by_type(t))
        return result

    def memory_view(self) -> list[HCIRNode]:
        """Episodes, Skills, Values — episodic and procedural memory."""
        mem_types = {
            HCIRNodeType.EPISODE,
            HCIRNodeType.SKILL,
            HCIRNodeType.VALUE,
        }
        result: list[HCIRNode] = []
        for t in mem_types:
            result.extend(self.nodes_by_type(t))
        return result

    def discovery_view(self) -> list[HCIRNode]:
        """Hypotheses, Evidence, Experiments, Claims, Contradictions, Unknowns — epistemic state."""
        discovery_types = {
            HCIRNodeType.HYPOTHESIS,
            HCIRNodeType.EVIDENCE,
            HCIRNodeType.EXPERIMENT,
            HCIRNodeType.CLAIM,
            HCIRNodeType.CONTRADICTION,
            HCIRNodeType.UNKNOWN,
            HCIRNodeType.RESEARCH_PROGRAM,
            HCIRNodeType.PREDICTION,
            HCIRNodeType.PREDICTION_ERROR,
        }
        result: list[HCIRNode] = []
        for t in discovery_types:
            result.extend(self.nodes_by_type(t))
        return result

    # ── Internal Indexing ────────────────────────────────────────────

    def _index_node(self, node: HCIRNode) -> None:
        self._idx_by_type[node.node_type].add(node.id)
        self._idx_by_category[node.category].add(node.id)
        self._idx_by_lifecycle[node.lifecycle].add(node.id)
        self._idx_by_scope.setdefault(node.scope.tenant_id, set()).add(node.id)
        for tag in node.tags:
            self._idx_by_tag.setdefault(tag, set()).add(node.id)

    def _deindex_node(self, node: HCIRNode) -> None:
        self._idx_by_type[node.node_type].discard(node.id)
        self._idx_by_category[node.category].discard(node.id)
        self._idx_by_lifecycle[node.lifecycle].discard(node.id)
        scope_set = self._idx_by_scope.get(node.scope.tenant_id)
        if scope_set:
            scope_set.discard(node.id)
        for tag in node.tags:
            tag_set = self._idx_by_tag.get(tag)
            if tag_set:
                tag_set.discard(node.id)
