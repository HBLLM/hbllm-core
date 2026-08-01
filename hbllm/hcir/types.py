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
        user_id: User within tenant.
        device_id: Device within user scope.
        cluster_id: Distributed cluster identifier.
        simulation_id: Simulation branch scope (empty = real world).
        security_level: Access control classification.
    """

    tenant_id: str = "default"
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

    STANDARD = "standard"        # Normal reasoning — produce answers
    DISCOVERY = "discovery"      # Scientific cognition — reduce uncertainty
    DIAGNOSTIC = "diagnostic"    # Root-cause analysis — isolate failures
    CREATIVE = "creative"        # Divergent thinking — maximize novelty
    CRITICAL = "critical"        # Adversarial review — maximize rigor


# ═══════════════════════════════════════════════════════════════════════════
# Epistemic Types — scientific reasoning primitives
# ═══════════════════════════════════════════════════════════════════════════


class FalsificationStatus(StrEnum):
    """Popperian falsification status for beliefs and hypotheses.

    Every belief and hypothesis must be falsifiable.  This enum tracks
    where each stands in the scientific lifecycle::

        UNTESTED → CORROBORATED → (WEAKENED → FALSIFIED | SUPERSEDED)

    A belief that cannot be falsified is not scientific — it's dogma.
    """

    UNTESTED = "untested"            # No prediction has been tested
    CORROBORATED = "corroborated"    # Predictions confirmed, not yet falsified
    WEAKENED = "weakened"            # Some predictions failed
    FALSIFIED = "falsified"          # Critical prediction failed
    SUPERSEDED = "superseded"        # Replaced by a better hypothesis


class EvidenceStrength(StrEnum):
    """Qualitative evidence strength classification.

    Ordered from weakest to strongest.  This is the epistemic
    hierarchy — not all evidence is created equal::

        ANECDOTAL < OBSERVATIONAL < CORRELATIONAL <
        EXPERIMENTAL < META_ANALYTIC < REPLICATED

    The discovery engine uses this to weight evidence during
    belief revision and hypothesis evaluation.
    """

    ANECDOTAL = "anecdotal"            # Single observation, no controls
    OBSERVATIONAL = "observational"    # Systematic observation, no intervention
    CORRELATIONAL = "correlational"    # Statistical relationship identified
    EXPERIMENTAL = "experimental"      # Controlled experiment
    META_ANALYTIC = "meta_analytic"    # Aggregation across multiple studies
    REPLICATED = "replicated"          # Independently reproduced results


class ExperimentStatus(StrEnum):
    """Lifecycle of an experiment from design to completion.

    Tracks the full experiment lifecycle::

        DESIGNED → APPROVED → RUNNING → COMPLETED | FAILED | CANCELLED
    """

    DESIGNED = "designed"      # Experiment plan created
    APPROVED = "approved"      # Safety/governance review passed
    RUNNING = "running"        # Currently executing
    COMPLETED = "completed"    # Finished with results
    FAILED = "failed"          # Execution failed (not the same as negative result)
    CANCELLED = "cancelled"    # Abandoned before completion


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

    SIMULATION = "simulation"          # Pure computational simulation
    DIGITAL = "digital"                # Software-based experiment (A/B test, etc.)
    OBSERVATIONAL = "observational"    # Passive real-world observation
    CONTROLLED = "controlled"          # Controlled real-world experiment
    PHYSICAL = "physical"              # Direct physical manipulation
