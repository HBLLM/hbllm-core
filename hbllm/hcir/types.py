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
    """

    created_by: str = ""
    timestamp: Timestamp = Field(default_factory=time.time)
    engine: str = ""
    prompt_hash: str = ""
    reasoning_step: int = 0
    simulation_branch: str = "main"


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
