"""Perceptual Base Abstractions — HBLLM Perception §A8.

Modality-neutral primitives shared by all perception modalities
(vision, audio, touch, IMU, biosignals, ...).

These are NOT abstract base classes with forced inheritance.
They are structural contracts that modality-specific types compose with.

Hierarchy:
    PerceptualObservation  — "a sensor received a signal"
    PerceptualAssessment   — "here is what perception thinks about it"

Both are thin. All modality-specific semantics remain in specialized types.

Design rationale:
    Vision and audio independently developed identical patterns:
        observation_id, segment_id, timestamp, duration, provenance
    This module extracts that common substrate so future modalities
    (touch, depth, IMU) don't reinvent it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.types import Provenance
from hbllm.perception.providers.provider_provenance import ProviderProvenance


def _new_observation_id(prefix: str = "obs") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _new_segment_id(prefix: str = "seg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ═══════════════════════════════════════════════════════════════════════════
# Perceptual Observation — "a sensor received a signal"
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PerceptualObservation:
    """Modality-neutral observation base.

    Common to all perception modalities. Records the fact that
    a sensor received a signal at a specific time.

    Modality-specific types (AcousticObservation, VisualEvidence)
    compose with this — they don't inherit from it.

    Attributes:
        observation_id: Unique ID for this specific sensor reading.
        segment_id: Identifies the input segment that was analyzed.
        timestamp: When this observation was made (epoch seconds).
        duration: Duration of the observed signal in seconds.
        provenance: How this observation was created.

    """

    observation_id: str = field(
        default_factory=lambda: _new_observation_id("pobs"),
    )
    segment_id: str = field(
        default_factory=lambda: _new_segment_id("pseg"),
    )
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    provenance: Provenance = field(default_factory=Provenance)


# ═══════════════════════════════════════════════════════════════════════════
# Perceptual Assessment — "here is what perception thinks"
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PerceptualAssessment:
    """Modality-neutral assessment base.

    Common structure for all perception modalities' assessment output.
    Each modality specializes the candidates and epistemic profile.

    Attributes:
        modality: Which modality produced this ("audio", "vision", ...).
        observation: The underlying perceptual observation.
        candidates: Modality-specific candidate evidence list.
        epistemic_profile: Modality-specific confidence dimensions.
        provider_provenance: Which provider(s) contributed.
        timestamp: When this assessment was produced.
        proposed_label: User-provided label (for learning).
        proposed_context: User-provided context.

    """

    modality: str = ""
    observation: PerceptualObservation = field(
        default_factory=PerceptualObservation,
    )
    candidates: list[Any] = field(default_factory=list)
    epistemic_profile: Any = None
    provider_provenance: ProviderProvenance = field(
        default_factory=ProviderProvenance,
    )
    timestamp: float = field(default_factory=time.time)
    proposed_label: str | None = None
    proposed_context: str | None = None
