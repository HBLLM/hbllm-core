"""
Semantic Ambiguity Adapter Types.

Defines strongly-typed representations of ambiguous linguistic instructions,
causal subgoals, ambiguity confidence classifications, and semantic benchmark tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SemanticTier(StrEnum):
    """Canonical linguistic ambiguity benchmark evaluation tiers."""

    TIER_1_CANONICAL_EXPLICIT = "tier_1_canonical_explicit"
    TIER_2_SYNONYM_PARAPHRASE = "tier_2_synonym_paraphrase"
    TIER_3_UNDERSPECIFIED_ELLIPTICAL = "tier_3_underspecified_elliptical"
    TIER_4_CONFLICTING_CORRECTION = "tier_4_conflicting_correction"
    TIER_5_ABSTRACT_INTENT = "tier_5_abstract_intent"


@dataclass
class CausalSubgoal:
    """Explicit executable causal subgoal."""

    verb: str  # "pick", "place", "toggle", "clean"
    target: str  # "red_key", "mug_1", "projector"
    destination: str | None = None  # "blue_box", "table", "on"


@dataclass
class SemanticObservation:
    """Observation containing natural language instruction and physical scene state."""

    instruction: str
    scene_objects: dict[str, dict[str, Any]]  # id -> {location, state, color}
    completed_subgoals: list[str] = field(default_factory=list)
    pending_subgoals: list[str] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 20
    done: bool = False
    won: bool = False
    ambiguity_flagged: bool = False
    info: dict[str, Any] = field(default_factory=dict)
