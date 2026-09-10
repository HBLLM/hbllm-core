"""Typed data structures and enumerations for the Developmental Learning Adapter (A23).

Defines the primitives for the BabyWorld simulator, sensor representations,
the three-layer Blank-Brain cognitive profile, and causal hypotheses.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BabyObjectType(str, Enum):
    """Raw physical entity geometry/category in BabyWorld."""

    BALL = "ball"
    BLOCK = "block"
    BOX = "box"
    CONTAINER = "container"
    DOOR = "door"
    BUTTON = "button"
    RAMP = "ramp"
    SURFACE = "surface"
    TOOL = "tool"
    OBSTACLE = "obstacle"


class BabyRelationType(str, Enum):
    """Spatial and physical relational links between entities."""

    ON = "ON"
    INSIDE = "INSIDE"
    NEAR = "NEAR"
    FAR = "FAR"
    TOUCHING = "TOUCHING"
    BLOCKING = "BLOCKING"
    SUPPORTING = "SUPPORTING"
    CONNECTED = "CONNECTED"
    MOVING = "MOVING"


class BabyActionType(str, Enum):
    """Primitive motor actions available to the embodied agent."""

    LOOK = "LOOK"
    MOVE = "MOVE"
    REACH = "REACH"
    GRASP = "GRASP"
    RELEASE = "RELEASE"
    PUSH = "PUSH"
    PULL = "PULL"
    PLACE = "PLACE"
    OPEN = "OPEN"
    CLOSE = "CLOSE"


@dataclass
class Vector2D:
    """2D spatial coordinates and extents."""

    x: float
    y: float

    def distance_to(self, other: Vector2D) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class BabyObjectState:
    """Complete physical ground-truth state of an object in BabyWorld."""

    id: str
    object_type: BabyObjectType
    color: str
    mass: float  # e.g. 1.0 (light) vs 10.0 (heavy)
    size: Vector2D  # width, height
    position: Vector2D
    velocity: Vector2D = field(default_factory=lambda: Vector2D(0.0, 0.0))
    is_fixed: bool = False  # Fixed to floor/table or movable
    is_open: bool | None = None  # For containers / doors
    contained_in: str | None = None  # ID of container if inside
    held_by_agent: bool = False
    is_occluded: bool = False  # True if hidden behind an occluder from agent's eye
    surface_friction: float = 1.0


@dataclass
class SensoryObservation:
    """Multi-modal raw sensory observation produced by BabyWorld."""

    step_index: int
    # Vision: List of perceived visual detections (neutral IDs, raw perceptual features only)
    vision: list[dict[str, Any]]
    # Depth: Distance to visible surfaces along rays or per detected entity
    depth: dict[str, float]
    # Audio: Sound frequency and volume signals (if emitted by collisions or buttons)
    audio: list[dict[str, Any]]
    # Touch: Contact sensor reading (force / tactile contact True/False)
    touch: bool
    # Proprioception: Agent hand/effector position, velocity, and effort expended
    proprioception: dict[str, Any]
    # Occlusion events: Signals if line-of-sight is blocked
    occluded_entity_ids: list[str] = field(default_factory=list)


@dataclass
class DevelopmentalProfile:
    """Phase 2: Three-layer developmental initialization profile.

    Enforces the scientific boundary between innate cognitive machinery
    and developmentally learned knowledge.
    """

    # --- Layer 1: Innate Machinery (Active and Preserved) ---
    hcir_graph_enabled: bool = True
    event_sourcing_enabled: bool = True
    epistemic_bookkeeping_enabled: bool = True
    prediction_machinery_enabled: bool = True
    simulation_sandbox_enabled: bool = True
    action_interface_enabled: bool = True
    dual_store_memory_enabled: bool = True
    active_inference_enabled: bool = True

    # --- Layer 2: Developmentally Learned (Strictly Empty at Initialization) ---
    semantic_concepts: dict[str, Any] = field(default_factory=dict)
    object_categories: dict[str, Any] = field(default_factory=dict)
    causal_rules: list[dict[str, Any]] = field(default_factory=list)
    affordances: dict[str, list[str]] = field(default_factory=dict)
    spatial_schemas: list[dict[str, Any]] = field(default_factory=list)
    procedural_skills: dict[str, Any] = field(default_factory=dict)
    lexical_mapping: dict[str, str] = field(default_factory=dict)

    def verify_blank_brain(self) -> bool:
        """Verify that all learned knowledge layers are strictly empty."""
        return (
            len(self.semantic_concepts) == 0
            and len(self.object_categories) == 0
            and len(self.causal_rules) == 0
            and len(self.affordances) == 0
            and len(self.spatial_schemas) == 0
            and len(self.procedural_skills) == 0
            and len(self.lexical_mapping) == 0
        )


class BeliefTransitionType(str, Enum):
    """Immutable event types for event-sourced developmental belief logging."""

    HYPOTHESIS_CREATED = "hypothesis_created"
    HYPOTHESIS_TESTED = "hypothesis_tested"
    HYPOTHESIS_FALSIFIED = "hypothesis_falsified"
    HYPOTHESIS_CONFIRMED = "hypothesis_confirmed"
    CONFIDENCE_CHANGED = "confidence_changed"
    RULE_GENERALIZED = "rule_generalized"
    RULE_REVISED = "rule_revised"


@dataclass
class BeliefTransitionEvent:
    """An immutable record of a developmental belief transition."""

    event_id: str = field(default_factory=lambda: f"bte_{uuid.uuid4().hex[:8]}")
    event_type: BeliefTransitionType = BeliefTransitionType.HYPOTHESIS_CREATED
    step_index: int = 0
    hypothesis_id: str = ""
    variable: str = ""  # e.g. "color", "mass", "shape"
    condition: str = ""  # e.g. "color == 'red'", "mass < 5.0"
    prior_confidence: float = 0.0
    posterior_confidence: float = 0.0
    is_falsified: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class CausalHypothesis:
    """State tracker for candidate causal relationships."""

    hypothesis_id: str = field(default_factory=lambda: f"hyp_{uuid.uuid4().hex[:6]}")
    action: BabyActionType = BabyActionType.PUSH
    variable: str = "color"  # Candidate variable being tested
    operator: str = "=="
    value: Any = "red"
    consequence: str = "MOVES"
    confidence: float = 0.5  # Prior belief
    interventions_tested: int = 0
    falsified: bool = False
    confirmed: bool = False
    supporting_episodes: list[str] = field(default_factory=list)
    counterexamples: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return f"{self.action.value}(x) ∧ ({self.variable} {self.operator} {self.value}) => {self.consequence}"
