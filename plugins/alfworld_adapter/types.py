"""
ALFWorld Adapter Types.

Defines strongly-typed representations of ALFWorld task types, objects,
receptacles, affordances, observations, and goals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ALFWorldTaskType(StrEnum):
    """The 6 canonical ALFWorld / ALFRED household task categories."""

    PICK_AND_PLACE = "pick_and_place"
    EXAMINE_IN_LIGHT = "examine_in_light"
    CLEAN_AND_PLACE = "clean_and_place"
    HEAT_AND_PLACE = "heat_and_place"
    COOL_AND_PLACE = "cool_and_place"
    PICK_TWO_AND_PLACE = "pick_two_and_place"


class ALFWorldAffordance(StrEnum):
    """Physical affordance capabilities of receptacles and objects."""

    OPENABLE = "openable"
    RECEPTACLE = "receptacle"
    HEATABLE = "heatable"
    COOLABLE = "coolable"
    CLEANABLE = "cleanable"
    TOGGLEABLE = "toggleable"


@dataclass
class ALFWorldObject:
    """An interactable household object."""

    id: str
    name: str
    object_type: str
    parent_receptacle: str | None = None
    is_clean: bool = False
    is_hot: bool = False
    is_cold: bool = False
    is_lit: bool = False


@dataclass
class ALFWorldReceptacle:
    """A container, furniture, or apparatus."""

    id: str
    name: str
    receptacle_type: str
    is_openable: bool = False
    is_open: bool = False
    affordances: set[ALFWorldAffordance] = field(default_factory=set)
    contained_objects: list[str] = field(default_factory=list)


@dataclass
class ALFWorldGoal:
    """Parsed goal directive from natural language mission."""

    task_type: ALFWorldTaskType
    target_object_type: str
    target_receptacle_type: str | None = None
    apparatus_receptacle_type: str | None = None
    count_required: int = 1
    raw_instruction: str = ""


@dataclass
class ALFWorldObservation:
    """State observation in text and structured formats."""

    text_obs: str
    current_location: str | None = None
    inventory: list[str] = field(default_factory=list)
    admissible_commands: list[str] = field(default_factory=list)
    goal_instruction: str = ""
    step_count: int = 0
    raw_obs: Any = None
