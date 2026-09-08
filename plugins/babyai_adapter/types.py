"""
Type definitions and canonical MiniGrid / BabyAI specifications.

Conforms strictly to the official Farama MiniGrid and BabyAI schemas:
- Object encodings: 0: unseen, 1: empty, 2: wall, 3: floor, 4: door, 5: key, 6: ball, 7: box, 8: goal, 9: lava, 10: agent
- Color encodings: 0: red, 1: green, 2: blue, 3: purple, 4: yellow, 5: grey
- State encodings: 0: open, 1: closed, 2: locked
- Actions: 0: left, 1: right, 2: forward, 3: pickup, 4: drop, 5: toggle, 6: done
- Directions: 0: East, 1: South, 2: West, 3: North
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class MiniGridObjectType(IntEnum):
    UNSEEN = 0
    EMPTY = 1
    WALL = 2
    FLOOR = 3
    DOOR = 4
    KEY = 5
    BALL = 6
    BOX = 7
    GOAL = 8
    LAVA = 9
    AGENT = 10


MiniGridObject = MiniGridObjectType


class MiniGridColor(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2
    PURPLE = 3
    YELLOW = 4
    GREY = 5


class MiniGridState(IntEnum):
    OPEN = 0
    CLOSED = 1
    LOCKED = 2


class MiniGridAction(IntEnum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    PICKUP = 3
    DROP = 4
    TOGGLE = 5
    DONE = 6


class MiniGridDirection(IntEnum):
    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3


IDX_TO_OBJECT = {e.value: e.name.lower() for e in MiniGridObjectType}
OBJECT_TO_IDX = {name: idx for idx, name in IDX_TO_OBJECT.items()}

IDX_TO_COLOR = {e.value: e.name.lower() for e in MiniGridColor}
COLOR_TO_IDX = {name: idx for idx, name in IDX_TO_COLOR.items()}

IDX_TO_STATE = {e.value: e.name.lower() for e in MiniGridState}
STATE_TO_IDX = {name: idx for idx, name in IDX_TO_STATE.items()}

IDX_TO_ACTION = {e.value: e.name.lower() for e in MiniGridAction}
ACTION_TO_IDX = {name: idx for idx, name in IDX_TO_ACTION.items()}

# Direction vectors: (dx, dy) in world coordinates (x right/east, y down/south)
DIR_TO_VEC = {
    MiniGridDirection.EAST: (1, 0),
    MiniGridDirection.SOUTH: (0, 1),
    MiniGridDirection.WEST: (-1, 0),
    MiniGridDirection.NORTH: (0, -1),
}


@dataclass
class MiniGridObservation:
    """Standard Gym/MiniGrid observation package."""

    image: Any  # ndarray or list with shape (H, W, 3)
    direction: int
    mission: str
    step_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BabyAIGoal:
    """Semantic goal parsed from a BabyAI mission instruction."""

    action: str  # e.g., "go_to", "pickup", "open", "put_next"
    target_type: str = "ball"  # e.g., "ball", "box", "key", "door"
    target_color: str | None = None  # e.g., "red", "green", "blue"
    target_id: str | None = None  # Bound entity ID once identified in graph

    # Tier 4: PutNext fixed landmark target
    fixed_type: str | None = None  # e.g., "box", "key", "ball", "door"
    fixed_color: str | None = None  # e.g., "green", "blue"
    fixed_id: str | None = None

    # Tier 6: Sequential & compound subgoals
    subgoals: list[BabyAIGoal] = field(default_factory=list)
    sequence_mode: str = "single"  # "single", "sequence", "and", "after"
    current_subgoal_idx: int = 0

    language: str = "en"
    raw_instruction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_compound(self) -> bool:
        """Return True if this goal contains multiple sequential subgoals."""
        return len(self.subgoals) > 0

    def get_active_subgoal(self) -> BabyAIGoal:
        """Return the current active subgoal (or self if single)."""
        if self.is_compound() and self.current_subgoal_idx < len(self.subgoals):
            return self.subgoals[self.current_subgoal_idx]
        return self

    def advance_subgoal(self) -> bool:
        """Advance to next subgoal. Returns True if advanced, False if all completed."""
        if self.is_compound():
            self.current_subgoal_idx += 1
            return self.current_subgoal_idx < len(self.subgoals)
        return False

    def is_all_completed(self) -> bool:
        """Return True if all subgoals have been marked completed."""
        if self.is_compound():
            return self.current_subgoal_idx >= len(self.subgoals)
        return False

    def matches_attributes(
        self,
        entity_type: str,
        color: str | None = None,
        entity_id: str | None = None,
    ) -> bool:
        """Check if an entity satisfies this goal's type and color requirements."""
        if self.target_id and entity_id and self.target_id != entity_id:
            return False
        if self.target_type and self.target_type != entity_type:
            return False
        if self.target_color and color and self.target_color != color:
            return False
        return True

    def matches_fixed_attributes(self, entity_type: str, color: str | None = None) -> bool:
        """Check if an entity satisfies the PutNext fixed landmark requirements."""
        if self.fixed_type and self.fixed_type != entity_type:
            return False
        if self.fixed_color and color and self.fixed_color != color:
            return False
        return True
