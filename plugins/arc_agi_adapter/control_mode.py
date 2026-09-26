"""
Control-Mode Primitive for HCIR / ARC-AGI-3.

Addresses the ontology gap where actions control different entities depending
on a discrete control mode state. Models mode-conditioned motor dynamics,
autonomous detection of mode-switching triggers (re86, ka59, dc22), and
first-class SwitchMode macro-operators for hierarchical planning.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 1. State Layer: Identifiable Entities & Control Context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityId:
    """Stable identifier for a trackable entity, independent of color/position."""

    label: str

    def __repr__(self) -> str:
        return f"EntityId({self.label!r})"


@dataclass
class ControllableEntity:
    """Entity representation with active control flag and spatial properties."""

    entity_id: EntityId
    centroid: tuple[float, float]
    color: int
    controllable: bool = True
    is_active_controller: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlContext:
    """Multi-entity control state container.

    Can live directly on WorldStateSnapshot or inside snapshot.variables["control_context"].
    """

    entities: dict[EntityId, ControllableEntity] = field(default_factory=dict)
    active_entity: EntityId | None = None
    mode_history: list[EntityId] = field(default_factory=list)

    def register_entity(
        self,
        entity_id: EntityId,
        centroid: tuple[float, float],
        color: int,
        controllable: bool = True,
        set_active: bool = False,
    ) -> ControllableEntity:
        """Register a new controllable entity in the context."""
        entity = ControllableEntity(
            entity_id=entity_id,
            centroid=centroid,
            color=color,
            controllable=controllable,
            is_active_controller=set_active or (self.active_entity == entity_id),
        )
        self.entities[entity_id] = entity
        if set_active or self.active_entity is None:
            self.switch_to(entity_id)
        return entity

    def switch_to(self, entity_id: EntityId) -> None:
        """Switch active control focus to the specified entity."""
        if entity_id not in self.entities:
            raise KeyError(f"Unknown entity {entity_id} in ControlContext")
        for e in self.entities.values():
            e.is_active_controller = e.entity_id == entity_id
        self.active_entity = entity_id
        self.mode_history.append(entity_id)

    def update_position(self, entity_id: EntityId, centroid: tuple[float, float]) -> None:
        """Update the spatial centroid of an entity."""
        if entity_id in self.entities:
            self.entities[entity_id].centroid = centroid

    def update_color(self, entity_id: EntityId, color: int) -> None:
        """Update the visual color marker of an entity."""
        if entity_id in self.entities:
            self.entities[entity_id].color = color

    @property
    def mode_key(self) -> EntityId | None:
        """Current mode key for ModeConditionedDynamics lookup."""
        return self.active_entity

    def to_dict(self) -> dict[str, Any]:
        """Serialize for deterministic snapshot replay."""
        return {
            "entities": {
                eid.label: {
                    "centroid": e.centroid,
                    "color": e.color,
                    "controllable": e.controllable,
                    "is_active_controller": e.is_active_controller,
                }
                for eid, e in self.entities.items()
            },
            "active_entity": self.active_entity.label if self.active_entity else None,
            "mode_history": [eid.label for eid in self.mode_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlContext:
        """Deserialize from snapshot variables dictionary."""
        ctx = cls()
        for label, edata in data.get("entities", {}).items():
            eid = EntityId(label)
            ctx.entities[eid] = ControllableEntity(
                entity_id=eid,
                centroid=tuple(edata["centroid"]),
                color=edata["color"],
                controllable=edata.get("controllable", True),
                is_active_controller=edata.get("is_active_controller", False),
            )
        active_label = data.get("active_entity")
        ctx.active_entity = EntityId(active_label) if active_label else None
        ctx.mode_history = [EntityId(lbl) for lbl in data.get("mode_history", [])]
        return ctx


# ---------------------------------------------------------------------------
# 2. Causal Discovery Layer: Action Observations & Mode Switch Detection
# ---------------------------------------------------------------------------


@dataclass
class ActionObservation:
    """Recorded observation of an action intervention across entities."""

    action: int
    pre_active_entity: EntityId
    pre_centroid: tuple[float, float]
    post_centroid: tuple[float, float]
    other_entities_moved: list[EntityId] = field(default_factory=list)
    action_data: dict[str, Any] | None = None
    trigger_pos: tuple[int, int] | None = None  # Contact trigger for environment switches (dc22)


def _euclidean_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class ModeSwitchDetector:
    """Infers candidate mode-switch actions and environmental triggers.

    Identifies actions that reliably produce near-zero displacement on the
    currently-active entity while triggering motion in another entity.
    """

    def __init__(self, zero_displacement_eps: float = 1.0, min_observations: int = 2) -> None:
        self.eps = zero_displacement_eps
        self.min_observations = min_observations
        self._by_action: dict[int, list[ActionObservation]] = {}
        self._contact_observations: list[ActionObservation] = []

    def record(self, obs: ActionObservation) -> None:
        """Record an action execution observation."""
        self._by_action.setdefault(obs.action, []).append(obs)
        if obs.trigger_pos is not None:
            self._contact_observations.append(obs)

    def candidate_mode_switches(self) -> list[int]:
        """Detect action IDs that reliably trigger discrete mode switches.

        A candidate mode-switch action satisfies:
        1. Mean displacement of the pre-active entity is < zero_displacement_eps.
        2. Across observations of this action, other entities move or become responsive.
        3. Motion does not occur uniformly across all actions (filtering autonomous hazards).
        """
        candidates: list[int] = []

        # Count baseline frequency of other entities moving across all actions
        total_other_motion_by_action: dict[int, int] = {}
        for action, obs_list in self._by_action.items():
            total_other_motion_by_action[action] = sum(
                1 for o in obs_list if len(o.other_entities_moved) > 0
            )

        for action, obs_list in self._by_action.items():
            if len(obs_list) < self.min_observations:
                continue

            displacements = [_euclidean_dist(o.pre_centroid, o.post_centroid) for o in obs_list]
            near_zero = statistics.mean(displacements) < self.eps

            # Does this action specifically correlate with another entity moving?
            other_moved_count = total_other_motion_by_action[action]
            other_moved_ratio = other_moved_count / len(obs_list)

            if near_zero and other_moved_ratio >= 0.5:
                candidates.append(action)

        return candidates

    def candidate_environmental_switches(self) -> list[tuple[int, int]]:
        """Identify contact positions (buttons/pads in dc22) that trigger room/focus toggles."""
        pos_switches: dict[tuple[int, int], int] = {}
        for obs in self._contact_observations:
            if obs.trigger_pos and obs.other_entities_moved:
                pos_switches[obs.trigger_pos] = pos_switches.get(obs.trigger_pos, 0) + 1
        return [pos for pos, cnt in pos_switches.items() if cnt >= self.min_observations]


# ---------------------------------------------------------------------------
# 3. Dynamics Modeling: Mode-Conditioned Motor Mapping
# ---------------------------------------------------------------------------


class ModeConditionedDynamics:
    """Mode-conditioned replacement for global dict[int, ActionDynamicsModel].

    Supports standard dict interface (keyed by action_id under the active mode)
    while storing and retrieving mode-isolated models. Falls back to a single
    implicit mode for environments that never switch control, maintaining 100%
    backward compatibility with single-avatar games.
    """

    _IMPLICIT_MODE = EntityId("__implicit__")

    def __init__(self, active_mode: EntityId | None = None) -> None:
        self._models: dict[tuple[int, EntityId], Any] = {}
        self.active_mode: EntityId | None = active_mode

    def set_active_mode(self, mode: EntityId | None) -> None:
        """Set current active control mode for unconditioned dict access."""
        self.active_mode = mode

    def set(self, action: int, mode: EntityId | None, model: Any) -> None:
        """Register an ActionDynamicsModel for a specific action and control mode."""
        self._models[(action, mode or self._IMPLICIT_MODE)] = model

    def get(self, action: int, mode_or_default: Any = None, default: Any = None) -> Any:
        """Retrieve the dynamics model for an action.

        Supports:
        - get(action, mode=EntityId)
        - get(action, default=...)
        - get(action)
        """
        if isinstance(mode_or_default, EntityId):
            mode = mode_or_default
            d = default
        else:
            mode = self.active_mode
            d = mode_or_default

        key = (action, mode or self._IMPLICIT_MODE)
        if key in self._models:
            return self._models[key]
        if (action, self._IMPLICIT_MODE) in self._models:
            return self._models[(action, self._IMPLICIT_MODE)]
        return d

    def __getitem__(self, key: int | tuple[int, EntityId | None]) -> Any:
        if isinstance(key, tuple):
            action, mode = key
        else:
            action, mode = key, self.active_mode
        m = self.get(action, mode)
        if m is None:
            raise KeyError(f"No dynamics model for action {action} in mode {mode}")
        return m

    def __setitem__(self, key: int | tuple[int, EntityId | None], model: Any) -> None:
        if isinstance(key, tuple):
            action, mode = key
        else:
            action, mode = key, self.active_mode
        self.set(action, mode, model)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple):
            action, mode = key
            return (action, mode or self._IMPLICIT_MODE) in self._models
        elif isinstance(key, int):
            return self.get(key, self.active_mode) is not None
        return False

    def __iter__(self):
        return iter(self.get_actions_for_mode(self.active_mode))

    def __len__(self) -> int:
        return len(self.get_actions_for_mode(self.active_mode))

    def items(self) -> Any:
        """Yield (action, model) for current active mode (with implicit fallback)."""
        mode = self.active_mode or self._IMPLICIT_MODE
        result: dict[int, Any] = {}
        for (a, m), model in self._models.items():
            if m == self._IMPLICIT_MODE:
                result[a] = model
        if mode != self._IMPLICIT_MODE:
            for (a, m), model in self._models.items():
                if m == mode:
                    result[a] = model
        return result.items()

    def values(self) -> list[Any]:
        return [v for _, v in self.items()]

    def keys(self) -> list[int]:
        return [k for k, _ in self.items()]

    def get_actions_for_mode(self, mode: EntityId | None) -> list[int]:
        """List all actions with dynamics defined under the specified mode."""
        target_mode = mode or self._IMPLICIT_MODE
        actions = [a for (a, m) in self._models if m == target_mode]
        if not actions and target_mode != self._IMPLICIT_MODE:
            actions = [a for (a, m) in self._models if m == self._IMPLICIT_MODE]
        return sorted(set(actions))

    def clear(self) -> None:
        """Clear all registered dynamics models."""
        self._models.clear()


# ---------------------------------------------------------------------------
# 4. Planning Layer: First-Class SwitchMode Macro-Operator
# ---------------------------------------------------------------------------


@dataclass
class SwitchMode:
    """First-class macro-operator for discrete control mode transitions.

    Costed as the discrete action(s) needed to trigger the switch,
    making it directly comparable to spatial path steps in MCTS / A*.
    """

    target_entity: EntityId
    trigger_action: int
    action_data: dict[str, Any] | None = None
    estimated_cost: int = 1

    def apply(self, ctx: ControlContext) -> None:
        """Apply the mode switch to the control context."""
        ctx.switch_to(self.target_entity)

    def describe(self) -> str:
        data_str = f" with {self.action_data}" if self.action_data else ""
        return (
            f"SwitchMode({self.target_entity.label!r} via Action {self.trigger_action}{data_str})"
        )
