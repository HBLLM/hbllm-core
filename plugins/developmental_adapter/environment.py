"""BabyWorld Environment — Deterministic Developmental Physics Simulator.

Provides a fully observable, inspectable simulator for early childhood
embodied cognition, sensory occlusion, and interventional causal discovery.
"""

from __future__ import annotations

import copy
import logging
import math
import random
from typing import Any

from .types import (
    BabyActionType,
    BabyObjectState,
    BabyObjectType,
    SensoryObservation,
    Vector2D,
)

logger = logging.getLogger(__name__)


class BabyWorldEnvironment:
    """Deterministic 2D physical environment for developmental cognition."""

    # Physics Constants
    DEFAULT_PUSH_FORCE: float = 5.0
    MASS_THRESHOLD: float = 5.0  # Objects with mass < 5.0 move when pushed with standard force
    REACH_DISTANCE: float = 1.5
    TABLE_EXTENT: tuple[float, float] = (4.0, 4.0)

    def __init__(self, seed: int | None = 42) -> None:
        self.rng = random.Random(seed)
        self.step_index: int = 0
        self.agent_position: Vector2D = Vector2D(0.0, 0.0)
        self.agent_held_object_id: str | None = None
        self.objects: dict[str, BabyObjectState] = {}
        self.occluders: list[str] = []
        self._state_snapshots: list[dict[str, Any]] = []

    def reset(self, scenario: str = "confounded_train_world") -> SensoryObservation:
        """Reset environment to a designated experimental scenario."""
        self.step_index = 0
        self.agent_position = Vector2D(0.0, 0.0)
        self.agent_held_object_id = None
        self.objects.clear()
        self.occluders.clear()
        self._state_snapshots.clear()

        if scenario == "confounded_train_world":
            self._setup_confounded_train_world()
        elif scenario == "unseen_entities_world":
            self._setup_unseen_entities_world()
        elif scenario == "unseen_environment_world":
            self._setup_unseen_environment_world()
        elif scenario == "occlusion_permanence_world":
            self._setup_occlusion_permanence_world()
        else:
            self._setup_confounded_train_world()

        return self.get_sensory_observation()

    def _setup_confounded_train_world(self) -> None:
        """Phase 5 / Canonical A23.5 Confounded Training World.

        Observational Correlation:
        - ALL RED objects are LIGHT (mass < 5.0) and therefore MOVE.
        - ALL BLUE objects are HEAVY (mass >= 5.0) and therefore DO NOT MOVE.

        Hidden True Causal Variable:
        - Mass < MASS_THRESHOLD (5.0), completely independent of color.
        """
        # 1. Red Light Ball (Moves)
        self.objects["obj_red_ball"] = BabyObjectState(
            id="obj_red_ball",
            object_type=BabyObjectType.BALL,
            color="red",
            mass=1.2,
            size=Vector2D(0.4, 0.4),
            position=Vector2D(1.0, 1.0),
        )
        # 2. Red Light Block (Moves)
        self.objects["obj_red_block"] = BabyObjectState(
            id="obj_red_block",
            object_type=BabyObjectType.BLOCK,
            color="red",
            mass=2.5,
            size=Vector2D(0.5, 0.5),
            position=Vector2D(1.2, 0.2),
        )
        # 3. Blue Heavy Ball (Does Not Move)
        self.objects["obj_blue_ball"] = BabyObjectState(
            id="obj_blue_ball",
            object_type=BabyObjectType.BALL,
            color="blue",
            mass=12.0,
            size=Vector2D(0.4, 0.4),
            position=Vector2D(0.8, -1.0),
        )
        # 4. Blue Heavy Block (Does Not Move)
        self.objects["obj_blue_block"] = BabyObjectState(
            id="obj_blue_block",
            object_type=BabyObjectType.BLOCK,
            color="blue",
            mass=15.0,
            size=Vector2D(0.6, 0.6),
            position=Vector2D(1.5, -0.5),
        )
        # Additional correlated objects establishing strong observational correlation
        self.objects["obj_red_cylinder"] = BabyObjectState(
            id="obj_red_cylinder",
            object_type=BabyObjectType.BLOCK,
            color="red",
            mass=1.8,
            size=Vector2D(0.3, 0.6),
            position=Vector2D(0.9, 0.6),
        )
        self.objects["obj_red_small_box"] = BabyObjectState(
            id="obj_red_small_box",
            object_type=BabyObjectType.BOX,
            color="red",
            mass=2.2,
            size=Vector2D(0.4, 0.4),
            position=Vector2D(1.4, 0.8),
        )
        self.objects["obj_blue_heavy_cylinder"] = BabyObjectState(
            id="obj_blue_heavy_cylinder",
            object_type=BabyObjectType.BLOCK,
            color="blue",
            mass=13.5,
            size=Vector2D(0.4, 0.7),
            position=Vector2D(1.1, -1.3),
        )
        self.objects["obj_blue_heavy_box"] = BabyObjectState(
            id="obj_blue_heavy_box",
            object_type=BabyObjectType.BOX,
            color="blue",
            mass=16.0,
            size=Vector2D(0.5, 0.5),
            position=Vector2D(1.6, -1.1),
        )
        # 5. Counterfactual Test Objects for Interventional Verification
        # Blue Light Ball (Contradicts Color Hypothesis: Blue, but Moves!)
        self.objects["obj_blue_light_ball"] = BabyObjectState(
            id="obj_blue_light_ball",
            object_type=BabyObjectType.BALL,
            color="blue",
            mass=1.1,
            size=Vector2D(0.4, 0.4),
            position=Vector2D(0.5, 1.2),
        )
        # Red Heavy Block (Contradicts Color Hypothesis: Red, but Does Not Move!)
        self.objects["obj_red_heavy_block"] = BabyObjectState(
            id="obj_red_heavy_block",
            object_type=BabyObjectType.BLOCK,
            color="red",
            mass=14.5,
            size=Vector2D(0.5, 0.5),
            position=Vector2D(1.3, -1.2),
        )

    def _setup_unseen_entities_world(self) -> None:
        """Level 2 Generalization World: Unseen Entities (Novel colors & shapes)."""
        # Green Cylinder (Light -> Moves)
        self.objects["obj_green_cylinder"] = BabyObjectState(
            id="obj_green_cylinder",
            object_type=BabyObjectType.BLOCK,
            color="green",
            mass=1.8,
            size=Vector2D(0.3, 0.7),
            position=Vector2D(0.9, 0.8),
        )
        # Yellow Cone (Heavy -> Does Not Move)
        self.objects["obj_yellow_cone"] = BabyObjectState(
            id="obj_yellow_cone",
            object_type=BabyObjectType.BLOCK,
            color="yellow",
            mass=11.2,
            size=Vector2D(0.4, 0.6),
            position=Vector2D(1.1, -0.7),
        )
        # Purple Torus (Light -> Moves)
        self.objects["obj_purple_torus"] = BabyObjectState(
            id="obj_purple_torus",
            object_type=BabyObjectType.BALL,
            color="purple",
            mass=0.9,
            size=Vector2D(0.5, 0.5),
            position=Vector2D(0.7, 0.3),
        )

    def _setup_unseen_environment_world(self) -> None:
        """Level 3 Generalization World: Unseen Environment layout with ramp and container."""
        self.objects["obj_container"] = BabyObjectState(
            id="obj_container",
            object_type=BabyObjectType.CONTAINER,
            color="gray",
            mass=20.0,
            size=Vector2D(1.2, 1.2),
            position=Vector2D(1.8, 1.0),
            is_fixed=True,
            is_open=True,
        )
        self.objects["obj_unseen_block"] = BabyObjectState(
            id="obj_unseen_block",
            object_type=BabyObjectType.BLOCK,
            color="orange",
            mass=2.2,
            size=Vector2D(0.4, 0.4),
            position=Vector2D(0.8, 0.9),
        )
        self.objects["obj_heavy_box"] = BabyObjectState(
            id="obj_heavy_box",
            object_type=BabyObjectType.BOX,
            color="brown",
            mass=18.0,
            size=Vector2D(0.8, 0.8),
            position=Vector2D(1.0, -1.0),
            is_fixed=False,
            is_open=False,
        )

    def _setup_occlusion_permanence_world(self) -> None:
        """D1 Object Permanence: Box acts as visual occluder."""
        # Visual Occluder Screen
        self.objects["obj_occluder_screen"] = BabyObjectState(
            id="obj_occluder_screen",
            object_type=BabyObjectType.OBSTACLE,
            color="black",
            mass=50.0,
            size=Vector2D(1.5, 0.2),
            position=Vector2D(1.0, 0.0),
            is_fixed=True,
        )
        self.occluders.append("obj_occluder_screen")

        # Hidden Ball behind screen
        self.objects["obj_hidden_ball"] = BabyObjectState(
            id="obj_hidden_ball",
            object_type=BabyObjectType.BALL,
            color="green",
            mass=1.0,
            size=Vector2D(0.4, 0.4),
            position=Vector2D(1.8, 0.0),  # Behind screen along x-axis line of sight
            is_occluded=True,
        )

    def step(
        self,
        action: BabyActionType,
        target_id: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[SensoryObservation, float, bool, dict[str, Any]]:
        """Execute primitive embodied action in the environment."""
        self.step_index += 1
        params = parameters or {}
        reward = 0.0
        consequences: dict[str, Any] = {"action": action.value, "target_id": target_id}

        target = self.objects.get(target_id) if target_id else None

        if action == BabyActionType.MOVE:
            dx = float(params.get("dx", 0.0))
            dy = float(params.get("dy", 0.0))
            self.agent_position.x += dx
            self.agent_position.y += dy
            consequences["new_agent_pos"] = self.agent_position.to_tuple()

        elif action == BabyActionType.REACH:
            if target:
                dist = self.agent_position.distance_to(target.position)
                reachable = dist <= self.REACH_DISTANCE
                consequences["reachable"] = reachable
                consequences["distance"] = dist

        elif action == BabyActionType.GRASP:
            if target:
                dist = self.agent_position.distance_to(target.position)
                if dist <= self.REACH_DISTANCE and not target.is_fixed and target.mass < 10.0:
                    self.agent_held_object_id = target.id
                    target.held_by_agent = True
                    consequences["grasped"] = True
                else:
                    consequences["grasped"] = False

        elif action == BabyActionType.RELEASE:
            if self.agent_held_object_id and self.agent_held_object_id in self.objects:
                held_obj = self.objects[self.agent_held_object_id]
                held_obj.held_by_agent = False
                self.agent_held_object_id = None
                consequences["released"] = True

        elif action == BabyActionType.PUSH:
            if target:
                dist = self.agent_position.distance_to(target.position)
                force = float(params.get("force", self.DEFAULT_PUSH_FORCE))
                consequences["applied_force"] = force
                consequences["target_mass"] = target.mass

                # True Physical Causal Law:
                # An object moves if force exceeds resistance (mass * friction) and is not fixed
                effective_resistance = target.mass * target.surface_friction
                if (
                    not target.is_fixed
                    and dist <= self.REACH_DISTANCE
                    and force > effective_resistance
                ):
                    # Object moves away from agent
                    angle = math.atan2(
                        target.position.y - self.agent_position.y,
                        target.position.x - self.agent_position.x,
                    )
                    displacement = min(1.0, force / (target.mass + 1.0))
                    target.position.x += displacement * math.cos(angle)
                    target.position.y += displacement * math.sin(angle)
                    target.velocity = Vector2D(
                        displacement * math.cos(angle), displacement * math.sin(angle)
                    )
                    consequences["moved"] = True
                    consequences["displacement"] = displacement
                else:
                    target.velocity = Vector2D(0.0, 0.0)
                    consequences["moved"] = False
                    consequences["displacement"] = 0.0

        elif action == BabyActionType.OPEN:
            if target and target.is_open is not None:
                dist = self.agent_position.distance_to(target.position)
                if dist <= self.REACH_DISTANCE:
                    target.is_open = True
                    consequences["opened"] = True

        elif action == BabyActionType.CLOSE:
            if target and target.is_open is not None:
                dist = self.agent_position.distance_to(target.position)
                if dist <= self.REACH_DISTANCE:
                    target.is_open = False
                    consequences["closed"] = True

        # Update occlusion states
        self._update_occlusion_states()

        obs = self.get_sensory_observation()
        return obs, reward, False, consequences

    def _update_occlusion_states(self) -> None:
        """Calculate line-of-sight visual occlusions."""
        for obj_id, obj in self.objects.items():
            if obj_id in self.occluders:
                obj.is_occluded = False
                continue

            # Check if any occluder lies strictly between agent and object
            is_blocked = False
            for occ_id in self.occluders:
                occ = self.objects.get(occ_id)
                if not occ:
                    continue
                # Line segment collision check
                if self.agent_position.distance_to(occ.position) < self.agent_position.distance_to(
                    obj.position
                ):
                    # Ray projection approximation
                    dx_obj = obj.position.x - self.agent_position.x
                    dy_obj = obj.position.y - self.agent_position.y
                    dx_occ = occ.position.x - self.agent_position.x
                    dy_occ = occ.position.y - self.agent_position.y
                    dot = dx_obj * dx_occ + dy_obj * dy_occ
                    if dot > 0 and abs(dx_obj * dy_occ - dy_obj * dx_occ) < 0.3:
                        is_blocked = True
                        break

            obj.is_occluded = is_blocked

    def get_sensory_observation(self) -> SensoryObservation:
        """Produce multi-modal sensory observation.

        CRITICAL SCIENTIFIC INVARIANT:
        Emits ONLY raw sensory percepts with neutral entity keys.
        Contains ZERO high-level semantic concepts ('BALL', 'CONTAINER')
        or affordances ('ROLLABLE', 'PUSHABLE').
        """
        vision_percepts: list[dict[str, Any]] = []
        depth_readings: dict[str, float] = {}
        occluded_ids: list[str] = []

        for obj_id, obj in self.objects.items():
            if obj.is_occluded:
                occluded_ids.append(obj_id)
                continue

            dist = self.agent_position.distance_to(obj.position)
            depth_readings[obj_id] = round(dist, 3)

            # Raw visual percept: Neutral ID, shape geometry, color string, bounding extents
            vision_percepts.append(
                {
                    "percept_id": obj_id,
                    "shape": obj.object_type.value,  # e.g. "ball", "block" (geometric descriptor)
                    "color": obj.color,
                    "size_extent": obj.size.to_tuple(),
                    "spatial_coordinates": obj.position.to_tuple(),
                    "velocity": obj.velocity.to_tuple(),
                    "mass_sensation": obj.mass,  # Tactile/inertial resistance estimate
                    "is_held": obj.held_by_agent,
                }
            )

        proprioception = {
            "effector_position": self.agent_position.to_tuple(),
            "holding_entity_id": self.agent_held_object_id,
            "effort_expended": 1.0 if self.agent_held_object_id else 0.0,
        }

        has_touch = any(
            self.agent_position.distance_to(o.position) <= 0.6 for o in self.objects.values()
        )

        return SensoryObservation(
            step_index=self.step_index,
            vision=vision_percepts,
            depth=depth_readings,
            audio=[],
            touch=has_touch,
            proprioception=proprioception,
            occluded_entity_ids=occluded_ids,
        )

    # ── Interventional Sandbox / Counterfactual Rollback ─────────────────────

    def save_state(self) -> int:
        """Create an immutable snapshot of world state."""
        snapshot = {
            "step_index": self.step_index,
            "agent_position": copy.deepcopy(self.agent_position),
            "agent_held_object_id": self.agent_held_object_id,
            "objects": copy.deepcopy(self.objects),
            "occluders": list(self.occluders),
        }
        self._state_snapshots.append(snapshot)
        return len(self._state_snapshots) - 1

    def restore_state(self, snapshot_idx: int = -1) -> None:
        """Restore environment to a prior snapshot."""
        if not self._state_snapshots:
            return
        snapshot = self._state_snapshots[snapshot_idx]
        self.step_index = snapshot["step_index"]
        self.agent_position = copy.deepcopy(snapshot["agent_position"])
        self.agent_held_object_id = snapshot["agent_held_object_id"]
        self.objects = copy.deepcopy(snapshot["objects"])
        self.occluders = list(snapshot["occluders"])

    def intervene_property(self, object_id: str, property_name: str, value: Any) -> bool:
        """Execute a formal Pearlian do(property = value) intervention."""
        obj = self.objects.get(object_id)
        if not obj:
            return False
        if hasattr(obj, property_name):
            setattr(obj, property_name, value)
            return True
        return False
