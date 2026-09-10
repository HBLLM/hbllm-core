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

    def __init__(
        self, seed: int | None = 42, scenario: str | None = None, random_seed: int | None = None
    ) -> None:
        actual_seed = seed if random_seed is None else random_seed
        self.rng = random.Random(actual_seed)
        self.step_index: int = 0
        self.agent_position: Vector2D = Vector2D(0.0, 0.0)
        self.agent_held_object_id: str | None = None
        self.objects: dict[str, BabyObjectState] = {}
        self.occluders: list[str] = []
        self._state_snapshots: list[dict[str, Any]] = []
        if scenario is not None:
            self.reset(scenario=scenario)

    @property
    def agent_hand_position(self) -> Vector2D:
        """Proprioceptive effector/hand position of the embodied agent."""
        return self.agent_position

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
        elif scenario == "randomized_confounded_world":
            self._setup_randomized_confounded_world()
        elif scenario == "friction_confounded_world":
            self._setup_friction_confounded_world()
        elif scenario == "affordance_discovery_world":
            self._setup_affordance_discovery_world()
        elif scenario == "containment_world":
            self._setup_containment_world()
        elif scenario == "tool_use_world":
            self._setup_tool_use_world()
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

    def _setup_randomized_confounded_world(self, mode_override: int | None = None) -> str:
        """A23.5-E2: Causal Variable Invariance.

        Dynamically varies the surface confounder across episodes:
        Mode 0: Red -> Light, Blue -> Heavy
        Mode 1: Blue -> Light, Red -> Heavy
        Mode 2: Block -> Light, Ball -> Heavy
        Mode 3: Ball -> Light, Block -> Heavy
        Mode 4: Purple -> Light, Cyan -> Heavy
        Mode 5: Orange -> Light, Black -> Heavy

        The true causal invariant is always: mass < MASS_THRESHOLD (5.0).
        """
        mode = mode_override if mode_override is not None else self.rng.randint(0, 5)

        if mode == 0:
            confound_desc = "color:red->light,blue->heavy"
            pos_color, neg_color = "red", "blue"
            pos_type, neg_type = BabyObjectType.BALL, BabyObjectType.BLOCK
        elif mode == 1:
            confound_desc = "color:blue->light,red->heavy"
            pos_color, neg_color = "blue", "red"
            pos_type, neg_type = BabyObjectType.BLOCK, BabyObjectType.BALL
        elif mode == 2:
            confound_desc = "shape:block->light,ball->heavy"
            pos_color, neg_color = "green", "green"
            pos_type, neg_type = BabyObjectType.BLOCK, BabyObjectType.BALL
        elif mode == 3:
            confound_desc = "shape:ball->light,block->heavy"
            pos_color, neg_color = "yellow", "yellow"
            pos_type, neg_type = BabyObjectType.BALL, BabyObjectType.BLOCK
        elif mode == 4:
            confound_desc = "color:purple->light,cyan->heavy"
            pos_color, neg_color = "purple", "cyan"
            pos_type, neg_type = BabyObjectType.BALL, BabyObjectType.BLOCK
        else:
            confound_desc = "color:orange->light,black->heavy"
            pos_color, neg_color = "orange", "black"
            pos_type, neg_type = BabyObjectType.BLOCK, BabyObjectType.BALL

        # 4 Correlated Light objects (Moves)
        for i in range(4):
            self.objects[f"obj_pos_{i}"] = BabyObjectState(
                id=f"obj_pos_{i}",
                object_type=pos_type,
                color=pos_color,
                mass=1.2 + i * 0.4,
                size=Vector2D(0.4, 0.4),
                position=Vector2D(0.5 + i * 0.3, 0.5 + i * 0.2),
            )

        # 4 Correlated Heavy objects (Does not move)
        for i in range(4):
            self.objects[f"obj_neg_{i}"] = BabyObjectState(
                id=f"obj_neg_{i}",
                object_type=neg_type,
                color=neg_color,
                mass=11.0 + i * 1.5,
                size=Vector2D(0.5, 0.5),
                position=Vector2D(0.5 + i * 0.3, -0.5 - i * 0.2),
            )

        # 2 Contrastive Decoupling Probes:
        # A: Negative surface feature, but LIGHT (Moves!) -> Falsifies surface correlation
        self.objects["obj_contrast_light"] = BabyObjectState(
            id="obj_contrast_light",
            object_type=neg_type,
            color=neg_color,
            mass=1.1,
            size=Vector2D(0.4, 0.4),
            position=Vector2D(1.5, 0.0),
        )
        # B: Positive surface feature, but HEAVY (Does not move!) -> Falsifies surface correlation
        self.objects["obj_contrast_heavy"] = BabyObjectState(
            id="obj_contrast_heavy",
            object_type=pos_type,
            color=pos_color,
            mass=14.0,
            size=Vector2D(0.5, 0.5),
            position=Vector2D(1.5, -1.0),
        )

        return confound_desc

    def _setup_friction_confounded_world(self) -> str:
        """A23.5-E3 Novel Causal Mechanism: Surface Friction Confounder.

        Physical Setup:
        - All objects have identical mass (5.0 kg), so mass CANNOT explain motion differences.
        - Push Force = 5.0 N.
        - True Physical Law: Moves if Force (5.0) > Mass (5.0) * Friction (μ).
          => Moves if μ < 1.0.

        Observational Confound:
        - Green objects are Smooth (μ in [0.15, 0.30] < 1.0 => MOVE).
        - Yellow objects are Rough (μ in [1.80, 2.50] > 1.0 => STATIONARY).
        - Spurious Correlation: color == 'green' correlates 100% with movement.

        Contrastive Decoupling Probes:
        - obj_contrast_yellow_smooth: Yellow, but μ = 0.20 => MOVES! (Falsifies color)
        - obj_contrast_green_rough: Green, but μ = 2.00 => STATIONARY! (Falsifies color)
        """
        # 1. Four Correlated Smooth Green Objects (Movers)
        self.objects["obj_green_smooth_ball"] = BabyObjectState(
            id="obj_green_smooth_ball",
            object_type=BabyObjectType.BALL,
            color="green",
            mass=5.0,
            surface_friction=0.20,
            size=Vector2D(0.4, 0.4),
            position=Vector2D(1.0, 1.0),
        )
        self.objects["obj_green_smooth_block"] = BabyObjectState(
            id="obj_green_smooth_block",
            object_type=BabyObjectType.BLOCK,
            color="green",
            mass=5.0,
            surface_friction=0.25,
            size=Vector2D(0.5, 0.5),
            position=Vector2D(1.2, 0.2),
        )
        self.objects["obj_green_smooth_cylinder"] = BabyObjectState(
            id="obj_green_smooth_cylinder",
            object_type=BabyObjectType.BLOCK,
            color="green",
            mass=5.0,
            surface_friction=0.18,
            size=Vector2D(0.3, 0.6),
            position=Vector2D(0.9, 0.6),
        )
        self.objects["obj_green_smooth_box"] = BabyObjectState(
            id="obj_green_smooth_box",
            object_type=BabyObjectType.BOX,
            color="green",
            mass=5.0,
            surface_friction=0.30,
            size=Vector2D(0.4, 0.4),
            position=Vector2D(1.4, 0.8),
        )

        # 2. Four Correlated Rough Yellow Objects (Non-Movers)
        self.objects["obj_yellow_rough_ball"] = BabyObjectState(
            id="obj_yellow_rough_ball",
            object_type=BabyObjectType.BALL,
            color="yellow",
            mass=5.0,
            surface_friction=2.00,
            size=Vector2D(0.4, 0.4),
            position=Vector2D(0.8, -1.0),
        )
        self.objects["obj_yellow_rough_block"] = BabyObjectState(
            id="obj_yellow_rough_block",
            object_type=BabyObjectType.BLOCK,
            color="yellow",
            mass=5.0,
            surface_friction=2.20,
            size=Vector2D(0.6, 0.6),
            position=Vector2D(1.5, -0.5),
        )
        self.objects["obj_yellow_rough_cylinder"] = BabyObjectState(
            id="obj_yellow_rough_cylinder",
            object_type=BabyObjectType.BLOCK,
            color="yellow",
            mass=5.0,
            surface_friction=1.80,
            size=Vector2D(0.4, 0.7),
            position=Vector2D(1.1, -1.3),
        )
        self.objects["obj_yellow_rough_box"] = BabyObjectState(
            id="obj_yellow_rough_box",
            object_type=BabyObjectType.BOX,
            color="yellow",
            mass=5.0,
            surface_friction=2.50,
            size=Vector2D(0.5, 0.5),
            position=Vector2D(1.6, -1.1),
        )

        # 3. Two Contrastive Decoupling Probes
        self.objects["obj_contrast_yellow_smooth"] = BabyObjectState(
            id="obj_contrast_yellow_smooth",
            object_type=BabyObjectType.BALL,
            color="yellow",
            mass=5.0,
            surface_friction=0.20,
            size=Vector2D(0.4, 0.4),
            position=Vector2D(0.5, 1.2),
        )
        self.objects["obj_contrast_green_rough"] = BabyObjectState(
            id="obj_contrast_green_rough",
            object_type=BabyObjectType.BLOCK,
            color="green",
            mass=5.0,
            surface_friction=2.00,
            size=Vector2D(0.5, 0.5),
            position=Vector2D(1.3, -1.2),
        )

        return "friction:green->smooth(0.2),yellow->rough(2.0)"

    def generate_observational_demonstrations(self) -> list[dict[str, Any]]:
        """Generate standardized observational demonstration episodes on training objects.

        Simulates observational trials (analogous to adult demonstrations in infant cognitive studies).
        Standardized push actions are applied to correlated training objects, and the resulting physical
        motion (displacement, moved: True/False) is recorded.
        These observational episodes contain NO active agent intervention and establish
        the empirical correlation before the agent begins interventional reasoning.
        """
        demos: list[dict[str, Any]] = []
        # Target observational objects (exclude contrastive/test probes)
        obs_ids = [
            oid
            for oid in self.objects.keys()
            if not any(k in oid for k in ("contrast", "light_ball", "heavy_block"))
        ]

        saved_state = self.save_state()
        for oid in obs_ids:
            obj = self.objects[oid]
            # In demonstrations, push force is applied directly adjacent to the object
            self.agent_position = Vector2D(obj.position.x - 0.2, obj.position.y)
            _, _, _, consequences = self.step(BabyActionType.PUSH, target_id=oid)
            moved = consequences.get("moved", False)
            disp = consequences.get("displacement", 0.0)

            demos.append(
                {
                    "action": BabyActionType.PUSH,
                    "percept_id": oid,
                    "features": {
                        "color": obj.color,
                        "shape": obj.object_type.value,
                        "size_extent": obj.size.to_tuple(),
                        "mass_sensation": obj.mass,
                        "surface_friction": obj.surface_friction,
                    },
                    "outcome": "MOVES" if moved else "STATIONARY",
                    "moved": moved,
                    "displacement": disp,
                }
            )
            self.restore_state(saved_state)

        return demos

    def _setup_affordance_discovery_world(self) -> None:
        """Stage D4 Affordance Discovery World: Spheres, blocks, containers, and immovable obstacles."""
        self.objects["obj_ball_red"] = BabyObjectState(
            id="obj_ball_red",
            object_type=BabyObjectType.BALL,
            color="red",
            mass=1.0,
            size=Vector2D(0.4, 0.4),
            position=Vector2D(0.4, 0.2),
            rollable=True,
        )
        self.objects["obj_ball_blue"] = BabyObjectState(
            id="obj_ball_blue",
            object_type=BabyObjectType.BALL,
            color="blue",
            mass=1.5,
            size=Vector2D(0.4, 0.4),
            position=Vector2D(0.4, -0.2),
            rollable=True,
        )
        self.objects["obj_block_green"] = BabyObjectState(
            id="obj_block_green",
            object_type=BabyObjectType.BLOCK,
            color="green",
            mass=2.0,
            size=Vector2D(0.5, 0.5),
            position=Vector2D(0.3, 0.3),
            rollable=False,
        )
        self.objects["obj_block_yellow"] = BabyObjectState(
            id="obj_block_yellow",
            object_type=BabyObjectType.BLOCK,
            color="yellow",
            mass=3.0,
            size=Vector2D(0.5, 0.5),
            position=Vector2D(0.3, -0.3),
            rollable=False,
        )
        self.objects["obj_box_open"] = BabyObjectState(
            id="obj_box_open",
            object_type=BabyObjectType.BOX,
            color="brown",
            mass=4.0,
            size=Vector2D(0.6, 0.6),
            position=Vector2D(0.4, 0.0),
            is_container=True,
            is_open=True,
            rollable=False,
        )
        self.objects["obj_heavy_pillar"] = BabyObjectState(
            id="obj_heavy_pillar",
            object_type=BabyObjectType.BLOCK,
            color="gray",
            mass=50.0,
            size=Vector2D(0.8, 0.8),
            position=Vector2D(1.0, 1.0),
            is_fixed=True,
            rollable=False,
        )

    def _setup_containment_world(self) -> None:
        """Stage D2 Spatial Containment & Transport World."""
        self.objects["obj_container_box"] = BabyObjectState(
            id="obj_container_box",
            object_type=BabyObjectType.BOX,
            color="blue",
            mass=3.0,
            size=Vector2D(0.7, 0.7),
            position=Vector2D(0.4, 0.1),
            is_container=True,
            is_open=True,
            contained_object_ids=["obj_toy_ball", "obj_toy_cube"],
        )
        self.objects["obj_toy_ball"] = BabyObjectState(
            id="obj_toy_ball",
            object_type=BabyObjectType.BALL,
            color="red",
            mass=0.5,
            size=Vector2D(0.2, 0.2),
            position=Vector2D(0.4, 0.1),
            contained_in="obj_container_box",
            rollable=True,
        )
        self.objects["obj_toy_cube"] = BabyObjectState(
            id="obj_toy_cube",
            object_type=BabyObjectType.BLOCK,
            color="yellow",
            mass=0.6,
            size=Vector2D(0.2, 0.2),
            position=Vector2D(0.4, 0.1),
            contained_in="obj_container_box",
            rollable=False,
        )
        self.objects["obj_outside_ball"] = BabyObjectState(
            id="obj_outside_ball",
            object_type=BabyObjectType.BALL,
            color="green",
            mass=0.8,
            size=Vector2D(0.3, 0.3),
            position=Vector2D(0.4, -0.6),
            contained_in=None,
            rollable=True,
        )

    def _setup_tool_use_world(self) -> None:
        """Stage D5 Tool Use & Compositional Causal Chains World."""
        # Distant target: distance = 2.2 > REACH_DISTANCE (1.5)
        self.objects["obj_distant_reward"] = BabyObjectState(
            id="obj_distant_reward",
            object_type=BabyObjectType.BALL,
            color="gold",
            mass=0.8,
            size=Vector2D(0.3, 0.3),
            position=Vector2D(2.2, 0.0),
            rollable=True,
        )
        # Functional Tool: Reachable (distance = 0.61 <= 1.5), Length = 1.0 (1.5 + 1.0 = 2.5 > 2.2)
        self.objects["obj_stick_tool"] = BabyObjectState(
            id="obj_stick_tool",
            object_type=BabyObjectType.BLOCK,
            color="brown",
            mass=0.7,
            size=Vector2D(0.1, 1.0),
            position=Vector2D(0.6, 0.1),
            is_tool=True,
            tool_length=1.0,
            rollable=False,
        )
        # Distractor 1: Ineffective tool (too short, length = 0.2 -> max reach 1.7 < 2.2)
        self.objects["obj_short_twig"] = BabyObjectState(
            id="obj_short_twig",
            object_type=BabyObjectType.BLOCK,
            color="brown",
            mass=0.3,
            size=Vector2D(0.1, 0.2),
            position=Vector2D(0.6, -0.2),
            is_tool=True,
            tool_length=0.2,
            rollable=False,
        )
        # Distractor 2: Ungraspable heavy boulder (mass 25.0 > 10.0 limit)
        self.objects["obj_heavy_boulder"] = BabyObjectState(
            id="obj_heavy_boulder",
            object_type=BabyObjectType.BLOCK,
            color="gray",
            mass=25.0,
            size=Vector2D(0.6, 0.6),
            position=Vector2D(0.6, 0.35),
            is_tool=False,
            tool_length=0.0,
            rollable=False,
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
        parameter: Any = None,
    ) -> tuple[SensoryObservation, float, bool, dict[str, Any]]:
        """Execute primitive embodied action in the environment."""
        self.step_index += 1
        params = dict(parameters or {})
        if parameter is not None:
            if isinstance(parameter, dict):
                params.update(parameter)
            elif isinstance(parameter, Vector2D):
                params["position"] = parameter
            elif isinstance(parameter, (tuple, list)):
                params["position"] = parameter
            else:
                params["param"] = parameter

        reward = 0.0
        consequences: dict[str, Any] = {"action": action.value, "target_id": target_id}

        target = self.objects.get(target_id) if target_id else None

        if action == BabyActionType.MOVE:
            if "position" in params:
                pos = params["position"]
                self.agent_position = (
                    Vector2D(pos.x, pos.y)
                    if isinstance(pos, Vector2D)
                    else Vector2D(pos[0], pos[1])
                )
            elif "dx" in params or "dy" in params:
                dx = float(params.get("dx", 0.0))
                dy = float(params.get("dy", 0.0))
                self.agent_position.x += dx
                self.agent_position.y += dy
            elif target:
                self.agent_position = Vector2D(target.position.x, target.position.y)
            consequences["new_agent_pos"] = self.agent_position.to_tuple()

            # Moving with held object
            if self.agent_held_object_id and self.agent_held_object_id in self.objects:
                self.objects[self.agent_held_object_id].position = Vector2D(
                    self.agent_position.x, self.agent_position.y
                )

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

                    # Containment Transport: Contained objects move synchronously with container
                    if target.contained_object_ids:
                        for cid in target.contained_object_ids:
                            cobj = self.objects.get(cid)
                            if cobj:
                                cobj.position.x += displacement * math.cos(angle)
                                cobj.position.y += displacement * math.sin(angle)
                                cobj.velocity = Vector2D(
                                    displacement * math.cos(angle), displacement * math.sin(angle)
                                )
                else:
                    target.velocity = Vector2D(0.0, 0.0)
                    consequences["moved"] = False
                    consequences["displacement"] = 0.0

        elif action == BabyActionType.ROLL:
            if target:
                dist = self.agent_position.distance_to(target.position)
                force = float(params.get("force", self.DEFAULT_PUSH_FORCE))
                is_spherical = target.rollable or target.object_type == BabyObjectType.BALL
                if not target.is_fixed and dist <= self.REACH_DISTANCE and is_spherical:
                    angle = math.atan2(
                        target.position.y - self.agent_position.y,
                        target.position.x - self.agent_position.x,
                    )
                    displacement = min(2.0, (force * 1.5) / (target.mass + 0.5))
                    target.position.x += displacement * math.cos(angle)
                    target.position.y += displacement * math.sin(angle)
                    target.velocity = Vector2D(
                        displacement * math.cos(angle), displacement * math.sin(angle)
                    )
                    consequences["rolled"] = True
                    consequences["displacement"] = displacement
                else:
                    target.velocity = Vector2D(0.0, 0.0)
                    consequences["rolled"] = False
                    consequences["displacement"] = 0.0

        elif action == BabyActionType.PLACE:
            container_candidate_id = target_id
            if (
                container_candidate_id == self.agent_held_object_id
                or (container_candidate_id and container_candidate_id not in self.objects)
            ) and params.get("param") in self.objects:
                container_candidate_id = params.get("param")
            elif not container_candidate_id and params.get("param") in self.objects:
                container_candidate_id = params.get("param")

            container_target = (
                self.objects.get(container_candidate_id) if container_candidate_id else target
            )

            if self.agent_held_object_id and self.agent_held_object_id in self.objects:
                held_obj = self.objects[self.agent_held_object_id]
                if container_target and (
                    container_target.is_container
                    or container_target.object_type
                    in (BabyObjectType.CONTAINER, BabyObjectType.BOX)
                ):
                    dist = self.agent_position.distance_to(container_target.position)
                    if dist <= self.REACH_DISTANCE and (
                        container_target.is_open is None or container_target.is_open
                    ):
                        held_obj.held_by_agent = False
                        held_obj.contained_in = container_target.id
                        held_obj.position = Vector2D(
                            container_target.position.x, container_target.position.y
                        )
                        if held_obj.id not in container_target.contained_object_ids:
                            container_target.contained_object_ids.append(held_obj.id)
                        self.agent_held_object_id = None
                        consequences["placed_in_container"] = True
                        consequences["container_id"] = container_target.id
                    else:
                        consequences["placed_in_container"] = False
                else:
                    held_obj.held_by_agent = False
                    self.agent_held_object_id = None
                    consequences["placed_on_floor"] = True

        elif action in (BabyActionType.PULL, BabyActionType.EXTEND):
            if target:
                held_tool = (
                    self.objects.get(self.agent_held_object_id)
                    if self.agent_held_object_id
                    else None
                )
                tool_len = held_tool.tool_length if (held_tool and held_tool.is_tool) else 0.0
                effective_reach = self.REACH_DISTANCE + tool_len
                dist = self.agent_position.distance_to(target.position)
                if dist <= effective_reach and not target.is_fixed and target.mass < 15.0:
                    pull_dist = min(dist - 0.4, 1.2)
                    angle = math.atan2(
                        self.agent_position.y - target.position.y,
                        self.agent_position.x - target.position.x,
                    )
                    target.position.x += pull_dist * math.cos(angle)
                    target.position.y += pull_dist * math.sin(angle)
                    consequences["pulled"] = True
                    consequences["new_distance"] = self.agent_position.distance_to(target.position)
                    consequences["tool_used"] = held_tool.id if held_tool else None
                    consequences["effective_reach"] = effective_reach
                else:
                    consequences["pulled"] = False
                    consequences["out_of_reach"] = dist > effective_reach

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
                    "surface_friction": obj.surface_friction,  # Surface texture/friction estimate
                    "is_held": obj.held_by_agent,
                    "is_container": obj.is_container,
                    "contained_in": obj.contained_in,
                    "is_tool": obj.is_tool,
                    "tool_length": obj.tool_length,
                    "rollable": obj.rollable or obj.object_type == BabyObjectType.BALL,
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
