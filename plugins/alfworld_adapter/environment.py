"""
ALFWorld Environment Wrapper.

Provides dual-mode execution:
1. Native ALFWorld TextWorld environment if installed.
2. High-fidelity StandaloneALFWorldEnv implementing all 6 task categories,
   receptacle containment, physical affordance transformations, and admissible text commands.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

from .types import (
    ALFWorldAffordance,
    ALFWorldGoal,
    ALFWorldObject,
    ALFWorldObservation,
    ALFWorldReceptacle,
    ALFWorldTaskType,
)

logger = logging.getLogger(__name__)


class StandaloneALFWorldEnv:
    """
    High-fidelity, zero-dependency ALFWorld text environment.
    Simulates household scenes with affordances (open, clean, heat, cool, toggle),
    receptacle containment, inventory state, and task verification.
    """

    def __init__(
        self,
        task_type: ALFWorldTaskType = ALFWorldTaskType.PICK_AND_PLACE,
        seed: int | None = None,
    ) -> None:
        self.task_type = task_type
        self.rng = random.Random(seed)
        self.step_count = 0
        self.max_steps = 50
        self.inventory: list[ALFWorldObject] = []
        self.current_location: str | None = None
        self.receptacles: dict[str, ALFWorldReceptacle] = {}
        self.objects: dict[str, ALFWorldObject] = {}
        self.goal: ALFWorldGoal | None = None
        self.reset(seed=seed, task_type=task_type)

    def reset(
        self,
        seed: int | None = None,
        task_type: ALFWorldTaskType | None = None,
    ) -> tuple[ALFWorldObservation, dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)
        if task_type is not None:
            self.task_type = task_type

        self.step_count = 0
        self.inventory = []
        self._build_scene()
        self.current_location = self.rng.choice(list(self.receptacles.keys()))
        obs = self._get_obs(
            "You are in the middle of a room. Looking quickly around you, you see various receptacles."
        )
        return obs, {"goal": self.goal}

    def _build_scene(self) -> None:
        """Construct scene objects, receptacles, affordances, and task goal."""
        self.receptacles = {
            "countertop 1": ALFWorldReceptacle(
                "countertop 1",
                "countertop 1",
                "countertop",
                affordances={ALFWorldAffordance.RECEPTACLE},
            ),
            "sinkbasin 1": ALFWorldReceptacle(
                "sinkbasin 1",
                "sinkbasin 1",
                "sinkbasin",
                affordances={ALFWorldAffordance.RECEPTACLE, ALFWorldAffordance.CLEANABLE},
            ),
            "microwave 1": ALFWorldReceptacle(
                "microwave 1",
                "microwave 1",
                "microwave",
                is_openable=True,
                affordances={
                    ALFWorldAffordance.RECEPTACLE,
                    ALFWorldAffordance.OPENABLE,
                    ALFWorldAffordance.HEATABLE,
                },
            ),
            "fridge 1": ALFWorldReceptacle(
                "fridge 1",
                "fridge 1",
                "fridge",
                is_openable=True,
                affordances={
                    ALFWorldAffordance.RECEPTACLE,
                    ALFWorldAffordance.OPENABLE,
                    ALFWorldAffordance.COOLABLE,
                },
            ),
            "cabinet 1": ALFWorldReceptacle(
                "cabinet 1",
                "cabinet 1",
                "cabinet",
                is_openable=True,
                affordances={ALFWorldAffordance.RECEPTACLE, ALFWorldAffordance.OPENABLE},
            ),
            "desk 1": ALFWorldReceptacle(
                "desk 1", "desk 1", "desk", affordances={ALFWorldAffordance.RECEPTACLE}
            ),
            "drawer 1": ALFWorldReceptacle(
                "drawer 1",
                "drawer 1",
                "drawer",
                is_openable=True,
                affordances={ALFWorldAffordance.RECEPTACLE, ALFWorldAffordance.OPENABLE},
            ),
            "desklamp 1": ALFWorldReceptacle(
                "desklamp 1", "desklamp 1", "desklamp", affordances={ALFWorldAffordance.TOGGLEABLE}
            ),
        }

        self.objects = {
            "apple 1": ALFWorldObject(
                "apple 1", "apple 1", "apple", parent_receptacle="countertop 1"
            ),
            "mug 1": ALFWorldObject("mug 1", "mug 1", "mug", parent_receptacle="countertop 1"),
            "soapbar 1": ALFWorldObject(
                "soapbar 1", "soapbar 1", "soapbar", parent_receptacle="cabinet 1"
            ),
            "sponge 1": ALFWorldObject(
                "sponge 1", "sponge 1", "sponge", parent_receptacle="drawer 1"
            ),
            "book 1": ALFWorldObject("book 1", "book 1", "book", parent_receptacle="desk 1"),
            "pen 1": ALFWorldObject("pen 1", "pen 1", "pen", parent_receptacle="desk 1"),
            "pen 2": ALFWorldObject("pen 2", "pen 2", "pen", parent_receptacle="cabinet 1"),
        }

        # Populate receptacle contained_objects
        for obj in self.objects.values():
            if obj.parent_receptacle and obj.parent_receptacle in self.receptacles:
                self.receptacles[obj.parent_receptacle].contained_objects.append(obj.id)

        # Create goal based on task_type
        if self.task_type == ALFWorldTaskType.PICK_AND_PLACE:
            self.goal = ALFWorldGoal(
                task_type=self.task_type,
                target_object_type="sponge",
                target_receptacle_type="countertop 1",
                raw_instruction="put a sponge on countertop 1",
            )
        elif self.task_type == ALFWorldTaskType.CLEAN_AND_PLACE:
            self.goal = ALFWorldGoal(
                task_type=self.task_type,
                target_object_type="soapbar",
                target_receptacle_type="countertop 1",
                apparatus_receptacle_type="sinkbasin 1",
                raw_instruction="clean soapbar with sinkbasin 1 and put on countertop 1",
            )
        elif self.task_type == ALFWorldTaskType.HEAT_AND_PLACE:
            self.goal = ALFWorldGoal(
                task_type=self.task_type,
                target_object_type="apple",
                target_receptacle_type="desk 1",
                apparatus_receptacle_type="microwave 1",
                raw_instruction="heat apple with microwave 1 and put on desk 1",
            )
        elif self.task_type == ALFWorldTaskType.COOL_AND_PLACE:
            self.goal = ALFWorldGoal(
                task_type=self.task_type,
                target_object_type="mug",
                target_receptacle_type="desk 1",
                apparatus_receptacle_type="fridge 1",
                raw_instruction="cool mug with fridge 1 and put on desk 1",
            )
        elif self.task_type == ALFWorldTaskType.EXAMINE_IN_LIGHT:
            self.goal = ALFWorldGoal(
                task_type=self.task_type,
                target_object_type="book",
                apparatus_receptacle_type="desklamp 1",
                raw_instruction="examine book with desklamp 1",
            )
        elif self.task_type == ALFWorldTaskType.PICK_TWO_AND_PLACE:
            self.goal = ALFWorldGoal(
                task_type=self.task_type,
                target_object_type="pen",
                target_receptacle_type="desk 1",
                count_required=2,
                raw_instruction="put two pens on desk 1",
            )

    def _get_admissible_commands(self) -> list[str]:
        commands = ["look", "inventory"]
        for recep in self.receptacles.keys():
            commands.append(f"go to {recep}")

        if self.current_location:
            rec = self.receptacles.get(self.current_location)
            if rec:
                if rec.is_openable:
                    if rec.is_open:
                        commands.append(f"close {rec.name}")
                    else:
                        commands.append(f"open {rec.name}")

                # If open or not openable, can take visible objects
                if not rec.is_openable or rec.is_open:
                    for oid in rec.contained_objects:
                        commands.append(f"take {oid} from {rec.name}")

                # Put held items
                for held in self.inventory:
                    if not rec.is_openable or rec.is_open:
                        commands.append(f"put {held.id} in/on {rec.name}")

                    if ALFWorldAffordance.CLEANABLE in rec.affordances:
                        commands.append(f"clean {held.id} with {rec.name}")
                    if ALFWorldAffordance.HEATABLE in rec.affordances:
                        commands.append(f"heat {held.id} with {rec.name}")
                    if ALFWorldAffordance.COOLABLE in rec.affordances:
                        commands.append(f"cool {held.id} with {rec.name}")

                if ALFWorldAffordance.TOGGLEABLE in rec.affordances:
                    commands.append(f"use {rec.name}")

        return commands

    def _get_obs(self, text_prefix: str = "") -> ALFWorldObservation:
        rec = self.receptacles.get(self.current_location) if self.current_location else None
        details = ""
        if rec:
            details += f" You are facing {rec.name}."
            if rec.is_openable:
                details += f" The {rec.name} is {'open' if rec.is_open else 'closed'}."
            if not rec.is_openable or rec.is_open:
                if rec.contained_objects:
                    details += (
                        f" On/in the {rec.name}, you see: {', '.join(rec.contained_objects)}."
                    )
                else:
                    details += f" The {rec.name} is empty."

        text = (text_prefix + details).strip()
        inv_ids = [obj.id for obj in self.inventory]

        return ALFWorldObservation(
            text_obs=text,
            current_location=self.current_location,
            inventory=inv_ids,
            admissible_commands=self._get_admissible_commands(),
            goal_instruction=self.goal.raw_instruction if self.goal else "",
            step_count=self.step_count,
        )

    def step(self, action: str) -> tuple[ALFWorldObservation, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        action = action.strip()
        text_feedback = ""
        reward = 0.0

        # Command matching
        if action.startswith("go to "):
            target_rec = action[len("go to ") :].strip()
            if target_rec in self.receptacles:
                self.current_location = target_rec
                text_feedback = f"You arrive at {target_rec}."
            else:
                text_feedback = "Nothing happens."

        elif action.startswith("open "):
            target_rec = action[len("open ") :].strip()
            if target_rec in self.receptacles and self.current_location == target_rec:
                rec = self.receptacles[target_rec]
                if rec.is_openable:
                    rec.is_open = True
                    text_feedback = f"You open {target_rec}."
                else:
                    text_feedback = f"You cannot open {target_rec}."
            else:
                text_feedback = f"You must go to {target_rec} first."

        elif action.startswith("close "):
            target_rec = action[len("close ") :].strip()
            if target_rec in self.receptacles and self.current_location == target_rec:
                rec = self.receptacles[target_rec]
                rec.is_open = False
                text_feedback = f"You close {target_rec}."
            else:
                text_feedback = f"You must go to {target_rec} first."

        elif action.startswith("take "):
            m = re.match(r"take (.+) from (.+)", action)
            if m:
                obj_id, target_rec = m.group(1).strip(), m.group(2).strip()
                if self.current_location == target_rec and target_rec in self.receptacles:
                    rec = self.receptacles[target_rec]
                    if obj_id in rec.contained_objects:
                        rec.contained_objects.remove(obj_id)
                        obj = self.objects[obj_id]
                        obj.parent_receptacle = None
                        self.inventory.append(obj)
                        text_feedback = f"You take {obj_id} from {target_rec}."
                    else:
                        text_feedback = f"{obj_id} is not in {target_rec}."
                else:
                    text_feedback = f"You are not at {target_rec}."
            else:
                text_feedback = "Invalid take command."

        elif action.startswith("put "):
            m = (
                re.match(r"put (.+) in/on (.+)", action)
                or re.match(r"put (.+) on (.+)", action)
                or re.match(r"put (.+) in (.+)", action)
            )
            if m:
                obj_id, target_rec = m.group(1).strip(), m.group(2).strip()
                if self.current_location == target_rec and target_rec in self.receptacles:
                    held = next((o for o in self.inventory if o.id == obj_id), None)
                    if held:
                        self.inventory.remove(held)
                        held.parent_receptacle = target_rec
                        self.receptacles[target_rec].contained_objects.append(held.id)
                        text_feedback = f"You put {obj_id} in/on {target_rec}."
                    else:
                        text_feedback = f"You are not holding {obj_id}."
                else:
                    text_feedback = f"You are not at {target_rec}."
            else:
                text_feedback = "Invalid put command."

        elif action.startswith("clean "):
            m = re.match(r"clean (.+) with (.+)", action)
            if m:
                obj_id, target_rec = m.group(1).strip(), m.group(2).strip()
                held = next((o for o in self.inventory if o.id == obj_id), None)
                if held and self.current_location == target_rec:
                    rec = self.receptacles[target_rec]
                    if ALFWorldAffordance.CLEANABLE in rec.affordances:
                        held.is_clean = True
                        text_feedback = f"You clean {obj_id} using {target_rec}."
                    else:
                        text_feedback = f"{target_rec} cannot clean objects."
                else:
                    text_feedback = "Action failed."

        elif action.startswith("heat "):
            m = re.match(r"heat (.+) with (.+)", action)
            if m:
                obj_id, target_rec = m.group(1).strip(), m.group(2).strip()
                held = next((o for o in self.inventory if o.id == obj_id), None)
                if held and self.current_location == target_rec:
                    rec = self.receptacles[target_rec]
                    if ALFWorldAffordance.HEATABLE in rec.affordances:
                        held.is_hot = True
                        text_feedback = f"You heat {obj_id} using {target_rec}."
                    else:
                        text_feedback = f"{target_rec} cannot heat objects."
                else:
                    text_feedback = "Action failed."

        elif action.startswith("cool "):
            m = re.match(r"cool (.+) with (.+)", action)
            if m:
                obj_id, target_rec = m.group(1).strip(), m.group(2).strip()
                held = next((o for o in self.inventory if o.id == obj_id), None)
                if held and self.current_location == target_rec:
                    rec = self.receptacles[target_rec]
                    if ALFWorldAffordance.COOLABLE in rec.affordances:
                        held.is_cold = True
                        text_feedback = f"You cool {obj_id} using {target_rec}."
                    else:
                        text_feedback = f"{target_rec} cannot cool objects."
                else:
                    text_feedback = "Action failed."

        elif action.startswith("use ") or action.startswith("toggle "):
            prefix = "use " if action.startswith("use ") else "toggle "
            target = action[len(prefix) :].strip()
            if target in self.receptacles and self.current_location == target:
                rec = self.receptacles[target]
                if ALFWorldAffordance.TOGGLEABLE in rec.affordances:
                    # If holding target object for examine_in_light
                    for held in self.inventory:
                        held.is_lit = True
                    text_feedback = f"You turn on {target}."
                else:
                    text_feedback = f"You cannot use {target}."
            else:
                text_feedback = f"You are not at {target}."

        elif action == "inventory":
            inv_str = ", ".join(o.id for o in self.inventory) if self.inventory else "nothing"
            text_feedback = f"You are carrying: {inv_str}."

        elif action == "look":
            text_feedback = "You look around."

        else:
            text_feedback = "Unknown command."

        # Verify Goal Completion
        terminated = self._check_goal_satisfied()
        if terminated:
            reward = 1.0

        truncated = self.step_count >= self.max_steps
        obs = self._get_obs(text_feedback)
        return obs, reward, terminated, truncated, {"success": terminated}

    def _check_goal_satisfied(self) -> bool:
        if not self.goal:
            return False

        t = self.goal.task_type
        target_type = self.goal.target_object_type
        target_rec = self.goal.target_receptacle_type

        if t == ALFWorldTaskType.PICK_AND_PLACE:
            if target_rec and target_rec in self.receptacles:
                for oid in self.receptacles[target_rec].contained_objects:
                    if self.objects[oid].object_type == target_type:
                        return True

        elif t == ALFWorldTaskType.CLEAN_AND_PLACE:
            if target_rec and target_rec in self.receptacles:
                for oid in self.receptacles[target_rec].contained_objects:
                    obj = self.objects[oid]
                    if obj.object_type == target_type and obj.is_clean:
                        return True

        elif t == ALFWorldTaskType.HEAT_AND_PLACE:
            if target_rec and target_rec in self.receptacles:
                for oid in self.receptacles[target_rec].contained_objects:
                    obj = self.objects[oid]
                    if obj.object_type == target_type and obj.is_hot:
                        return True

        elif t == ALFWorldTaskType.COOL_AND_PLACE:
            if target_rec and target_rec in self.receptacles:
                for oid in self.receptacles[target_rec].contained_objects:
                    obj = self.objects[oid]
                    if obj.object_type == target_type and obj.is_cold:
                        return True

        elif t == ALFWorldTaskType.EXAMINE_IN_LIGHT:
            for obj in self.inventory:
                if obj.object_type == target_type and obj.is_lit:
                    return True

        elif t == ALFWorldTaskType.PICK_TWO_AND_PLACE:
            if target_rec and target_rec in self.receptacles:
                count = sum(
                    1
                    for oid in self.receptacles[target_rec].contained_objects
                    if self.objects[oid].object_type == target_type
                )
                if count >= self.goal.count_required:
                    return True

        return False


class NativeALFWorldWrapper:
    """
    Dual-mode wrapper wrapping authentic upstream alfworld.agents.environment package.

    NOTE ON ALFWorld UPSTREAM PACKAGING & ABI:
    The official 'alfworld' package depends on TextWorld and Jericho (a C++ Z-Machine emulator).
    Upstream Jericho wheels are distributed for Python <=3.10 and fail compilation on
    Python 3.12+ due to removed C-API macros.
    When executed in a Python 3.8-3.10 environment with 'alfworld' installed,
    this wrapper delegates directly to authentic AlfredTWEnv.
    When running in Python 3.12+ or without alfworld, HBLLM provides the high-fidelity
    StandaloneALFWorldEnv standalone simulator.
    """

    is_native: bool = True

    TASK_PREFIX_MAP = {
        ALFWorldTaskType.PICK_AND_PLACE: "pick_and_place_simple",
        ALFWorldTaskType.EXAMINE_IN_LIGHT: "look_at_obj_in_light",
        ALFWorldTaskType.CLEAN_AND_PLACE: "pick_clean_then_place_in_recep",
        ALFWorldTaskType.HEAT_AND_PLACE: "pick_heat_then_place_in_recep",
        ALFWorldTaskType.COOL_AND_PLACE: "pick_cool_then_place_in_recep",
        ALFWorldTaskType.PICK_TWO_AND_PLACE: "pick_two_obj_and_place",
    }

    def __init__(
        self,
        task_type: ALFWorldTaskType = ALFWorldTaskType.PICK_AND_PLACE,
        seed: int | None = None,
        config_path: str | None = None,
    ) -> None:
        try:
            import alfworld.agents.environment as alf_env  # type: ignore
        except ImportError as err:
            raise ImportError(
                "alfworld is required for NativeALFWorldWrapper. "
                "Install via 'pip install alfworld' in Python 3.8-3.10, "
                "or use StandaloneALFWorldEnv."
            ) from err

        self._alf_env = alf_env
        self.task_type = task_type
        self.seed = seed
        self.step_count = 0
        self.max_steps = 50

        # Initialize native environment if config available
        self.env = None
        if config_path:
            self.env = alf_env.AlfredTWEnv(config_path, train_eval="eval_out_of_distribution")

    def reset(
        self,
        seed: int | None = None,
        task_type: ALFWorldTaskType | None = None,
    ) -> tuple[ALFWorldObservation, dict[str, Any]]:
        """Reset native environment or raise if config uninitialized."""
        if seed is not None:
            self.seed = seed
        if task_type is not None:
            self.task_type = task_type
        self.step_count = 0

        if self.env is not None:
            obs_text, info = self.env.reset()
            obs = ALFWorldObservation(
                feedback=str(obs_text),
                inventory_text=str(info.get("inventory", "")),
                admissible_commands=list(info.get("admissible_commands", [])),
                won=False,
                step_count=0,
                max_steps=self.max_steps,
            )
            return obs, info

        raise RuntimeError("NativeALFWorldWrapper requires an ALFWorld configuration file.")

    def step(self, action: str) -> tuple[ALFWorldObservation, float, bool, bool, dict[str, Any]]:
        """Step native environment with text command."""
        self.step_count += 1
        if self.env is not None:
            obs_text, reward, done, info = self.env.step(action)
            won = bool(info.get("won", False) or reward > 0)
            truncated = self.step_count >= self.max_steps
            obs = ALFWorldObservation(
                feedback=str(obs_text),
                inventory_text=str(info.get("inventory", "")),
                admissible_commands=list(info.get("admissible_commands", [])),
                won=won,
                step_count=self.step_count,
                max_steps=self.max_steps,
            )
            return obs, float(reward), done or won, truncated, info

        raise RuntimeError("NativeALFWorldWrapper requires an ALFWorld configuration file.")


def make_alfworld_env(
    task_type: ALFWorldTaskType = ALFWorldTaskType.PICK_AND_PLACE,
    seed: int | None = None,
    prefer_native: bool = False,
    config_path: str | None = None,
) -> StandaloneALFWorldEnv | NativeALFWorldWrapper:
    """Instantiate ALFWorld environment with dual-mode native/standalone selection."""
    if prefer_native:
        try:
            return NativeALFWorldWrapper(task_type=task_type, seed=seed, config_path=config_path)
        except Exception as e:
            logger.warning(
                "Native alfworld unavailable (%s), falling back to StandaloneALFWorldEnv",
                e,
            )
    return StandaloneALFWorldEnv(task_type=task_type, seed=seed)
