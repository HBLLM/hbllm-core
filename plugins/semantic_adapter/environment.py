"""
Semantic Ambiguity Environment Simulator.

Generates ambiguous, elliptical, paraphrased, and high-level human directives
requiring semantic disambiguation and causal subgoal decomposition.
"""

from __future__ import annotations

import logging
from typing import Any

from .types import CausalSubgoal, SemanticObservation, SemanticTier

logger = logging.getLogger(__name__)

TIER_SCENARIOS = {
    SemanticTier.TIER_1_CANONICAL_EXPLICIT: {
        "instruction": "Pick up the red key and unlock the blue door.",
        "subgoals": ["pick:red_key", "unlock:blue_door"],
        "objects": {
            "red_key": {"location": "floor", "type": "key", "color": "red"},
            "blue_door": {"state": "locked", "type": "door", "color": "blue"},
        },
    },
    SemanticTier.TIER_2_SYNONYM_PARAPHRASE: {
        "instruction": "Stash the crimson opener inside the azure container.",
        "subgoals": ["pick:red_key", "place:red_key:blue_box"],
        "objects": {
            "red_key": {
                "location": "table",
                "type": "key",
                "color": "red",
                "alias": "crimson opener",
            },
            "blue_box": {
                "location": "shelf",
                "type": "box",
                "color": "blue",
                "alias": "azure container",
            },
        },
    },
    SemanticTier.TIER_3_UNDERSPECIFIED_ELLIPTICAL: {
        "instruction": "Tidy up the workbench.",
        "subgoals": ["stow:screwdriver", "stow:hammer", "wipe:counter"],
        "objects": {
            "screwdriver": {"location": "workbench", "type": "tool"},
            "hammer": {"location": "workbench", "type": "tool"},
            "workbench": {"state": "dusty", "type": "surface"},
        },
    },
    SemanticTier.TIER_4_CONFLICTING_CORRECTION: {
        "instruction": "Bring me the blue mug... wait, scratch that, bring the red mug instead.",
        "subgoals": ["pick:red_mug", "deliver:user"],
        "objects": {
            "blue_mug": {"location": "counter", "color": "blue"},
            "red_mug": {"location": "counter", "color": "red"},
        },
    },
    SemanticTier.TIER_5_ABSTRACT_INTENT: {
        "instruction": "Prepare the conference room for the client presentation.",
        "subgoals": ["turn_on:projector", "close:blinds", "align:chairs"],
        "objects": {
            "projector": {"state": "off"},
            "blinds": {"state": "open"},
            "chairs": {"state": "disordered"},
        },
    },
}


class StandaloneSemanticEnv:
    """Environment validating natural language directive understanding and execution."""

    def __init__(
        self,
        tier: SemanticTier | str = SemanticTier.TIER_1_CANONICAL_EXPLICIT,
        seed: int = 42,
        max_steps: int = 10,
    ) -> None:
        self.tier = SemanticTier(tier)
        self.seed = seed
        self.max_steps = max_steps
        self.step_count = 0
        self.instruction = ""
        self.objects: dict[str, dict[str, Any]] = {}
        self.pending_subgoals: list[str] = []
        self.completed_subgoals: list[str] = []

        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> SemanticObservation:
        """Reset environment to tier scenario."""
        if seed is not None:
            self.seed = seed
        self.step_count = 0
        scenario = TIER_SCENARIOS[self.tier]
        self.instruction = scenario["instruction"]
        self.pending_subgoals = list(scenario["subgoals"])
        self.completed_subgoals = []
        self.objects = {k: dict(v) for k, v in scenario["objects"].items()}

        return self._get_obs()

    def step(
        self, action: CausalSubgoal | str
    ) -> tuple[SemanticObservation, float, bool, dict[str, Any]]:
        """Apply executed subgoal action."""
        self.step_count += 1

        action_str = ""
        if isinstance(action, CausalSubgoal):
            action_str = f"{action.verb}:{action.target}"
            if action.destination:
                action_str += f":{action.destination}"
        else:
            action_str = str(action)

        reward = 0.0
        if self.pending_subgoals and action_str == self.pending_subgoals[0]:
            # Successfully matched expected causal subgoal!
            subgoal = self.pending_subgoals.pop(0)
            self.completed_subgoals.append(subgoal)
            reward = 1.0

        won = len(self.pending_subgoals) == 0
        done = won or self.step_count >= self.max_steps
        if won:
            reward += 10.0

        info = {
            "won": won,
            "completed": list(self.completed_subgoals),
            "remaining": list(self.pending_subgoals),
            "steps": self.step_count,
        }

        obs = self._get_obs(done=done, won=won, info=info)
        return obs, reward, done, info

    def _get_obs(
        self,
        done: bool = False,
        won: bool = False,
        info: dict[str, Any] | None = None,
    ) -> SemanticObservation:
        """Construct observation."""
        return SemanticObservation(
            instruction=self.instruction,
            scene_objects={k: dict(v) for k, v in self.objects.items()},
            completed_subgoals=list(self.completed_subgoals),
            pending_subgoals=list(self.pending_subgoals),
            step_count=self.step_count,
            max_steps=self.max_steps,
            done=done,
            won=won,
            info=info or {},
        )


def make_semantic_env(
    tier: SemanticTier | str = SemanticTier.TIER_1_CANONICAL_EXPLICIT,
    seed: int = 42,
    max_steps: int = 10,
) -> StandaloneSemanticEnv:
    """Factory creating semantic ambiguity evaluation environments."""
    return StandaloneSemanticEnv(tier=tier, seed=seed, max_steps=max_steps)
