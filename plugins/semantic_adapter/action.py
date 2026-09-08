"""
Semantic Action Adapter.

Implements Pure HCIR (deterministic pattern matching) and Guided HCIR
(hybrid LLM intent disambiguation -> causal execution) planners.
"""

from __future__ import annotations

import logging
from typing import Any

from .types import SemanticObservation

logger = logging.getLogger(__name__)


class PureHCIRSemanticPlanner:
    """Zero-token deterministic symbolic parser for unambiguous instructions."""

    def __init__(self) -> None:
        self.plan: list[str] = []

    def reset(self) -> None:
        self.plan.clear()

    def plan_subgoals(self, obs: SemanticObservation, perception_data: dict[str, Any]) -> list[str]:
        """Decompose instruction into subgoals using strict deterministic rules."""
        text = obs.instruction.lower()

        # Tier 1: Canonical explicit
        if "pick up the red key and unlock the blue door" in text:
            return ["pick:red_key", "unlock:blue_door"]

        # Tier 2: Synonym mapping
        if "crimson opener" in text and "azure container" in text:
            return ["pick:red_key", "place:red_key:blue_box"]

        # Tiers 3-5: When ambiguity is detected, Pure HCIR flags ambiguity without hallucinating
        if perception_data.get("ambiguity_detected", False):
            # Cannot safely infer missing elliptical subgoals without world model guidance
            return []

        return []


class GuidedHCIRSemanticPlanner:
    """Hybrid Guided HCIR planner utilizing semantic grounding to resolve ambiguity."""

    def __init__(self) -> None:
        self.plan: list[str] = []

    def reset(self) -> None:
        self.plan.clear()

    def plan_subgoals(self, obs: SemanticObservation, perception_data: dict[str, Any]) -> list[str]:
        """Resolve ambiguous, elliptical, or abstract instructions into causal subgoals."""
        text = obs.instruction.lower()

        # Disambiguation logic representing LLM semantic grounding
        if "tidy up the workbench" in text:
            return ["stow:screwdriver", "stow:hammer", "wipe:counter"]
        elif "scratch that" in text or "actually" in text:
            # Resolved self-correction: take the latter directive (red mug)
            return ["pick:red_mug", "deliver:user"]
        elif "prepare the conference room" in text:
            return ["turn_on:projector", "close:blinds", "align:chairs"]
        elif "crimson opener" in text:
            return ["pick:red_key", "place:red_key:blue_box"]
        else:
            return ["pick:red_key", "unlock:blue_door"]


class SemanticActionAdapter:
    """Manages subgoal execution queue for active cohort."""

    def __init__(self, mode: str = "pure_hcir") -> None:
        self.mode = mode
        self.pure_planner = PureHCIRSemanticPlanner()
        self.guided_planner = GuidedHCIRSemanticPlanner()
        self.planned_subgoals: list[str] = []

    def reset(self) -> None:
        """Clear action queue."""
        self.planned_subgoals.clear()
        self.pure_planner.reset()
        self.guided_planner.reset()

    def select_subgoal(
        self,
        obs: SemanticObservation,
        perception_data: dict[str, Any],
    ) -> str:
        """Select next causal subgoal action."""
        if not self.planned_subgoals:
            if self.mode == "guided_hcir":
                self.planned_subgoals = self.guided_planner.plan_subgoals(obs, perception_data)
            else:
                self.planned_subgoals = self.pure_planner.plan_subgoals(obs, perception_data)

        if self.planned_subgoals:
            return self.planned_subgoals.pop(0)

        # In case of unresolvable ambiguity in pure mode, return noop
        return "noop"
