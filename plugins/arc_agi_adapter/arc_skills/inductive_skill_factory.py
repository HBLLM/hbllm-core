"""Inductive Skill Factory: Synthesis of Reusable Declarative Skills from Observed Winning Episodes.

Phase 3 of the AGI Inductive Learning Architecture:
Transforms harvested winning macro action sequences and structural grid perceptions
into compiled, reusable DeclarativeNeuroSymbolicSkill instances.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    AllOf,
    DeclarativeNeuroSymbolicSkill,
    EntityCountPredicate,
    GridDimensionPredicate,
    PanelConstraint,
    SubgoalSequence,
    SymbolicPredicate,
    SymbolicSubgoal,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent
from plugins.arc_agi_adapter.arc_skills.perceptual_context import PerceptualSkillContext

logger = logging.getLogger(__name__)


class InductiveSkillFactory:
    """Creates new DeclarativeNeuroSymbolicSkills from observed winning episodes."""

    def induce_skill(
        self,
        initial_grid: np.ndarray,
        winning_actions: list[tuple[int, dict[str, int] | None]],
        available_actions: list[int],
        game_id: str,
        level: int,
    ) -> DeclarativeNeuroSymbolicSkill | None:
        """Induce a new DeclarativeNeuroSymbolicSkill from a successful episode."""
        if not winning_actions:
            return None

        if initial_grid.ndim == 3:
            initial_grid = initial_grid[-1]

        # 1. Analyze initial grid to extract perceptual structural features
        pctx = PerceptualSkillContext.from_grid(initial_grid, available_actions=available_actions)

        # 2. Build signature predicates dynamically from observed perceptual geometry
        predicates: list[SymbolicPredicate] = [
            ActionAffordancePredicate(exact=set(available_actions)),
            GridDimensionPredicate(
                exact_shape=(int(initial_grid.shape[0]), int(initial_grid.shape[1]))
            ),
        ]

        # Entity count predicate from perception
        entity_count = len(pctx.entities)
        if entity_count >= 3:
            predicates.append(
                EntityCountPredicate(
                    min_count=max(1, entity_count - 3),
                    max_count=entity_count + 10,
                )
            )

        # Panel dividers constraint if detected
        if pctx.dividers:
            predicates.append(
                PanelConstraint(
                    min_row_ratio=0.0,
                    max_row_ratio=1.0,
                    min_distinct_colors=max(2, len(np.unique(initial_grid)) - 1),
                )
            )

        signature = AllOf(*predicates)
        skill_identifier = f"induced_{game_id}_L{level}"
        recorded_plan = list(winning_actions)

        # 3. Create a concrete DeclarativeNeuroSymbolicSkill subclass
        class InducedSkill(DeclarativeNeuroSymbolicSkill):
            skill_name: str = skill_identifier
            semantic_intent: SpatialActionIntent = SpatialActionIntent.MANIPULATE
            program = SubgoalSequence(
                SymbolicSubgoal(
                    intent=SpatialActionIntent.MANIPULATE,
                    target_query={"role": "macro_winner"},
                )
            )

            def __init__(
                self, sig: SymbolicPredicate, plan_actions: list[tuple[int, dict[str, int] | None]]
            ) -> None:
                super().__init__()
                self.signature = sig
                self._recorded_plan = plan_actions

            def plan(
                self,
                grid: np.ndarray,
                current_level: int = 0,
                metadata: dict[str, Any] | None = None,
            ) -> list[tuple[int, dict[str, int] | None]]:
                return list(self._recorded_plan)

        instantiated_skill = InducedSkill(signature, recorded_plan)
        logger.info(
            "InductiveSkillFactory: Successfully induced skill '%s' (%d actions) with signature: %s",
            skill_identifier,
            len(recorded_plan),
            signature,
        )
        return instantiated_skill
