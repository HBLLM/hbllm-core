"""Compositional Language Understanding Engine (Stage D11).

Parses multi-token instructions by composing grounded lexical items,
resolves referents in the perceived scene, synthesizes structured goals,
and executes multi-step plans zero-shot.
"""

from __future__ import annotations

import logging
from typing import Any

from .blank_brain import BlankBrainSubstrate
from .environment import BabyWorldEnvironment
from .goal_planning import GoalDirectedPlanningEngine
from .language_grounding import LanguageGroundingEngine
from .types import (
    BeliefTransitionEvent,
    BeliefTransitionType,
    LexicalCategory,
    PlanExecutionResult,
    PredicateGoal,
)

logger = logging.getLogger(__name__)


class CompositionalLanguageEngine:
    """Parses novel compound sentences into executable developmental goals."""

    def __init__(
        self,
        substrate: BlankBrainSubstrate,
        env: BabyWorldEnvironment,
        grounding_engine: LanguageGroundingEngine,
        planning_engine: GoalDirectedPlanningEngine,
    ) -> None:
        self.substrate = substrate
        self.env = env
        self.grounding = grounding_engine
        self.planner = planning_engine

    def parse_instruction_to_goal(self, instruction: str) -> PredicateGoal | None:
        """Parse natural instruction into structured PredicateGoal via grounded lexicon."""
        grounded_items = self.grounding.ground_utterance(instruction)
        if not grounded_items:
            logger.warning(f"No grounded lexical tokens found in '{instruction}'")
            return None

        # Extract verbs, nouns, adjectives, prepositions
        verbs = [item for item in grounded_items if item.category == LexicalCategory.VERB]
        nouns = [item for item in grounded_items if item.category == LexicalCategory.NOUN]
        adjectives = [item for item in grounded_items if item.category == LexicalCategory.ADJECTIVE]
        prepositions = [
            item for item in grounded_items if item.category == LexicalCategory.PREPOSITION
        ]

        # Resolve subject entity
        subject_id = self._resolve_entity_reference(nouns, adjectives, index=0)
        if not subject_id and self.env.objects:
            # Fallback to first object
            subject_id = list(self.env.objects.keys())[0]

        # Resolve target/destination entity if preposition present
        target_id = None
        predicate = "REACHABLE"  # Default

        if prepositions:
            prep_sym = prepositions[0].grounded_symbol
            if prep_sym in ("INSIDE", "IN"):
                predicate = "INSIDE"
                target_id = self._resolve_entity_reference(nouns, adjectives, index=1)
                # Fallback to first container in environment if not resolved
                if not target_id:
                    for oid, obj in self.env.objects.items():
                        if obj.is_container or obj.object_type.value in ("container", "box"):
                            target_id = oid
                            break
            elif prep_sym in ("ON", "ONTO"):
                predicate = "ON"
                target_id = self._resolve_entity_reference(nouns, adjectives, index=1)
            elif prep_sym == "INSTRUMENT_WITH":
                predicate = "REACHABLE"

        # Check verb actions
        if verbs:
            verb_sym = verbs[0].grounded_symbol
            if verb_sym in ("OPEN", "CLOSE"):
                predicate = "STATE"
                return PredicateGoal(
                    predicate=predicate,
                    subject_id=subject_id or "",
                    target_value=(verb_sym == "OPEN"),
                )
            if verb_sym in ("PUSH", "MOVE", "PULL") and predicate == "REACHABLE":
                predicate = "REACHABLE"

        goal = PredicateGoal(
            predicate=predicate,
            subject_id=subject_id or "",
            target_id=target_id,
        )

        # Log transition
        if hasattr(self.substrate, "profile") and hasattr(
            self.substrate.profile, "belief_transitions"
        ):
            self.substrate.profile.belief_transitions.append(
                BeliefTransitionEvent(
                    event_type=BeliefTransitionType.COMPOSITION_PARSED,
                    variable="syntax_to_goal",
                    condition=f"'{instruction}' => {predicate}({subject_id}, {target_id})",
                    posterior_confidence=1.0,
                    evidence={"tokens": [i.token for i in grounded_items]},
                )
            )

        return goal

    def execute_instruction(self, instruction: str) -> PlanExecutionResult:
        """Parse natural language instruction and execute zero-shot plan."""
        goal = self.parse_instruction_to_goal(instruction)
        if not goal:
            return PlanExecutionResult(
                goal=PredicateGoal(predicate="UNKNOWN", subject_id=""),
                steps=[],
                success=False,
                replan_count=0,
                wasted_actions=0,
            )

        return self.planner.execute_with_replanning(goal)

    def _resolve_entity_reference(
        self, nouns: list[Any], adjectives: list[Any], index: int = 0
    ) -> str | None:
        """Match noun+adjective specifications against active environment objects."""
        target_noun = nouns[index].grounded_symbol if index < len(nouns) else None
        target_adj = adjectives[index].grounded_symbol if index < len(adjectives) else None

        for oid, obj in self.env.objects.items():
            type_match = (
                target_noun is None
                or obj.object_type.value == target_noun
                or (target_noun == "box" and obj.is_container)
            )
            adj_match = (
                target_adj is None
                or obj.color == target_adj
                or (target_adj == "heavy" and obj.mass >= 5.0)
                or (target_adj == "light" and obj.mass <= 2.5)
            )
            if type_match and adj_match:
                return oid

        # Partial match on noun only
        if target_noun:
            for oid, obj in self.env.objects.items():
                if obj.object_type.value == target_noun:
                    return oid

        return None
