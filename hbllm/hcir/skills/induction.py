"""Skill Induction Engine — Automated Declarative Neuro-Symbolic Skill Acquisition.

Analyzes committed transaction traces from TransactionManager to synthesize
declarative, reusable DeclarativeNeuroSymbolicSkill programs with composable
SymbolicPredicate signatures and SubgoalSequence programs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    AllOf,
    DeclarativeNeuroSymbolicSkill,
    SubgoalSequence,
    SymbolicPredicate,
    SymbolicSubgoal,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp

logger = logging.getLogger(__name__)


class SkillInductionEngine:
    """Automated induction engine compiling transaction execution traces into declarative skills."""

    def __init__(self, min_support: int = 1) -> None:
        self.min_support = min_support

    def induce_from_transactions(
        self,
        transactions: Sequence[HCIRTransaction],
        skill_name: str = "induced_skill",
        description: str = "",
    ) -> DeclarativeNeuroSymbolicSkill | None:
        """Induce a DeclarativeNeuroSymbolicSkill from committed transaction sequences.

        Scans operations for action intents, navigation subgoals, and environmental
        affordances, extracting the minimal common predicate signature and
        the canonical subgoal execution sequence.
        """
        if not transactions:
            return None

        action_intents: list[str] = []
        action_ids: set[int] = set()
        subgoals: list[SymbolicSubgoal] = []
        target_roles: list[str] = []

        for tx in transactions:
            for op in tx.operations:
                data = op.node_data or {}
                # 1. Capture action nodes or action proposals
                if op.op in (TransactionOp.ADD_NODE, TransactionOp.UPSERT_NODE):
                    role = data.get("role") or data.get("target_role")
                    intent_str = (
                        data.get("action_intent")
                        or data.get("intent")
                        or data.get("event_kind", "")
                    )
                    act_id = data.get("action_id") or data.get("action")

                    if isinstance(act_id, int):
                        action_ids.add(act_id)
                    elif isinstance(act_id, str) and act_id.isdigit():
                        action_ids.add(int(act_id))

                    if role:
                        target_roles.append(str(role))

                    if "navigate" in intent_str.lower() or "move" in intent_str.lower():
                        action_intents.append("navigate")
                        subgoals.append(
                            SymbolicSubgoal(
                                intent=SpatialActionIntent.NAVIGATE,
                                target_query={"role": str(role) if role else "target"},
                            )
                        )
                    elif (
                        "interact" in intent_str.lower()
                        or "push" in intent_str.lower()
                        or "toggle" in intent_str.lower()
                    ):
                        action_intents.append("interact")
                        subgoals.append(
                            SymbolicSubgoal(
                                intent=SpatialActionIntent.INTERACT,
                                target_query={"role": str(role) if role else "interactable"},
                            )
                        )
                    elif "goal" in intent_str.lower():
                        action_intents.append("reach_goal")
                        subgoals.append(
                            SymbolicSubgoal(
                                intent=SpatialActionIntent.ACTUATE,
                                target_query={"role": "goal"},
                            )
                        )

        # Build composable signature predicates
        predicates: list[SymbolicPredicate] = []
        if action_ids:
            predicates.append(ActionAffordancePredicate(required=action_ids))

        signature: SymbolicPredicate = AllOf(*predicates) if predicates else AllOf()

        # Deduplicate consecutive identical subgoals while preserving sequence
        deduped_subgoals: list[SymbolicSubgoal] = []
        for sg in subgoals:
            if not deduped_subgoals or (
                deduped_subgoals[-1].intent != sg.intent
                or deduped_subgoals[-1].target_query != sg.target_query
            ):
                deduped_subgoals.append(sg)

        program = SubgoalSequence(*deduped_subgoals) if deduped_subgoals else None

        skill = DeclarativeNeuroSymbolicSkill()
        skill.skill_name = skill_name
        skill.description = description or (
            f"Induced skill '{skill_name}' with {len(deduped_subgoals)} subgoals "
            f"from {len(transactions)} transaction traces"
        )
        skill.signature = signature
        skill.program = program

        logger.info(
            "SkillInductionEngine synthesized '%s': signature=%s, program_len=%d",
            skill_name,
            signature,
            len(deduped_subgoals),
        )
        return skill

    def induce_from_trajectory(
        self,
        actions: Sequence[int],
        metadata: dict[str, Any] | None = None,
        skill_name: str = "induced_trajectory_skill",
    ) -> DeclarativeNeuroSymbolicSkill:
        """Directly induce a skill from an observed action sequence and context metadata."""
        meta = metadata or {}
        required_actions = set(actions)
        signature = ActionAffordancePredicate(required=required_actions)

        subgoals = []
        for act in actions:
            subgoals.append(
                SymbolicSubgoal(
                    intent=SpatialActionIntent.NAVIGATE,
                    target_query={"action": act, **meta},
                )
            )

        skill = DeclarativeNeuroSymbolicSkill()
        skill.skill_name = skill_name
        skill.description = f"Trajectory skill for action sequence of length {len(actions)}"
        skill.signature = signature
        skill.program = SubgoalSequence(*subgoals)
        return skill
