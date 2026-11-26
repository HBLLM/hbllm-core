"""Skill Induction Engine — Automated Declarative Neuro-Symbolic Skill Acquisition."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from hbllm.hcir.graph import SkillNode
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
        """Induce a DeclarativeNeuroSymbolicSkill from committed transaction sequences."""
        if not transactions:
            return None

        action_ids: set[int] = set()
        subgoals: list[SymbolicSubgoal] = []
        for tx in transactions:
            for op in tx.operations:
                if op.op not in (TransactionOp.ADD_NODE, TransactionOp.UPSERT_NODE):
                    continue
                data = op.node_data or {}
                role = data.get("role") or data.get("target_role")
                intent_l = str(
                    data.get("action_intent") or data.get("intent") or data.get("event_kind", "")
                ).lower()
                act_id = data.get("action_id") or data.get("action")
                if isinstance(act_id, int) or (isinstance(act_id, str) and act_id.isdigit()):
                    action_ids.add(int(act_id))
                if "navigate" in intent_l or "move" in intent_l:
                    sg = SymbolicSubgoal(
                        intent=SpatialActionIntent.NAVIGATE,
                        target_query={"role": str(role) if role else "target"},
                    )
                elif any(k in intent_l for k in ("interact", "push", "toggle")):
                    sg = SymbolicSubgoal(
                        intent=SpatialActionIntent.INTERACT,
                        target_query={"role": str(role) if role else "interactable"},
                    )
                elif "goal" in intent_l:
                    sg = SymbolicSubgoal(
                        intent=SpatialActionIntent.ACTUATE, target_query={"role": "goal"}
                    )
                else:
                    continue
                if not subgoals or (
                    subgoals[-1].intent != sg.intent or subgoals[-1].target_query != sg.target_query
                ):
                    subgoals.append(sg)

        signature: SymbolicPredicate = (
            AllOf(ActionAffordancePredicate(required=action_ids)) if action_ids else AllOf()
        )
        skill = DeclarativeNeuroSymbolicSkill()
        skill.skill_name = skill_name
        skill.description = description or (
            f"Induced skill '{skill_name}' with {len(subgoals)} subgoals "
            f"from {len(transactions)} transaction traces"
        )
        skill.signature = signature
        skill.program = SubgoalSequence(*subgoals) if subgoals else None
        logger.info(
            "SkillInductionEngine synthesized '%s': signature=%s, program_len=%d",
            skill_name,
            signature,
            len(subgoals),
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
        skill = DeclarativeNeuroSymbolicSkill()
        skill.skill_name = skill_name
        skill.description = f"Trajectory skill for action sequence of length {len(actions)}"
        skill.signature = ActionAffordancePredicate(required=set(actions))
        skill.program = SubgoalSequence(
            *[
                SymbolicSubgoal(
                    intent=SpatialActionIntent.NAVIGATE, target_query={"action": act, **meta}
                )
                for act in actions
            ]
        )
        return skill

    def induce_skill_node_from_transactions(
        self,
        transactions: Sequence[HCIRTransaction],
        skill_name: str = "induced_skill",
        description: str = "",
        tenant_id: str = "default",
    ) -> SkillNode | None:
        """Induce a parameterized homoiconic `SkillNode` directly from transaction traces."""
        skill = self.induce_from_transactions(
            transactions, skill_name=skill_name, description=description
        )
        return None if skill is None else skill.to_skill_node(tenant_id=tenant_id)

    def induce_skill_node_from_trajectory(
        self,
        actions: Sequence[int],
        metadata: dict[str, Any] | None = None,
        skill_name: str = "induced_trajectory_skill",
        tenant_id: str = "default",
    ) -> SkillNode:
        """Induce a parameterized homoiconic `SkillNode` from an action trajectory."""
        return self.induce_from_trajectory(
            actions, metadata=metadata, skill_name=skill_name
        ).to_skill_node(tenant_id=tenant_id)
