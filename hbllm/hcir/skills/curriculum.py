"""Curricular Skill Graph — Hierarchical Declarative Neuro-Symbolic Skill Organization.

Maintains an indexed Directed Acyclic Graph (DAG) of declarative skills partitioned
by cognitive abstraction tiers (Micro, Meso, Macro), supporting dynamic skill retrieval,
hierarchical composition, and continuous Bayesian confidence calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    AllOf,
    CompositeDeclarativeSkill,
    DeclarativeNeuroSymbolicSkill,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


class SkillTier(StrEnum):
    """Abstraction layer for declarative neuro-symbolic skills."""

    MICRO = "micro"  # Primitive spatial movement, single-step navigation, corridor transit
    MESO = "meso"  # Multi-step object interaction (keys, doors, block pushing, holding bay)
    MACRO = "macro"  # Composite multi-stage sequences completing entire operational phases


@dataclass
class CurricularSkillNode:
    """A node in the curricular skill graph encapsulating a declarative skill and its topology."""

    skill: DeclarativeNeuroSymbolicSkill
    tier: SkillTier = SkillTier.MESO
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 1.0
    dependencies: set[str] = field(default_factory=set)  # Prerequisite skill names
    subskills: list[str] = field(default_factory=list)  # Component subskill names
    target_roles: set[str] = field(
        default_factory=set
    )  # Entity roles addressed (e.g. "goal", "key")
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def skill_name(self) -> str:
        return self.skill.skill_name

    def record_outcome(self, success: bool) -> None:
        """Update Bayesian confidence based on operational execution feedback."""
        if success:
            self.success_count += 1
            self.confidence = min(1.0, self.confidence + 0.05)
        else:
            self.failure_count += 1
            self.confidence = max(0.05, self.confidence - 0.20)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export curricular skill state as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"curricular_skill:{self.skill_name}",
            operator="==",
            value=f"tier:{self.tier.value}:conf:{self.confidence:.2f}",
            relation=f"deps:{len(self.dependencies)}:subs:{len(self.subskills)}",
            target_entity=self.skill_name,
        )

    def to_transaction_operation(self) -> TransactionOperation:
        """Propose atomic HCIR transaction updating skill node in cognitive workspace."""
        return TransactionOperation(
            op=TransactionOp.UPSERT_NODE,
            node_id=f"curricular_skill_{self.skill_name}",
            node_data={
                "id": f"curricular_skill_{self.skill_name}",
                "node_type": "skill",
                "category": "procedural",
                "skill_name": self.skill_name,
                "tier": self.tier.value,
                "confidence": self.confidence,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "dependencies": list(self.dependencies),
                "subskills": list(self.subskills),
                "target_roles": list(self.target_roles),
            },
        )


class CurricularSkillGraph:
    """Hierarchical graph organizing and indexing declarative neuro-symbolic skills."""

    def __init__(self) -> None:
        self.nodes: dict[str, CurricularSkillNode] = {}

    def register_skill(
        self,
        skill: DeclarativeNeuroSymbolicSkill,
        tier: SkillTier = SkillTier.MESO,
        dependencies: set[str] | None = None,
        subskills: list[str] | None = None,
        target_roles: set[str] | None = None,
        confidence: float = 1.0,
    ) -> CurricularSkillNode:
        """Register or update a declarative skill in the curricular graph."""
        name = skill.skill_name
        deps = dependencies or set()
        subs = subskills or []
        roles = set(target_roles or set())

        # Auto-extract target roles from program subgoals if not explicitly passed
        if not roles and skill.program and skill.program.subgoals:
            for sg in skill.program.subgoals:
                if isinstance(sg.target_query, dict):
                    role = sg.target_query.get("role")
                    if role:
                        roles.add(str(role))

        node = CurricularSkillNode(
            skill=skill,
            tier=tier,
            confidence=confidence,
            dependencies=deps,
            subskills=subs,
            target_roles=roles,
        )
        self.nodes[name] = node
        logger.info(
            "CurricularSkillGraph: Registered skill '%s' [tier=%s, deps=%d, subs=%d, roles=%s]",
            name,
            tier.value,
            len(deps),
            len(subs),
            roles,
        )
        return node

    def get_skill(self, skill_name: str) -> DeclarativeNeuroSymbolicSkill | None:
        """Retrieve declarative skill by name."""
        node = self.nodes.get(skill_name)
        return node.skill if node else None

    def get_node(self, skill_name: str) -> CurricularSkillNode | None:
        """Retrieve curricular skill node by name."""
        return self.nodes.get(skill_name)

    def find_matching_skills(
        self,
        grid: np.ndarray,
        available_actions: list[int] | set[int],
        target_role: str | None = None,
        tier: SkillTier | None = None,
        min_confidence: float = 0.2,
    ) -> list[CurricularSkillNode]:
        """Query matching skills based on signature affordance, target role, and confidence."""
        matches: list[CurricularSkillNode] = []
        action_list = list(available_actions)

        for node in self.nodes.values():
            if node.confidence < min_confidence:
                continue
            if tier is not None and node.tier != tier:
                continue
            if target_role is not None and target_role not in node.target_roles:
                continue

            # Verify signature against grid and actions
            if node.skill.can_handle(grid, action_list):
                matches.append(node)

        # Sort descending by confidence and tier complexity
        matches.sort(
            key=lambda n: (
                n.confidence,
                2 if n.tier == SkillTier.MACRO else (1 if n.tier == SkillTier.MESO else 0),
            ),
            reverse=True,
        )
        return matches

    def compose_macro_skill(
        self,
        macro_name: str,
        subskill_names: list[str],
        description: str = "",
    ) -> DeclarativeNeuroSymbolicSkill | None:
        """Hierarchically compose existing subskills into a unified macro-skill."""
        missing = [name for name in subskill_names if name not in self.nodes]
        if missing:
            logger.warning("Cannot compose '%s': missing subskills %s", macro_name, missing)
            return None

        component_skills = [self.nodes[name].skill for name in subskill_names]

        # Combine signatures (Union of action affordances)
        required_actions: set[int] = set()
        subgoals: list[SymbolicSubgoal] = []
        target_roles: set[str] = set()

        for c_skill in component_skills:
            if isinstance(c_skill.signature, ActionAffordancePredicate):
                required_actions.update(c_skill.signature.required)
            elif isinstance(c_skill.signature, AllOf):
                for pred in c_skill.signature.predicates:
                    if isinstance(pred, ActionAffordancePredicate):
                        required_actions.update(pred.required)
            if c_skill.program and c_skill.program.subgoals:
                subgoals.extend(c_skill.program.subgoals)

        for name in subskill_names:
            target_roles.update(self.nodes[name].target_roles)

        macro_skill = CompositeDeclarativeSkill(skills=component_skills)
        macro_skill.skill_name = macro_name
        macro_skill.description = (
            description or f"Composed macro-skill '{macro_name}' from {subskill_names}"
        )
        if required_actions:
            macro_skill.signature = ActionAffordancePredicate(required=required_actions)
        if subgoals:
            macro_skill.program = SubgoalSequence(*subgoals)

        self.register_skill(
            skill=macro_skill,
            tier=SkillTier.MACRO,
            subskills=list(subskill_names),
            target_roles=target_roles,
        )
        return macro_skill

    def decompose_macro_skill(self, macro_name: str) -> list[DeclarativeNeuroSymbolicSkill]:
        """Decompose a macro-skill into its ordered constituent subskills."""
        node = self.nodes.get(macro_name)
        if not node or not node.subskills:
            return []
        return [self.nodes[name].skill for name in node.subskills if name in self.nodes]

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose batch transactional mutations capturing the entire curricular graph."""
        return [node.to_transaction_operation() for node in self.nodes.values()]

    def to_causal_predicates(self) -> list[CausalPredicate]:
        """Export all curricular skills as formal causal predicates."""
        return [node.to_causal_predicate() for node in self.nodes.values()]
