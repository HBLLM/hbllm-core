"""Relational Graph Neural-Symbolic Policy Distillation.

Extracts invariant relational production rules from successful (and catastrophic)
episodic traces, compressing high-dimensional spatiotemporal actions into compact
neuro-symbolic declarative programs (Garnelo & Shanahan 2019, Gentner 1997).

Invariant rule structure:
    IF Premise(Agent, TargetEntity, SpatialRelation) => DO(Action) [or INHIBIT(Action)]
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hbllm.hcir.skills.declarative import (
    AllOf,
    DeclarativeNeuroSymbolicSkill,
    SpatialActionIntent,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_episodic import EpisodicTransition

logger = logging.getLogger(__name__)


class SpatialRelationType(StrEnum):
    """Categorical spatial relations between agent and environment entities."""

    ADJACENT_CARDINAL = "adjacent_cardinal"  # Manhattan distance == 1
    ALIGNED_LINE_OF_SIGHT = "aligned_line_of_sight"  # Same row or col, unobstructed
    WITHIN_VICINITY = "within_vicinity"  # Manhattan distance <= threshold
    COINCIDENT = "coincident"  # Exact same cell (stepping on item/trigger)


@dataclass(frozen=True)
class RelationalPremise:
    """A symbolic relational precondition over spatial entities."""

    source_role: EntityRole | str
    target_role: EntityRole | str
    relation_type: SpatialRelationType
    relative_offset: tuple[int, int] | None = None

    def match(
        self,
        source_pos: tuple[int, int],
        target_pos: tuple[int, int],
        vicinity_radius: int = 3,
    ) -> bool:
        """Check if source and target coordinates satisfy this relational premise."""
        sr, sc = source_pos
        tr, tc = target_pos
        dr, dc = tr - sr, tc - sc
        manhattan_dist = abs(dr) + abs(dc)

        if self.relation_type == SpatialRelationType.COINCIDENT:
            return manhattan_dist == 0

        if self.relation_type == SpatialRelationType.ADJACENT_CARDINAL:
            if manhattan_dist == 1:
                if self.relative_offset is not None:
                    return (dr, dc) == self.relative_offset
                return True
            return False

        if self.relation_type == SpatialRelationType.ALIGNED_LINE_OF_SIGHT:
            return dr == 0 or dc == 0

        if self.relation_type == SpatialRelationType.WITHIN_VICINITY:
            return manhattan_dist <= vicinity_radius

        return False


@dataclass
class DistilledPolicyRule:
    """A compact neuro-symbolic production rule distilled from experience."""

    rule_id: str
    premise: RelationalPremise
    recommended_action: Any
    is_inhibition: bool = False
    support_count: int = 1
    confidence: float = 1.0
    entropy_reduction: float = 0.0
    description: str = ""

    def evaluate(
        self,
        avatar_pos: tuple[int, int],
        entities: Sequence[SpatialEntity],
    ) -> bool:
        """Check if this rule triggers given the current avatar position and entities."""
        for ent in entities:
            ent_role = getattr(ent, "role", EntityRole.UNKNOWN)
            if ent_role != self.premise.target_role and self.premise.target_role != "*":
                continue
            cells = (
                getattr(ent, "cells", None)
                or (ent.properties.get("cells") if hasattr(ent, "properties") else None)
                or ([ent.grid_pos] if hasattr(ent, "grid_pos") else [])
            )
            for c in cells:
                if self.premise.match(avatar_pos, c):
                    return True
        return False

    def to_causal_predicate(self) -> CausalPredicate:
        """Export distilled rule as a formal Pearl causal predicate."""
        action_str = f"{'INHIBIT' if self.is_inhibition else 'DO'}:{self.recommended_action}"
        return CausalPredicate(
            variable=f"relational_rule:{self.rule_id}",
            operator="=>",
            value=action_str,
            relation=f"{self.premise.source_role}->{self.premise.relation_type}->{self.premise.target_role}",
            target_entity=str(self.premise.target_role),
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Convert distilled rule into transactional HCIR graph state mutations."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"distilled_rule_{self.rule_id}",
                node_data={
                    "rule_id": self.rule_id,
                    "premise_source": str(self.premise.source_role),
                    "premise_target": str(self.premise.target_role),
                    "relation_type": str(self.premise.relation_type),
                    "relative_offset": list(self.premise.relative_offset)
                    if self.premise.relative_offset
                    else None,
                    "action": str(self.recommended_action),
                    "is_inhibition": self.is_inhibition,
                    "support": self.support_count,
                    "confidence": self.confidence,
                },
            )
        ]


class RelationalPolicyDistiller:
    """Distills episodic transition sequences into invariant relational production rules."""

    @classmethod
    def distill_trajectory(
        cls,
        trajectory: Sequence[EpisodicTransition],
        entities: Sequence[SpatialEntity] | None = None,
        min_support: int = 1,
        min_confidence: float = 0.60,
    ) -> list[DistilledPolicyRule]:
        """Extract high-confidence relational rules from an episodic trajectory."""
        if not trajectory or not entities:
            return []

        # Count state-relation-action occurrences
        relation_action_counts: dict[tuple[RelationalPremise, Any, bool], int] = defaultdict(int)
        relation_total_counts: dict[tuple[RelationalPremise, bool], int] = defaultdict(int)

        # Build map of entity locations
        entity_role_map: list[tuple[EntityRole | str, tuple[int, int]]] = []
        for ent in entities:
            role = getattr(ent, "role", EntityRole.UNKNOWN)
            cells = (
                getattr(ent, "cells", None)
                or (ent.properties.get("cells") if hasattr(ent, "properties") else None)
                or ([ent.grid_pos] if hasattr(ent, "grid_pos") else [])
            )
            for c in cells:
                entity_role_map.append((role, c))

        for t in trajectory:
            if t.avatar_pos is None or t.action_id is None:
                continue

            ar, ac = t.avatar_pos
            is_fatal = t.is_lost

            # Find active relational premises with surrounding entities
            for role, (er, ec) in entity_role_map:
                dr, dc = er - ar, ec - ac
                dist = abs(dr) + abs(dc)

                premise: RelationalPremise | None = None
                if dist == 0:
                    premise = RelationalPremise(
                        source_role=EntityRole.AGENT,
                        target_role=role,
                        relation_type=SpatialRelationType.COINCIDENT,
                    )
                elif dist == 1:
                    premise = RelationalPremise(
                        source_role=EntityRole.AGENT,
                        target_role=role,
                        relation_type=SpatialRelationType.ADJACENT_CARDINAL,
                        relative_offset=(dr, dc),
                    )
                elif dr == 0 or dc == 0:
                    premise = RelationalPremise(
                        source_role=EntityRole.AGENT,
                        target_role=role,
                        relation_type=SpatialRelationType.ALIGNED_LINE_OF_SIGHT,
                    )

                if premise is not None:
                    key = (premise, t.action_id, is_fatal)
                    relation_action_counts[key] += 1
                    relation_total_counts[(premise, is_fatal)] += 1

        distilled_rules: list[DistilledPolicyRule] = []
        idx = 0

        for (premise, action, is_fatal), count in relation_action_counts.items():
            if count < min_support:
                continue
            total = relation_total_counts.get((premise, is_fatal), count)
            confidence = float(count / max(1, total))

            if confidence >= min_confidence:
                idx += 1
                rule = DistilledPolicyRule(
                    rule_id=f"rule_{idx}_{premise.target_role}_{action}",
                    premise=premise,
                    recommended_action=action,
                    is_inhibition=is_fatal,
                    support_count=count,
                    confidence=confidence,
                    description=f"When {premise.relation_type} to {premise.target_role}, "
                    f"{'inhibit' if is_fatal else 'execute'} action {action}",
                )
                distilled_rules.append(rule)

        # Sort rules by confidence and support
        distilled_rules.sort(key=lambda r: (r.confidence, r.support_count), reverse=True)
        return distilled_rules

    @classmethod
    def compile_to_declarative_skill(
        cls,
        rules: Sequence[DistilledPolicyRule],
        skill_name: str = "DistilledRelationalSkill",
    ) -> DeclarativeNeuroSymbolicSkill:
        """Compile a collection of distilled rules into an executable DeclarativeNeuroSymbolicSkill."""
        subgoals: list[SymbolicSubgoal] = []

        for r in rules:
            if not r.is_inhibition:
                subgoals.append(
                    SymbolicSubgoal(
                        intent=SpatialActionIntent.NAVIGATE,
                        target_query={
                            "role": str(r.premise.target_role),
                            "action": r.recommended_action,
                            "relation": str(r.premise.relation_type),
                        },
                    )
                )

        skill = DeclarativeNeuroSymbolicSkill()
        skill.skill_name = skill_name
        skill.description = f"Neuro-Symbolic skill distilled from {len(rules)} relational rules"
        skill.signature = AllOf()
        skill.program = SubgoalSequence(*subgoals) if subgoals else None
        setattr(skill, "_distilled_rules", list(rules))
        return skill
