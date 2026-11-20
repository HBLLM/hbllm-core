"""Pearl Causal Predicates, Logic Operators, and Belief Events."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class BeliefTransitionType(str, Enum):
    """Immutable event types for event-sourced developmental belief logging."""

    HYPOTHESIS_CREATED = "hypothesis_created"
    HYPOTHESIS_TESTED = "hypothesis_tested"
    HYPOTHESIS_FALSIFIED = "hypothesis_falsified"
    HYPOTHESIS_CONFIRMED = "hypothesis_confirmed"
    CONFIDENCE_CHANGED = "confidence_changed"
    RULE_GENERALIZED = "rule_generalized"
    RULE_REVISED = "rule_revised"
    AFFORDANCE_DISCOVERED = "affordance_discovered"
    SPATIAL_SCHEMA_INDUCED = "spatial_schema_induced"
    TOOL_COMPOSED = "tool_composed"
    GOAL_SYNTHESIZED = "goal_synthesized"
    PLAN_EXECUTED = "plan_executed"
    PLAN_REPLANNED = "plan_replanned"
    CURIOSITY_EXPLORATION = "curiosity_exploration"
    CONCEPT_INDUCED = "concept_induced"
    LEXICON_GROUNDED = "lexicon_grounded"
    COMPOSITION_PARSED = "composition_parsed"
    MEMORY_CONSOLIDATED = "memory_consolidated"
    METACOGNITION_CALIBRATED = "metacognition_calibrated"
    CROSS_WORLD_TRANSFERRED = "cross_world_transferred"
    CROSS_DOMAIN_TRANSFERRED = "cross_domain_transferred"


class LogicOperator(str, Enum):
    """Boolean logic operators for composite causal preconditions."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"


@dataclass
class CausalPredicate:
    """Unified representation for atomic, relational, and composite causal preconditions."""

    # Atomic evaluation: e.g. variable="mass", operator="<", value=5.0
    variable: str = ""
    operator: str = "=="
    value: Any = None

    # Relational evaluation: e.g. relation="contained_in", target_entity="box_1"
    relation: str = ""
    target_entity: str = ""

    # Composite evaluation: e.g. logic_op=AND, children=[pred1, pred2]
    logic_op: LogicOperator | None = None
    children: list[CausalPredicate] = field(default_factory=list)

    # Causal lifecycle & confidence tracking (G8)
    causal_status: str = "HYPOTHESISED"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_composite(self) -> bool:
        """True if predicate represents a compound boolean expression."""
        return self.logic_op is not None and bool(self.children)

    def is_relational(self) -> bool:
        """True if predicate represents an inter-entity relational edge."""
        return bool(self.relation)

    def evaluate(
        self,
        features: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Recursively evaluate predicate against features and optional context."""
        # 1. Composite boolean evaluation
        if self.logic_op == LogicOperator.AND:
            if not self.children:
                return True
            return all(c.evaluate(features, context) for c in self.children)
        elif self.logic_op == LogicOperator.OR:
            if not self.children:
                return False
            return any(c.evaluate(features, context) for c in self.children)
        elif self.logic_op == LogicOperator.NOT:
            if not self.children:
                return True
            return not self.children[0].evaluate(features, context)

        # 2. Relational evaluation
        if self.is_relational():
            rel_val = features.get(self.relation)
            if rel_val is None and context:
                rel_val = context.get(self.relation)
                if rel_val is None and "relations" in context:
                    relations = context["relations"]
                    if isinstance(relations, dict):
                        rel_val = relations.get(self.relation)
                    elif isinstance(relations, list):
                        for item in relations:
                            if (
                                isinstance(item, (tuple, list))
                                and len(item) == 2
                                and item[0] == self.relation
                            ):
                                rel_val = item[1]
                                break
            if rel_val is None:
                return False

            target = self.target_entity if self.target_entity != "" else self.value
            if self.operator == "==":
                return str(rel_val) == str(target)
            elif self.operator == "!=":
                return str(rel_val) != str(target)
            elif self.operator == "in":
                if isinstance(rel_val, (list, set, tuple)):
                    return target in rel_val
                return str(target) in str(rel_val)
            elif self.operator == "contains":
                if isinstance(rel_val, (list, set, tuple)):
                    return target in rel_val
                return str(target) in str(rel_val)
            return False

        # 3. Atomic attribute evaluation
        val = features.get(self.variable)
        if val is None:
            return False
        if self.operator == "==":
            return str(val) == str(self.value)
        elif self.operator == "!=":
            return str(val) != str(self.value)
        elif self.operator == "in":
            if isinstance(self.value, (list, set, tuple)):
                return val in self.value
            return str(val) in str(self.value)
        elif self.operator == "contains":
            if isinstance(val, (list, set, tuple)):
                return self.value in val
            return str(self.value) in str(val)
        try:
            fval = float(val)
            fthresh = float(self.value)
            if self.operator == "<":
                return fval < fthresh
            elif self.operator == "<=":
                return fval <= fthresh
            elif self.operator == ">":
                return fval > fthresh
            elif self.operator == ">=":
                return fval >= fthresh
        except (ValueError, TypeError):
            return False
        return False

    def describe(self) -> str:
        """Generate human-readable and formal logical description."""
        if self.logic_op == LogicOperator.AND:
            parts = [
                f"({c.describe()})" if c.is_composite() else c.describe() for c in self.children
            ]
            return " ∧ ".join(parts)
        elif self.logic_op == LogicOperator.OR:
            parts = [
                f"({c.describe()})" if c.is_composite() else c.describe() for c in self.children
            ]
            return " ∨ ".join(parts)
        elif self.logic_op == LogicOperator.NOT:
            child_desc = self.children[0].describe() if self.children else "true"
            return f"¬({child_desc})"
        elif self.is_relational():
            target = self.target_entity if self.target_entity != "" else self.value
            if self.operator == "==":
                return f"{self.relation}(x, {target})"
            return f"{self.relation}(x, {target}) {self.operator} true"
        return f"{self.variable} {self.operator} {self.value}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize predicate tree to dictionary."""
        return {
            "variable": self.variable,
            "operator": self.operator,
            "value": self.value,
            "relation": self.relation,
            "target_entity": self.target_entity,
            "logic_op": self.logic_op.value if self.logic_op else None,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalPredicate:
        """Deserialize dictionary into predicate tree."""
        lop = LogicOperator(data["logic_op"]) if data.get("logic_op") else None
        children = [cls.from_dict(c) for c in data.get("children", [])]
        return cls(
            variable=data.get("variable", ""),
            operator=data.get("operator", "=="),
            value=data.get("value"),
            relation=data.get("relation", ""),
            target_entity=data.get("target_entity", ""),
            logic_op=lop,
            children=children,
        )


@dataclass
class BeliefTransitionEvent:
    """An immutable record of a developmental or cognitive belief transition."""

    event_id: str = field(default_factory=lambda: f"bte_{uuid.uuid4().hex[:8]}")
    event_type: BeliefTransitionType = BeliefTransitionType.HYPOTHESIS_CREATED
    step_index: int = 0
    hypothesis_id: str = ""
    variable: str = ""  # e.g. "color", "mass", "shape", "feature_x"
    condition: str = ""  # e.g. "color == 'red'", "mass < 5.0"
    prior_confidence: float = 0.0
    posterior_confidence: float = 0.0
    is_falsified: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class CausalHypothesis:
    """State tracker for candidate causal relationships."""

    hypothesis_id: str = field(default_factory=lambda: f"hyp_{uuid.uuid4().hex[:6]}")
    action: Any = "PUSH"
    variable: str = "color"
    operator: str = "=="
    value: Any = "red"
    consequence: str = "MOVES"
    confidence: float = 0.5
    interventions_tested: int = 0
    falsified: bool = False
    confirmed: bool = False
    supporting_episodes: list[str] = field(default_factory=list)
    counterexamples: list[str] = field(default_factory=list)
    predicate: CausalPredicate | None = None

    def get_predicate(self) -> CausalPredicate:
        """Return existing predicate or synthesize atomic CausalPredicate."""
        if self.predicate is not None:
            return self.predicate
        return CausalPredicate(
            variable=self.variable,
            operator=self.operator,
            value=self.value,
        )

    def describe(self) -> str:
        """Format causal hypothesis into symbolic statement."""
        act_val = self.action.value if isinstance(self.action, Enum) else str(self.action)
        if self.predicate is not None:
            return f"{act_val}(x) ∧ ({self.predicate.describe()}) => {self.consequence}"
        return (
            f"{act_val}(x) ∧ ({self.variable} {self.operator} {self.value}) => {self.consequence}"
        )
