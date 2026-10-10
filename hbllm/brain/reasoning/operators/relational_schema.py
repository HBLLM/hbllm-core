"""Relational Causal Schema Induction Operator over HCIR.

Transforms ground observation transitions into generalized, first-order relational
causal schemas via abductive-inductive abstraction:

    Ground Transition:
        Pre:   [adjacent_to(agent, apple_1), clear(apple_1)]
        Act:   pickup(apple_1)
        Post:  [holds(agent, apple_1), !clear(apple_1)]
            ↓
    First-Order Variabilization & Generalization:
        Schema: PickupObject(?agent: Agent, ?item: Object)
        Pre:    [adjacent_to(?agent, ?item), clear(?item)]
        Eff+:   [holds(?agent, ?item)]
        Eff-:   [clear(?item)]
        Bayesian Confidence: Beta(alpha=1, beta=1) updated per observation.

Independence Level: L1 (100% deterministic, 0 LLM tokens).
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_PRED_PATTERN = re.compile(r"^(!)?([a-zA-Z0-9_]+)(?:\((.*)\))?$")


@dataclass(frozen=True)
class PredicateTerm:
    """A term inside a relational predicate (constant or typed variable)."""

    name: str
    is_variable: bool = False
    term_type: str = "any"

    @classmethod
    def from_str(cls, s: str, type_hint: str = "any") -> PredicateTerm:
        s = s.strip()
        is_var = s.startswith("?")
        if ":" in s:
            parts = s.split(":", 1)
            name = parts[0].strip()
            term_type = parts[1].strip()
            is_var = name.startswith("?")
            return cls(name=name, is_variable=is_var, term_type=term_type)
        return cls(name=s, is_variable=is_var, term_type=type_hint)

    def to_string(self) -> str:
        if self.term_type != "any":
            return f"{self.name}:{self.term_type}"
        return self.name


@dataclass
class FirstOrderPredicate:
    """First-order relational predicate: name(arg1, arg2, ...)."""

    name: str
    terms: list[PredicateTerm] = field(default_factory=list)
    negated: bool = False

    @classmethod
    def from_string(cls, s: str, type_hints: dict[str, str] | None = None) -> FirstOrderPredicate:
        s = s.strip()
        hints = type_hints or {}
        match = _PRED_PATTERN.match(s)
        if not match:
            negated = s.startswith("!")
            clean = s[1:] if negated else s
            return cls(name=clean, terms=[], negated=negated)

        negated = match.group(1) is not None
        pred_name = match.group(2)
        args_str = match.group(3)
        terms: list[PredicateTerm] = []

        if args_str:
            raw_args = [a.strip() for a in args_str.split(",") if a.strip()]
            for arg in raw_args:
                hint = hints.get(arg, "any")
                terms.append(PredicateTerm.from_str(arg, type_hint=hint))

        return cls(name=pred_name, terms=terms, negated=negated)

    def to_string(self, show_types: bool = False) -> str:
        prefix = "!" if self.negated else ""
        if not self.terms:
            return f"{prefix}{self.name}"
        args_str = ", ".join(t.to_string() if show_types else t.name for t in self.terms)
        return f"{prefix}{self.name}({args_str})"

    def is_ground(self) -> bool:
        return not any(t.is_variable for t in self.terms)

    def bind(self, bindings: dict[str, str]) -> FirstOrderPredicate:
        """Return a new predicate with variables replaced by bound values."""
        new_terms: list[PredicateTerm] = []
        for t in self.terms:
            if t.is_variable and t.name in bindings:
                bound_val = bindings[t.name]
                new_terms.append(
                    PredicateTerm(name=bound_val, is_variable=False, term_type=t.term_type)
                )
            else:
                new_terms.append(t)
        return FirstOrderPredicate(name=self.name, terms=new_terms, negated=self.negated)

    def unify(
        self,
        ground_pred: FirstOrderPredicate,
        current_bindings: dict[str, str] | None = None,
    ) -> dict[str, str] | None:
        """Unify this predicate (pattern) with a ground predicate."""
        if self.name != ground_pred.name or self.negated != ground_pred.negated:
            return None
        if len(self.terms) != len(ground_pred.terms):
            return None

        bindings = dict(current_bindings or {})
        for pattern_term, ground_term in zip(self.terms, ground_pred.terms, strict=False):
            if pattern_term.is_variable:
                var_name = pattern_term.name
                if var_name in bindings:
                    if bindings[var_name] != ground_term.name:
                        return None
                else:
                    bindings[var_name] = ground_term.name
            else:
                if pattern_term.name != ground_term.name:
                    return None

        return bindings


@dataclass
class CausalSchema:
    """A generalized, parameterized causal schema learned from experience."""

    schema_id: str
    name: str
    parameters: list[PredicateTerm]
    preconditions: list[FirstOrderPredicate]
    effects_add: list[FirstOrderPredicate]
    effects_del: list[FirstOrderPredicate]
    success_count: int = 0
    failure_count: int = 0
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    exemplars: list[str] = field(default_factory=list)
    brier_history: list[float] = field(default_factory=list)

    @property
    def total_trials(self) -> int:
        return self.success_count + self.failure_count

    @property
    def confidence(self) -> float:
        """Posterior expected probability of causal satisfaction: (alpha + s) / (alpha + beta + s + f)."""
        return (self.alpha_prior + self.success_count) / (
            self.alpha_prior + self.beta_prior + self.success_count + self.failure_count
        )

    @property
    def mean_brier_score(self) -> float:
        """Mean squared probability error across recorded validations."""
        if not self.brier_history:
            return 0.0
        return sum(self.brier_history) / len(self.brier_history)

    def record_outcome(self, success: bool, exemplar_id: str = "") -> None:
        """Update Bayesian Dirichlet/Beta parameters and track Brier score."""
        pred_p = self.confidence
        actual = 1.0 if success else 0.0
        brier = (pred_p - actual) ** 2
        self.brier_history.append(brier)

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        if exemplar_id and exemplar_id not in self.exemplars:
            self.exemplars.append(exemplar_id)

    def instantiate(
        self, bindings: dict[str, str]
    ) -> tuple[list[FirstOrderPredicate], list[FirstOrderPredicate], list[FirstOrderPredicate]]:
        """Produce concrete ground preconditions, adds, and dels given variable bindings."""
        ground_pre = [p.bind(bindings) for p in self.preconditions]
        ground_add = [p.bind(bindings) for p in self.effects_add]
        ground_del = [p.bind(bindings) for p in self.effects_del]
        return ground_pre, ground_add, ground_del

    def unifies_with_goal(self, goal_predicate: FirstOrderPredicate) -> list[dict[str, str]]:
        """Find all variable bindings that would allow this schema to produce the goal predicate."""
        matching_bindings: list[dict[str, str]] = []
        for eff in self.effects_add:
            bindings = eff.unify(goal_predicate)
            if bindings is not None:
                matching_bindings.append(bindings)
        return matching_bindings

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "name": self.name,
            "parameters": [p.to_string() for p in self.parameters],
            "preconditions": [p.to_string(show_types=True) for p in self.preconditions],
            "effects_add": [p.to_string(show_types=True) for p in self.effects_add],
            "effects_del": [p.to_string(show_types=True) for p in self.effects_del],
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "confidence": self.confidence,
            "mean_brier_score": self.mean_brier_score,
            "exemplars": self.exemplars,
        }


class RelationalSchemaInducer:
    """Online Inductive & Abductive Causal Schema Engine for HCIR.

    Ingests state transitions, abstracts entity instances into typed relational variables,
    calculates empirical Bayesian confidence distributions, and validates causal laws.
    """

    def __init__(
        self,
        min_instances_to_validate: int = 3,
        confidence_threshold: float = 0.75,
    ) -> None:
        self.min_instances_to_validate = min_instances_to_validate
        self.confidence_threshold = confidence_threshold
        self.candidate_schemas: dict[str, CausalSchema] = {}
        self.validated_schemas: dict[str, CausalSchema] = {}

    def _variabilize(
        self,
        pred_strings: list[str],
        entity_types: dict[str, str] | None = None,
    ) -> tuple[list[FirstOrderPredicate], dict[str, str], list[PredicateTerm]]:
        """Convert list of ground condition strings into typed variabilized predicates."""
        types = entity_types or {}
        var_map: dict[str, str] = {}  # ground_name -> "?var_name"
        var_types: dict[str, str] = {}  # "?var_name" -> "Type"
        predicates: list[FirstOrderPredicate] = []

        # 1. Identify all ground arguments and allocate variable names
        var_counter = 1
        for s in pred_strings:
            p = FirstOrderPredicate.from_string(s)
            for term in p.terms:
                if not term.is_variable and term.name not in var_map:
                    t_type = types.get(term.name, "Entity")
                    var_name = f"?x{var_counter}"
                    var_map[term.name] = var_name
                    var_types[var_name] = t_type
                    var_counter += 1

        # 2. Re-create predicates with variables substituted
        for s in pred_strings:
            p = FirstOrderPredicate.from_string(s)
            new_terms: list[PredicateTerm] = []
            for term in p.terms:
                if term.name in var_map:
                    v_name = var_map[term.name]
                    new_terms.append(
                        PredicateTerm(name=v_name, is_variable=True, term_type=var_types[v_name])
                    )
                else:
                    new_terms.append(term)
            predicates.append(FirstOrderPredicate(name=p.name, terms=new_terms, negated=p.negated))

        param_terms = [
            PredicateTerm(name=v_name, is_variable=True, term_type=t_type)
            for v_name, t_type in var_types.items()
        ]
        return predicates, var_map, param_terms

    def observe_transition(
        self,
        action_name: str,
        pre_state_conditions: set[str],
        post_state_conditions: set[str],
        entity_types: dict[str, str] | None = None,
        exemplar_id: str = "",
    ) -> CausalSchema:
        """Observe state transition (Pre, Action, Post) and update or induce causal schema."""
        exemplar = exemplar_id or str(uuid.uuid4())[:8]

        # 1. Calculate deltas
        added = post_state_conditions - pre_state_conditions
        deleted = pre_state_conditions - post_state_conditions

        # 2. Extract contextually relevant preconditions (those sharing entities with deltas)
        delta_entities: set[str] = set()
        for c in added | deleted:
            p = FirstOrderPredicate.from_string(c)
            for t in p.terms:
                delta_entities.add(t.name)

        relevant_pre: list[str] = []
        for c in pre_state_conditions:
            p = FirstOrderPredicate.from_string(c)
            if any(t.name in delta_entities for t in p.terms) or not p.terms:
                relevant_pre.append(c)

        # 3. Variabilize the transition components
        var_pre, var_map_pre, params = self._variabilize(relevant_pre, entity_types)
        var_add, _, _ = self._variabilize(list(added), entity_types)
        var_del, _, _ = self._variabilize(list(deleted), entity_types)

        # Normalize schema signature
        action_base = action_name.split("(")[0].strip()
        schema_key = f"{action_base}__{len(var_pre)}__{len(var_add)}__{len(var_del)}"

        # 4. Update existing or create candidate schema
        target_schema = self.validated_schemas.get(schema_key) or self.candidate_schemas.get(
            schema_key
        )

        if target_schema is None:
            target_schema = CausalSchema(
                schema_id=str(uuid.uuid4())[:8],
                name=f"Schema_{action_base}_{schema_key}",
                parameters=params,
                preconditions=var_pre,
                effects_add=var_add,
                effects_del=var_del,
            )
            self.candidate_schemas[schema_key] = target_schema

        # Record empirical outcome
        target_schema.record_outcome(success=True, exemplar_id=exemplar)

        # 5. Check validation threshold
        if (
            target_schema.confidence >= self.confidence_threshold
            and target_schema.total_trials >= self.min_instances_to_validate
            and schema_key not in self.validated_schemas
        ):
            self.validated_schemas[schema_key] = target_schema
            self.candidate_schemas.pop(schema_key, None)
            logger.info(
                "CausalSchema %s promoted to VALIDATED (Confidence=%.2f)",
                target_schema.name,
                target_schema.confidence,
            )

        return target_schema

    def find_schemas_for_goal(
        self,
        goal_condition: str,
        only_validated: bool = False,
    ) -> list[tuple[CausalSchema, dict[str, str]]]:
        """Query schemas capable of producing the specified goal condition."""
        goal_pred = FirstOrderPredicate.from_string(goal_condition)
        results: list[tuple[CausalSchema, dict[str, str]]] = []

        pool = list(self.validated_schemas.values())
        if not only_validated:
            pool.extend(self.candidate_schemas.values())

        for schema in pool:
            for bindings in schema.unifies_with_goal(goal_pred):
                results.append((schema, bindings))

        # Sort by posterior confidence descending
        results.sort(key=lambda x: x[0].confidence, reverse=True)
        return results

    def get_catalog_summary(self) -> dict[str, Any]:
        return {
            "validated_count": len(self.validated_schemas),
            "candidate_count": len(self.candidate_schemas),
            "validated": [s.to_dict() for s in self.validated_schemas.values()],
            "candidates": [s.to_dict() for s in self.candidate_schemas.values()],
        }
