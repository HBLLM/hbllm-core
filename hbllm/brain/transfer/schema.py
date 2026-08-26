"""Relational Schema Data Models and Bayesian Reliability for A20.

Defines reusable structural graph templates with typed variable roles,
physical/geometric constraints, higher-order relations, action templates,
and Beta-evidence reliability updates.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SchemaLifecycleStatus(str, Enum):
    """Lifecycle maturity of an induced relational schema."""

    CANDIDATE = "candidate"  # Proposed from recurring subgraph observation
    VERIFIED = "verified"  # Confirmed predictive validity in source domain
    TRANSFERABLE = "transferable"  # Sufficiently validated for cross-domain transfer
    SPECIALIZED = "specialized"  # Boundary constraints narrowed after negative outcome
    DEPRECATED = "deprecated"  # High contradiction rate, retired from transfer


@dataclass
class SchemaRole:
    """A parameterized variable role within a RelationalSchema."""

    role_id: str  # e.g. "Base", "Payload", "Container", "Tool", "Target"
    type_requirement: str = ""  # Optional broad category, e.g. "physical_entity"
    required_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaRelation:
    """A directed topological relationship between roles."""

    source_role: str
    edge_type: str  # e.g. "LOCATED_ON", "LOCATED_IN", "SUPPORTS", "CAUSES", "DISPLACES"
    target_role: str


@dataclass
class SchemaConstraint:
    """A physical, geometric, or state precondition required for schema applicability."""

    role_id: str
    property_key: str  # e.g. "geometry", "surface", "is_closed"
    expected_value: Any  # e.g. "flat", False


@dataclass
class ActionTemplate:
    """A parameterized action operator linked to the relational schema."""

    operator_name: str  # e.g. "STACK", "PUT_IN", "PUSH", "MOVE"
    role_parameters: dict[str, str] = field(default_factory=dict)  # param_key -> role_id


@dataclass
class ConsequenceTemplate:
    """A predicted state transition or relational consequence."""

    consequence_type: str  # e.g. "stable_support", "contained", "displacement"
    predicted_edge_type: str = ""
    source_role: str = ""
    target_role: str = ""


@dataclass
class RelationalSchema:
    """A generalized, reusable structural pattern extracted from grounded HCIR experience."""

    schema_id: str = field(default_factory=lambda: f"schema_{uuid.uuid4().hex[:8]}")
    name: str = ""
    roles: list[SchemaRole] = field(default_factory=list)
    relations: list[SchemaRelation] = field(default_factory=list)
    higher_order_relations: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[SchemaConstraint] = field(default_factory=list)
    action_templates: list[ActionTemplate] = field(default_factory=list)
    predicted_consequences: list[ConsequenceTemplate] = field(default_factory=list)

    # Provenance & evidence tracking
    source_episode_ids: list[str] = field(default_factory=list)
    alpha_success: float = 3.0  # Beta prior α
    beta_failure: float = 1.0  # Beta prior β
    status: SchemaLifecycleStatus = SchemaLifecycleStatus.TRANSFERABLE
    specialization_rules: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Bayesian reliability estimate: α / (α + β)."""
        total = self.alpha_success + self.beta_failure
        return round(self.alpha_success / total, 4) if total > 0.0 else 0.50

    @property
    def is_transferable(self) -> bool:
        return (
            self.status
            in (
                SchemaLifecycleStatus.VERIFIED,
                SchemaLifecycleStatus.TRANSFERABLE,
                SchemaLifecycleStatus.SPECIALIZED,
            )
            and self.confidence >= 0.60
        )

    def evaluate_constraint_compatibility(
        self, entity_properties_map: dict[str, dict[str, Any]]
    ) -> tuple[bool, list[str]]:
        """Verify that bound target entity properties satisfy schema physical constraints.

        Args:
            entity_properties_map: Mapping of role_id -> entity properties dict.

        Returns:
            (is_valid, list of violated constraint descriptions)
        """
        violations: list[str] = []
        for c in self.constraints:
            props = entity_properties_map.get(c.role_id, {})
            actual_val = props.get(c.property_key)
            if actual_val is None or str(actual_val).lower() != str(c.expected_value).lower():
                violations.append(
                    f"Constraint violation on role '{c.role_id}': expected {c.property_key}={c.expected_value}, got {actual_val}"
                )

        return len(violations) == 0, violations

    def record_outcome(self, is_success: bool, failed_constraint: str | None = None) -> None:
        """Update Bayesian reliability and apply specialization rules on failure."""
        if is_success:
            self.alpha_success += 1.0
        else:
            self.beta_failure += 1.0
            if failed_constraint:
                self.specialization_rules.append(failed_constraint)
                self.status = SchemaLifecycleStatus.SPECIALIZED

        # Deprecate if failure rate is overwhelmingly high
        if self.confidence < 0.35 and (self.alpha_success + self.beta_failure) > 10:
            self.status = SchemaLifecycleStatus.DEPRECATED
