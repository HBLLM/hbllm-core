"""HCIR Relational Transfer Bridge for A23.5-E4.

Lifts developmentally induced schemas (containment, tool use, causal push)
acquired by the blank-brain substrate into declarative relational schemas,
and executes zero-shot structural analogical matching directly over ``CognitiveGraph``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.graph import CognitiveGraph, HCIRNode

logger = logging.getLogger(__name__)


class SchemaLifecycleStatus(StrEnum):
    """Epistemic maturity of an induced relational schema."""

    CANDIDATE = "candidate"
    VALIDATED = "validated"
    TRANSFERABLE = "transferable"
    DEPRECATED = "deprecated"


class MappingStatus(StrEnum):
    """Outcome of aligning a relational schema against a target HCIR graph."""

    APPLICABLE = "applicable"
    PARTIAL = "partial"
    REJECTED = "rejected"


@dataclass
class SchemaRole:
    role_id: str
    type_requirement: str = "physical_entity"
    required_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaRelation:
    source_role: str
    edge_type: str
    target_role: str


@dataclass
class SchemaConstraint:
    role_id: str
    property_key: str
    expected_value: Any
    operator: str = "eq"
    is_required: bool = True


@dataclass
class ActionTemplate:
    operator_name: str
    role_parameters: dict[str, str] = field(default_factory=dict)


@dataclass
class ConsequenceTemplate:
    consequence_type: str
    predicted_edge_type: str
    source_role: str
    target_role: str


@dataclass
class RelationalSchema:
    """Declarative relational schema transferred across HCIR graphs."""

    schema_id: str
    name: str
    roles: list[SchemaRole] = field(default_factory=list)
    relations: list[SchemaRelation] = field(default_factory=list)
    constraints: list[SchemaConstraint] = field(default_factory=list)
    action_templates: list[ActionTemplate] = field(default_factory=list)
    predicted_consequences: list[ConsequenceTemplate] = field(default_factory=list)
    status: SchemaLifecycleStatus = SchemaLifecycleStatus.TRANSFERABLE
    alpha_success: float = 1.0
    beta_failure: float = 1.0

    @property
    def is_transferable(self) -> bool:
        return self.status == SchemaLifecycleStatus.TRANSFERABLE

    @property
    def confidence(self) -> float:
        denom = max(1e-9, self.alpha_success + self.beta_failure)
        return self.alpha_success / denom


def _evaluate_constraint(constraint: SchemaConstraint, props: dict[str, Any]) -> str | None:
    """Evaluate a SchemaConstraint against node properties; return violation message or None."""
    if constraint.property_key not in props:
        if constraint.is_required:
            return f"Missing required property '{constraint.property_key}' on role {constraint.role_id}"
        return None

    actual = props[constraint.property_key]
    expected = constraint.expected_value
    op = constraint.operator
    passed = False
    try:
        if op == "eq":
            passed = actual == expected
        elif op == "neq":
            passed = actual != expected
        elif op == "lte":
            passed = float(actual) <= float(expected)
        elif op == "lt":
            passed = float(actual) < float(expected)
        elif op == "gte":
            passed = float(actual) >= float(expected)
        elif op == "gt":
            passed = float(actual) > float(expected)
        else:
            passed = actual == expected
    except (TypeError, ValueError):
        passed = False

    if not passed:
        return (
            f"Constraint violated on {constraint.role_id}.{constraint.property_key}: "
            f"expected {op} {expected!r}, got {actual!r}"
        )
    return None


def _node_matches_type(node: HCIRNode, type_requirement: str) -> bool:
    """Match an HCIRNode against a schema role's entity type or boolean property marker."""
    req = type_requirement.lower()
    props = getattr(node, "properties", {}) or {}
    entity_type = str(
        getattr(node, "entity_type", "") or props.get("entity_type", "") or ""
    ).lower()
    if entity_type == req:
        return True
    if bool(props.get(f"is_{req}", False)):
        return True
    if req == "physical_entity":
        return (
            (
                not props.get("is_agent")
                and not props.get("is_container")
                and not props.get("is_tool")
                and entity_type not in ("agent", "container", "tool")
            )
            or bool(props.get("is_physical", False))
            or entity_type == "physical_entity"
        )
    return str(getattr(node, "node_type", "")).lower() == req


class A20RelationalTransferBridge:
    """Bridges developmental cognitive discoveries (A23) to HCIR Relational Generalization."""

    def __init__(self) -> None:
        self.transfer_history: list[dict[str, Any]] = []

    @staticmethod
    def lift_containment_schema(
        discovered_schema: dict[str, Any] | None = None,
    ) -> RelationalSchema:
        """Lift a developmentally discovered containment schema into a RelationalSchema."""
        schema_id = (
            discovered_schema.get("schema_id", f"schema_containment_{uuid.uuid4().hex[:6]}")
            if discovered_schema
            else f"schema_containment_{uuid.uuid4().hex[:6]}"
        )

        roles = [
            SchemaRole(
                role_id="Container",
                type_requirement="container",
                required_properties={"is_closed": False},
            ),
            SchemaRole(
                role_id="Payload",
                type_requirement="physical_entity",
            ),
        ]
        relations = [
            SchemaRelation(source_role="Payload", edge_type="LOCATED_IN", target_role="Container"),
            SchemaRelation(source_role="Container", edge_type="SUPPORTS", target_role="Payload"),
        ]
        constraints = [
            SchemaConstraint(
                role_id="Container",
                property_key="is_closed",
                expected_value=False,
                operator="eq",
                is_required=True,
            ),
        ]
        actions = [
            ActionTemplate(
                operator_name="PUT_IN",
                role_parameters={"item_id": "Payload", "container_id": "Container"},
            ),
            ActionTemplate(
                operator_name="MOVE",
                role_parameters={"target_id": "Container"},
            ),
        ]
        consequences = [
            ConsequenceTemplate(
                consequence_type="contained",
                predicted_edge_type="LOCATED_IN",
                source_role="Payload",
                target_role="Container",
            ),
            ConsequenceTemplate(
                consequence_type="synchronous_displacement",
                predicted_edge_type="MOVES_WITH",
                source_role="Payload",
                target_role="Container",
            ),
        ]

        return RelationalSchema(
            schema_id=schema_id,
            name="Developmental-Containment-Transport",
            roles=roles,
            relations=relations,
            constraints=constraints,
            action_templates=actions,
            predicted_consequences=consequences,
            status=SchemaLifecycleStatus.TRANSFERABLE,
            alpha_success=5.0,
            beta_failure=1.0,
        )

    @staticmethod
    def lift_tool_reach_schema(
        discovered_tool: dict[str, Any] | None = None,
    ) -> RelationalSchema:
        """Lift a developmentally discovered tool schema into a RelationalSchema."""
        schema_id = (
            discovered_tool.get("schema_id", f"schema_tool_{uuid.uuid4().hex[:6]}")
            if discovered_tool
            else f"schema_tool_{uuid.uuid4().hex[:6]}"
        )

        roles = [
            SchemaRole(role_id="Agent", type_requirement="agent"),
            SchemaRole(
                role_id="Tool",
                type_requirement="tool",
                required_properties={"is_rigid": True},
            ),
            SchemaRole(role_id="Target", type_requirement="physical_entity"),
        ]
        relations = [
            SchemaRelation(source_role="Tool", edge_type="EXTENDS_REACH", target_role="Agent"),
            SchemaRelation(source_role="Tool", edge_type="MANIPULATES", target_role="Target"),
        ]
        constraints = [
            SchemaConstraint(
                role_id="Tool",
                property_key="is_rigid",
                expected_value=True,
                operator="eq",
                is_required=True,
            ),
        ]
        actions = [
            ActionTemplate(
                operator_name="USE_TOOL",
                role_parameters={
                    "agent_id": "Agent",
                    "tool_id": "Tool",
                    "target_id": "Target",
                },
            ),
        ]
        consequences = [
            ConsequenceTemplate(
                consequence_type="displacement",
                predicted_edge_type="DISPLACES",
                source_role="Target",
                target_role="Tool",
            ),
        ]

        return RelationalSchema(
            schema_id=schema_id,
            name="Developmental-Tool-Reach-Extension",
            roles=roles,
            relations=relations,
            constraints=constraints,
            action_templates=actions,
            predicted_consequences=consequences,
            status=SchemaLifecycleStatus.TRANSFERABLE,
            alpha_success=4.0,
            beta_failure=1.0,
        )

    @staticmethod
    def lift_causal_rule_schema(
        causal_rule: dict[str, Any],
    ) -> RelationalSchema:
        """Lift an interventional causal discovery rule into a RelationalSchema."""
        prop = causal_rule.get("precondition", {}).get("property", "mass_sensation")
        raw_op = causal_rule.get("precondition", {}).get("operator", "<")
        val = causal_rule.get("precondition", {}).get("value", 5.0)

        op_map = {
            "<": "lte",
            "<=": "lte",
            ">": "gte",
            ">=": "gte",
            "==": "eq",
            "=": "eq",
            "eq": "eq",
            "!=": "neq",
            "neq": "neq",
        }
        constraint_op = op_map.get(raw_op, "lte")

        roles = [
            SchemaRole(role_id="Agent", type_requirement="agent"),
            SchemaRole(role_id="Target", type_requirement="physical_entity"),
        ]
        relations = [
            SchemaRelation(source_role="Agent", edge_type="PUSHES", target_role="Target"),
        ]
        constraints = [
            SchemaConstraint(
                role_id="Target",
                property_key=prop,
                expected_value=val,
                operator=constraint_op,
                is_required=True,
            ),
        ]
        actions = [
            ActionTemplate(
                operator_name="PUSH",
                role_parameters={"agent_id": "Agent", "target_id": "Target"},
            ),
        ]
        consequences = [
            ConsequenceTemplate(
                consequence_type="displacement",
                predicted_edge_type="DISPLACES",
                source_role="Target",
                target_role="Agent",
            ),
        ]

        return RelationalSchema(
            schema_id=causal_rule.get("rule_id", f"schema_rule_{uuid.uuid4().hex[:6]}"),
            name=f"Developmental-Causal-Push-{prop}",
            roles=roles,
            relations=relations,
            constraints=constraints,
            action_templates=actions,
            predicted_consequences=consequences,
            status=SchemaLifecycleStatus.TRANSFERABLE,
            alpha_success=causal_rule.get("empirical_support_count", 3.0) + 1.0,
            beta_failure=1.0,
        )

    def transfer_to_target_domain(
        self,
        schema: RelationalSchema,
        target_graph: CognitiveGraph,
    ) -> dict[str, Any]:
        """Execute analogical matching and zero-shot action synthesis over target HCIR graph."""
        nodes = list(target_graph.all_nodes())
        role_mapping: dict[str, str] = {}
        bound_nodes: dict[str, HCIRNode] = {}
        used_node_ids: set[str] = set()
        violations: list[str] = []

        constraints_by_role: dict[str, list[SchemaConstraint]] = {}
        for constraint in schema.constraints:
            constraints_by_role.setdefault(constraint.role_id, []).append(constraint)

        for role in schema.roles:
            type_candidates = [
                node
                for node in nodes
                if node.id not in used_node_ids and _node_matches_type(node, role.type_requirement)
            ]
            if not type_candidates:
                violations.append(
                    f"No node found matching role {role.role_id}:{role.type_requirement}"
                )
                continue

            role_constraints = constraints_by_role.get(role.role_id, [])
            selected_node: HCIRNode | None = None
            candidate_violations: list[str] = []

            for cand in type_candidates:
                props = getattr(cand, "properties", {}) or {}
                errs = [
                    msg
                    for c in role_constraints
                    if (msg := _evaluate_constraint(c, props)) is not None
                ]
                if not errs:
                    selected_node = cand
                    break
                candidate_violations = errs

            if selected_node is None:
                violations.extend(candidate_violations)
                selected_node = type_candidates[0]

            role_mapping[role.role_id] = selected_node.id
            bound_nodes[role.role_id] = selected_node
            used_node_ids.add(selected_node.id)

        all_roles_bound = len(role_mapping) == len(schema.roles)
        if not all_roles_bound or violations:
            status = MappingStatus.REJECTED
            is_applicable = False
            is_rejected = True
            candidate_actions: list[tuple[str, dict[str, str]]] = []
        else:
            status = MappingStatus.APPLICABLE
            is_applicable = True
            is_rejected = False
            candidate_actions = [
                (
                    act.operator_name,
                    {
                        param_name: role_mapping[role_id]
                        for param_name, role_id in act.role_parameters.items()
                        if role_id in role_mapping
                    },
                )
                for act in schema.action_templates
            ]

        role_score = len(role_mapping) / max(1, len(schema.roles))
        edge_hits = 0
        if schema.relations:
            for rel in schema.relations:
                src_id = role_mapping.get(rel.source_role)
                tgt_id = role_mapping.get(rel.target_role)
                if src_id and tgt_id:
                    for edge in target_graph.edges_from(src_id):
                        if (
                            edge.target_id == tgt_id
                            and str(edge.edge_type).lower() == rel.edge_type.lower()
                        ):
                            edge_hits += 1
                            break
        rel_bonus = (edge_hits / max(1, len(schema.relations))) * 0.25
        alignment_score = 0.0 if is_rejected else min(1.0, role_score * 0.75 + rel_bonus)

        record = {
            "schema_id": schema.schema_id,
            "schema_name": schema.name,
            "mapping_status": status.value,
            "is_applicable": is_applicable,
            "is_rejected": is_rejected,
            "role_mapping": role_mapping,
            "candidate_actions": candidate_actions,
            "score": round(alignment_score, 4),
            "confidence": round(schema.confidence if is_applicable else 0.0, 4),
            "violations": violations,
        }
        self.transfer_history.append(record)
        return record
