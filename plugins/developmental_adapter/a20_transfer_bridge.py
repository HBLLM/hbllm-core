"""A20 Relational Transfer Bridge for A23.5-E4.

Lifts developmentally induced schemas (containment, tool use, causal push)
acquired by the blank-brain substrate into formal A20 RelationalSchema models,
and executes zero-shot analogical transfer via StructureMappingEngine and AnalogicalTransferEngine.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from hbllm.brain.transfer.engine import AnalogicalTransferEngine
from hbllm.brain.transfer.mapper import MappingStatus, StructureMappingEngine
from hbllm.brain.transfer.schema import (
    ActionTemplate,
    ConsequenceTemplate,
    RelationalSchema,
    SchemaConstraint,
    SchemaLifecycleStatus,
    SchemaRelation,
    SchemaRole,
)
from hbllm.hcir.graph import CognitiveGraph

logger = logging.getLogger(__name__)


class A20RelationalTransferBridge:
    """Bridges developmental cognitive discoveries (A23) to A20 Relational Generalization."""

    def __init__(
        self,
        mapper: StructureMappingEngine | None = None,
        transfer_engine: AnalogicalTransferEngine | None = None,
    ) -> None:
        self.mapper = mapper or StructureMappingEngine()
        self.transfer_engine = transfer_engine or AnalogicalTransferEngine(mapper=self.mapper)

    @staticmethod
    def lift_containment_schema(
        discovered_schema: dict[str, Any] | None = None,
    ) -> RelationalSchema:
        """Lift a developmentally discovered containment schema into an A20 RelationalSchema."""
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
        """Lift a developmentally discovered tool schema into an A20 RelationalSchema."""
        schema_id = (
            discovered_tool.get("schema_id", f"schema_tool_{uuid.uuid4().hex[:6]}")
            if discovered_tool
            else f"schema_tool_{uuid.uuid4().hex[:6]}"
        )

        roles = [
            SchemaRole(
                role_id="Agent",
                type_requirement="agent",
            ),
            SchemaRole(
                role_id="Tool",
                type_requirement="tool",
                required_properties={"is_rigid": True},
            ),
            SchemaRole(
                role_id="Target",
                type_requirement="physical_entity",
            ),
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
        """Lift an interventional causal discovery rule into an A20 RelationalSchema."""
        prop = causal_rule.get("precondition", {}).get("property", "mass_sensation")
        raw_op = causal_rule.get("precondition", {}).get("operator", "<")
        val = causal_rule.get("precondition", {}).get("value", 5.0)

        # Map developmental causal operator to A20 SchemaConstraint operator
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
        """Execute analogical transfer and zero-shot action synthesis onto target graph."""
        transfer, cond_hyp, mapping = self.transfer_engine.transfer_schema_to_domain(
            schema=schema,
            target_graph=target_graph,
        )

        is_applicable = mapping.status == MappingStatus.APPLICABLE
        is_rejected = mapping.status == MappingStatus.REJECTED

        candidate_actions = transfer.candidate_actions if transfer else []
        role_map = transfer.role_mapping if transfer else {}

        return {
            "schema_id": schema.schema_id,
            "schema_name": schema.name,
            "mapping_status": mapping.status.value,
            "is_applicable": is_applicable,
            "is_rejected": is_rejected,
            "role_mapping": role_map,
            "candidate_actions": candidate_actions,
            "score": mapping.relational_alignment_score,
            "confidence": transfer.confidence if transfer else 0.0,
            "violations": mapping.violated_constraints,
        }
