"""Analogical Transfer Engine and Zero-Shot Action Synthesis for A20.

Translates structural mappings into hypothetical analogical projections and executable action plans.
Ensures transferred relations carry epistemic provenance (`source = ANALOGICAL_TRANSFER`)
and updates schema Bayesian reliability based on physical execution outcomes.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.transfer.extractor import RelationalSchemaExtractor
from hbllm.brain.transfer.mapper import (
    MappingStatus,
    StructuralMappingResult,
    StructureMappingEngine,
)
from hbllm.brain.transfer.schema import RelationalSchema
from hbllm.hcir.graph import CognitiveGraph

logger = logging.getLogger(__name__)


@dataclass
class AnalogicalTransfer:
    """A projected analogical transfer from a source schema to a target domain."""

    transfer_id: str = field(default_factory=lambda: f"trans_{uuid.uuid4().hex[:8]}")
    source_schema_id: str = ""
    source_schema_name: str = ""
    role_mapping: dict[str, str] = field(default_factory=dict)  # role_id -> target_node_id
    projected_relations: list[tuple[str, str, str]] = field(default_factory=list)  # (src_node, edge_type, tgt_node)
    projected_predictions: list[dict[str, Any]] = field(default_factory=list)
    candidate_actions: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    provenance_source: str = "ANALOGICAL_TRANSFER"
    confidence: float = 0.75


@dataclass
class ConditionalAnalogicalHypothesis:
    """An incomplete analogical mapping detailing missing conditions for A19 epistemic probing."""

    hypothesis_id: str = field(default_factory=lambda: f"cond_ana_{uuid.uuid4().hex[:8]}")
    schema_id: str = ""
    partial_role_bindings: dict[str, str] = field(default_factory=dict)
    missing_roles: list[str] = field(default_factory=list)
    missing_conditions: list[str] = field(default_factory=list)
    confidence: float = 0.40


class AnalogicalTransferEngine:
    """Coordinates schema retrieval, structure mapping, and zero-shot plan synthesis."""

    def __init__(
        self,
        extractor: RelationalSchemaExtractor | None = None,
        mapper: StructureMappingEngine | None = None,
    ) -> None:
        self.extractor = extractor or RelationalSchemaExtractor()
        self.mapper = mapper or StructureMappingEngine()

    def transfer_schema_to_domain(
        self,
        schema: RelationalSchema,
        target_graph: CognitiveGraph,
        candidate_node_ids: list[str] | None = None,
    ) -> tuple[AnalogicalTransfer | None, ConditionalAnalogicalHypothesis | None, StructuralMappingResult]:
        """Perform structure mapping and project analogical inferences onto target domain."""
        mapping_res = self.mapper.map_schema_to_target(
            schema=schema,
            target_graph=target_graph,
            candidate_node_ids=candidate_node_ids,
        )

        if mapping_res.status == MappingStatus.REJECTED:
            return None, None, mapping_res

        if mapping_res.status == MappingStatus.PARTIALLY_APPLICABLE:
            cond_hyp = ConditionalAnalogicalHypothesis(
                schema_id=schema.schema_id,
                partial_role_bindings=mapping_res.role_bindings,
                missing_roles=mapping_res.missing_roles,
                missing_conditions=mapping_res.violated_constraints,
                confidence=round(schema.confidence * 0.5, 4),
            )
            return None, cond_hyp, mapping_res

        # Status is APPLICABLE: synthesize candidate actions and projected relations
        candidate_actions: list[tuple[str, dict[str, Any]]] = []
        for action_tmpl in schema.action_templates:
            params: dict[str, Any] = {}
            for param_key, role_id in action_tmpl.role_parameters.items():
                bound_node = mapping_res.role_bindings.get(role_id)
                if bound_node:
                    params[param_key] = bound_node
            candidate_actions.append((action_tmpl.operator_name, params))

        # Build projected relations with explicit analogical provenance
        projected_rels: list[tuple[str, str, str]] = []
        for rel in schema.relations:
            src = mapping_res.role_bindings.get(rel.source_role, "")
            tgt = mapping_res.role_bindings.get(rel.target_role, "")
            if src and tgt:
                projected_rels.append((src, rel.edge_type, tgt))

        transfer = AnalogicalTransfer(
            source_schema_id=schema.schema_id,
            source_schema_name=schema.name,
            role_mapping=mapping_res.role_bindings,
            projected_relations=projected_rels,
            projected_predictions=mapping_res.transferred_predictions,
            candidate_actions=candidate_actions,
            provenance_source="ANALOGICAL_TRANSFER",
            confidence=schema.confidence,
        )

        return transfer, None, mapping_res

    def match_best_schema(
        self,
        target_graph: CognitiveGraph,
        candidate_node_ids: list[str] | None = None,
    ) -> tuple[AnalogicalTransfer | None, StructuralMappingResult | None]:
        """Find the highest-scoring applicable schema in the schema library."""
        best_transfer: AnalogicalTransfer | None = None
        best_result: StructuralMappingResult | None = None
        best_score = -1.0

        for schema in self.extractor.all_schemas():
            if not schema.is_transferable:
                continue

            transfer, _, mapping_res = self.transfer_schema_to_domain(
                schema=schema,
                target_graph=target_graph,
                candidate_node_ids=candidate_node_ids,
            )

            if transfer and mapping_res.status == MappingStatus.APPLICABLE:
                score = mapping_res.relational_alignment_score + mapping_res.systematicity_score
                if score > best_score:
                    best_score = score
                    best_transfer = transfer
                    best_result = mapping_res

        return best_transfer, best_result

    def record_transfer_outcome(
        self,
        schema_id: str,
        is_success: bool,
        failed_constraint: str | None = None,
    ) -> None:
        """Update schema reliability in library based on physical execution feedback."""
        schema = self.extractor.get_schema(schema_id)
        if schema:
            schema.record_outcome(is_success=is_success, failed_constraint=failed_constraint)
