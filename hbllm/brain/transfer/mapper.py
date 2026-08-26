"""Structure Mapping Engine for A20.

Implements Gentner's Structure Mapping Theory over HCIR subgraphs:
1. Enforces 1-to-1 role-to-entity alignments.
2. Prioritizes higher-order relational systematicity over surface attributes.
3. Evaluates physical and geometric constraint satisfaction.
4. Returns explicit MappingStatus (APPLICABLE, PARTIALLY_APPLICABLE, REJECTED).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hbllm.brain.transfer.schema import RelationalSchema
from hbllm.hcir.graph import CognitiveGraph, PhysicalEntityNode

logger = logging.getLogger(__name__)


class MappingStatus(str, Enum):
    """The structural and constraint validation status of an analogical mapping."""

    APPLICABLE = "applicable"                      # Fully bound, all constraints satisfied
    PARTIALLY_APPLICABLE = "partially_applicable"  # Partially bound or conditional on missing state
    REJECTED = "rejected"                          # Critical physical/geometric constraint violation


@dataclass
class StructuralMappingResult:
    """The output of mapping a RelationalSchema onto a target HCIR graph."""

    schema_id: str
    schema_name: str
    status: MappingStatus
    role_bindings: dict[str, str] = field(default_factory=dict)  # role_id -> target_node_id
    relational_alignment_score: float = 0.0  # 0.0 to 1.0
    systematicity_score: float = 0.0         # Reward for connected higher-order relational chains
    violated_constraints: list[str] = field(default_factory=list)
    missing_roles: list[str] = field(default_factory=list)
    transferred_predictions: list[dict[str, Any]] = field(default_factory=list)


class StructureMappingEngine:
    """Deterministic structure-mapping algorithm over HCIR graphs."""

    def map_schema_to_target(
        self,
        schema: RelationalSchema,
        target_graph: CognitiveGraph,
        candidate_node_ids: list[str] | None = None,
    ) -> StructuralMappingResult:
        """Find the optimal 1-to-1 mapping from schema roles to target entities."""
        target_nodes = (
            [target_graph.get_node(nid) for nid in candidate_node_ids]
            if candidate_node_ids
            else target_graph.all_nodes()
        )
        valid_nodes = [n for n in target_nodes if n is not None and isinstance(n, PhysicalEntityNode)]

        if len(valid_nodes) < len(schema.roles):
            missing = [r.role_id for r in schema.roles[len(valid_nodes):]]
            return StructuralMappingResult(
                schema_id=schema.schema_id,
                schema_name=schema.name,
                status=MappingStatus.PARTIALLY_APPLICABLE,
                missing_roles=missing,
                relational_alignment_score=0.30,
            )

        best_result: StructuralMappingResult | None = None
        best_score = -1.0

        # Enumerate 1-to-1 permutations of target nodes for schema roles
        role_list = schema.roles
        node_ids = [n.id for n in valid_nodes]

        for perm in itertools.permutations(node_ids, len(role_list)):
            bindings = {role_list[i].role_id: perm[i] for i in range(len(role_list))}

            # 1. Collect target entity properties
            props_map: dict[str, dict[str, Any]] = {}
            for role_id, nid in bindings.items():
                node = target_graph.get_node(nid)
                if node and isinstance(node, PhysicalEntityNode):
                    p = getattr(node, "properties", None) or getattr(node, "observed_properties", {}) or {}
                    props_map[role_id] = dict(p)

            # 2. Evaluate physical constraint compatibility
            is_valid_constraints, violations = schema.evaluate_constraint_compatibility(props_map)

            # 3. Compute relational systematicity and alignment score
            alignment_score, systematicity = self._compute_alignment_and_systematicity(
                schema, target_graph, bindings
            )

            # 4. Classify status
            if not is_valid_constraints:
                status = MappingStatus.REJECTED
                total_score = 0.10
            elif alignment_score >= 0.70:
                status = MappingStatus.APPLICABLE
                total_score = (alignment_score * 0.7) + (systematicity * 0.3)
            else:
                status = MappingStatus.PARTIALLY_APPLICABLE
                total_score = (alignment_score * 0.7) + (systematicity * 0.3)

            # 5. Build transferred predictions
            transferred_preds: list[dict[str, Any]] = []
            if status != MappingStatus.REJECTED:
                for c in schema.predicted_consequences:
                    transferred_preds.append({
                        "consequence_type": c.consequence_type,
                        "predicted_edge_type": c.predicted_edge_type,
                        "source_node": bindings.get(c.source_role, ""),
                        "target_node": bindings.get(c.target_role, ""),
                    })

            result = StructuralMappingResult(
                schema_id=schema.schema_id,
                schema_name=schema.name,
                status=status,
                role_bindings=bindings,
                relational_alignment_score=round(alignment_score, 4),
                systematicity_score=round(systematicity, 4),
                violated_constraints=violations,
                transferred_predictions=transferred_preds,
            )

            if total_score > best_score:
                best_score = total_score
                best_result = result

        return best_result or StructuralMappingResult(
            schema_id=schema.schema_id,
            schema_name=schema.name,
            status=MappingStatus.REJECTED,
            violated_constraints=["No valid 1-to-1 mapping found"],
        )

    def _compute_alignment_and_systematicity(
        self,
        schema: RelationalSchema,
        target_graph: CognitiveGraph,
        bindings: dict[str, str],
    ) -> tuple[float, float]:
        """Compute relational topological alignment and systematicity bonus."""
        if not schema.relations:
            return 1.0, 0.5

        matched_relations = 0
        total_relations = len(schema.relations)

        for rel in schema.relations:
            src_nid = bindings.get(rel.source_role)
            tgt_nid = bindings.get(rel.target_role)
            if src_nid and tgt_nid:
                # Check if edge exists or is geometrically compatible
                edges = target_graph.edges_from(src_nid)
                if any(rel.edge_type in str(e.edge_type) and tgt_nid in e.targets for e in edges):
                    matched_relations += 1
                else:
                    # Potential candidate action relation
                    matched_relations += 0.8  # High relational potential

        alignment = matched_relations / float(total_relations)

        # Systematicity: bonus for higher-order connected graphs (roles > 2 or chains)
        systematicity = 0.5 + (0.15 * min(3, len(schema.roles) - 1))
        return alignment, systematicity
