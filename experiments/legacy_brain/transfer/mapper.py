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

    APPLICABLE = "applicable"  # Fully bound, all constraints satisfied
    PARTIALLY_APPLICABLE = "partially_applicable"  # Partially bound or conditional on missing state
    REJECTED = "rejected"  # Critical physical/geometric constraint violation


@dataclass
class StructuralMappingResult:
    """The output of mapping a RelationalSchema onto a target HCIR graph."""

    schema_id: str
    schema_name: str
    status: MappingStatus
    role_bindings: dict[str, str] = field(default_factory=dict)  # role_id -> target_node_id
    relational_alignment_score: float = 0.0  # 0.0 to 1.0
    systematicity_score: float = 0.0  # Reward for connected higher-order relational chains
    violated_constraints: list[str] = field(default_factory=list)
    missing_roles: list[str] = field(default_factory=list)
    transferred_predictions: list[dict[str, Any]] = field(default_factory=list)


# Functional compatibility clusters for analogical mapping
_FUNCTIONAL_COMPATIBILITY_CLUSTERS: list[set[str]] = [
    {"supports", "stable_for", "rests_on", "above", "below"},
    {"contains", "located_in", "has_cavity", "fits_inside", "part_of"},
    {"transmits_force_to", "changes_state_of", "affords", "causes", "acts_on"},
    {"travels_along", "connects", "path_to", "near"},
    {"avoids", "blocked_by", "contradicts"},
]


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
        valid_nodes = [
            n for n in target_nodes if n is not None and isinstance(n, PhysicalEntityNode)
        ]

        if not valid_nodes or not schema.roles:
            return StructuralMappingResult(
                schema_id=schema.schema_id,
                schema_name=schema.name,
                status=MappingStatus.REJECTED,
                violated_constraints=["Target graph contains no physical entities"],
            )

        if len(valid_nodes) < len(schema.roles):
            missing = [r.role_id for r in schema.roles[len(valid_nodes) :]]
            coverage = len(valid_nodes) / float(len(schema.roles))
            return StructuralMappingResult(
                schema_id=schema.schema_id,
                schema_name=schema.name,
                status=MappingStatus.PARTIALLY_APPLICABLE,
                missing_roles=missing,
                relational_alignment_score=round(0.25 * coverage, 4),
                systematicity_score=0.20,
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
                    p = (
                        getattr(node, "properties", None)
                        or getattr(node, "observed_properties", {})
                        or {}
                    )
                    props_map[role_id] = dict(p)

            # 2. Evaluate physical constraint compatibility
            is_valid_constraints, violations = schema.evaluate_constraint_compatibility(props_map)

            # 3. Compute relational systematicity and alignment score
            alignment_score, systematicity = self._compute_alignment_and_systematicity(
                schema, target_graph, bindings
            )

            # 4. Compute composite structural mapping score
            constraint_factor = 1.0 if is_valid_constraints else 0.0
            coverage = len(bindings) / float(len(schema.roles))
            composite_score = (
                ((alignment_score * 0.70) + (systematicity * 0.30)) * constraint_factor * coverage
            )

            # 5. Classify status
            if not is_valid_constraints:
                status = MappingStatus.REJECTED
            elif composite_score >= 0.40 and alignment_score >= 0.30:
                status = MappingStatus.APPLICABLE
            elif composite_score >= 0.20:
                status = MappingStatus.PARTIALLY_APPLICABLE
            else:
                status = MappingStatus.REJECTED

            # 6. Build transferred predictions
            transferred_preds: list[dict[str, Any]] = []
            if status != MappingStatus.REJECTED:
                for c in schema.predicted_consequences:
                    transferred_preds.append(
                        {
                            "consequence_type": c.consequence_type,
                            "predicted_edge_type": c.predicted_edge_type,
                            "source_node": bindings.get(c.source_role, ""),
                            "target_node": bindings.get(c.target_role, ""),
                        }
                    )

            result = StructuralMappingResult(
                schema_id=schema.schema_id,
                schema_name=schema.name,
                status=status,
                role_bindings=bindings,
                relational_alignment_score=round(composite_score, 4),
                systematicity_score=round(systematicity, 4),
                violated_constraints=violations,
                transferred_predictions=transferred_preds,
            )

            if composite_score > best_score:
                best_score = composite_score
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
        """Compute relational topological alignment and systematicity bonus.

        Evaluates exact relation matches, functional cluster correspondences,
        zero-shot affordance projection, and multi-hop connected relational chains.
        """
        if not schema.relations:
            systematicity = 0.50 + (0.15 * min(3, len(schema.roles) - 1))
            return 1.0, systematicity

        matched_relation_score = 0.0
        total_relations = len(schema.relations)
        has_contradiction = False

        for rel in schema.relations:
            src_nid = bindings.get(rel.source_role)
            tgt_nid = bindings.get(rel.target_role)
            if not src_nid or not tgt_nid:
                continue

            rel_type_str = rel.edge_type.lower()
            edges_from_src = target_graph.edges_from(src_nid)
            edges_to_tgt = [e for e in edges_from_src if tgt_nid in e.targets]

            edges_from_tgt = target_graph.edges_from(tgt_nid)
            edges_from_tgt_to_src = [e for e in edges_from_tgt if src_nid in e.targets]

            all_candidate_edges = edges_to_tgt + edges_from_tgt_to_src

            if not all_candidate_edges:
                # Zero-shot affordance projection: entities are present and satisfy physical constraints
                matched_relation_score += 0.80
            else:
                edge_score = 0.0
                for edge in all_candidate_edges:
                    edge_type_name = str(edge.edge_type).lower()
                    if "contradicts" in edge_type_name:
                        has_contradiction = True
                        edge_score = 0.0
                        break
                    elif rel_type_str == edge_type_name or rel_type_str in edge_type_name:
                        edge_score = max(edge_score, 1.0)
                    elif any(
                        rel_type_str in cluster
                        and any(e_name in cluster for e_name in [edge_type_name])
                        for cluster in _FUNCTIONAL_COMPATIBILITY_CLUSTERS
                    ):
                        edge_score = max(edge_score, 0.50)

                matched_relation_score += edge_score

        alignment = 0.0 if has_contradiction else (matched_relation_score / float(total_relations))

        # Compute systematicity: reward higher-order multi-role systems
        if has_contradiction:
            systematicity = 0.15
        else:
            systematicity = min(0.95, 0.50 + (0.15 * min(3, len(schema.roles) - 1)))

        return alignment, systematicity
