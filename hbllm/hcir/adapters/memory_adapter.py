"""
Memory Adapter — bridges HBLLM memory subsystems ↔ HCIR graph views.

Maps the existing memory hierarchy into HCIR graph views:

    Episodic Memory  →  EpisodeNode    (memory_view)
    Semantic Memory  →  ConceptNode    (knowledge_view)
    Procedural       →  ProcedureNode  (knowledge_view)
    Value Memory     →  ValueNode      (memory_view)
    Knowledge Graph  →  FactNode/BeliefNode + edges (knowledge_view)

Direction:
    Memory → HCIR:  ``import_*()`` methods
    HCIR → Memory:  ``export_*()`` methods (for consolidation)
"""

from __future__ import annotations

import logging

from hbllm.hcir.graph import (
    BeliefNode,
    ConceptNode,
    EpisodeNode,
    HCIREdge,
    HCIREdgeType,
    NodeLifecycle,
    SkillNode,
    ValueNode,
)
from hbllm.hcir.types import Provenance, Scope, UncertaintyVector
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


class MemoryAdapter:
    """Bidirectional adapter between HBLLM memory systems and HCIR graph.

    Provides import methods for each memory type.  Imported entities
    become first-class graph nodes with full HCIR semantics (uncertainty,
    attention, provenance, scope).

    Usage::

        adapter = MemoryAdapter()
        adapter.import_episode(
            workspace=ws,
            session_id="session_42",
            summary="User asked about solar dehydrators",
            outcome="Provided design specifications",
            reward=0.8,
        )
    """

    def import_episode(
        self,
        workspace: HCIRWorkspaceState,
        session_id: str,
        summary: str,
        outcome: str = "",
        reward: float = 0.0,
        tenant_id: str = "default",
        author: str = "memory_adapter",
    ) -> EpisodeNode:
        """Import an episodic memory into the HCIR graph."""
        node = EpisodeNode(
            id=f"ep_{session_id}",
            summary=summary,
            outcome=outcome,
            reward=reward,
            lifecycle=NodeLifecycle.ACTIVE,
            provenance=Provenance(created_by=author),
            scope=Scope(tenant_id=tenant_id),
            tags=["episodic", "imported"],
        )
        workspace.upsert_node(node, author=author)
        logger.debug("Imported episode: %s", node.id)
        return node

    def import_concept(
        self,
        workspace: HCIRWorkspaceState,
        label: str,
        definition: str = "",
        domain: str = "",
        confidence: float = 0.8,
        tenant_id: str = "default",
        author: str = "memory_adapter",
    ) -> ConceptNode:
        """Import a semantic memory concept into the HCIR graph."""
        node_id = f"concept_{label.lower().replace(' ', '_')}"
        node = ConceptNode(
            id=node_id,
            label=label,
            definition=definition,
            domain=domain,
            lifecycle=NodeLifecycle.ACTIVE,
            uncertainty=UncertaintyVector(confidence=confidence),
            provenance=Provenance(created_by=author),
            scope=Scope(tenant_id=tenant_id),
            tags=["semantic", "imported", domain] if domain else ["semantic", "imported"],
        )
        workspace.upsert_node(node, author=author)
        logger.debug("Imported concept: %s", node.id)
        return node

    def import_skill(
        self,
        workspace: HCIRWorkspaceState,
        skill_name: str,
        description: str = "",
        success_rate: float = 0.0,
        invocation_count: int = 0,
        tenant_id: str = "default",
        author: str = "memory_adapter",
    ) -> SkillNode:
        """Import a procedural skill into the HCIR graph."""
        node_id = f"skill_{skill_name.lower().replace(' ', '_')}"
        node = SkillNode(
            id=node_id,
            skill_name=skill_name,
            description=description,
            success_rate=success_rate,
            invocation_count=invocation_count,
            lifecycle=NodeLifecycle.ACTIVE,
            provenance=Provenance(created_by=author),
            scope=Scope(tenant_id=tenant_id),
            tags=["procedural", "imported"],
        )
        workspace.upsert_node(node, author=author)
        logger.debug("Imported skill: %s", node.id)
        return node

    def import_value(
        self,
        workspace: HCIRWorkspaceState,
        dimension: str,
        weight: float = 0.5,
        tenant_id: str = "default",
        author: str = "memory_adapter",
    ) -> ValueNode:
        """Import a value/alignment preference into the HCIR graph."""
        node_id = f"value_{dimension.lower().replace(' ', '_')}"
        node = ValueNode(
            id=node_id,
            dimension=dimension,
            weight=weight,
            lifecycle=NodeLifecycle.ACTIVE,
            provenance=Provenance(created_by=author),
            scope=Scope(tenant_id=tenant_id),
            tags=["value", "imported"],
        )
        workspace.upsert_node(node, author=author)
        logger.debug("Imported value: %s", node.id)
        return node

    def import_knowledge_triple(
        self,
        workspace: HCIRWorkspaceState,
        subject: str,
        relation: str,
        obj: str,
        confidence: float = 0.8,
        tenant_id: str = "default",
        author: str = "memory_adapter",
    ) -> tuple[BeliefNode, BeliefNode, HCIREdge]:
        """Import a knowledge graph triple as HCIR nodes + edge.

        Example: (\"Python\", \"is_a\", \"programming_language\")
        becomes two BeliefNodes connected by a typed edge.
        """
        subj_id = f"belief_{subject.lower().replace(' ', '_')}"
        obj_id = f"belief_{obj.lower().replace(' ', '_')}"

        subj_node = BeliefNode(
            id=subj_id,
            claim=subject,
            lifecycle=NodeLifecycle.ACTIVE,
            uncertainty=UncertaintyVector(confidence=confidence),
            provenance=Provenance(created_by=author),
            scope=Scope(tenant_id=tenant_id),
            tags=["knowledge_graph", "imported"],
        )
        workspace.upsert_node(subj_node, author=author)

        obj_node = BeliefNode(
            id=obj_id,
            claim=obj,
            lifecycle=NodeLifecycle.ACTIVE,
            uncertainty=UncertaintyVector(confidence=confidence),
            provenance=Provenance(created_by=author),
            scope=Scope(tenant_id=tenant_id),
            tags=["knowledge_graph", "imported"],
        )
        workspace.upsert_node(obj_node, author=author)

        # Map relation string to edge type
        edge_type_map = {
            "is_a": HCIREdgeType.PART_OF,
            "contains": HCIREdgeType.PART_OF,
            "uses": HCIREdgeType.DEPENDS_ON,
            "requires": HCIREdgeType.REQUIRES,
            "causes": HCIREdgeType.CAUSES,
            "supports": HCIREdgeType.SUPPORTS,
            "contradicts": HCIREdgeType.CONTRADICTS,
            "similar_to": HCIREdgeType.SIMILAR_TO,
            "derived_from": HCIREdgeType.DERIVED_FROM,
        }
        edge_type = edge_type_map.get(relation, HCIREdgeType.SUPPORTS)

        edge_id = f"kg_{subj_id}_{relation}_{obj_id}"
        # Remove old edge if exists (upsert semantics)
        workspace.remove_edge(edge_id)
        edge = HCIREdge(
            id=edge_id,
            edge_type=edge_type,
            sources=[subj_id],
            targets=[obj_id],
            properties={"relation": relation},
            provenance=Provenance(created_by=author),
        )
        workspace.add_edge(edge, author=author)

        logger.debug("Imported triple: %s -[%s]-> %s", subject, relation, obj)
        return subj_node, obj_node, edge
