"""
HCIR Memory Backend — HCIR-native memory implementing all subsystem operations.

Makes HCIR the canonical storage backend for all memory types via a
5-phase migration strategy that minimizes operational risk:

    Phase A: Read-Through     → HCIR populated from legacy stores
    Phase B: Dual-Write       → Writes go to both; compare for divergence
    Phase C: Shadow-Read      → HCIR answers in parallel; legacy is official
    Phase D: HCIR Primary     → HCIR is authoritative; legacy for rollback
    Phase E: Legacy Removed   → Retire old stores

Replaces:
    EpisodicMemory (SQLite)    → EpisodeNode queries on persistent workspace
    SemanticMemory (in-memory) → ConceptNode queries
    ProceduralMemory (SQLite)  → SkillNode/ProcedureNode queries
    ValueMemory (SQLite)       → ValueNode queries
    KnowledgeGraph (JSON)      → BeliefNode + HCIREdge queries

Usage::

    backend = HCIRMemoryBackend(tiered_workspace)
    entry_id = await backend.store_episode(summary="User asked about weather", ...)
    results = await backend.recall_episodes(query="weather", limit=5)
"""

from __future__ import annotations

import logging
import uuid
from enum import StrEnum
from typing import Any

from hbllm.hcir.graph import (
    BeliefNode,
    ConceptNode,
    EpisodeNode,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
    SkillNode,
    ValueNode,
)
from hbllm.hcir.query import GraphQuery
from hbllm.hcir.types import Provenance, Scope
from hbllm.hcir.workspace_tiers import TieredWorkspace, WorkspaceTier

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Migration Phase
# ═══════════════════════════════════════════════════════════════════════════


class MigrationPhase(StrEnum):
    """5-phase migration strategy for memory backends."""

    READ_THROUGH = "read_through"  # Phase A: HCIR populated from legacy
    DUAL_WRITE = "dual_write"  # Phase B: Write both, compare
    SHADOW_READ = "shadow_read"  # Phase C: HCIR reads in parallel
    HCIR_PRIMARY = "hcir_primary"  # Phase D: HCIR authoritative
    LEGACY_REMOVED = "legacy_removed"  # Phase E: Old stores retired


# ═══════════════════════════════════════════════════════════════════════════
# HCIR Memory Backend
# ═══════════════════════════════════════════════════════════════════════════


class HCIRMemoryBackend:
    """HCIR-native memory backend implementing all subsystem operations.

    This is the HCIR side of the memory migration.  It stores and
    retrieves memory entries as typed graph nodes in the persistent
    workspace tier.

    The ``migration_phase`` controls how this backend interacts with
    legacy stores:

        - ``READ_THROUGH``: Legacy is authoritative.  HCIR is populated
          from legacy results for warming.
        - ``DUAL_WRITE``: Writes go to both HCIR and legacy.  Divergence
          is logged as warnings.
        - ``SHADOW_READ``: HCIR reads are executed in parallel with legacy.
          Legacy results are returned, but HCIR results are compared.
        - ``HCIR_PRIMARY``: HCIR results are authoritative.  Legacy is
          available for rollback only.
        - ``LEGACY_REMOVED``: Legacy stores are fully retired.

    Usage::

        backend = HCIRMemoryBackend(workspace)
        entry_id = await backend.store_episode(
            summary="User asked about weather",
            outcome="Provided forecast",
            tenant_id="t1",
        )
    """

    def __init__(
        self,
        tiered_workspace: TieredWorkspace,
        migration_phase: MigrationPhase = MigrationPhase.READ_THROUGH,
    ) -> None:
        self._workspace = tiered_workspace
        self._migration_phase = migration_phase
        self._divergence_count: int = 0

    @property
    def migration_phase(self) -> MigrationPhase:
        return self._migration_phase

    @migration_phase.setter
    def migration_phase(self, phase: MigrationPhase) -> None:
        logger.info("Memory migration phase changed: %s → %s", self._migration_phase, phase)
        self._migration_phase = phase

    @property
    def divergence_count(self) -> int:
        """Number of divergences detected during dual-write/shadow-read."""
        return self._divergence_count

    @property
    def is_hcir_authoritative(self) -> bool:
        """True if HCIR is the primary source of truth."""
        return self._migration_phase in (
            MigrationPhase.HCIR_PRIMARY,
            MigrationPhase.LEGACY_REMOVED,
        )

    # ── Episodic Memory ──────────────────────────────────────────────

    async def store_episode(
        self,
        summary: str,
        outcome: str = "",
        reward: float = 0.0,
        tenant_id: str = "default",
        session_id: str = "",
        goal_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Store an episode in the persistent workspace.

        Maps to: ``EpisodeNode`` in persistent tier.
        """
        node_id = f"ep_{uuid.uuid4().hex[:10]}"
        node = EpisodeNode(
            id=node_id,
            summary=summary,
            outcome=outcome,
            reward=reward,
            scope=Scope(tenant_id=tenant_id),
            provenance=Provenance(
                created_by="hcir_memory_backend",
                session_id=session_id,
                goal_id=goal_id,
                reason="Episodic memory storage",
            ),
            tags=["episodic", "hcir_native"],
        )
        self._workspace.persistent.upsert_node(node, author="hcir_memory_backend")
        logger.debug("Stored episode %s: %s", node_id, summary[:50])
        return node_id

    async def recall_episodes(
        self,
        query: str = "",
        tenant_id: str = "default",
        limit: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Recall episodes from the persistent workspace.

        Queries ``EpisodeNode`` entries in the persistent tier,
        optionally filtered by text content.
        """
        graph_query = GraphQuery(
            node_type=HCIRNodeType.EPISODE,
            scope_tenant=tenant_id,
            text_contains=query if query else None,
            limit=limit,
        )
        result = self._workspace.persistent.query(graph_query)
        return [
            {
                "id": node.id,
                "summary": node.summary if isinstance(node, EpisodeNode) else "",
                "outcome": node.outcome if isinstance(node, EpisodeNode) else "",
                "reward": node.reward if isinstance(node, EpisodeNode) else 0.0,
            }
            for node in result.nodes
        ]

    # ── Semantic Memory ──────────────────────────────────────────────

    async def store_concept(
        self,
        label: str,
        definition: str = "",
        domain: str = "",
        tenant_id: str = "default",
        **kwargs: Any,
    ) -> str:
        """Store a concept in the persistent workspace.

        Maps to: ``ConceptNode`` in persistent tier.
        """
        node_id = f"sem_{uuid.uuid4().hex[:10]}"
        node = ConceptNode(
            id=node_id,
            label=label,
            definition=definition,
            domain=domain,
            scope=Scope(tenant_id=tenant_id),
            provenance=Provenance(
                created_by="hcir_memory_backend",
                reason="Semantic memory storage",
            ),
            tags=["semantic", "hcir_native"],
        )
        self._workspace.persistent.upsert_node(node, author="hcir_memory_backend")
        return node_id

    async def recall_concepts(
        self,
        query: str = "",
        tenant_id: str = "default",
        limit: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Recall concepts from the persistent workspace."""
        graph_query = GraphQuery(
            node_type=HCIRNodeType.CONCEPT,
            scope_tenant=tenant_id,
            text_contains=query if query else None,
            limit=limit,
        )
        result = self._workspace.persistent.query(graph_query)
        return [
            {
                "id": node.id,
                "label": node.label if isinstance(node, ConceptNode) else "",
                "definition": node.definition if isinstance(node, ConceptNode) else "",
                "domain": node.domain if isinstance(node, ConceptNode) else "",
            }
            for node in result.nodes
        ]

    # ── Procedural Memory ────────────────────────────────────────────

    async def store_skill(
        self,
        skill_name: str,
        description: str = "",
        success_rate: float = 0.5,
        tenant_id: str = "default",
        **kwargs: Any,
    ) -> str:
        """Store a skill in the persistent workspace.

        Maps to: ``SkillNode`` in persistent tier.
        """
        node_id = f"skill_{uuid.uuid4().hex[:10]}"
        node = SkillNode(
            id=node_id,
            skill_name=skill_name,
            description=description,
            success_rate=success_rate,
            scope=Scope(tenant_id=tenant_id),
            provenance=Provenance(
                created_by="hcir_memory_backend",
                reason="Procedural memory storage",
            ),
            tags=["procedural", "hcir_native"],
        )
        self._workspace.persistent.upsert_node(node, author="hcir_memory_backend")
        return node_id

    async def recall_skills(
        self,
        query: str = "",
        tenant_id: str = "default",
        limit: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Recall skills from the persistent workspace."""
        graph_query = GraphQuery(
            node_type=HCIRNodeType.SKILL,
            scope_tenant=tenant_id,
            text_contains=query if query else None,
            limit=limit,
        )
        result = self._workspace.persistent.query(graph_query)
        return [
            {
                "id": node.id,
                "skill_name": node.skill_name if isinstance(node, SkillNode) else "",
                "description": node.description if isinstance(node, SkillNode) else "",
                "success_rate": node.success_rate if isinstance(node, SkillNode) else 0.0,
            }
            for node in result.nodes
        ]

    # ── Value Memory ─────────────────────────────────────────────────

    async def store_value(
        self,
        dimension: str,
        weight: float = 0.5,
        tenant_id: str = "default",
        **kwargs: Any,
    ) -> str:
        """Store a value alignment marker in the persistent workspace.

        Maps to: ``ValueNode`` in persistent tier.
        """
        node_id = f"val_{uuid.uuid4().hex[:10]}"
        node = ValueNode(
            id=node_id,
            dimension=dimension,
            weight=weight,
            scope=Scope(tenant_id=tenant_id),
            provenance=Provenance(
                created_by="hcir_memory_backend",
                reason="Value memory storage",
            ),
            tags=["value", "hcir_native"],
        )
        self._workspace.persistent.upsert_node(node, author="hcir_memory_backend")
        return node_id

    # ── Knowledge Graph (Beliefs + Edges) ────────────────────────────

    async def store_belief(
        self,
        claim: str,
        belief_type: str = "factual",
        evidence_sources: list[str] | None = None,
        tenant_id: str = "default",
        **kwargs: Any,
    ) -> str:
        """Store a belief with evidence in the persistent workspace.

        Maps to: ``BeliefNode`` in persistent tier.
        """
        node_id = f"belief_{uuid.uuid4().hex[:10]}"
        node = BeliefNode(
            id=node_id,
            claim=claim,
            belief_type=belief_type,
            evidence_sources=evidence_sources or [],
            scope=Scope(tenant_id=tenant_id),
            provenance=Provenance(
                created_by="hcir_memory_backend",
                reason="Knowledge graph belief storage",
            ),
            tags=["knowledge_graph", "belief", "hcir_native"],
        )
        self._workspace.persistent.upsert_node(node, author="hcir_memory_backend")
        return node_id

    async def link_beliefs(
        self,
        source_id: str,
        target_id: str,
        edge_type: HCIREdgeType = HCIREdgeType.SUPPORTS,
        tenant_id: str = "default",
    ) -> str:
        """Create a typed edge between two belief nodes."""
        edge_id = f"e_{uuid.uuid4().hex[:8]}"
        edge = HCIREdge(
            id=edge_id,
            sources=[source_id],
            targets=[target_id],
            edge_type=edge_type,
        )
        self._workspace.persistent.add_edge(edge, author="hcir_memory_backend")
        return edge_id

    # ── Cross-Memory Queries ─────────────────────────────────────────

    async def search_across_memory_types(
        self,
        query: str,
        tenant_id: str = "default",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search across all memory types using a single graph query.

        This is the primary advantage of HCIR-backed memory: one graph,
        one query engine, cross-memory-type results.
        """
        # Query across all tiers for text matching
        graph_query = GraphQuery(
            scope_tenant=tenant_id,
            text_contains=query,
            limit=limit,
        )
        result = self._workspace.query_across_tiers(
            graph_query,
            tiers=[WorkspaceTier.PERSISTENT, WorkspaceTier.BRAIN],
        )

        results: list[dict[str, Any]] = []
        for node in result.nodes:
            entry: dict[str, Any] = {"id": node.id, "type": node.node_type.value}
            if isinstance(node, EpisodeNode):
                entry["content"] = node.summary
                entry["memory_type"] = "episodic"
            elif isinstance(node, ConceptNode):
                entry["content"] = f"{node.label}: {node.definition}"
                entry["memory_type"] = "semantic"
            elif isinstance(node, SkillNode):
                entry["content"] = f"{node.skill_name}: {node.description}"
                entry["memory_type"] = "procedural"
            elif isinstance(node, BeliefNode):
                entry["content"] = node.claim
                entry["memory_type"] = "knowledge_graph"
            elif isinstance(node, ValueNode):
                entry["content"] = f"{node.dimension}: {node.weight}"
                entry["memory_type"] = "value"
            else:
                entry["content"] = str(node.id)
                entry["memory_type"] = "other"
            results.append(entry)
        return results

    # ── Migration Support ────────────────────────────────────────────

    def record_divergence(
        self,
        memory_type: str,
        hcir_result: Any,
        legacy_result: Any,
    ) -> None:
        """Record a divergence between HCIR and legacy results.

        Used during dual-write and shadow-read phases.
        """
        self._divergence_count += 1
        logger.warning(
            "Memory divergence #%d in %s: hcir=%s legacy=%s",
            self._divergence_count,
            memory_type,
            type(hcir_result).__name__,
            type(legacy_result).__name__,
        )

    async def stats(self, tenant_id: str = "default") -> dict[str, Any]:
        """Get HCIR memory statistics."""
        persistent = self._workspace.persistent

        def _count_type(node_type: HCIRNodeType) -> int:
            q = GraphQuery(node_type=node_type, scope_tenant=tenant_id)
            return persistent.query(q).total_matches

        return {
            "backend": "hcir",
            "migration_phase": self._migration_phase.value,
            "divergence_count": self._divergence_count,
            "episodic_count": _count_type(HCIRNodeType.EPISODE),
            "concept_count": _count_type(HCIRNodeType.CONCEPT),
            "skill_count": _count_type(HCIRNodeType.SKILL),
            "belief_count": _count_type(HCIRNodeType.BELIEF),
            "value_count": _count_type(HCIRNodeType.VALUE),
        }
