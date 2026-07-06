"""
Belief Graph — explainable provenance for memory beliefs.

Every semantic memory has a ``BeliefRecord`` that tracks:
    - **Who** created this belief (source_node)
    - **Why** it was created (trigger, reason)
    - **How confident** it is (confidence, reinforcement_count)
    - **What supports/contradicts** it (edges to other beliefs)
    - **When** it was last reinforced or challenged

The ``BeliefGraph`` provides:
    - ``explain(memory_id)`` → human-readable explanation chain
    - ``get_contested_beliefs()`` → beliefs with contradictions (for REM sleep)
    - ``add_support/add_contradiction`` → build the provenance graph

Architecture::

    MemCube "Earth is round"
        ↓ has
    BeliefRecord
        ├── supports ← "Satellite photos confirm"
        ├── supports ← "Ship disappearing over horizon"
        └── contradicts ← "Flat-earth claim" (confidence=0.1)

    explain("Earth is round"):
        "I believe 'Earth is round' because:
         - Created by 'perception' at 2024-01-01 (reason: 'Scientific consensus')
         - Supported by 'Satellite photos confirm' (strength=0.9)
         - Supported by 'Ship disappearing over horizon' (strength=0.7)
         - Contradicted by 'Flat-earth claim' (confidence=0.1, weakly contested)
         Overall confidence: 0.95 (reinforced 7×, corrected 0×)"

Usage::

    from hbllm.memory.belief_graph import BeliefGraph, BeliefRecord

    graph = BeliefGraph()
    await graph.record_belief(BeliefRecord(
        id="bel_001",
        memory_id="mem_001",
        created_by="perception",
        created_at=time.time(),
        reason="User stated this",
        trigger="user_input",
    ))
    await graph.add_support("mem_001", "mem_002", strength=0.8)
    explanation = await graph.explain("mem_001")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Belief Edge Types
# ═══════════════════════════════════════════════════════════════════════════


class BeliefEdgeType(StrEnum):
    """Relationship types between beliefs."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVES_FROM = "derives_from"
    CORRECTS = "corrects"
    REINFORCES = "reinforces"


# ═══════════════════════════════════════════════════════════════════════════
# Belief Edge
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BeliefEdge:
    """A directed edge in the belief graph.

    Attributes:
        source_memory_id: The memory providing evidence.
        target_memory_id: The memory being supported/contradicted.
        edge_type: Relationship type.
        strength: Edge strength [0.0, 1.0].
        created_at: When this edge was created.
        reason: Why this relationship exists.
    """

    source_memory_id: str
    target_memory_id: str
    edge_type: BeliefEdgeType
    strength: float = 0.5
    created_at: float = field(default_factory=time.time)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_memory_id,
            "target": self.target_memory_id,
            "type": self.edge_type.value,
            "strength": round(self.strength, 3),
            "reason": self.reason,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Belief Record
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BeliefRecord:
    """Provenance record for a memory's belief status.

    Tracks who created the belief, why, how many times it's been
    reinforced or corrected, and its current confidence.

    Attributes:
        id: Unique belief record ID.
        memory_id: The MemCube this belief is about.
        created_by: Node that created this belief.
        created_at: When the belief was formed.
        reason: Why this belief was formed.
        trigger: What triggered belief formation.
        confidence: Current confidence [0.0, 1.0].
        reinforcement_count: Times this belief was reinforced.
        correction_count: Times this belief was corrected.
        edges: Edges to supporting/contradicting beliefs.
        tenant_id: Multi-tenant isolation.
    """

    id: str
    memory_id: str
    created_by: str
    created_at: float
    reason: str
    trigger: str
    confidence: float = 1.0
    reinforcement_count: int = 0
    correction_count: int = 0
    edges: list[BeliefEdge] = field(default_factory=list)
    tenant_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "reason": self.reason,
            "trigger": self.trigger,
            "confidence": round(self.confidence, 3),
            "reinforcement_count": self.reinforcement_count,
            "correction_count": self.correction_count,
            "edge_count": len(self.edges),
            "tenant_id": self.tenant_id,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Belief Graph
# ═══════════════════════════════════════════════════════════════════════════


class BeliefGraph:
    """Directed graph of belief provenance and support.

    Provides explainability for memory beliefs: why something is
    believed, what supports it, and what contradicts it.
    """

    def __init__(self) -> None:
        self._records: dict[str, BeliefRecord] = {}  # memory_id → record
        self._edges: list[BeliefEdge] = []

    async def record_belief(self, record: BeliefRecord) -> None:
        """Register a belief record for a memory.

        If a belief already exists for this memory_id, it is replaced.

        Args:
            record: The belief record to register.
        """
        self._records[record.memory_id] = record
        logger.debug(
            "Belief recorded: memory=%s, confidence=%.2f, by=%s",
            record.memory_id,
            record.confidence,
            record.created_by,
        )

    async def get_belief(self, memory_id: str) -> BeliefRecord | None:
        """Get the belief record for a memory.

        Args:
            memory_id: The memory to look up.

        Returns:
            The BeliefRecord, or None if no belief exists.
        """
        return self._records.get(memory_id)

    async def add_support(
        self,
        memory_id: str,
        supporting_id: str,
        strength: float = 0.5,
        reason: str = "",
    ) -> None:
        """Add a supporting edge to a belief.

        Supporting evidence increases the belief's confidence.

        Args:
            memory_id: The belief being supported.
            supporting_id: The memory providing support.
            strength: Support strength [0.0, 1.0].
            reason: Why this supports the belief.
        """
        edge = BeliefEdge(
            source_memory_id=supporting_id,
            target_memory_id=memory_id,
            edge_type=BeliefEdgeType.SUPPORTS,
            strength=strength,
            reason=reason,
        )
        self._edges.append(edge)

        record = self._records.get(memory_id)
        if record:
            record.edges.append(edge)
            # Support increases confidence
            record.confidence = min(
                1.0,
                record.confidence + strength * 0.1,
            )

    async def add_contradiction(
        self,
        memory_id: str,
        contradicting_id: str,
        strength: float = 0.5,
        reason: str = "",
    ) -> None:
        """Add a contradicting edge to a belief.

        Contradiction decreases the belief's confidence.

        Args:
            memory_id: The belief being contradicted.
            contradicting_id: The memory providing contradiction.
            strength: Contradiction strength [0.0, 1.0].
            reason: Why this contradicts the belief.
        """
        edge = BeliefEdge(
            source_memory_id=contradicting_id,
            target_memory_id=memory_id,
            edge_type=BeliefEdgeType.CONTRADICTS,
            strength=strength,
            reason=reason,
        )
        self._edges.append(edge)

        record = self._records.get(memory_id)
        if record:
            record.edges.append(edge)
            # Contradiction decreases confidence
            record.confidence = max(
                0.0,
                record.confidence - strength * 0.15,
            )

    async def reinforce(self, memory_id: str) -> None:
        """Reinforce a belief (increment count, bump confidence).

        Args:
            memory_id: The belief to reinforce.
        """
        record = self._records.get(memory_id)
        if record:
            record.reinforcement_count += 1
            record.confidence = min(1.0, record.confidence + 0.02)

    async def correct(self, memory_id: str, correction: str) -> None:
        """Record a correction to a belief.

        Args:
            memory_id: The belief being corrected.
            correction: Description of the correction.
        """
        record = self._records.get(memory_id)
        if record:
            record.correction_count += 1
            # Corrections slightly reduce confidence (need re-validation)
            record.confidence = max(0.0, record.confidence - 0.1)

    async def explain(self, memory_id: str, depth: int = 5) -> str:
        """Generate a human-readable explanation of a belief.

        Args:
            memory_id: The memory to explain.
            depth: Maximum depth for traversing support chain.

        Returns:
            Multi-line explanation string.
        """
        record = self._records.get(memory_id)
        if not record:
            return f"No belief record found for memory '{memory_id}'."

        lines: list[str] = []
        lines.append(f"I believe this (confidence={record.confidence:.2f}) because:")
        lines.append(
            f"  - Created by '{record.created_by}' "
            f"(reason: '{record.reason}', trigger: '{record.trigger}')"
        )

        # Group edges by type
        supports = [e for e in record.edges if e.edge_type == BeliefEdgeType.SUPPORTS]
        contradictions = [e for e in record.edges if e.edge_type == BeliefEdgeType.CONTRADICTS]
        derivations = [e for e in record.edges if e.edge_type == BeliefEdgeType.DERIVES_FROM]

        for edge in supports:
            lines.append(
                f"  + Supported by '{edge.source_memory_id}' (strength={edge.strength:.1f})"
            )
        for edge in contradictions:
            lines.append(
                f"  - Contradicted by '{edge.source_memory_id}' (strength={edge.strength:.1f})"
            )
        for edge in derivations:
            lines.append(f"  → Derived from '{edge.source_memory_id}'")

        lines.append(
            f"  Reinforced {record.reinforcement_count}×, corrected {record.correction_count}×"
        )

        if contradictions and record.confidence < 0.5:
            lines.append("  ⚠ This belief is CONTESTED.")

        return "\n".join(lines)

    async def get_contested_beliefs(
        self,
        tenant_id: str | None = None,
    ) -> list[BeliefRecord]:
        """Get beliefs that have contradictions.

        Used by REM sleep to resolve contested beliefs.

        Args:
            tenant_id: Optional tenant filter.

        Returns:
            List of belief records with at least one contradiction.
        """
        contested: list[BeliefRecord] = []
        for record in self._records.values():
            if tenant_id and record.tenant_id != tenant_id:
                continue
            has_contradiction = any(e.edge_type == BeliefEdgeType.CONTRADICTS for e in record.edges)
            if has_contradiction:
                contested.append(record)

        # Sort by lowest confidence first (most contested)
        contested.sort(key=lambda r: r.confidence)
        return contested

    async def get_strong_beliefs(
        self,
        min_confidence: float = 0.8,
        tenant_id: str | None = None,
    ) -> list[BeliefRecord]:
        """Get high-confidence beliefs.

        Args:
            min_confidence: Minimum confidence threshold.
            tenant_id: Optional tenant filter.

        Returns:
            High-confidence belief records.
        """
        result: list[BeliefRecord] = []
        for record in self._records.values():
            if tenant_id and record.tenant_id != tenant_id:
                continue
            if record.confidence >= min_confidence:
                result.append(record)
        result.sort(key=lambda r: r.confidence, reverse=True)
        return result

    def stats(self) -> dict[str, Any]:
        """Graph statistics."""
        return {
            "total_beliefs": len(self._records),
            "total_edges": len(self._edges),
            "contested_count": sum(
                1
                for r in self._records.values()
                if any(e.edge_type == BeliefEdgeType.CONTRADICTS for e in r.edges)
            ),
        }
