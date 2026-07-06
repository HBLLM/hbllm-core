"""
Evidence Packet — Structured evidence for cognitive reasoning.

Insulates reasoners from the internals of ``BeliefGraph`` by
providing a clean, self-contained evidence format.

Instead of passing raw ``BeliefRecord`` objects to reasoners,
we construct ``EvidencePacket`` instances that contain everything
a reasoner needs to evaluate trustworthiness:

    - The fact itself
    - Confidence level
    - Supporting evidence
    - Contradictions
    - Source lineage (who said this, and why)
    - Freshness (how recent)
    - Importance

This allows reasoning to naturally handle conflicting evidence
rather than treating all retrieved memories as equally trustworthy.

Architecture::

    BeliefGraph
        ↓ construct
    EvidencePacket
        ↓ inject
    Reasoner
        ↓
    "This fact has 0.7 confidence, supported by 3 sources,
     contradicted by 1 source, last updated 2 hours ago"

Usage::

    from hbllm.brain.evidence import EvidencePacket, EvidenceBuilder

    packet = EvidencePacket(
        fact="User prefers dark mode",
        confidence=0.85,
        supporting_evidence=["User said 'I like dark mode'"],
        contradictions=["App settings show light mode"],
        source_lineage=["perception → workspace → belief_graph"],
        freshness=0.9,
        importance=0.6,
    )

    # Or build from BeliefGraph:
    packets = EvidenceBuilder.from_belief_graph(belief_graph, memory_ids)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# EvidencePacket
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EvidencePacket:
    """Structured evidence for cognitive reasoning.

    A self-contained unit of evidence that reasoners can evaluate
    without knowing about belief graphs or memory event stores.

    Attributes:
        fact: The factual content being evaluated.
        confidence: Belief confidence [0.0, 1.0].
        supporting_evidence: Human-readable descriptions of support.
        contradictions: Human-readable descriptions of contradictions.
        source_lineage: Chain of sources that produced this evidence.
        freshness: How recent this evidence is [0.0, 1.0].
            1.0 = just observed, 0.0 = very old.
        importance: How important this evidence is [0.0, 1.0].
        memory_id: Reference to the source memory (if applicable).
        metadata: Additional context.
    """

    fact: str
    confidence: float = 1.0
    supporting_evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    source_lineage: list[str] = field(default_factory=list)
    freshness: float = 1.0
    importance: float = 0.5
    memory_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_contested(self) -> bool:
        """Whether this evidence has contradictions."""
        return len(self.contradictions) > 0

    @property
    def support_count(self) -> int:
        """Number of supporting evidence items."""
        return len(self.supporting_evidence)

    @property
    def reliability_score(self) -> float:
        """Composite reliability: confidence × freshness × support ratio.

        Higher = more reliable for reasoning.
        """
        support_ratio = self.support_count / max(1, self.support_count + len(self.contradictions))
        return self.confidence * self.freshness * support_ratio

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact": self.fact,
            "confidence": round(self.confidence, 3),
            "support_count": self.support_count,
            "contradiction_count": len(self.contradictions),
            "freshness": round(self.freshness, 3),
            "importance": round(self.importance, 3),
            "reliability": round(self.reliability_score, 3),
            "is_contested": self.is_contested,
        }


# ═══════════════════════════════════════════════════════════════════════════
# EvidenceBuilder — construct EvidencePackets from BeliefGraph
# ═══════════════════════════════════════════════════════════════════════════


class EvidenceBuilder:
    """Constructs ``EvidencePacket`` instances from a ``BeliefGraph``.

    This is the bridge between the memory/belief subsystem and
    the reasoning subsystem. Reasoners never import BeliefGraph
    directly — they receive EvidencePackets.
    """

    @staticmethod
    async def from_belief_graph(
        belief_graph: Any,
        memory_ids: list[str],
        max_age_seconds: float = 86400.0,
    ) -> list[EvidencePacket]:
        """Build evidence packets from belief graph records.

        Args:
            belief_graph: A ``BeliefGraph`` instance.
            memory_ids: Memory IDs to build evidence for.
            max_age_seconds: Maximum age for freshness calculation.

        Returns:
            List of EvidencePacket instances.
        """
        packets: list[EvidencePacket] = []
        now = time.time()

        for mid in memory_ids:
            record = await belief_graph.get_belief(mid)
            if record is None:
                continue

            # Calculate freshness from record age
            age = now - record.created_at
            freshness = max(0.0, 1.0 - age / max_age_seconds)

            # Get explanation chain
            explanation = await belief_graph.explain(mid)

            # Gather support and contradiction descriptions
            supporting: list[str] = []
            contradictions: list[str] = []
            edges = belief_graph._edges.get(mid, [])
            for edge in edges:
                if edge.edge_type == "support":
                    supporting.append(
                        f"Supported by {edge.target_id} (strength={edge.strength:.2f})"
                    )
                elif edge.edge_type == "contradiction":
                    contradictions.append(
                        f"Contradicted by {edge.target_id} (strength={edge.strength:.2f})"
                    )

            packet = EvidencePacket(
                fact=record.reason or f"Belief {mid}",
                confidence=record.confidence,
                supporting_evidence=supporting,
                contradictions=contradictions,
                source_lineage=[record.created_by, explanation],
                freshness=freshness,
                importance=record.confidence,  # Approximate importance
                memory_id=mid,
            )
            packets.append(packet)

        return packets

    @staticmethod
    def from_raw(
        fact: str,
        confidence: float = 1.0,
        source: str = "unknown",
    ) -> EvidencePacket:
        """Create a simple evidence packet from raw data.

        Useful when evidence comes from outside the belief graph
        (e.g., direct user input, tool output).

        Args:
            fact: The factual content.
            confidence: Confidence level.
            source: Where this evidence came from.

        Returns:
            An EvidencePacket.
        """
        return EvidencePacket(
            fact=fact,
            confidence=confidence,
            source_lineage=[source],
            freshness=1.0,
            importance=confidence,
        )

    @staticmethod
    def merge_evidence(packets: list[EvidencePacket]) -> EvidencePacket:
        """Merge multiple evidence packets about the same fact.

        Combines support, contradictions, and averages confidence.

        Args:
            packets: Evidence packets to merge.

        Returns:
            A merged EvidencePacket.

        Raises:
            ValueError: If no packets provided.
        """
        if not packets:
            raise ValueError("Cannot merge empty evidence list")

        if len(packets) == 1:
            return packets[0]

        all_support = []
        all_contradictions = []
        all_lineage = []
        total_confidence = 0.0
        max_freshness = 0.0
        max_importance = 0.0

        for p in packets:
            all_support.extend(p.supporting_evidence)
            all_contradictions.extend(p.contradictions)
            all_lineage.extend(p.source_lineage)
            total_confidence += p.confidence
            max_freshness = max(max_freshness, p.freshness)
            max_importance = max(max_importance, p.importance)

        return EvidencePacket(
            fact=packets[0].fact,
            confidence=total_confidence / len(packets),
            supporting_evidence=all_support,
            contradictions=all_contradictions,
            source_lineage=list(set(all_lineage)),
            freshness=max_freshness,
            importance=max_importance,
            memory_id=packets[0].memory_id,
        )
