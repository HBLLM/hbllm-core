"""Belief Store — persistent storage for beliefs and contradictions.

Separation of concerns:
    BeliefStore       → WHERE beliefs live (storage, retrieval, indexing)
    BeliefRevisionEngine → HOW beliefs change (reasoning, evidence integration)

Beliefs are the atomic units of knowledge in HBLLM's cognitive architecture.
They bridge raw experience and abstract concepts:

    Mechanisms → Beliefs → Concepts

Belief types:
    factual     — "Python uses reference counting for GC"
    causal      — "Unsanitized input causes injection attacks"
    procedural  — "To deploy, run CI then merge to main"
    strategic   — "Prefer composition over inheritance"
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────────────────────


class BeliefType(StrEnum):
    """Classification of beliefs for downstream reasoning."""

    FACTUAL = "factual"  # Observable facts about the world
    CAUSAL = "causal"  # X causes/prevents/enables Y
    PROCEDURAL = "procedural"  # How to do something
    STRATEGIC = "strategic"  # When/why to choose approach X over Y


class BeliefStatus(StrEnum):
    """Lifecycle status of a belief."""

    ACTIVE = "active"  # Currently held
    CONTESTED = "contested"  # Has active contradictions
    SUPERSEDED = "superseded"  # Replaced by newer belief
    DECAYED = "decayed"  # Confidence dropped below threshold


@dataclass
class Belief:
    """A single persistent belief — the atomic unit of knowledge.

    Unlike BeliefHypothesis (which is part of a BeliefState group),
    a Belief is a standalone entity with its own identity,
    typed classification, and evidence chain.
    """

    belief_id: str = field(default_factory=lambda: f"blf_{uuid.uuid4().hex[:12]}")
    concept: str = ""  # What concept does this belief relate to?
    claim: str = ""  # The actual belief statement
    belief_type: BeliefType = BeliefType.FACTUAL
    confidence: float = 0.5  # 0.0 to 1.0
    status: BeliefStatus = BeliefStatus.ACTIVE
    domain: str = ""  # Which knowledge domain?

    # Evidence provenance
    evidence_sources: list[str] = field(default_factory=list)
    # e.g., "execution:skill_123", "research:session_abc", "contradiction:ctr_xyz"

    # Contradiction links
    contradiction_ids: list[str] = field(default_factory=list)

    # Lifecycle
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    reinforcement_count: int = 0

    def reinforce(self, source: str, delta: float = 0.05) -> None:
        """Reinforce this belief with supporting evidence."""
        self.confidence = min(1.0, self.confidence + delta)
        if source not in self.evidence_sources:
            self.evidence_sources.append(source)
        self.reinforcement_count += 1
        self.last_updated = time.time()

    def weaken(self, source: str, delta: float = 0.05) -> None:
        """Weaken this belief with contradicting evidence."""
        self.confidence = max(0.0, self.confidence - delta)
        if source not in self.evidence_sources:
            self.evidence_sources.append(source)
        self.last_updated = time.time()
        if self.confidence < 0.1:
            self.status = BeliefStatus.DECAYED

    def link_contradiction(self, contradiction_id: str) -> None:
        """Link a contradiction to this belief."""
        if contradiction_id not in self.contradiction_ids:
            self.contradiction_ids.append(contradiction_id)
            self.status = BeliefStatus.CONTESTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_id": self.belief_id,
            "concept": self.concept,
            "claim": self.claim,
            "belief_type": self.belief_type.value,
            "confidence": self.confidence,
            "status": self.status.value,
            "domain": self.domain,
            "evidence_sources": self.evidence_sources,
            "contradiction_ids": self.contradiction_ids,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "reinforcement_count": self.reinforcement_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Belief:
        return cls(
            belief_id=d.get("belief_id", f"blf_{uuid.uuid4().hex[:12]}"),
            concept=d.get("concept", ""),
            claim=d.get("claim", ""),
            belief_type=BeliefType(d.get("belief_type", "factual")),
            confidence=d.get("confidence", 0.5),
            status=BeliefStatus(d.get("status", "active")),
            domain=d.get("domain", ""),
            evidence_sources=d.get("evidence_sources", []),
            contradiction_ids=d.get("contradiction_ids", []),
            created_at=d.get("created_at", time.time()),
            last_updated=d.get("last_updated", time.time()),
            reinforcement_count=d.get("reinforcement_count", 0),
        )


# ── Belief Store ─────────────────────────────────────────────────────────────


class BeliefStore:
    """SQLite-backed persistent store for beliefs and contradictions.

    Responsibilities:
        - Store, retrieve, update beliefs
        - Persist contradictions as first-class objects
        - Index beliefs by concept, domain, type, and status
        - Provide priority-ordered contradiction retrieval

    Does NOT do reasoning — that's BeliefRevisionEngine's job.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self._db_path = Path(data_dir) / "belief_store.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # In-memory cache for hot beliefs
        self._cache: dict[str, Belief] = {}
        self._max_cache = 1000

        # Stats
        self._beliefs_stored = 0
        self._contradictions_stored = 0
        self._lookups = 0

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS beliefs (
                    belief_id TEXT PRIMARY KEY,
                    concept TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    belief_type TEXT NOT NULL DEFAULT 'factual',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    status TEXT NOT NULL DEFAULT 'active',
                    domain TEXT DEFAULT '',
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_updated REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persistent_contradictions (
                    contradiction_id TEXT PRIMARY KEY,
                    concept TEXT NOT NULL,
                    claim_a TEXT NOT NULL,
                    claim_b TEXT NOT NULL,
                    severity REAL NOT NULL DEFAULT 0.5,
                    resolved INTEGER DEFAULT 0,
                    resolution TEXT DEFAULT '',
                    data TEXT NOT NULL,
                    detected_at REAL NOT NULL,
                    resolved_at REAL
                )
            """)
            # Indexes for common queries
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_beliefs_concept "
                "ON beliefs(concept)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_beliefs_domain "
                "ON beliefs(domain)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_beliefs_type "
                "ON beliefs(belief_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_beliefs_status "
                "ON beliefs(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_contradictions_resolved "
                "ON persistent_contradictions(resolved)"
            )

    # ── Belief CRUD ──────────────────────────────────────────────────────

    def store_belief(
        self,
        claim: str,
        concept: str,
        confidence: float = 0.5,
        source: str = "",
        belief_type: BeliefType = BeliefType.FACTUAL,
        domain: str = "",
    ) -> Belief:
        """Store a new belief or reinforce an existing matching one.

        If a belief with a similar claim already exists for the concept,
        it is reinforced rather than duplicated.
        """
        existing = self._find_matching_belief(concept, claim)
        if existing:
            existing.reinforce(source)
            self._persist_belief(existing)
            return existing

        belief = Belief(
            concept=concept,
            claim=claim,
            belief_type=belief_type,
            confidence=confidence,
            domain=domain,
            evidence_sources=[source] if source else [],
        )
        self._persist_belief(belief)
        self._cache[belief.belief_id] = belief
        self._beliefs_stored += 1
        return belief

    def get_belief(self, belief_id: str) -> Belief | None:
        """Get a specific belief by ID."""
        self._lookups += 1
        if belief_id in self._cache:
            return self._cache[belief_id]

        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT data FROM beliefs WHERE belief_id = ?",
                    (belief_id,),
                ).fetchone()
                if row:
                    belief = Belief.from_dict(json.loads(row[0]))
                    self._cache[belief.belief_id] = belief
                    return belief
        except Exception as e:
            logger.debug("Failed to get belief %s: %s", belief_id, e)
        return None

    def get_beliefs_for_concept(self, concept: str) -> list[Belief]:
        """Get all beliefs related to a concept."""
        self._lookups += 1
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT data FROM beliefs WHERE concept = ? AND status != 'decayed' "
                    "ORDER BY confidence DESC",
                    (concept.lower(),),
                ).fetchall()
                return [Belief.from_dict(json.loads(r[0])) for r in rows]
        except Exception as e:
            logger.debug("Failed to get beliefs for concept %s: %s", concept, e)
            return []

    def get_beliefs_by_domain(self, domain: str) -> list[Belief]:
        """Get all active beliefs in a domain."""
        self._lookups += 1
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT data FROM beliefs WHERE domain = ? AND status = 'active' "
                    "ORDER BY confidence DESC",
                    (domain,),
                ).fetchall()
                return [Belief.from_dict(json.loads(r[0])) for r in rows]
        except Exception as e:
            logger.debug("Failed to get beliefs for domain %s: %s", domain, e)
            return []

    def get_beliefs_by_type(
        self,
        belief_type: BeliefType,
        min_confidence: float = 0.0,
    ) -> list[Belief]:
        """Get all beliefs of a specific type above a confidence threshold."""
        self._lookups += 1
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT data FROM beliefs WHERE belief_type = ? "
                    "AND confidence >= ? AND status = 'active' "
                    "ORDER BY confidence DESC",
                    (belief_type.value, min_confidence),
                ).fetchall()
                return [Belief.from_dict(json.loads(r[0])) for r in rows]
        except Exception as e:
            logger.debug("Failed to get beliefs by type %s: %s", belief_type, e)
            return []

    def update_belief(self, belief: Belief) -> None:
        """Persist an updated belief."""
        belief.last_updated = time.time()
        self._persist_belief(belief)
        self._cache[belief.belief_id] = belief

    def get_contested_beliefs(self) -> list[Belief]:
        """Get all beliefs with active contradictions."""
        self._lookups += 1
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT data FROM beliefs WHERE status = 'contested' "
                    "ORDER BY confidence ASC",
                ).fetchall()
                return [Belief.from_dict(json.loads(r[0])) for r in rows]
        except Exception as e:
            logger.debug("Failed to get contested beliefs: %s", e)
            return []

    def decay_beliefs(self, rate: float = 0.01, threshold: float = 0.05) -> int:
        """Apply confidence decay to all active beliefs.

        Returns number of beliefs decayed below threshold.
        """
        decayed = 0
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT data FROM beliefs WHERE status = 'active'",
                ).fetchall()
                for row in rows:
                    belief = Belief.from_dict(json.loads(row[0]))
                    belief.confidence = max(0.0, belief.confidence - rate)
                    if belief.confidence < threshold:
                        belief.status = BeliefStatus.DECAYED
                        decayed += 1
                    belief.last_updated = time.time()
                    self._persist_belief(belief)
        except Exception as e:
            logger.debug("Failed to decay beliefs: %s", e)
        return decayed

    # ── Contradiction Persistence ────────────────────────────────────────

    def store_contradiction(
        self,
        claim_a: str,
        claim_b: str,
        concept: str,
        severity: float = 0.5,
    ) -> str:
        """Persist a contradiction as a first-class object.

        Returns the contradiction_id.
        """
        ctr_id = f"ctr_{uuid.uuid4().hex[:10]}"
        now = time.time()
        data = {
            "contradiction_id": ctr_id,
            "concept": concept,
            "claim_a": claim_a,
            "claim_b": claim_b,
            "severity": severity,
            "resolved": False,
            "resolution": "",
            "detected_at": now,
        }
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT INTO persistent_contradictions
                       (contradiction_id, concept, claim_a, claim_b,
                        severity, resolved, data, detected_at)
                       VALUES (?, ?, ?, ?, ?, 0, ?, ?)""",
                    (ctr_id, concept, claim_a, claim_b, severity, json.dumps(data), now),
                )
            self._contradictions_stored += 1

            # Link contradiction to matching beliefs
            for belief in self.get_beliefs_for_concept(concept):
                if self._claim_matches(belief.claim, claim_a) or self._claim_matches(
                    belief.claim, claim_b
                ):
                    belief.link_contradiction(ctr_id)
                    self.update_belief(belief)

        except Exception as e:
            logger.warning("Failed to persist contradiction: %s", e)
        return ctr_id

    def resolve_contradiction(self, contradiction_id: str, resolution: str) -> None:
        """Mark a contradiction as resolved."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "UPDATE persistent_contradictions SET resolved = 1, "
                    "resolution = ?, resolved_at = ? WHERE contradiction_id = ?",
                    (resolution, time.time(), contradiction_id),
                )
        except Exception as e:
            logger.warning("Failed to resolve contradiction: %s", e)

    def get_unresolved_contradictions(self) -> list[dict[str, Any]]:
        """Get all unresolved contradictions."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT data FROM persistent_contradictions WHERE resolved = 0 "
                    "ORDER BY severity DESC",
                ).fetchall()
                return [json.loads(r[0]) for r in rows]
        except Exception as e:
            logger.debug("Failed to get unresolved contradictions: %s", e)
            return []

    def get_contradictions_by_priority(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get contradictions ordered by severity × age (priority).

        Older, more severe contradictions bubble up — same anti-starvation
        principle as CognitivePriorityScheduler.
        """
        contradictions = self.get_unresolved_contradictions()
        now = time.time()
        for c in contradictions:
            age_hours = max(0.001, (now - c.get("detected_at", now)) / 3600.0)
            c["priority"] = c.get("severity", 0.5) * (1.0 + 0.1 * age_hours)
        contradictions.sort(key=lambda c: c.get("priority", 0), reverse=True)
        return contradictions[:limit]

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                total_beliefs = conn.execute(
                    "SELECT COUNT(*) FROM beliefs"
                ).fetchone()[0]
                active_beliefs = conn.execute(
                    "SELECT COUNT(*) FROM beliefs WHERE status = 'active'"
                ).fetchone()[0]
                contested_beliefs = conn.execute(
                    "SELECT COUNT(*) FROM beliefs WHERE status = 'contested'"
                ).fetchone()[0]
                total_contradictions = conn.execute(
                    "SELECT COUNT(*) FROM persistent_contradictions"
                ).fetchone()[0]
                unresolved = conn.execute(
                    "SELECT COUNT(*) FROM persistent_contradictions WHERE resolved = 0"
                ).fetchone()[0]
        except Exception:
            total_beliefs = active_beliefs = contested_beliefs = 0
            total_contradictions = unresolved = 0

        return {
            "total_beliefs": total_beliefs,
            "active_beliefs": active_beliefs,
            "contested_beliefs": contested_beliefs,
            "total_contradictions": total_contradictions,
            "unresolved_contradictions": unresolved,
            "beliefs_stored": self._beliefs_stored,
            "contradictions_stored": self._contradictions_stored,
            "lookups": self._lookups,
        }

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _persist_belief(self, belief: Belief) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO beliefs
                       (belief_id, concept, claim, belief_type, confidence,
                        status, domain, data, created_at, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        belief.belief_id,
                        belief.concept.lower(),
                        belief.claim,
                        belief.belief_type.value,
                        belief.confidence,
                        belief.status.value,
                        belief.domain,
                        json.dumps(belief.to_dict()),
                        belief.created_at,
                        belief.last_updated,
                    ),
                )
        except Exception as e:
            logger.warning("Failed to persist belief: %s", e)

    def _find_matching_belief(self, concept: str, claim: str) -> Belief | None:
        """Find an existing belief with a similar claim for the concept."""
        beliefs = self.get_beliefs_for_concept(concept)
        for b in beliefs:
            if self._claim_matches(b.claim, claim):
                return b
        return None

    def _claim_matches(self, a: str, b: str) -> bool:
        """Simple Jaccard similarity check between two claims."""
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words or not b_words:
            return False
        intersection = a_words & b_words
        union = a_words | b_words
        return len(intersection) / len(union) > 0.5 if union else False
