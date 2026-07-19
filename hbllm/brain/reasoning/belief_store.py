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

from hbllm.security import TenantSQLiteRepository

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


class BeliefStore(TenantSQLiteRepository):
    """SQLite-backed persistent store for beliefs and contradictions.

    Responsibilities:
        - Store, retrieve, update beliefs
        - Persist contradictions as first-class objects
        - Index beliefs by concept, domain, type, and status
        - Provide priority-ordered contradiction retrieval

    Does NOT do reasoning — that's BeliefRevisionEngine's job.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        db_path = Path(data_dir) / "belief_store.db"
        super().__init__(db_path)
        self._db_path = db_path
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
            # Check if beliefs table exists
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='beliefs'"
            )
            beliefs_exists = cur.fetchone() is not None

            if not beliefs_exists:
                conn.execute("BEGIN TRANSACTION")
                try:
                    conn.execute("""
                        CREATE TABLE beliefs (
                            belief_id TEXT PRIMARY KEY,
                            tenant_id TEXT DEFAULT '__legacy__',
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
                        CREATE TABLE persistent_contradictions (
                            contradiction_id TEXT PRIMARY KEY,
                            tenant_id TEXT DEFAULT '__legacy__',
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
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_beliefs_tenant_concept ON beliefs(tenant_id, concept)"
                    )
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_domain ON beliefs(domain)")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_beliefs_type ON beliefs(belief_type)"
                    )
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_status ON beliefs(status)")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_contradictions_tenant ON persistent_contradictions(tenant_id, concept)"
                    )
                    conn.execute("PRAGMA user_version = 2")
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            else:
                # Tables exist. Check if tenant_id column exists in beliefs
                cur = conn.execute("PRAGMA table_info(beliefs)")
                beliefs_columns = [row[1] for row in cur.fetchall()]

                # Check if tenant_id column exists in persistent_contradictions
                cur = conn.execute("PRAGMA table_info(persistent_contradictions)")
                contradictions_columns = [row[1] for row in cur.fetchall()]

                conn.execute("BEGIN TRANSACTION")
                try:
                    if "tenant_id" not in beliefs_columns:
                        conn.execute(
                            "ALTER TABLE beliefs ADD COLUMN tenant_id TEXT DEFAULT '__legacy__'"
                        )
                        conn.execute(
                            "CREATE INDEX IF NOT EXISTS idx_beliefs_tenant_concept ON beliefs(tenant_id, concept)"
                        )
                    if "tenant_id" not in contradictions_columns:
                        conn.execute(
                            "ALTER TABLE persistent_contradictions ADD COLUMN tenant_id TEXT DEFAULT '__legacy__'"
                        )
                        conn.execute(
                            "CREATE INDEX IF NOT EXISTS idx_contradictions_tenant ON persistent_contradictions(tenant_id, concept)"
                        )
                    conn.execute("PRAGMA user_version = 2")
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

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
        tenant_id = self.current_tenant()
        cache_key = f"{tenant_id}:{belief_id}" if tenant_id else belief_id
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            with sqlite3.connect(self._db_path) as conn:
                if tenant_id:
                    row = self.execute_tenant(
                        conn,
                        "SELECT data FROM beliefs WHERE belief_id = ? AND tenant_id = ?",
                        (belief_id, tenant_id),
                    ).fetchone()
                else:
                    row = self.execute_tenant(
                        conn,
                        "SELECT data FROM beliefs WHERE belief_id = ?",
                        (belief_id,),
                        required_capability="belief_maintenance",
                    ).fetchone()
                if row:
                    belief = Belief.from_dict(json.loads(row[0]))
                    if tenant_id:
                        self._cache[cache_key] = belief
                    return belief
        except Exception as e:
            logger.debug("Failed to get belief %s: %s", belief_id, e)
        return None

    def get_beliefs_for_concept(self, concept: str) -> list[Belief]:
        """Get all beliefs related to a concept."""
        self._lookups += 1
        tenant_id = self.current_tenant()
        try:
            with sqlite3.connect(self._db_path) as conn:
                if tenant_id:
                    rows = self.execute_tenant(
                        conn,
                        "SELECT data FROM beliefs WHERE concept = ? AND tenant_id = ? AND status != 'decayed' "
                        "ORDER BY confidence DESC",
                        (concept.lower(), tenant_id),
                    ).fetchall()
                else:
                    rows = self.execute_tenant(
                        conn,
                        "SELECT data FROM beliefs WHERE concept = ? AND status != 'decayed' "
                        "ORDER BY confidence DESC",
                        (concept.lower(),),
                        required_capability="belief_maintenance",
                    ).fetchall()
                return [Belief.from_dict(json.loads(r[0])) for r in rows]
        except Exception as e:
            logger.debug("Failed to get beliefs for concept %s: %s", concept, e)
            return []

    def get_beliefs_by_domain(self, domain: str) -> list[Belief]:
        """Get all active beliefs in a domain."""
        self._lookups += 1
        tenant_id = self.current_tenant()
        try:
            with sqlite3.connect(self._db_path) as conn:
                if tenant_id:
                    rows = self.execute_tenant(
                        conn,
                        "SELECT data FROM beliefs WHERE domain = ? AND tenant_id = ? AND status = 'active' "
                        "ORDER BY confidence DESC",
                        (domain, tenant_id),
                    ).fetchall()
                else:
                    rows = self.execute_tenant(
                        conn,
                        "SELECT data FROM beliefs WHERE domain = ? AND status = 'active' "
                        "ORDER BY confidence DESC",
                        (domain,),
                        required_capability="belief_maintenance",
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
        tenant_id = self.current_tenant()
        try:
            with sqlite3.connect(self._db_path) as conn:
                if tenant_id:
                    rows = self.execute_tenant(
                        conn,
                        "SELECT data FROM beliefs WHERE belief_type = ? AND tenant_id = ? "
                        "AND confidence >= ? AND status = 'active' "
                        "ORDER BY confidence DESC",
                        (belief_type.value, tenant_id, min_confidence),
                    ).fetchall()
                else:
                    rows = self.execute_tenant(
                        conn,
                        "SELECT data FROM beliefs WHERE belief_type = ? "
                        "AND confidence >= ? AND status = 'active' "
                        "ORDER BY confidence DESC",
                        (belief_type.value, min_confidence),
                        required_capability="belief_maintenance",
                    ).fetchall()
                return [Belief.from_dict(json.loads(r[0])) for r in rows]
        except Exception as e:
            logger.debug("Failed to get beliefs by type %s: %s", belief_type, e)
            return []

    def update_belief(self, belief: Belief) -> None:
        """Persist an updated belief."""
        belief.last_updated = time.time()
        self._persist_belief(belief)
        tenant_id = self.current_tenant()
        cache_key = f"{tenant_id}:{belief.belief_id}" if tenant_id else belief.belief_id
        self._cache[cache_key] = belief

    def get_contested_beliefs(self) -> list[Belief]:
        """Get all beliefs with active contradictions."""
        return self.get_beliefs_by_status(BeliefStatus.CONTESTED)

    def get_beliefs_by_status(self, status: BeliefStatus) -> list[Belief]:
        """Get all beliefs with a specific status."""
        self._lookups += 1
        tenant_id = self.current_tenant()
        try:
            with sqlite3.connect(self._db_path) as conn:
                if tenant_id:
                    rows = self.execute_tenant(
                        conn,
                        "SELECT data FROM beliefs WHERE status = ? AND tenant_id = ? ORDER BY confidence DESC",
                        (status.value, tenant_id),
                    ).fetchall()
                else:
                    rows = self.execute_tenant(
                        conn,
                        "SELECT data FROM beliefs WHERE status = ? ORDER BY confidence DESC",
                        (status.value,),
                        required_capability="belief_maintenance",
                    ).fetchall()
                return [Belief.from_dict(json.loads(r[0])) for r in rows]
        except Exception as e:
            logger.debug("Failed to get beliefs by status %s: %s", status, e)
            return []

    def get_strongest(self, n: int = 10) -> list[Belief]:
        """Get the highest-confidence active beliefs."""
        self._lookups += 1
        tenant_id = self.current_tenant()
        try:
            with sqlite3.connect(self._db_path) as conn:
                if tenant_id:
                    rows = self.execute_tenant(
                        conn,
                        "SELECT data FROM beliefs WHERE status = 'active' AND tenant_id = ? "
                        "ORDER BY confidence DESC LIMIT ?",
                        (tenant_id, n),
                    ).fetchall()
                else:
                    rows = self.execute_tenant(
                        conn,
                        "SELECT data FROM beliefs WHERE status = 'active' "
                        "ORDER BY confidence DESC LIMIT ?",
                        (n,),
                        required_capability="belief_maintenance",
                    ).fetchall()
                return [Belief.from_dict(json.loads(r[0])) for r in rows]
        except Exception as e:
            logger.debug("Failed to get strongest beliefs: %s", e)
            return []

    def decay_beliefs(self, rate: float = 0.01, threshold: float = 0.05) -> int:
        """Apply confidence decay to all active beliefs.

        Returns number of beliefs decayed below threshold.
        """
        decayed = 0
        tenant_id = self.current_tenant()
        try:
            with sqlite3.connect(self._db_path) as conn:
                if tenant_id:
                    rows = self.execute_tenant(
                        conn,
                        "SELECT data FROM beliefs WHERE status = 'active' AND tenant_id = ?",
                        (tenant_id,),
                    ).fetchall()
                else:
                    rows = self.execute_tenant(
                        conn,
                        "SELECT data FROM beliefs WHERE status = 'active'",
                        required_capability="belief_maintenance",
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
        tenant_id = self.current_tenant() or "__legacy__"
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
                self.execute_tenant(
                    conn,
                    """INSERT INTO persistent_contradictions
                       (contradiction_id, tenant_id, concept, claim_a, claim_b,
                        severity, resolved, data, detected_at)
                       VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                    (ctr_id, tenant_id, concept, claim_a, claim_b, severity, json.dumps(data), now),
                    required_capability="belief_maintenance",
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
        tenant_id = self.current_tenant()
        try:
            with sqlite3.connect(self._db_path) as conn:
                if tenant_id:
                    self.execute_tenant(
                        conn,
                        "UPDATE persistent_contradictions SET resolved = 1, "
                        "resolution = ?, resolved_at = ? WHERE contradiction_id = ? AND tenant_id = ?",
                        (resolution, time.time(), contradiction_id, tenant_id),
                    )
                else:
                    self.execute_tenant(
                        conn,
                        "UPDATE persistent_contradictions SET resolved = 1, "
                        "resolution = ?, resolved_at = ? WHERE contradiction_id = ?",
                        (resolution, time.time(), contradiction_id),
                        required_capability="belief_maintenance",
                    )
        except Exception as e:
            logger.warning("Failed to resolve contradiction: %s", e)

    def get_unresolved_contradictions(self) -> list[dict[str, Any]]:
        """Get all unresolved contradictions."""
        tenant_id = self.current_tenant()
        try:
            with sqlite3.connect(self._db_path) as conn:
                if tenant_id:
                    rows = self.execute_tenant(
                        conn,
                        "SELECT data FROM persistent_contradictions WHERE resolved = 0 AND tenant_id = ? "
                        "ORDER BY severity DESC",
                        (tenant_id,),
                    ).fetchall()
                else:
                    rows = self.execute_tenant(
                        conn,
                        "SELECT data FROM persistent_contradictions WHERE resolved = 0 "
                        "ORDER BY severity DESC",
                        required_capability="belief_maintenance",
                    ).fetchall()
                return [json.loads(r[0]) for r in rows]
        except Exception as e:
            logger.debug("Failed to get unresolved contradictions: %s", e)
            return []

    def get_contradictions_by_priority(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get contradictions ordered by severity × age (priority)."""
        contradictions = self.get_unresolved_contradictions()
        now = time.time()
        for c in contradictions:
            age_hours = max(0.001, (now - c.get("detected_at", now)) / 3600.0)
            c["priority"] = c.get("severity", 0.5) * (1.0 + 0.1 * age_hours)
        contradictions.sort(key=lambda c: c.get("priority", 0), reverse=True)
        return contradictions[:limit]

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        tenant_id = self.current_tenant()
        try:
            with sqlite3.connect(self._db_path) as conn:
                if tenant_id:
                    total_beliefs = self.execute_tenant(
                        conn, "SELECT COUNT(*) FROM beliefs WHERE tenant_id = ?", (tenant_id,)
                    ).fetchone()[0]
                    active_beliefs = self.execute_tenant(
                        conn,
                        "SELECT COUNT(*) FROM beliefs WHERE status = 'active' AND tenant_id = ?",
                        (tenant_id,),
                    ).fetchone()[0]
                    contested_beliefs = self.execute_tenant(
                        conn,
                        "SELECT COUNT(*) FROM beliefs WHERE status = 'contested' AND tenant_id = ?",
                        (tenant_id,),
                    ).fetchone()[0]
                    total_contradictions = self.execute_tenant(
                        conn,
                        "SELECT COUNT(*) FROM persistent_contradictions WHERE tenant_id = ?",
                        (tenant_id,),
                    ).fetchone()[0]
                    unresolved = self.execute_tenant(
                        conn,
                        "SELECT COUNT(*) FROM persistent_contradictions WHERE resolved = 0 AND tenant_id = ?",
                        (tenant_id,),
                    ).fetchone()[0]
                else:
                    total_beliefs = self.execute_tenant(
                        conn,
                        "SELECT COUNT(*) FROM beliefs",
                        required_capability="belief_maintenance",
                    ).fetchone()[0]
                    active_beliefs = self.execute_tenant(
                        conn,
                        "SELECT COUNT(*) FROM beliefs WHERE status = 'active'",
                        required_capability="belief_maintenance",
                    ).fetchone()[0]
                    contested_beliefs = self.execute_tenant(
                        conn,
                        "SELECT COUNT(*) FROM beliefs WHERE status = 'contested'",
                        required_capability="belief_maintenance",
                    ).fetchone()[0]
                    total_contradictions = self.execute_tenant(
                        conn,
                        "SELECT COUNT(*) FROM persistent_contradictions",
                        required_capability="belief_maintenance",
                    ).fetchone()[0]
                    unresolved = self.execute_tenant(
                        conn,
                        "SELECT COUNT(*) FROM persistent_contradictions WHERE resolved = 0",
                        required_capability="belief_maintenance",
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
        tenant_id = self.current_tenant() or "__legacy__"
        try:
            with sqlite3.connect(self._db_path) as conn:
                self.execute_tenant(
                    conn,
                    """INSERT OR REPLACE INTO beliefs
                       (belief_id, tenant_id, concept, claim, belief_type, confidence,
                        status, domain, data, created_at, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        belief.belief_id,
                        tenant_id,
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
                    required_capability="belief_maintenance",
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
