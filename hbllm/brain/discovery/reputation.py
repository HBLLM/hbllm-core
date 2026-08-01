"""Source Reputation Tracker — epistemic trust for knowledge sources.

Tracks the reliability of knowledge sources over time.  The system
learns which sources are trustworthy, which reasoning patterns work,
and which hypothesis origins historically succeed.

This is the internal epistemic model that humans use subconsciously:
"Source A's claims are confirmed 90% of the time; Source B only 30%."

Architecture::

    Claim from Source A → confirmed    → reputation ↑
    Claim from Source A → contradicted → reputation ↓
    
    New claim from Source A:
        "Source A has 0.87 reputation → weight this claim higher"

    New claim from Source B:
        "Source B has 0.34 reputation → weight this claim lower"

Usage::

    tracker = SourceReputationTracker(data_dir=Path("./research"))
    
    # Record outcomes
    await tracker.record_outcome("arxiv:2301.12345", "claim_001", confirmed=True)
    await tracker.record_outcome("arxiv:2301.12345", "claim_002", confirmed=True)
    await tracker.record_outcome("blog:random", "claim_003", confirmed=False)
    
    # Query reputation
    score = await tracker.get_reputation("arxiv:2301.12345")  # → 0.85
    
    # Get top sources
    tops = await tracker.get_top_sources(domain="neuroscience")
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SourceReputation:
    """Reputation record for a knowledge source."""

    source_id: str = ""
    display_name: str = ""
    domain: str = ""  # Primary knowledge domain

    # Tracking
    total_claims: int = 0
    confirmed_claims: int = 0
    refuted_claims: int = 0
    pending_claims: int = 0

    # Computed scores
    confirmation_rate: float = 0.5  # confirmed / total (smoothed)
    reputation_score: float = 0.5   # Weighted reputation [0.0, 1.0]

    # History
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    score_history: list[dict[str, Any]] = field(default_factory=list)
    # Each entry: {"timestamp": float, "score": float, "reason": str}

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "display_name": self.display_name,
            "domain": self.domain,
            "total_claims": self.total_claims,
            "confirmed_claims": self.confirmed_claims,
            "refuted_claims": self.refuted_claims,
            "pending_claims": self.pending_claims,
            "confirmation_rate": self.confirmation_rate,
            "reputation_score": self.reputation_score,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "score_history": self.score_history,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceReputation:
        return cls(
            source_id=d.get("source_id", ""),
            display_name=d.get("display_name", ""),
            domain=d.get("domain", ""),
            total_claims=d.get("total_claims", 0),
            confirmed_claims=d.get("confirmed_claims", 0),
            refuted_claims=d.get("refuted_claims", 0),
            pending_claims=d.get("pending_claims", 0),
            confirmation_rate=d.get("confirmation_rate", 0.5),
            reputation_score=d.get("reputation_score", 0.5),
            first_seen=d.get("first_seen", time.time()),
            last_seen=d.get("last_seen", time.time()),
            score_history=d.get("score_history", []),
        )


@dataclass
class ReputationConfig:
    """Configuration for reputation scoring."""

    # Laplace smoothing — prevents 0/0 or 1/1 extremes for low counts
    smoothing_alpha: float = 2.0
    smoothing_beta: float = 2.0

    # Minimum claims before reputation is considered reliable
    min_claims_for_confidence: int = 5

    # Time decay — older outcomes contribute less
    decay_halflife_days: float = 365.0

    # Maximum history entries to keep per source
    max_history_entries: int = 100


# ═══════════════════════════════════════════════════════════════════════════
# Source Reputation Tracker
# ═══════════════════════════════════════════════════════════════════════════


class SourceReputationTracker:
    """Tracks the reliability of knowledge sources over time.

    Uses a Bayesian-smoothed confirmation rate with time decay
    to compute reputation scores.  Sources with more claims
    and higher confirmation rates get higher reputations.

    The tracker is domain-aware — a source may be reliable in
    one domain but not another.
    """

    def __init__(
        self,
        data_dir: str | Path,
        config: ReputationConfig | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "source_reputation.db"
        self._config = config or ReputationConfig()
        self._cache: dict[str, SourceReputation] = {}
        self._init_db()

    # ── Core Operations ───────────────────────────────────────────────

    async def record_outcome(
        self,
        source_id: str,
        claim_id: str,
        confirmed: bool,
        domain: str = "",
        display_name: str = "",
    ) -> SourceReputation:
        """Record whether a claim from a source was confirmed or refuted.

        Updates the source's reputation score using Bayesian smoothing.
        """
        rep = await self._get_or_create(source_id, domain, display_name)

        rep.total_claims += 1
        if confirmed:
            rep.confirmed_claims += 1
        else:
            rep.refuted_claims += 1
        rep.last_seen = time.time()

        # Bayesian-smoothed confirmation rate
        alpha = self._config.smoothing_alpha
        beta = self._config.smoothing_beta
        rep.confirmation_rate = (rep.confirmed_claims + alpha) / (
            rep.total_claims + alpha + beta
        )

        # Reputation score — confirmation rate weighted by claim count confidence
        claim_confidence = min(
            1.0,
            rep.total_claims / self._config.min_claims_for_confidence,
        )
        # Blend between prior (0.5) and observed rate based on how many claims we have
        rep.reputation_score = (
            (1.0 - claim_confidence) * 0.5  # Prior
            + claim_confidence * rep.confirmation_rate  # Observed
        )

        # Record history
        rep.score_history.append({
            "timestamp": time.time(),
            "score": rep.reputation_score,
            "reason": f"claim {claim_id} {'confirmed' if confirmed else 'refuted'}",
        })
        # Trim history
        if len(rep.score_history) > self._config.max_history_entries:
            rep.score_history = rep.score_history[-self._config.max_history_entries:]

        self._cache[source_id] = rep
        self._persist(rep)

        logger.debug(
            "Source %s reputation: %.3f (confirmed=%d, refuted=%d)",
            source_id, rep.reputation_score,
            rep.confirmed_claims, rep.refuted_claims,
        )
        return rep

    async def get_reputation(self, source_id: str) -> float:
        """Get the reliability score for a source [0.0, 1.0].

        Returns 0.5 (neutral prior) for unknown sources.
        """
        rep = self._cache.get(source_id)
        if rep is None:
            rep = self._load(source_id)
        if rep is None:
            return 0.5  # Neutral prior for unknown sources
        return rep.reputation_score

    async def get_source_details(self, source_id: str) -> SourceReputation | None:
        """Get full reputation details for a source."""
        rep = self._cache.get(source_id)
        if rep is None:
            rep = self._load(source_id)
        return rep

    async def get_top_sources(
        self,
        domain: str = "",
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the most reliable sources, optionally filtered by domain."""
        all_reps = self._load_all()

        if domain:
            all_reps = [r for r in all_reps if r.domain == domain]

        # Only include sources with enough data
        qualified = [
            r for r in all_reps
            if r.total_claims >= self._config.min_claims_for_confidence
        ]

        qualified.sort(key=lambda r: r.reputation_score, reverse=True)
        return [
            (r.source_id, r.reputation_score)
            for r in qualified[:limit]
        ]

    async def get_unreliable_sources(
        self,
        threshold: float = 0.3,
        domain: str = "",
    ) -> list[tuple[str, float]]:
        """Return sources with low reputation scores."""
        all_reps = self._load_all()

        if domain:
            all_reps = [r for r in all_reps if r.domain == domain]

        unreliable = [
            r for r in all_reps
            if r.reputation_score < threshold
            and r.total_claims >= self._config.min_claims_for_confidence
        ]

        unreliable.sort(key=lambda r: r.reputation_score)
        return [(r.source_id, r.reputation_score) for r in unreliable]

    # ── Persistence ───────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS source_reputations (
                    source_id TEXT PRIMARY KEY,
                    domain TEXT DEFAULT '',
                    reputation_score REAL DEFAULT 0.5,
                    total_claims INTEGER DEFAULT 0,
                    data TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sr_domain
                ON source_reputations(domain)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sr_score
                ON source_reputations(reputation_score DESC)
            """)

    def _persist(self, rep: SourceReputation) -> None:
        data = json.dumps(rep.to_dict())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO source_reputations
                   (source_id, domain, reputation_score, total_claims, data, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    rep.source_id,
                    rep.domain,
                    rep.reputation_score,
                    rep.total_claims,
                    data,
                    time.time(),
                ),
            )

    def _load(self, source_id: str) -> SourceReputation | None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT data FROM source_reputations WHERE source_id = ?",
                    (source_id,),
                ).fetchone()
            if row:
                rep = SourceReputation.from_dict(json.loads(row[0]))
                self._cache[source_id] = rep
                return rep
        except Exception as e:
            logger.warning("Failed to load source reputation %s: %s", source_id, e)
        return None

    def _load_all(self) -> list[SourceReputation]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("SELECT data FROM source_reputations").fetchall()
            return [SourceReputation.from_dict(json.loads(r[0])) for r in rows]
        except Exception as e:
            logger.warning("Failed to load source reputations: %s", e)
            return []

    async def _get_or_create(
        self,
        source_id: str,
        domain: str = "",
        display_name: str = "",
    ) -> SourceReputation:
        """Get existing reputation or create a new one."""
        rep = self._cache.get(source_id)
        if rep is None:
            rep = self._load(source_id)
        if rep is None:
            rep = SourceReputation(
                source_id=source_id,
                display_name=display_name or source_id,
                domain=domain,
            )
            self._cache[source_id] = rep
        return rep
