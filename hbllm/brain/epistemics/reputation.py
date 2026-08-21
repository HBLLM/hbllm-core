"""Source Reputation Tracker — epistemic trust for knowledge sources.

Tracks the reliability of knowledge sources over time.  The system
learns which sources are trustworthy, which reasoning patterns work,
and which hypothesis origins historically succeed.

A11 Architectural Invariant:
    Provider reputation decomposes into three INDEPENDENT scores:

    1. signal_quality      — SNR, sensor clarity, raw output quality
    2. cross_modal_concordance — agreement with other modalities
    3. empirical_accuracy   — ground-truth validated outcomes ONLY

    Cross-modal consensus updates only cross_modal_concordance,
    NEVER empirical_accuracy. empirical_accuracy may ONLY be
    updated by external ground truth:
    - Experiment verification
    - User confirmation
    - Tool execution
    - External verification

Provider reputation remains OUTSIDE the belief feedback loop::

                    PROVIDER
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   signal_quality  concordance  empirical_accuracy
                                      ▲
                                      │
                              external ground truth

Usage::

    tracker = SourceReputationTracker(data_dir=Path("./research"))

    # Record external ground truth outcomes
    await tracker.record_empirical_outcome(
        "vision_yolo", "claim_001",
        outcome=OutcomeType.EXPERIMENT, verified=True
    )

    # Record cross-modal concordance (does NOT affect empirical_accuracy)
    await tracker.record_concordance("vision_yolo", concordant=True)

    # Record sensor signal quality
    await tracker.record_signal_quality("vision_yolo", quality_score=0.85)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SourceReputation:
    """Reputation record for a knowledge source.

    Decomposes into three independent scores:
    - signal_quality: SNR, sensor clarity, raw output quality
    - cross_modal_concordance: agreement with other modalities
    - empirical_accuracy: ground-truth validated outcomes ONLY
    """

    source_id: str = ""
    display_name: str = ""
    domain: str = ""  # Primary knowledge domain

    # Three independent reputation dimensions
    signal_quality: float = 0.5  # SNR, sensor clarity [0.0, 1.0]
    cross_modal_concordance: float = 0.5  # Agreement with other modalities [0.0, 1.0]
    empirical_accuracy: float = 0.5  # Ground-truth validated outcomes ONLY [0.0, 1.0]

    # Legacy tracking (still used for empirical_accuracy computation)
    total_claims: int = 0
    confirmed_claims: int = 0
    refuted_claims: int = 0
    pending_claims: int = 0

    # Derived scores
    confirmation_rate: float = 0.5  # confirmed / total (smoothed)
    reputation_score: float = 0.5  # Composite reputation [0.0, 1.0]

    # Concordance tracking
    concordance_total: int = 0
    concordance_agreements: int = 0

    # Signal quality tracking
    signal_quality_samples: int = 0
    signal_quality_sum: float = 0.0

    # History
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    score_history: list[dict[str, Any]] = field(default_factory=list)
    # Each entry: {"timestamp": float, "score": float, "reason": str}

    def compute_composite_reputation(self) -> float:
        """Compute composite reputation from three independent dimensions."""
        return (
            0.3 * self.signal_quality
            + 0.2 * self.cross_modal_concordance
            + 0.5 * self.empirical_accuracy
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "display_name": self.display_name,
            "domain": self.domain,
            "signal_quality": self.signal_quality,
            "cross_modal_concordance": self.cross_modal_concordance,
            "empirical_accuracy": self.empirical_accuracy,
            "total_claims": self.total_claims,
            "confirmed_claims": self.confirmed_claims,
            "refuted_claims": self.refuted_claims,
            "pending_claims": self.pending_claims,
            "confirmation_rate": self.confirmation_rate,
            "reputation_score": self.reputation_score,
            "concordance_total": self.concordance_total,
            "concordance_agreements": self.concordance_agreements,
            "signal_quality_samples": self.signal_quality_samples,
            "signal_quality_sum": self.signal_quality_sum,
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
            signal_quality=d.get("signal_quality", 0.5),
            cross_modal_concordance=d.get("cross_modal_concordance", 0.5),
            empirical_accuracy=d.get("empirical_accuracy", 0.5),
            total_claims=d.get("total_claims", 0),
            confirmed_claims=d.get("confirmed_claims", 0),
            refuted_claims=d.get("refuted_claims", 0),
            pending_claims=d.get("pending_claims", 0),
            confirmation_rate=d.get("confirmation_rate", 0.5),
            reputation_score=d.get("reputation_score", 0.5),
            concordance_total=d.get("concordance_total", 0),
            concordance_agreements=d.get("concordance_agreements", 0),
            signal_quality_samples=d.get("signal_quality_samples", 0),
            signal_quality_sum=d.get("signal_quality_sum", 0.0),
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

        Updates the source's empirical_accuracy using Bayesian smoothing.
        This is a convenience method that delegates to record_empirical_outcome.
        """
        from hbllm.hcir.types import OutcomeType

        return await self.record_empirical_outcome(
            source_id=source_id,
            claim_id=claim_id,
            outcome=OutcomeType.EXTERNAL_VERIFICATION,
            verified=confirmed,
            domain=domain,
            display_name=display_name,
        )

    async def record_empirical_outcome(
        self,
        source_id: str,
        claim_id: str,
        outcome: Any,
        verified: bool,
        domain: str = "",
        display_name: str = "",
    ) -> SourceReputation:
        """Record an external ground-truth outcome for a source.

        Anti-circularity guard: Only OutcomeType values (EXPERIMENT,
        USER_CONFIRMATION, TOOL_EXECUTION, EXTERNAL_VERIFICATION) are
        accepted. Internal belief convergence CANNOT update empirical_accuracy.

        Args:
            source_id: The source identifier.
            claim_id: The claim being verified.
            outcome: Must be a valid OutcomeType.
            verified: Whether the claim was confirmed (True) or refuted (False).
            domain: Knowledge domain.
            display_name: Human-readable source name.

        Raises:
            ValueError: If outcome is not a valid OutcomeType.
        """
        from hbllm.hcir.types import OutcomeType

        # Anti-circularity guard
        valid_outcomes = set(OutcomeType)
        outcome_str = str(outcome)
        if outcome_str not in valid_outcomes:
            raise ValueError(
                f"Invalid outcome type '{outcome_str}' for empirical accuracy update. "
                f"Only external ground truth is accepted: {sorted(valid_outcomes)}. "
                f"Cross-modal consensus updates cross_modal_concordance only."
            )

        rep = await self._get_or_create(source_id, domain, display_name)

        rep.total_claims += 1
        if verified:
            rep.confirmed_claims += 1
        else:
            rep.refuted_claims += 1
        rep.last_seen = time.time()

        # Bayesian-smoothed confirmation rate
        alpha = self._config.smoothing_alpha
        beta = self._config.smoothing_beta
        rep.confirmation_rate = (rep.confirmed_claims + alpha) / (rep.total_claims + alpha + beta)

        # Update empirical_accuracy
        claim_confidence = min(
            1.0,
            rep.total_claims / self._config.min_claims_for_confidence,
        )
        rep.empirical_accuracy = (
            (1.0 - claim_confidence) * 0.5  # Prior
            + claim_confidence * rep.confirmation_rate  # Observed
        )

        # Update composite reputation
        rep.reputation_score = rep.compute_composite_reputation()

        # Record history
        rep.score_history.append(
            {
                "timestamp": time.time(),
                "score": rep.reputation_score,
                "dimension": "empirical_accuracy",
                "reason": f"claim {claim_id} {'confirmed' if verified else 'refuted'} via {outcome_str}",
            }
        )
        # Trim history
        if len(rep.score_history) > self._config.max_history_entries:
            rep.score_history = rep.score_history[-self._config.max_history_entries :]

        self._cache[source_id] = rep
        self._persist(rep)

        logger.debug(
            "Source %s empirical_accuracy: %.3f (confirmed=%d, refuted=%d, outcome=%s)",
            source_id,
            rep.empirical_accuracy,
            rep.confirmed_claims,
            rep.refuted_claims,
            outcome_str,
        )
        return rep

    async def record_concordance(
        self,
        source_id: str,
        concordant: bool,
        domain: str = "",
        display_name: str = "",
    ) -> SourceReputation:
        """Record cross-modal concordance for a source.

        Updates ONLY cross_modal_concordance. Does NOT affect empirical_accuracy.
        This is the correct place to record audio ↔ vision agreement.
        """
        rep = await self._get_or_create(source_id, domain, display_name)

        rep.concordance_total += 1
        if concordant:
            rep.concordance_agreements += 1
        rep.last_seen = time.time()

        # Bayesian-smoothed concordance rate
        alpha = self._config.smoothing_alpha
        beta = self._config.smoothing_beta
        rep.cross_modal_concordance = (rep.concordance_agreements + alpha) / (
            rep.concordance_total + alpha + beta
        )

        # Update composite reputation
        rep.reputation_score = rep.compute_composite_reputation()

        # Record history
        rep.score_history.append(
            {
                "timestamp": time.time(),
                "score": rep.reputation_score,
                "dimension": "cross_modal_concordance",
                "reason": f"concordance {'agreement' if concordant else 'disagreement'}",
            }
        )
        if len(rep.score_history) > self._config.max_history_entries:
            rep.score_history = rep.score_history[-self._config.max_history_entries :]

        self._cache[source_id] = rep
        self._persist(rep)

        logger.debug(
            "Source %s concordance: %.3f (agreements=%d, total=%d)",
            source_id,
            rep.cross_modal_concordance,
            rep.concordance_agreements,
            rep.concordance_total,
        )
        return rep

    async def record_signal_quality(
        self,
        source_id: str,
        quality_score: float,
        domain: str = "",
        display_name: str = "",
    ) -> SourceReputation:
        """Record a signal quality measurement for a source.

        Updates signal_quality via exponential moving average.
        Does NOT affect empirical_accuracy or cross_modal_concordance.
        """
        rep = await self._get_or_create(source_id, domain, display_name)

        rep.signal_quality_samples += 1
        rep.signal_quality_sum += quality_score
        rep.last_seen = time.time()

        # Exponential moving average with increasing weight on observations
        alpha = min(0.9, rep.signal_quality_samples / 10.0)
        rep.signal_quality = (
            (1.0 - alpha) * 0.5  # Prior
            + alpha * (rep.signal_quality_sum / rep.signal_quality_samples)  # Observed
        )

        # Update composite reputation
        rep.reputation_score = rep.compute_composite_reputation()

        self._cache[source_id] = rep
        self._persist(rep)

        logger.debug(
            "Source %s signal_quality: %.3f (samples=%d)",
            source_id,
            rep.signal_quality,
            rep.signal_quality_samples,
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
            r for r in all_reps if r.total_claims >= self._config.min_claims_for_confidence
        ]

        qualified.sort(key=lambda r: r.reputation_score, reverse=True)
        return [(r.source_id, r.reputation_score) for r in qualified[:limit]]

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
            r
            for r in all_reps
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
