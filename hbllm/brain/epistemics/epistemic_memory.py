"""Epistemic Memory — universal memory of the system's reasoning history.

Stores things the graph and workspace don't track::

    Past hypotheses  — abandoned, falsified, superseded (and *why*)
    Failed predictions — and what we learned from them
    Evidence retractions — retracted or superseded evidence
    Confidence trajectories — belief confidence over time
    Calibration data — predicted confidence vs actual outcome
    Unknown history — knowledge gaps: open → resolved → abandoned

Any epistemic consumer can use this memory.  Research, debugging,
calibration, and counterfactual analysis all query the same store.

Architecture::

    EpistemicMemory
        ├── SQLite backend (epistemic_memory.db)
        │   ├── hypothesis_history
        │   ├── prediction_history
        │   ├── evidence_history
        │   ├── confidence_snapshots
        │   ├── unknown_history
        │   └── calibration_data
        ├── Record methods (called by EpistemicLoop)
        └── Query methods (used by CalibrationEngine, etc.)

Usage::

    memory = EpistemicMemory(data_dir="/path/to/workspace")
    await memory.record_prediction_result(outcome)
    accuracy = await memory.get_prediction_accuracy()
    trajectory = await memory.get_confidence_trajectory(belief_id)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from hbllm.brain.epistemics.interfaces import (
    ConfidenceSnapshot,
    PredictionOutcome,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════════════

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hypothesis_history (
    id              TEXT PRIMARY KEY,
    claim           TEXT NOT NULL DEFAULT '',
    lifecycle       TEXT NOT NULL DEFAULT '',
    outcome         TEXT NOT NULL DEFAULT '',
    reason          TEXT NOT NULL DEFAULT '',
    program_id      TEXT NOT NULL DEFAULT '',
    domain          TEXT NOT NULL DEFAULT '',
    novelty         REAL NOT NULL DEFAULT 0.0,
    testability     REAL NOT NULL DEFAULT 0.0,
    final_confidence REAL NOT NULL DEFAULT 0.0,
    created_at      REAL NOT NULL DEFAULT 0.0,
    archived_at     REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS prediction_history (
    id              TEXT PRIMARY KEY,
    claim           TEXT NOT NULL DEFAULT '',
    predicted       TEXT NOT NULL DEFAULT '',
    observed        TEXT NOT NULL DEFAULT '',
    correct         INTEGER,
    confidence_delta REAL NOT NULL DEFAULT 0.0,
    hypothesis_id   TEXT NOT NULL DEFAULT '',
    program_id      TEXT NOT NULL DEFAULT '',
    domain          TEXT NOT NULL DEFAULT '',
    predicted_confidence REAL NOT NULL DEFAULT 0.0,
    created_at      REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS evidence_history (
    id              TEXT PRIMARY KEY,
    quality_score   REAL NOT NULL DEFAULT 0.0,
    weight          REAL NOT NULL DEFAULT 0.0,
    bias_flags      TEXT NOT NULL DEFAULT '[]',
    status          TEXT NOT NULL DEFAULT 'active',
    reason          TEXT NOT NULL DEFAULT '',
    source_uri      TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL DEFAULT 0.0,
    retracted_at    REAL
);

CREATE TABLE IF NOT EXISTS confidence_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    belief_id       TEXT NOT NULL,
    derived_confidence REAL NOT NULL DEFAULT 0.0,
    evidence_quality REAL NOT NULL DEFAULT 0.0,
    evidence_quantity REAL NOT NULL DEFAULT 0.0,
    reproducibility  REAL NOT NULL DEFAULT 0.0,
    prediction_accuracy REAL NOT NULL DEFAULT 0.0,
    model_agreement  REAL NOT NULL DEFAULT 0.0,
    source_trust     REAL NOT NULL DEFAULT 0.0,
    timestamp       REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS unknown_history (
    id              TEXT PRIMARY KEY,
    question        TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'open',
    resolution      TEXT NOT NULL DEFAULT '',
    program_id      TEXT NOT NULL DEFAULT '',
    importance      REAL NOT NULL DEFAULT 0.0,
    created_at      REAL NOT NULL DEFAULT 0.0,
    resolved_at     REAL
);

CREATE TABLE IF NOT EXISTS calibration_data (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_confidence REAL NOT NULL DEFAULT 0.0,
    actual_outcome  INTEGER NOT NULL DEFAULT 0,
    domain          TEXT NOT NULL DEFAULT '',
    source          TEXT NOT NULL DEFAULT '',
    timestamp       REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_pred_hist_hyp ON prediction_history(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_pred_hist_domain ON prediction_history(domain);
CREATE INDEX IF NOT EXISTS idx_conf_snap_belief ON confidence_snapshots(belief_id);
CREATE INDEX IF NOT EXISTS idx_conf_snap_time ON confidence_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_calib_domain ON calibration_data(domain);
CREATE INDEX IF NOT EXISTS idx_hyp_hist_program ON hypothesis_history(program_id);
CREATE INDEX IF NOT EXISTS idx_unknown_hist_program ON unknown_history(program_id);
"""


class EpistemicMemory:
    """Universal epistemic memory — the history of reasoning.

    Implements the ``IEpistemicMemory`` protocol.

    SQLite-backed persistent store for epistemic history.  Separate
    from the workspace database — the workspace stores *current* program
    state; memory stores *historical* reasoning data.

    Any consumer can query this memory:
    - EpistemicCalibrationEngine uses it for self-assessment
    - CounterfactualReasoner uses it for historical comparison
    - The epistemic loop records to it at each step
    """

    def __init__(self, data_dir: str) -> None:
        """Initialize epistemic memory with SQLite backend.

        Args:
            data_dir: Directory for the ``epistemic_memory.db`` file.
        """
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "epistemic_memory.db"
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("EpistemicMemory initialized at %s", self._db_path)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ═══════════════════════════════════════════════════════════════════
    # Record Methods (called by EpistemicLoop at each step)
    # ═══════════════════════════════════════════════════════════════════

    async def record_hypothesis_outcome(
        self,
        hypothesis_id: str,
        outcome: str,
        reason: str,
        *,
        claim: str = "",
        lifecycle: str = "",
        program_id: str = "",
        domain: str = "",
        novelty: float = 0.0,
        testability: float = 0.0,
        final_confidence: float = 0.0,
        created_at: float = 0.0,
    ) -> None:
        """Record a hypothesis's final outcome.

        Args:
            hypothesis_id: The hypothesis node ID.
            outcome: One of 'falsified', 'promoted', 'abandoned', 'superseded'.
            reason: Human-readable explanation of why.
        """
        self._conn.execute(
            """INSERT OR REPLACE INTO hypothesis_history
               (id, claim, lifecycle, outcome, reason, program_id,
                domain, novelty, testability, final_confidence,
                created_at, archived_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                hypothesis_id,
                claim,
                lifecycle,
                outcome,
                reason,
                program_id,
                domain,
                novelty,
                testability,
                final_confidence,
                created_at,
                time.time(),
            ),
        )
        self._conn.commit()
        logger.debug(
            "Recorded hypothesis outcome: %s → %s",
            hypothesis_id,
            outcome,
        )

    async def record_prediction_result(
        self,
        outcome: PredictionOutcome,
        *,
        claim: str = "",
        program_id: str = "",
        domain: str = "",
        predicted_confidence: float = 0.0,
    ) -> None:
        """Record a prediction outcome for calibration analysis.

        Args:
            outcome: The PredictionOutcome from the PredictionTracker.
        """
        correct_int = None
        if outcome.correct is True:
            correct_int = 1
        elif outcome.correct is False:
            correct_int = 0

        self._conn.execute(
            """INSERT OR REPLACE INTO prediction_history
               (id, claim, predicted, observed, correct,
                confidence_delta, hypothesis_id, program_id,
                domain, predicted_confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                outcome.prediction_id,
                claim,
                outcome.predicted,
                outcome.observed,
                correct_int,
                outcome.confidence_delta,
                outcome.hypothesis_id,
                program_id,
                domain,
                predicted_confidence,
                outcome.timestamp,
            ),
        )
        self._conn.commit()

        # Also record calibration data point
        if outcome.correct is not None:
            self._conn.execute(
                """INSERT INTO calibration_data
                   (predicted_confidence, actual_outcome, domain, source, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    predicted_confidence,
                    1 if outcome.correct else 0,
                    domain,
                    "prediction",
                    time.time(),
                ),
            )
            self._conn.commit()

        logger.debug(
            "Recorded prediction result: %s (correct=%s)",
            outcome.prediction_id,
            outcome.correct,
        )

    async def record_evidence_retraction(
        self,
        evidence_id: str,
        reason: str,
        *,
        quality_score: float = 0.0,
        weight: float = 0.0,
        bias_flags: list[str] | None = None,
        source_uri: str = "",
    ) -> None:
        """Record that evidence was retracted or superseded.

        Args:
            evidence_id: The evidence node ID.
            reason: Why the evidence was retracted.
        """
        self._conn.execute(
            """INSERT OR REPLACE INTO evidence_history
               (id, quality_score, weight, bias_flags, status,
                reason, source_uri, created_at, retracted_at)
               VALUES (?, ?, ?, ?, 'retracted', ?, ?, ?, ?)""",
            (
                evidence_id,
                quality_score,
                weight,
                json.dumps(bias_flags or []),
                reason,
                source_uri,
                time.time(),
                time.time(),
            ),
        )
        self._conn.commit()
        logger.debug("Recorded evidence retraction: %s", evidence_id)

    async def snapshot_belief_confidence(
        self,
        belief_id: str,
        snapshot: ConfidenceSnapshot | None = None,
        *,
        derived_confidence: float = 0.0,
        evidence_quality: float = 0.0,
        evidence_quantity: float = 0.0,
        reproducibility: float = 0.0,
        prediction_accuracy: float = 0.0,
        model_agreement: float = 0.0,
        source_trust: float = 0.0,
    ) -> None:
        """Record a point-in-time confidence snapshot.

        Args:
            belief_id: The belief node ID.
            snapshot: Optional pre-built ConfidenceSnapshot.
        """
        if snapshot is not None:
            dc = snapshot.derived_confidence
            eq = snapshot.evidence_quality
            eqn = snapshot.evidence_quantity
            rep = snapshot.reproducibility
            pa = snapshot.prediction_accuracy
            ma = snapshot.model_agreement
            st = snapshot.source_trust
            ts = snapshot.timestamp
        else:
            dc = derived_confidence
            eq = evidence_quality
            eqn = evidence_quantity
            rep = reproducibility
            pa = prediction_accuracy
            ma = model_agreement
            st = source_trust
            ts = time.time()

        self._conn.execute(
            """INSERT INTO confidence_snapshots
               (belief_id, derived_confidence, evidence_quality,
                evidence_quantity, reproducibility, prediction_accuracy,
                model_agreement, source_trust, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (belief_id, dc, eq, eqn, rep, pa, ma, st, ts),
        )
        self._conn.commit()

    async def record_unknown_resolved(
        self,
        unknown_id: str,
        resolution: str,
        *,
        question: str = "",
        program_id: str = "",
        importance: float = 0.0,
        created_at: float = 0.0,
    ) -> None:
        """Record that a knowledge gap was resolved."""
        self._conn.execute(
            """INSERT OR REPLACE INTO unknown_history
               (id, question, status, resolution, program_id,
                importance, created_at, resolved_at)
               VALUES (?, ?, 'resolved', ?, ?, ?, ?, ?)""",
            (
                unknown_id,
                question,
                resolution,
                program_id,
                importance,
                created_at,
                time.time(),
            ),
        )
        self._conn.commit()

    async def record_unknown_abandoned(
        self,
        unknown_id: str,
        reason: str,
        *,
        question: str = "",
        program_id: str = "",
        importance: float = 0.0,
        created_at: float = 0.0,
    ) -> None:
        """Record that a knowledge gap was abandoned."""
        self._conn.execute(
            """INSERT OR REPLACE INTO unknown_history
               (id, question, status, resolution, program_id,
                importance, created_at, resolved_at)
               VALUES (?, ?, 'abandoned', ?, ?, ?, ?, ?)""",
            (
                unknown_id,
                question,
                reason,
                program_id,
                importance,
                created_at,
                time.time(),
            ),
        )
        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════════
    # Query Methods
    # ═══════════════════════════════════════════════════════════════════

    async def get_hypothesis_history(
        self,
        program_id: str = "",
        outcome: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get historical hypothesis records.

        Args:
            program_id: Filter by research program (empty = all).
            outcome: Filter by outcome (empty = all).
            limit: Maximum results.

        Returns:
            List of hypothesis history dicts.
        """
        query = "SELECT * FROM hypothesis_history WHERE 1=1"
        params: list[Any] = []

        if program_id:
            query += " AND program_id = ?"
            params.append(program_id)
        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)

        query += " ORDER BY archived_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    async def get_prediction_accuracy(
        self,
        hypothesis_id: str = "",
        domain: str = "",
    ) -> float:
        """Compute prediction accuracy over stored history.

        Args:
            hypothesis_id: Filter by hypothesis (empty = all).
            domain: Filter by domain (empty = all).

        Returns:
            Accuracy as a float [0.0, 1.0].  Returns 0.0 if no data.
        """
        query = """SELECT
                       COUNT(*) as total,
                       SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_count
                   FROM prediction_history
                   WHERE correct IS NOT NULL"""
        params: list[Any] = []

        if hypothesis_id:
            query += " AND hypothesis_id = ?"
            params.append(hypothesis_id)
        if domain:
            query += " AND domain = ?"
            params.append(domain)

        row = self._conn.execute(query, params).fetchone()
        total = row["total"] if row else 0
        correct = row["correct_count"] if row else 0

        if total == 0:
            return 0.0
        return correct / total

    async def get_similar_past_hypotheses(
        self,
        claim: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find past hypotheses with similar claims.

        Uses simple substring matching.  For production, this should
        use semantic similarity via the embedding system.

        Args:
            claim: The claim text to search for.
            limit: Maximum results.

        Returns:
            List of matching hypothesis history dicts.
        """
        # Simple LIKE matching — replace with semantic search in production
        query = """SELECT * FROM hypothesis_history
                   WHERE claim LIKE ?
                   ORDER BY archived_at DESC
                   LIMIT ?"""
        pattern = f"%{claim[:50]}%"
        rows = self._conn.execute(query, (pattern, limit)).fetchall()
        return [dict(row) for row in rows]

    async def get_confidence_trajectory(
        self,
        belief_id: str,
    ) -> list[ConfidenceSnapshot]:
        """Get the confidence trajectory for a belief over time.

        Args:
            belief_id: The belief node ID.

        Returns:
            Time-ordered list of ConfidenceSnapshots.
        """
        rows = self._conn.execute(
            """SELECT * FROM confidence_snapshots
               WHERE belief_id = ?
               ORDER BY timestamp ASC""",
            (belief_id,),
        ).fetchall()

        return [
            ConfidenceSnapshot(
                belief_id=row["belief_id"],
                derived_confidence=row["derived_confidence"],
                evidence_quality=row["evidence_quality"],
                evidence_quantity=row["evidence_quantity"],
                reproducibility=row["reproducibility"],
                prediction_accuracy=row["prediction_accuracy"],
                model_agreement=row["model_agreement"],
                source_trust=row["source_trust"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    async def get_calibration_data(
        self,
        domain: str = "",
    ) -> list[dict[str, Any]]:
        """Get raw calibration data for calibration curve analysis.

        Args:
            domain: Filter by domain (empty = all).

        Returns:
            List of dicts with 'predicted_confidence' and 'actual_outcome'.
        """
        query = "SELECT * FROM calibration_data WHERE 1=1"
        params: list[Any] = []

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        query += " ORDER BY timestamp DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    async def get_prediction_history(
        self,
        hypothesis_id: str = "",
        correct_only: bool | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get prediction history records.

        Args:
            hypothesis_id: Filter by hypothesis (empty = all).
            correct_only: None=all, True=correct only, False=incorrect only.
            limit: Maximum results.

        Returns:
            List of prediction history dicts.
        """
        query = "SELECT * FROM prediction_history WHERE 1=1"
        params: list[Any] = []

        if hypothesis_id:
            query += " AND hypothesis_id = ?"
            params.append(hypothesis_id)
        if correct_only is True:
            query += " AND correct = 1"
        elif correct_only is False:
            query += " AND correct = 0"

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    async def get_unknown_history(
        self,
        program_id: str = "",
        status: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get knowledge gap history records."""
        query = "SELECT * FROM unknown_history WHERE 1=1"
        params: list[Any] = []

        if program_id:
            query += " AND program_id = ?"
            params.append(program_id)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY resolved_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # ═══════════════════════════════════════════════════════════════════
    # Summary Statistics
    # ═══════════════════════════════════════════════════════════════════

    async def get_hypothesis_survival_rate(self) -> float:
        """Fraction of hypotheses that survived testing (promoted vs total)."""
        row = self._conn.execute(
            """SELECT
                   COUNT(*) as total,
                   SUM(CASE WHEN outcome = 'promoted' THEN 1 ELSE 0 END) as promoted
               FROM hypothesis_history""",
        ).fetchone()
        total = row["total"] if row else 0
        promoted = row["promoted"] if row else 0
        if total == 0:
            return 0.0
        return promoted / total

    async def get_falsification_rate(self) -> float:
        """Fraction of hypotheses that were actively falsified."""
        row = self._conn.execute(
            """SELECT
                   COUNT(*) as total,
                   SUM(CASE WHEN outcome = 'falsified' THEN 1 ELSE 0 END) as falsified
               FROM hypothesis_history""",
        ).fetchone()
        total = row["total"] if row else 0
        falsified = row["falsified"] if row else 0
        if total == 0:
            return 0.0
        return falsified / total

    async def get_total_counts(self) -> dict[str, int]:
        """Get counts of all stored records."""
        counts: dict[str, int] = {}
        for table in (
            "hypothesis_history",
            "prediction_history",
            "evidence_history",
            "confidence_snapshots",
            "unknown_history",
            "calibration_data",
        ):
            row = self._conn.execute(
                f"SELECT COUNT(*) as c FROM {table}",
            ).fetchone()
            counts[table] = row["c"] if row else 0
        return counts
