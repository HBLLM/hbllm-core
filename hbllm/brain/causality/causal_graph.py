"""Causal Cognition Graph.

This module provides the third layer of the epistemic model:
1. EventLog -> What happened (Truth)
2. WorldState -> What is happening (Belief)
3. CausalGraph -> Why it happened (Inference)

It stores probabilistic causal links between events and actions to prevent
causal hallucination and enable learning.
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


@dataclass
class CausalLink:
    """A probabilistic causal relationship between two nodes (events/tasks)."""

    link_id: str = field(default_factory=lambda: f"causal_{uuid.uuid4().hex[:12]}")
    source_id: str = ""  # The cause (e.g. task_id)
    target_id: str = ""  # The effect (e.g. event_id)
    probability: float = 0.0
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_id": self.link_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "probability": self.probability,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CausalLink:
        return cls(
            link_id=d.get("link_id", ""),
            source_id=d.get("source_id", ""),
            target_id=d.get("target_id", ""),
            probability=d.get("probability", 0.0),
            created_at=d.get("created_at", 0.0),
            metadata=d.get("metadata", {}),
        )


class CausalGraph:
    """SQLite-backed storage for causal inferences."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "causal_graph.db"
        self.hallucination_threshold = 0.5  # Ignore links weaker than this
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS causal_links (
                    link_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    probability REAL NOT NULL,
                    created_at REAL NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_causal_source ON causal_links(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_causal_target ON causal_links(target_id)")

    def calculate_causal_probability(
        self,
        temporal_distance_s: float,
        source_trust: float,
        event_match_score: float,
        state_alignment_score: float,
        intervention_signal_strength: float
    ) -> float:
        """Probabilistic scoring function for causality.

        Args:
            temporal_distance_s: Time between cause and effect.
            source_trust: Reliability of the sensor reporting the effect.
            event_match_score: Semantic match between action intent and event outcome.
            state_alignment_score: How well the event aligns with current WorldState.
            intervention_signal_strength: Was this actively triggered by the system? (0 to 1)
        """
        # Time decays causality rapidly (assume actions manifest within 120s for now)
        time_decay = max(0.0, 1.0 - (temporal_distance_s / 120.0))

        base_prob = (
            time_decay * 0.3 +
            intervention_signal_strength * 0.3 +
            event_match_score * 0.2 +
            source_trust * 0.1 +
            state_alignment_score * 0.1
        )
        return min(1.0, max(0.0, base_prob))

    def infer_and_store(
        self,
        source_id: str,
        target_id: str,
        temporal_distance_s: float,
        source_trust: float,
        event_match_score: float,
        state_alignment_score: float,
        intervention_signal_strength: float,
        metadata: dict[str, Any] | None = None
    ) -> CausalLink | None:
        """Calculate probability and store link if above threshold."""
        prob = self.calculate_causal_probability(
            temporal_distance_s,
            source_trust,
            event_match_score,
            state_alignment_score,
            intervention_signal_strength
        )

        if prob < self.hallucination_threshold:
            logger.debug("Discarded weak causal link %s -> %s (prob: %.2f)", source_id, target_id, prob)
            return None

        link = CausalLink(
            source_id=source_id,
            target_id=target_id,
            probability=prob,
            metadata=metadata or {}
        )

        self._insert(link)
        logger.info("Inferred causal link: %s -> %s (prob: %.2f)", source_id, target_id, prob)
        return link

    def _insert(self, link: CausalLink) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO causal_links
                   (link_id, source_id, target_id, probability, created_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    link.link_id,
                    link.source_id,
                    link.target_id,
                    link.probability,
                    link.created_at,
                    json.dumps(link.metadata)
                )
            )

    def get_causes(self, target_id: str) -> list[CausalLink]:
        """Find what caused a specific event/task."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM causal_links WHERE target_id = ? ORDER BY probability DESC",
                (target_id,)
            ).fetchall()
            return [self._row_to_link(r) for r in rows]

    def get_effects(self, source_id: str) -> list[CausalLink]:
        """Find what effects an event/task caused."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM causal_links WHERE source_id = ? ORDER BY probability DESC",
                (source_id,)
            ).fetchall()
            return [self._row_to_link(r) for r in rows]

    def _row_to_link(self, row: sqlite3.Row) -> CausalLink:
        return CausalLink(
            link_id=row["link_id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            probability=row["probability"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"] or "{}")
        )
