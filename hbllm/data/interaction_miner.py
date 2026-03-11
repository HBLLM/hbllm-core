"""
Interaction Miner — automatically extracts training datasets from usage.

Pipeline:
1. Collects query-response pairs from live traffic
2. Filters for high-quality interactions (via reward signals)
3. Generates instruction-following datasets
4. Creates preference pairs from regeneration events
5. Synthesizes negative examples for safety training

This is the "dataset flywheel" — every user interaction improves the model.
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


@dataclass
class MinedSample:
    """A training sample extracted from interaction data."""
    instruction: str
    response: str
    quality_score: float
    source: str = "mined"
    data_type: str = "sft"  # sft | dpo | safety
    metadata: dict[str, Any] = field(default_factory=dict)


class InteractionMiner:
    """
    Mines training data from user interactions.

    Strategies:
    1. High-reward SFT: queries with thumbs-up or continued conversations
    2. Preference pairs: regeneration events → chosen/rejected
    3. Hard negatives: failed queries → safety training
    4. Knowledge extraction: factual Q&A → knowledge base
    """

    def __init__(self, data_dir: str = "data"):
        self._db_path = Path(data_dir) / "interaction_mine.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT DEFAULT '',
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    reward REAL DEFAULT 0.0,
                    regenerated INTEGER DEFAULT 0,
                    follow_up INTEGER DEFAULT 0,
                    tokens_used INTEGER DEFAULT 0,
                    model TEXT DEFAULT '',
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mined_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instruction TEXT NOT NULL,
                    response TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    data_type TEXT DEFAULT 'sft',
                    source TEXT DEFAULT 'mined',
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_reward "
                "ON interactions(reward DESC)"
            )

    # ─── Collection ──────────────────────────────────────────────────

    def record_interaction(
        self,
        query: str,
        response: str,
        reward: float = 0.0,
        regenerated: bool = False,
        follow_up: bool = False,
        tenant_id: str = "",
        tokens_used: int = 0,
        model: str = "",
    ) -> None:
        """Record a user interaction for future mining."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO interactions "
                "(tenant_id, query, response, reward, regenerated, follow_up, "
                "tokens_used, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (tenant_id, query, response, reward, int(regenerated),
                 int(follow_up), tokens_used, model, time.time()),
            )

    # ─── Mining Strategies ───────────────────────────────────────────

    def mine_sft_samples(
        self, min_reward: float = 0.5, min_length: int = 20, limit: int = 5000,
    ) -> list[MinedSample]:
        """
        Extract high-quality SFT training samples.

        Criteria: high reward + sufficient length + not regenerated.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute("""
                SELECT query, response, reward FROM interactions
                WHERE reward >= ? AND regenerated = 0
                AND LENGTH(response) > ?
                ORDER BY reward DESC LIMIT ?
            """, (min_reward, min_length, limit)).fetchall()

        samples = []
        for q, r, rew in rows:
            samples.append(MinedSample(
                instruction=q, response=r,
                quality_score=rew, data_type="sft",
            ))
        logger.info("Mined %d SFT samples (min_reward=%.2f)", len(samples), min_reward)
        return samples

    def mine_preference_pairs(self, limit: int = 2000) -> list[dict[str, str]]:
        """
        Extract preference pairs from regeneration events.

        When a user regenerates, the original = rejected, the follow-up = chosen.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            # Find queries that were regenerated (same query, different responses)
            rows = conn.execute("""
                SELECT a.query, b.response as chosen, a.response as rejected
                FROM interactions a
                JOIN interactions b ON a.query = b.query AND a.id != b.id
                WHERE a.regenerated = 1 AND b.regenerated = 0
                AND b.reward > a.reward
                ORDER BY b.reward DESC
                LIMIT ?
            """, (limit,)).fetchall()

        pairs = [
            {"query": r[0], "chosen": r[1], "rejected": r[2]}
            for r in rows
        ]
        logger.info("Mined %d preference pairs", len(pairs))
        return pairs

    def mine_hard_negatives(self, max_reward: float = -0.3, limit: int = 1000) -> list[MinedSample]:
        """
        Extract low-quality responses for safety/quality training.

        These become negative examples in contrastive learning.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute("""
                SELECT query, response, reward FROM interactions
                WHERE reward <= ?
                ORDER BY reward ASC LIMIT ?
            """, (max_reward, limit)).fetchall()

        return [
            MinedSample(
                instruction=q, response=r,
                quality_score=rew, data_type="safety",
                source="hard_negative",
            )
            for q, r, rew in rows
        ]

    def mine_knowledge(self, min_reward: float = 0.7, limit: int = 5000) -> list[dict[str, str]]:
        """
        Extract factual Q&A pairs for knowledge base enrichment.

        High-reward, non-regenerated Q&A pairs are likely factually accurate.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute("""
                SELECT query, response FROM interactions
                WHERE reward >= ? AND regenerated = 0
                AND query LIKE '%?%'
                ORDER BY reward DESC LIMIT ?
            """, (min_reward, limit)).fetchall()

        return [{"question": r[0], "answer": r[1]} for r in rows]

    # ─── Export ───────────────────────────────────────────────────────

    def export_dataset(
        self,
        output_path: str | None = None,
        min_reward: float = 0.3,
        format: str = "jsonl",
    ) -> list[dict]:
        """Export mined data as a training dataset."""
        samples = self.mine_sft_samples(min_reward=min_reward)
        dataset = [
            {"instruction": s.instruction, "output": s.response, "score": s.quality_score}
            for s in samples
        ]

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                for item in dataset:
                    f.write(json.dumps(item) + "\n")
            logger.info("Exported %d samples to %s", len(dataset), output_path)

        return dataset

    def stats(self) -> dict[str, Any]:
        """Return mining statistics."""
        with sqlite3.connect(str(self._db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            avg_reward = conn.execute("SELECT AVG(reward) FROM interactions").fetchone()[0]
            regen = conn.execute("SELECT COUNT(*) FROM interactions WHERE regenerated=1").fetchone()[0]
            mined = conn.execute("SELECT COUNT(*) FROM mined_samples").fetchone()[0]
        return {
            "total_interactions": total,
            "avg_reward": round(avg_reward or 0, 3),
            "regeneration_rate": round(regen / max(total, 1), 3),
            "mined_samples": mined,
        }
