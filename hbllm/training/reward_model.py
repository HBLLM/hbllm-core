"""
Reward Model — learns preference scores from interaction feedback.

Supports:
- Pairwise preference learning (chosen vs rejected)
- Scalar reward prediction
- Integration with DPO/PPO training loops
- Online reward updates from user feedback
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RewardSignal:
    """A single reward observation from user feedback."""
    query: str
    response: str
    reward: float  # -1.0 to 1.0
    source: str = "explicit"  # explicit | implicit | synthetic
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PreferencePair:
    """A preference pair for RLHF training."""
    query: str
    chosen: str
    rejected: str
    margin: float = 1.0  # confidence margin
    source: str = "human"
    timestamp: float = field(default_factory=time.time)


class RewardModel:
    """
    Learns reward scores from user interactions and feedback.

    Architecture:
    1. Collects explicit feedback (thumbs up/down, ratings)
    2. Infers implicit feedback (regeneration = negative, copy = positive)
    3. Trains a reward scorer for response quality ranking
    4. Feeds preference pairs to DPO/PPO optimizers
    """

    def __init__(self, data_dir: str = "data"):
        self._db_path = Path(data_dir) / "reward_model.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._reward_weights: dict[str, float] = {
            "relevance": 0.3,
            "helpfulness": 0.3,
            "safety": 0.2,
            "coherence": 0.2,
        }

    def _init_db(self):
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rewards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    reward REAL NOT NULL,
                    source TEXT DEFAULT 'explicit',
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    chosen TEXT NOT NULL,
                    rejected TEXT NOT NULL,
                    margin REAL DEFAULT 1.0,
                    source TEXT DEFAULT 'human',
                    created_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rewards_time ON rewards(created_at)"
            )

    # ─── Feedback Collection ─────────────────────────────────────────

    def record_reward(self, signal: RewardSignal) -> None:
        """Record explicit or implicit reward signal."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO rewards (query, response, reward, source, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (signal.query, signal.response, signal.reward,
                 signal.source, json.dumps(signal.metadata), signal.timestamp),
            )
        logger.debug("Recorded reward=%.2f source=%s", signal.reward, signal.source)

    def record_preference(self, pair: PreferencePair) -> None:
        """Record a preference pair for RLHF training."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO preferences (query, chosen, rejected, margin, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (pair.query, pair.chosen, pair.rejected,
                 pair.margin, pair.source, pair.timestamp),
            )

    # ─── Implicit Feedback Inference ─────────────────────────────────

    def infer_implicit_reward(
        self, query: str, response: str, signals: dict[str, Any],
    ) -> float:
        """
        Infer reward from implicit user behavior signals.

        Signals:
        - regenerated: user asked for regeneration (negative)
        - copied: user copied the response (positive)
        - follow_up: user continued conversation (positive)
        - time_on_response: reading time in seconds
        - thumbs: explicit thumbs up/down
        """
        reward = 0.0

        if signals.get("thumbs") == "up":
            reward += 1.0
        elif signals.get("thumbs") == "down":
            reward -= 1.0
        elif signals.get("regenerated"):
            reward -= 0.5
        elif signals.get("copied"):
            reward += 0.7
        elif signals.get("follow_up"):
            reward += 0.3

        # Time-based: quick dismissal = bad, reading = good
        read_time = signals.get("time_on_response", 0)
        if read_time > 5:
            reward += min(0.3, read_time / 60)

        reward = max(-1.0, min(1.0, reward))

        self.record_reward(RewardSignal(
            query=query, response=response, reward=reward,
            source="implicit", metadata=signals,
        ))
        return reward

    # ─── Reward Scoring ──────────────────────────────────────────────

    def score_response(self, query: str, response: str) -> float:
        """
        Score a response quality based on learned patterns.

        Uses heuristic features + historical reward statistics.
        """
        score = 0.5  # neutral baseline

        # Length heuristic: not too short, not too long
        resp_len = len(response.split())
        if resp_len < 5:
            score -= 0.2
        elif resp_len > 500:
            score -= 0.1
        else:
            score += 0.1

        # Query-response relevance (word overlap)
        query_words = set(query.lower().split())
        resp_words = set(response.lower().split())
        if query_words:
            overlap = len(query_words & resp_words) / len(query_words)
            score += overlap * 0.2

        # Historical average for similar queries
        historical = self._get_historical_reward(query)
        if historical is not None:
            score = score * 0.6 + historical * 0.4

        return max(0.0, min(1.0, score))

    def _get_historical_reward(self, query: str) -> float | None:
        """Get average historical reward for similar queries."""
        keywords = query.lower().split()[:3]
        if not keywords:
            return None
        pattern = f"%{keywords[0]}%"
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT AVG(reward) FROM rewards WHERE query LIKE ? "
                "ORDER BY created_at DESC LIMIT 100",
                (pattern,),
            ).fetchone()
        return row[0] if row and row[0] is not None else None

    # ─── Export for Training ─────────────────────────────────────────

    def export_preferences(self, min_margin: float = 0.5, limit: int = 10000) -> list[dict]:
        """Export preference pairs for DPO/RLHF training."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT query, chosen, rejected, margin FROM preferences "
                "WHERE margin >= ? ORDER BY created_at DESC LIMIT ?",
                (min_margin, limit),
            ).fetchall()
        return [
            {"query": r[0], "chosen": r[1], "rejected": r[2], "margin": r[3]}
            for r in rows
        ]

    def export_rewards(self, min_reward: float = 0.0, limit: int = 10000) -> list[dict]:
        """Export high-reward interactions for SFT training."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT query, response, reward FROM rewards "
                "WHERE reward >= ? ORDER BY reward DESC LIMIT ?",
                (min_reward, limit),
            ).fetchall()
        return [
            {"instruction": r[0], "response": r[1], "reward": r[2]}
            for r in rows
        ]

    def stats(self) -> dict[str, Any]:
        """Return reward model statistics."""
        with sqlite3.connect(str(self._db_path)) as conn:
            reward_count = conn.execute("SELECT COUNT(*) FROM rewards").fetchone()[0]
            pref_count = conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
            avg_reward = conn.execute("SELECT AVG(reward) FROM rewards").fetchone()[0]
        return {
            "total_rewards": reward_count,
            "total_preferences": pref_count,
            "avg_reward": round(avg_reward or 0, 3),
        }
