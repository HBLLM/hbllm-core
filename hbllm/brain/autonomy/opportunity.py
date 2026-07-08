"""Proactive Opportunity Framework Models.

Defines the first-class Opportunity object, aging policies,
and the history persistence manager.
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
class Opportunity:
    """First-class representation of a proactive opportunity.

    Standardizes how the system communicates candidates for autonomous behavior
    or proactive conversation.
    """

    id: str
    source: str
    category: str
    priority: float
    urgency: float
    confidence: float
    created_at: float
    expires_at: float | None = None
    reason: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    suggested_actions: list[str] = field(default_factory=list)

    # Aging policies
    aging_strategy: str = "none"  # "escalate", "decay", "none"
    aging_rate: float = 0.0  # priority shift rate per second

    def update_priority(self, now: float) -> float:
        """Apply the aging/decay policies to recalculate the priority.

        Args:
            now: The current epoch timestamp.

        Returns:
            The newly computed priority score bounded between 0.0 and 1.0.
        """
        if self.expires_at is not None and now >= self.expires_at:
            self.priority = 0.0
            return 0.0

        elapsed = now - self.created_at
        if elapsed <= 0:
            return self.priority

        if self.aging_strategy == "escalate":
            self.priority = min(1.0, self.priority + (self.aging_rate * elapsed))
        elif self.aging_strategy == "decay":
            self.priority = max(0.0, self.priority - (self.aging_rate * elapsed))

        return self.priority


class OpportunityHistory:
    """SQLite-backed history store to track and audit proactive opportunities."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS opportunity_history (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    category TEXT,
                    priority REAL,
                    urgency REAL,
                    confidence REAL,
                    created_at REAL,
                    expires_at REAL,
                    reason TEXT,
                    context TEXT,
                    suggested_actions TEXT,
                    status TEXT,
                    updated_at REAL
                )
                """
            )

    def log_opportunity(self, opp: Opportunity, status: str) -> None:
        """Log or update an opportunity's state in history.

        Args:
            opp: The Opportunity to log.
            status: Status of the opportunity ("created", "dismissed", "executed", "expired").
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO opportunity_history
                (id, source, category, priority, urgency, confidence, created_at, expires_at, reason, context, suggested_actions, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    opp.id,
                    opp.source,
                    opp.category,
                    opp.priority,
                    opp.urgency,
                    opp.confidence,
                    opp.created_at,
                    opp.expires_at,
                    opp.reason,
                    json.dumps(opp.context),
                    json.dumps(opp.suggested_actions),
                    status,
                    time.time(),
                ),
            )

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch history of opportunities."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM opportunity_history ORDER BY updated_at DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
