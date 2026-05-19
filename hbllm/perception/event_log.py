"""Event Log — Append-only SQLite truth ledger for perception events.

This is the absolute source of truth for the cognitive system.
It stores normalized events that have passed through the RealityEventBus
and EventNormalizer. By storing the log, we can rebuild the WorldStateEngine
probabilistic graph from scratch.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hbllm.perception.reality_bus import PerceptionEvent

logger = logging.getLogger(__name__)


class EventLog:
    """Append-only storage for normalized perception events."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "event_log.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS perception_events (
                    event_id TEXT PRIMARY KEY,
                    logical_clock INTEGER NOT NULL,
                    entity_id TEXT,
                    event_type TEXT,
                    sub_type TEXT,
                    modality TEXT,
                    origin TEXT,
                    confidence REAL,
                    source_trust REAL,
                    priority_hint INTEGER,
                    event_timestamp REAL,
                    ingest_timestamp REAL,
                    payload TEXT
                )
            """)
            # Indexes for fast replay and querying
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logical_clock "
                "ON perception_events(logical_clock)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entity_id "
                "ON perception_events(entity_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_timestamp "
                "ON perception_events(event_timestamp)"
            )

    def append(self, event: PerceptionEvent) -> None:
        """Append a normalized event to the log."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO perception_events
                       (event_id, logical_clock, entity_id, event_type, sub_type,
                        modality, origin, confidence, source_trust, priority_hint,
                        event_timestamp, ingest_timestamp, payload)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event.event_id,
                        event.logical_clock,
                        event.entity_id,
                        event.event_type,
                        event.sub_type,
                        event.modality.value,
                        event.origin.value,
                        event.confidence,
                        event.source_trust,
                        event.priority_hint,
                        event.event_timestamp,
                        event.ingest_timestamp,
                        json.dumps(event.payload),
                    ),
                )
        except sqlite3.IntegrityError:
            # Event already exists (likely due to deduplication handling elsewhere)
            logger.debug("Event %s already in log, ignoring.", event.event_id)
        except Exception as e:
            logger.error("Failed to append event to log: %s", e)

    def replay(self, start_clock: int = 0) -> Iterator[PerceptionEvent]:
        """Yield events sequentially for rebuilding state or time-travel debugging."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM perception_events WHERE logical_clock >= ? "
                "ORDER BY logical_clock ASC",
                (start_clock,),
            )
            for row in cursor:
                d = dict(row)
                d["payload"] = json.loads(d["payload"] or "{}")
                yield PerceptionEvent.from_dict(d)

    def get_latest_clock(self) -> int:
        """Get the highest logical clock currently stored."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(logical_clock) FROM perception_events"
            ).fetchone()
            return row[0] if row and row[0] is not None else 0
