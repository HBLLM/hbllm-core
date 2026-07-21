"""
SQLite Event Store — persistent, hash-chained event log for HCIR.

Stores ``GraphEvent`` records in an append-only SQLite table with
content hashing for tamper detection and snapshot version bookmarks.

Schema::

    hcir_events
    ├── id              INTEGER PRIMARY KEY AUTOINCREMENT
    ├── sequence        INTEGER UNIQUE
    ├── event_type      TEXT NOT NULL
    ├── author          TEXT NOT NULL DEFAULT ''
    ├── tenant_id       TEXT NOT NULL DEFAULT 'default'
    ├── transaction_id  TEXT DEFAULT ''
    ├── timestamp       REAL NOT NULL
    ├── data_json       TEXT NOT NULL DEFAULT '{}'
    ├── content_hash    TEXT NOT NULL
    └── previous_hash   TEXT NOT NULL DEFAULT ''
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from hbllm.hcir.stores import (
    EventType,
    GraphEvent,
    IEventStore,
)

logger = logging.getLogger(__name__)


_CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS hcir_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence        INTEGER UNIQUE NOT NULL,
    event_type      TEXT    NOT NULL,
    author          TEXT    NOT NULL DEFAULT '',
    tenant_id       TEXT    NOT NULL DEFAULT 'default',
    transaction_id  TEXT    DEFAULT '',
    timestamp       REAL    NOT NULL,
    data_json       TEXT    NOT NULL DEFAULT '{}',
    content_hash    TEXT    NOT NULL,
    previous_hash   TEXT    NOT NULL DEFAULT ''
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_hcir_events_type
    ON hcir_events(event_type);
CREATE INDEX IF NOT EXISTS idx_hcir_events_tenant
    ON hcir_events(tenant_id);
CREATE INDEX IF NOT EXISTS idx_hcir_events_timestamp
    ON hcir_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_hcir_events_transaction
    ON hcir_events(transaction_id);
"""

_CREATE_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS hcir_snapshots (
    version         INTEGER PRIMARY KEY,
    event_sequence  INTEGER NOT NULL,
    timestamp       REAL    NOT NULL,
    node_count      INTEGER NOT NULL DEFAULT 0,
    edge_count      INTEGER NOT NULL DEFAULT 0,
    content_hash    TEXT    NOT NULL DEFAULT ''
);
"""

_CREATE_NODES_INDEX = """
CREATE TABLE IF NOT EXISTS hcir_nodes_index (
    node_id         TEXT    PRIMARY KEY,
    node_type       TEXT    NOT NULL,
    category        TEXT    NOT NULL DEFAULT '',
    tenant_id       TEXT    NOT NULL DEFAULT 'default',
    lifecycle       TEXT    NOT NULL DEFAULT 'created',
    created_at      REAL    NOT NULL,
    updated_at      REAL    NOT NULL,
    data_json       TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_hcir_nodes_type
    ON hcir_nodes_index(node_type);
CREATE INDEX IF NOT EXISTS idx_hcir_nodes_tenant
    ON hcir_nodes_index(tenant_id);
CREATE INDEX IF NOT EXISTS idx_hcir_nodes_lifecycle
    ON hcir_nodes_index(lifecycle);
"""

_CREATE_EDGES_INDEX = """
CREATE TABLE IF NOT EXISTS hcir_edges_index (
    edge_id         TEXT    PRIMARY KEY,
    edge_type       TEXT    NOT NULL,
    tenant_id       TEXT    NOT NULL DEFAULT 'default',
    sources_json    TEXT    NOT NULL DEFAULT '[]',
    targets_json    TEXT    NOT NULL DEFAULT '[]',
    created_at      REAL    NOT NULL,
    data_json       TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_hcir_edges_type
    ON hcir_edges_index(edge_type);
"""


def _compute_event_hash(
    sequence: int,
    event_type: str,
    data_json: str,
    previous_hash: str,
) -> str:
    """Compute a deterministic hash for an event in the chain."""
    payload = f"{sequence}:{event_type}:{data_json}:{previous_hash}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class SQLiteEventStore(IEventStore):
    """Persistent event store with hash-chained integrity.

    The event log is append-only. Each event's ``content_hash``
    includes the ``previous_hash``, forming an immutable chain
    for tamper detection.

    Also maintains materialized indexes for nodes and edges
    (cache, not source of truth).

    Usage::

        store = SQLiteEventStore("/path/to/hcir.db")
        store.append(GraphEvent(
            sequence=1,
            event_type=EventType.NODE_ADDED,
            timestamp=time.time(),
            author="planner",
            data={"node_id": "g1"},
        ))
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._last_hash: str = ""
        self._sequence: int = 0
        self._initialize()

    def _initialize(self) -> None:
        """Create all tables and indexes."""
        self._conn.executescript(
            _CREATE_EVENTS_TABLE
            + _CREATE_INDEXES
            + _CREATE_SNAPSHOTS_TABLE
            + _CREATE_NODES_INDEX
            + _CREATE_EDGES_INDEX
        )
        self._conn.commit()
        # Recover last hash and sequence
        row = self._conn.execute(
            "SELECT sequence, content_hash FROM hcir_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            self._sequence = row["sequence"]
            self._last_hash = row["content_hash"]

    # ── IEventStore interface ────────────────────────────────────────

    def append(self, event: GraphEvent) -> None:
        """Append an event with hash chaining."""
        data_json = json.dumps(event.data, default=str)
        content_hash = _compute_event_hash(
            event.sequence, event.event_type, data_json, self._last_hash
        )
        self._conn.execute(
            """
            INSERT INTO hcir_events
                (sequence, event_type, author, tenant_id, timestamp,
                 data_json, content_hash, previous_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.sequence,
                event.event_type,
                event.author,
                event.data.get("tenant_id", "default"),
                event.timestamp,
                data_json,
                content_hash,
                self._last_hash,
            ),
        )
        self._conn.commit()
        self._last_hash = content_hash
        self._sequence = max(self._sequence, event.sequence)

    def get_events(
        self,
        from_sequence: int = 0,
        to_sequence: int | None = None,
        event_types: list[EventType] | None = None,
    ) -> list[GraphEvent]:
        """Retrieve events in sequence order."""
        query = "SELECT * FROM hcir_events WHERE sequence >= ?"
        params: list[Any] = [from_sequence]

        if to_sequence is not None:
            query += " AND sequence <= ?"
            params.append(to_sequence)

        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            query += f" AND event_type IN ({placeholders})"
            params.extend(event_types)

        query += " ORDER BY sequence ASC"
        cursor = self._conn.execute(query, params)

        events = []
        for row in cursor:
            events.append(GraphEvent(
                sequence=row["sequence"],
                event_type=row["event_type"],
                timestamp=row["timestamp"],
                author=row["author"],
                data=json.loads(row["data_json"]),
            ))
        return events

    def latest_sequence(self) -> int:
        return self._sequence

    def clear(self) -> None:
        """Clear all data (testing only)."""
        self._conn.executescript("""
            DELETE FROM hcir_events;
            DELETE FROM hcir_snapshots;
            DELETE FROM hcir_nodes_index;
            DELETE FROM hcir_edges_index;
        """)
        self._conn.commit()
        self._last_hash = ""
        self._sequence = 0

    # ── Snapshot management ──────────────────────────────────────────

    def save_snapshot(
        self,
        version: int,
        node_count: int = 0,
        edge_count: int = 0,
    ) -> None:
        """Record a snapshot bookmark at the current event sequence."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO hcir_snapshots
                (version, event_sequence, timestamp, node_count, edge_count, content_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (version, self._sequence, time.time(), node_count, edge_count, self._last_hash),
        )
        self._conn.commit()

    def get_latest_snapshot_version(self) -> int:
        """Return the latest snapshot version, or 0 if none."""
        row = self._conn.execute(
            "SELECT MAX(version) as v FROM hcir_snapshots"
        ).fetchone()
        return row["v"] if row and row["v"] is not None else 0

    # ── Node/Edge Index (materialized cache) ─────────────────────────

    def index_node(
        self,
        node_id: str,
        node_type: str,
        category: str = "",
        tenant_id: str = "default",
        lifecycle: str = "created",
        data: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a node into the materialized index."""
        now = time.time()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO hcir_nodes_index
                (node_id, node_type, category, tenant_id, lifecycle,
                 created_at, updated_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_id, node_type, category, tenant_id, lifecycle,
                now, now,
                json.dumps(data or {}, default=str),
            ),
        )
        self._conn.commit()

    def remove_node_index(self, node_id: str) -> None:
        self._conn.execute("DELETE FROM hcir_nodes_index WHERE node_id = ?", (node_id,))
        self._conn.commit()

    def index_edge(
        self,
        edge_id: str,
        edge_type: str,
        sources: list[str],
        targets: list[str],
        tenant_id: str = "default",
        data: dict[str, Any] | None = None,
    ) -> None:
        """Upsert an edge into the materialized index."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO hcir_edges_index
                (edge_id, edge_type, tenant_id, sources_json, targets_json,
                 created_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge_id, edge_type, tenant_id,
                json.dumps(sources), json.dumps(targets),
                time.time(),
                json.dumps(data or {}, default=str),
            ),
        )
        self._conn.commit()

    def remove_edge_index(self, edge_id: str) -> None:
        self._conn.execute("DELETE FROM hcir_edges_index WHERE edge_id = ?", (edge_id,))
        self._conn.commit()

    # ── Integrity verification ───────────────────────────────────────

    def verify_chain_integrity(self) -> bool:
        """Verify the hash chain hasn't been tampered with.

        Returns True if all event hashes are consistent.
        """
        cursor = self._conn.execute(
            "SELECT sequence, event_type, data_json, content_hash, previous_hash "
            "FROM hcir_events ORDER BY id ASC"
        )
        prev_hash = ""
        for row in cursor:
            expected = _compute_event_hash(
                row["sequence"], row["event_type"],
                row["data_json"], row["previous_hash"],
            )
            if expected != row["content_hash"]:
                logger.error(
                    "Hash chain broken at sequence %d: expected %s, got %s",
                    row["sequence"], expected, row["content_hash"],
                )
                return False
            if row["previous_hash"] != prev_hash:
                logger.error(
                    "Previous hash mismatch at sequence %d", row["sequence"]
                )
                return False
            prev_hash = row["content_hash"]
        return True

    @property
    def event_count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM hcir_events")
        return cursor.fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
