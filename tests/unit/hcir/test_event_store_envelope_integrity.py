"""Tests for SQLiteEventStore full-envelope hashing, compaction anchoring, and synchronous=FULL (H-11)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from hbllm.hcir.stores import EventType, GraphEvent
from hbllm.hcir.stores.sqlite_event_store import SQLiteEventStore


def _append_sample_events(store: SQLiteEventStore, count: int) -> None:
    for seq in range(1, count + 1):
        store.append(
            GraphEvent(
                sequence=seq,
                event_type=EventType.NODE_ADDED,
                timestamp=1700000000.0 + seq,
                author=f"agent_{seq}",
                data={"node_id": f"n_{seq}", "tenant_id": "tenant_alpha"},
            )
        )


def test_full_envelope_tamper_breaks_integrity(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    store = SQLiteEventStore(db_path, synchronous_full=True, tenant_id="tenant_alpha")
    sync_mode = store._conn.execute("PRAGMA synchronous").fetchone()[0]
    assert sync_mode == 2  # SQLite FULL

    _append_sample_events(store, 4)
    assert store.verify_chain_integrity() is True

    for col, val in (
        ("author", "spoofed_author"),
        ("tenant_id", "tenant_beta"),
        ("timestamp", time.time() + 9999.0),
    ):
        tamper_conn = sqlite3.connect(str(db_path))
        try:
            with tamper_conn:
                tamper_conn.execute("BEGIN")
                tamper_conn.execute(f"UPDATE hcir_events SET {col} = ? WHERE sequence = 2", (val,))
            # Re-verify must detect metadata tampering
            assert store.verify_chain_integrity() is False
        finally:
            tamper_conn.close()
        # Restore via fresh store test for subsequent column checks
        db_path = tmp_path / f"events_{col}.db"
        store = SQLiteEventStore(db_path, tenant_id="tenant_alpha")
        _append_sample_events(store, 4)


def test_compaction_anchors_previous_hash(tmp_path: Path) -> None:
    db_path = tmp_path / "compact.db"
    store = SQLiteEventStore(db_path, tenant_id="tenant_comp")
    _append_sample_events(store, 8)

    head_before = store._conn.execute(
        "SELECT content_hash FROM hcir_events ORDER BY sequence ASC"
    ).fetchall()
    removed = store.compact(max_events=4)
    assert removed > 0
    assert store.verify_chain_integrity() is True

    first_row = store._conn.execute(
        "SELECT event_type, previous_hash, data_json FROM hcir_events ORDER BY id ASC LIMIT 1"
    ).fetchone()
    assert str(first_row["event_type"]) == str(EventType.SNAPSHOT_CREATED)
    assert first_row["previous_hash"] != ""
    assert first_row["previous_hash"] in {r["content_hash"] for r in head_before}

    # Tampering with previous_hash anchor breaks integrity
    store._conn.execute(
        "UPDATE hcir_events SET previous_hash = '' WHERE id = (SELECT MIN(id) FROM hcir_events)"
    )
    store._conn.commit()
    assert store.verify_chain_integrity() is False
