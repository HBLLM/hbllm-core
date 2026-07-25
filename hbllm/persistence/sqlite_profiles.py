"""
SQLite Profiles — Workload-specific database PRAGMA configurations.

Provides fine-tuned SQLite connection profiles for different database access
patterns (high-throughput semantic memory, knowledge graph traversal, event logs, scheduler).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SQLiteProfile:
    """Configuration profile for SQLite database connections."""

    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    cache_size_kb: int = 8_000  # 8MB default cache
    mmap_size_mb: int = 64  # 64MB default memory-mapped I/O
    temp_store: str = "MEMORY"
    busy_timeout_ms: int = 5_000


PROFILES: dict[str, SQLiteProfile] = {
    "semantic_memory": SQLiteProfile(mmap_size_mb=512, cache_size_kb=64_000),
    "knowledge_graph": SQLiteProfile(mmap_size_mb=1024, cache_size_kb=64_000),
    "event_log": SQLiteProfile(mmap_size_mb=32, cache_size_kb=4_000),
    "scheduler": SQLiteProfile(synchronous="NORMAL", cache_size_kb=2_000),
    "default": SQLiteProfile(),
}


def get_profile(name: str) -> SQLiteProfile:
    """Get a registered SQLite profile by name, falling back to default."""
    return PROFILES.get(name, PROFILES["default"])


def open_connection(
    db_path: str | Path,
    profile: str | SQLiteProfile = "default",
    timeout: float = 5.0,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """Open a SQLite connection configured with the specified profile PRAGMAs."""
    prof = get_profile(profile) if isinstance(profile, str) else profile

    path_str = str(db_path)
    conn = sqlite3.connect(path_str, timeout=timeout, check_same_thread=check_same_thread)

    conn.execute(f"PRAGMA journal_mode={prof.journal_mode}")
    conn.execute(f"PRAGMA synchronous={prof.synchronous}")
    conn.execute(f"PRAGMA cache_size=-{prof.cache_size_kb}")
    conn.execute(f"PRAGMA temp_store={prof.temp_store}")
    conn.execute(f"PRAGMA mmap_size={prof.mmap_size_mb * 1024 * 1024}")
    conn.execute(f"PRAGMA busy_timeout={prof.busy_timeout_ms}")

    return conn
