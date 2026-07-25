"""
Kernel Persistence Service — Centralized database pool management.

Provides unified database pool management, profile configuration, and backend
abstraction for SQLite (and future PostgreSQL / libSQL) across all cognitive subsystems.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from hbllm.persistence.sqlite_profiles import SQLiteProfile, open_connection

logger = logging.getLogger(__name__)


class StorePool:
    """Connection pool container for a named store."""

    def __init__(
        self,
        name: str,
        db_path: str | Path,
        profile: str | SQLiteProfile = "default",
        max_connections: int = 5,
    ) -> None:
        self.name = name
        self.db_path = Path(db_path)
        self.profile = profile
        self.max_connections = max_connections
        self._connections: list[aiosqlite.Connection] = []

    async def acquire(self) -> aiosqlite.Connection:
        """Create or reuse an tuned aiosqlite connection."""
        conn = await aiosqlite.connect(str(self.db_path))
        conn.row_factory = aiosqlite.Row

        # Apply profile PRAGMAs
        prof = self.profile if isinstance(self.profile, SQLiteProfile) else SQLiteProfile()
        await conn.execute(f"PRAGMA journal_mode={prof.journal_mode};")
        await conn.execute(f"PRAGMA synchronous={prof.synchronous};")
        await conn.execute(f"PRAGMA cache_size=-{prof.cache_size_kb};")
        await conn.execute(f"PRAGMA mmap_size={prof.mmap_size_mb * 1024 * 1024};")

        return conn


class PersistenceService:
    """
    Kernel-managed Database Service.

    Centralizes store registration, connection pooling, and lifecycle management.
    """

    def __init__(self, data_dir: str | Path = "data", backend: str = "sqlite") -> None:
        self.data_dir = Path(data_dir)
        self.backend = backend
        self._stores: dict[str, StorePool] = {}

    def register_store(
        self,
        name: str,
        db_name: str,
        profile: str = "default",
        max_connections: int = 5,
    ) -> StorePool:
        """Register a named database store during boot."""
        db_path = self.data_dir / db_name
        self.data_dir.mkdir(parents=True, exist_ok=True)

        pool = StorePool(
            name=name,
            db_path=db_path,
            profile=profile,
            max_connections=max_connections,
        )
        self._stores[name] = pool
        logger.info(
            "[PersistenceService] Registered store '%s' at %s (profile=%s)", name, db_path, profile
        )
        return pool

    def get_sync_connection(self, name: str) -> sqlite3.Connection:
        """Get an optimized synchronous SQLite connection for a registered store."""
        pool = self._stores.get(name)
        if pool is None:
            # Fallback for unregistered stores
            db_path = self.data_dir / f"{name}.db"
            return open_connection(db_path, profile=name)

        return open_connection(pool.db_path, profile=pool.profile)

    @asynccontextmanager
    async def acquire_async(self, name: str) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Acquire an async database connection for a registered store."""
        pool = self._stores.get(name)
        if pool is None:
            pool = self.register_store(name, f"{name}.db")

        conn = await pool.acquire()
        try:
            yield conn
        finally:
            await conn.close()

    async def close_all(self) -> None:
        """Close all registered database pools."""
        self._stores.clear()
        logger.info("[PersistenceService] All persistence pools closed")
