"""
Async Database Pool for SQLite.

Provides a thread-safe, asyncio-compatible connection pool for SQLite databases
used in episodic, procedural, and value memory subsystems.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import aiosqlite

logger = logging.getLogger(__name__)


class DatabasePool:
    """
    Semaphore-based async connection pool for SQLite.

    Ensures safe concurrent access in multi-worker environments while
    enabling PRAGMA optimizations.
    """

    def __init__(self, db_path: str, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self._semaphore = asyncio.Semaphore(max_connections)
        self._connections: list[aiosqlite.Connection] = []

    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new tuned SQLite connection."""
        from hbllm.persistence.sqlite_profiles import open_async_connection

        conn = await open_async_connection(self.db_path)
        conn.row_factory = aiosqlite.Row
        return conn

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """
        Acquire a connection from the pool.

        Yields:
            aiosqlite.Connection: An active database connection.
        """
        await self._semaphore.acquire()
        conn = None
        try:
            if self._connections:
                conn = self._connections.pop()
            else:
                conn = await self._create_connection()
            yield conn
        except Exception:
            if conn is not None:
                try:
                    await conn.close()
                except Exception as e:
                    logger.debug("[Pool] non-critical error: %s", e)
                conn = None
            raise
        finally:
            if conn is not None:
                self._connections.append(conn)
            self._semaphore.release()

    async def close_all(self) -> None:
        """Close all connections in the pool and checkpoint WAL."""
        if not self._connections:
            return

        try:
            # Force WAL truncation to prevent unbounded growth
            await self._connections[-1].execute("PRAGMA wal_checkpoint(TRUNCATE);")
            await self._connections[-1].commit()
            logger.info("WAL checkpointed and truncated for %s", self.db_path)
        except (sqlite3.Error, OSError) as e:
            logger.error("Failed to checkpoint WAL for %s: %s", self.db_path, e)

        for conn in self._connections:
            try:
                await conn.close()
            except (sqlite3.Error, OSError) as e:
                logger.error("Error closing connection for %s: %s", self.db_path, e)
        self._connections.clear()
        logger.info("Closed DatabasePool for %s", self.db_path)
