"""
Database Connection Pool Manager.

Provides asyncpg connection pooling for PostgreSQL.
Falls back to a warning/No-Op mode if `HBLLM_DATABASE_URL` is not set, allowing
local SQLite components to continue functioning as default if preferred.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    import asyncpg  # type: ignore[import-untyped]
    from pgvector.asyncpg import register_vector  # type: ignore[import-untyped]

    _HAS_ASYNCPG = True
except ImportError:
    _HAS_ASYNCPG = False


class DBPool:
    """Singleton asyncpg connection pool manager."""

    _pool: Any | None = None
    _db_url: str | None = None

    @classmethod
    async def get_pool(cls) -> Any | None:
        """Get the connection pool, initializing it if necessary."""
        if cls._pool is not None:
            return cls._pool

        if not _HAS_ASYNCPG:
            logger.warning("asyncpg or pgvector not installed. PostgreSQL persistence unavailable.")
            return None

        cls._db_url = os.getenv("HBLLM_DATABASE_URL")
        if not cls._db_url:
            # We don't error out, allowing SQLite fallbacks
            return None

        try:

            async def init_connection(conn: Any) -> None:
                await register_vector(conn)
                import json

                await conn.set_type_codec(
                    "json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

            cls._pool = await asyncpg.create_pool(
                dsn=cls._db_url,
                min_size=2,
                max_size=20,
                init=init_connection,
                timeout=10.0,
            )
            logger.info("PostgreSQL connection pool initialized.")
            return cls._pool
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            return None

    @classmethod
    async def close(cls) -> None:
        """Close the connection pool."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
            logger.info("PostgreSQL connection pool closed.")
