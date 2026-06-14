"""
Database Connection Pool Manager.

Provides asyncpg connection pooling for PostgreSQL.
Falls back to a warning/No-Op mode if `HBLLM_DATABASE_URL` is not set, allowing
local SQLite components to continue functioning as default if preferred.

Per-tenant connection quotas prevent noisy neighbor exhaustion of the shared pool.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

try:
    import asyncpg  # type: ignore[import-untyped]
    from pgvector.asyncpg import register_vector  # type: ignore[import-untyped]

    _HAS_ASYNCPG = True
except ImportError:
    _HAS_ASYNCPG = False


class DBPool:
    """Singleton asyncpg connection pool manager with per-tenant quotas."""

    _pool: Any | None = None
    _db_url: str | None = None
    _tenant_semaphores: dict[str, asyncio.Semaphore] = {}
    _max_per_tenant: int = 5  # Max concurrent connections per tenant

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

        cls._max_per_tenant = int(os.getenv("HBLLM_DB_MAX_PER_TENANT", "5"))

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
            logger.info(
                "PostgreSQL connection pool initialized (max_per_tenant=%d).",
                cls._max_per_tenant,
            )
            return cls._pool
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            return None

    @classmethod
    def _get_semaphore(cls, tenant_id: str) -> asyncio.Semaphore:
        """Get or create a semaphore for a tenant."""
        if tenant_id not in cls._tenant_semaphores:
            cls._tenant_semaphores[tenant_id] = asyncio.Semaphore(cls._max_per_tenant)
        return cls._tenant_semaphores[tenant_id]

    @classmethod
    @asynccontextmanager
    async def acquire(cls, tenant_id: str = "default"):
        """Acquire a pooled connection with per-tenant quota enforcement.

        Usage::

            async with DBPool.acquire(tenant_id="t1") as conn:
                rows = await conn.fetch("SELECT ...")

        Raises:
            asyncio.TimeoutError: If tenant quota is exhausted for 10s.
        """
        pool = await cls.get_pool()
        if pool is None:
            yield None
            return

        sem = cls._get_semaphore(tenant_id)
        try:
            # Wait up to 10s for a tenant slot
            await asyncio.wait_for(sem.acquire(), timeout=10.0)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning(
                "[DBPool] Tenant %s exceeded max %d concurrent connections",
                tenant_id,
                cls._max_per_tenant,
            )
            raise

        try:
            async with pool.acquire() as conn:
                yield conn
        finally:
            sem.release()

    @classmethod
    async def close(cls) -> None:
        """Close the connection pool."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
            cls._tenant_semaphores.clear()
            logger.info("PostgreSQL connection pool closed.")
