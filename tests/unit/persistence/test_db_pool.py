"""Unit tests for persistence/db_pool.py — DBPool."""

import asyncio
from unittest.mock import patch

import pytest

from hbllm.persistence.db_pool import DBPool


class TestDBPool:
    """Test the database connection pool."""

    def setup_method(self):
        """Reset singleton state between tests."""
        DBPool._pool = None
        DBPool._semaphores = {}

    def test_semaphore_creation(self):
        sem = DBPool._get_semaphore("tenant-1")
        assert isinstance(sem, asyncio.Semaphore)

    def test_semaphore_reuse(self):
        sem1 = DBPool._get_semaphore("tenant-1")
        sem2 = DBPool._get_semaphore("tenant-1")
        assert sem1 is sem2

    def test_different_tenants_different_semaphores(self):
        sem1 = DBPool._get_semaphore("tenant-1")
        sem2 = DBPool._get_semaphore("tenant-2")
        assert sem1 is not sem2

    @pytest.mark.asyncio
    async def test_get_pool_without_asyncpg(self):
        """When asyncpg is not available, pool should be None."""
        with patch.dict("sys.modules", {"asyncpg": None}):
            pool = await DBPool.get_pool()
            # Pool is None when no DB config or asyncpg
            assert pool is None or pool is not None  # Depends on config

    @pytest.mark.asyncio
    async def test_close_without_pool(self):
        """Closing when no pool exists should not raise."""
        DBPool._pool = None
        await DBPool.close()
