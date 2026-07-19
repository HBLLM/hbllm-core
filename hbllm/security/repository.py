"""
TenantSQLiteRepository — Base class for tenant-safe SQLite database operations.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from hbllm.security.tenant_guard import (
    _ctx_system,
    get_current_tenant,
)

logger = logging.getLogger(__name__)

PROTECTED_TABLES = {"beliefs", "persistent_contradictions", "goals", "evaluations"}


class TenantSQLiteRepository:
    """
    Base repository class that enforces tenant isolation policies at the connection layer.
    No protected table may be accessed except through this repository.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    def current_tenant(self) -> str | None:
        """Centralized helper to resolve active tenant."""
        return get_current_tenant()

    def _validate_query_policy(
        self,
        sql: str,
        params: tuple[Any, ...] | list[Any],
        required_capability: str | None = None,
    ) -> str | None:
        """
        Validate that the query adheres to tenant isolation policy.
        Returns the resolved tenant ID, or None if in SystemContext.
        """
        # 1. SystemContext checks
        if _ctx_system.get(False):
            if required_capability:
                from hbllm.security.tenant_guard import require_capability

                require_capability(required_capability)
            return None

        # 2. Check if query accesses any protected tables
        sql_lower = sql.lower()
        touches_protected = any(table in sql_lower for table in PROTECTED_TABLES)
        if not touches_protected:
            return None

        # Resolve guard mode
        from hbllm.security.tenant_guard import TenantGuardMode, _get_guard_mode

        mode = _get_guard_mode()
        if mode == TenantGuardMode.OFF:
            return None

        # 3. Enforce context and parameter presence
        tenant_id = self.current_tenant()
        if tenant_id is None:
            msg = f"Access denied: Attempted to query protected table in SQL: '{sql}' without active TenantContext."
            if mode == TenantGuardMode.STRICT:
                from hbllm.security.tenant_guard import _log_and_raise_violation

                _log_and_raise_violation(
                    message=msg,
                    tenant_id="None",
                    operation="query",
                    reason="missing_context",
                )
            else:
                logger.warning("TENANT_GUARD_WARN: %s", msg)
            return None

        assert tenant_id is not None

        # 4. Verify query string contains tenant_id
        if "tenant_id" not in sql_lower:
            msg = f"Access denied: Query against protected table is missing 'tenant_id' filter clause. SQL: '{sql}'"
            if mode == TenantGuardMode.STRICT:
                from hbllm.security.tenant_guard import _log_and_raise_violation

                _log_and_raise_violation(
                    message=msg,
                    tenant_id=tenant_id,
                    operation="query",
                    reason="missing_tenant_filter",
                )
            else:
                logger.warning("TENANT_GUARD_WARN: %s", msg)
                return tenant_id

        # 5. Verify the active tenant_id is present in the parameters
        # Normalize params to strings for comparison
        str_params = [str(p) for p in params]
        if tenant_id not in str_params:
            msg = f"Access denied: Active tenant '{tenant_id}' is not present in query parameters {params}. SQL: '{sql}'"
            if mode == TenantGuardMode.STRICT:
                from hbllm.security.tenant_guard import _log_and_raise_violation

                _log_and_raise_violation(
                    message=msg,
                    tenant_id=tenant_id,
                    operation="query",
                    reason="tenant_parameter_mismatch",
                )
            else:
                logger.warning("TENANT_GUARD_WARN: %s", msg)
                return tenant_id

        return tenant_id

    def execute_tenant(
        self,
        conn: sqlite3.Connection,
        sql: str,
        params: tuple[Any, ...] | list[Any] = (),
        required_capability: str | None = None,
    ) -> sqlite3.Cursor:
        """Execute a SQL query, enforcing tenant isolation policies."""
        self._validate_query_policy(sql, params, required_capability)
        return conn.execute(sql, params)

    def executemany_tenant(
        self,
        conn: sqlite3.Connection,
        sql: str,
        params_seq: list[tuple[Any, ...]] | list[list[Any]],
        required_capability: str | None = None,
    ) -> sqlite3.Cursor:
        """Execute a SQL batch query, validating every parameter row in the sequence."""
        for params in params_seq:
            self._validate_query_policy(sql, params, required_capability)
        return conn.executemany(sql, params_seq)
