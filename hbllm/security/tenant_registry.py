"""
Tenant Registry — Central metadata database for tenant hierarchy relationships.

Maintains a local SQLite database at ~/.hbllm/tenants.db to map child
workspace tenants to parent developer/org tenants.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class TenantRegistry:
    """Manages SQLite-based registry for parent/child tenant mappings."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_dir = Path(os.path.expanduser("~/.hbllm"))
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = db_dir / "tenants.db"
        else:
            self.db_path = Path(db_path)

        self._init_db()

    def _init_db(self) -> None:
        """Create the tenants metadata table if not exists."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tenants (
                        id TEXT PRIMARY KEY,
                        parent_id TEXT,
                        name TEXT,
                        created_at REAL DEFAULT (strftime('%s', 'now'))
                    )
                    """
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error("Failed to initialize tenants database: %s", e)

    def register_tenant(
        self,
        tenant_id: str,
        parent_id: str | None = None,
        name: str | None = None,
    ) -> None:
        """Register or update a tenant and its optional parent relationship."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO tenants (id, parent_id, name)
                    VALUES (?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        parent_id = COALESCE(?, parent_id),
                        name = COALESCE(?, name)
                    """,
                    (tenant_id, parent_id, name, parent_id, name),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error("Failed to register tenant %s: %s", tenant_id, e)

    def get_parent_id(self, tenant_id: str) -> str | None:
        """Retrieve the parent ID for a given child tenant."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT parent_id FROM tenants WHERE id = ?", (tenant_id,))
                row = cursor.fetchone()
                return str(row[0]) if row and row[0] is not None else None
        except sqlite3.Error as e:
            logger.error("Failed to look up parent for tenant %s: %s", tenant_id, e)
            return None

    def get_children_ids(self, parent_id: str) -> list[str]:
        """Retrieve all registered child tenant IDs for a parent tenant."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT id FROM tenants WHERE parent_id = ?", (parent_id,))
                return [str(row[0]) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error("Failed to look up children for parent tenant %s: %s", parent_id, e)
            return []


_registry: TenantRegistry | None = None


def get_tenant_registry() -> TenantRegistry:
    """Retrieve the global instance of the TenantRegistry."""
    global _registry
    if _registry is None:
        _registry = TenantRegistry()
    return _registry
