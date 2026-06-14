"""
Role-Based Access Control (RBAC) — Fine-Grained Permission System.

Provides role definitions, permissions, and a guard for enforcing access
within a multi-tenant system. Roles are assigned per-tenant-per-user.

Roles:
  - owner:   Full access, can manage tenant and users
  - admin:   Administrative access, cannot delete tenant
  - member:  Standard access (chat, read/write memory)
  - viewer:  Read-only access
  - api_key: Programmatic access with scoped permissions

Usage::

    from hbllm.security.rbac import RBACGuard, Permission

    guard = RBACGuard()
    guard.assign_role("tenant_1", "user_42", Role.MEMBER)

    if guard.check("tenant_1", "user_42", Permission.CHAT_SEND):
        # allowed
        ...
"""

from __future__ import annotations

import logging
import sqlite3
import time
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Roles ─────────────────────────────────────────────────────────────


class Role(str, Enum):
    """System roles with hierarchical privileges."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"
    API_KEY = "api_key"


# ── Permissions ───────────────────────────────────────────────────────


class Permission(str, Enum):
    """Granular permission tokens."""

    # Chat
    CHAT_SEND = "chat:send"
    CHAT_READ = "chat:read"

    # Memory
    MEMORY_WRITE = "memory:write"
    MEMORY_READ = "memory:read"

    # Tools
    TOOL_EXECUTE = "tool:execute"
    TOOL_SHELL = "tool:shell"  # Elevated: host shell execution

    # Admin
    ADMIN_MANAGE_USERS = "admin:manage_users"
    ADMIN_VIEW_AUDIT = "admin:view_audit"
    ADMIN_MANAGE_CONFIG = "admin:manage_config"
    ADMIN_MANAGE_TENANT = "admin:manage_tenant"

    # Data
    DATA_EXPORT = "data:export"
    DATA_DELETE = "data:delete"


# ── Permission Matrix ────────────────────────────────────────────────

ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.OWNER: set(Permission),  # All permissions
    Role.ADMIN: {
        Permission.CHAT_SEND,
        Permission.CHAT_READ,
        Permission.MEMORY_WRITE,
        Permission.MEMORY_READ,
        Permission.TOOL_EXECUTE,
        Permission.ADMIN_MANAGE_USERS,
        Permission.ADMIN_VIEW_AUDIT,
        Permission.ADMIN_MANAGE_CONFIG,
        Permission.DATA_EXPORT,
    },
    Role.MEMBER: {
        Permission.CHAT_SEND,
        Permission.CHAT_READ,
        Permission.MEMORY_WRITE,
        Permission.MEMORY_READ,
        Permission.TOOL_EXECUTE,
    },
    Role.VIEWER: {
        Permission.CHAT_READ,
        Permission.MEMORY_READ,
    },
    Role.API_KEY: {
        Permission.CHAT_SEND,
        Permission.CHAT_READ,
        Permission.MEMORY_READ,
        Permission.TOOL_EXECUTE,
    },
}


# ── Guard ─────────────────────────────────────────────────────────────


class RBACGuard:
    """RBAC enforcement engine with SQLite-backed role assignments.

    Maintains a ``role_assignments`` table keyed by (tenant_id, user_id).
    The first user in a tenant is automatically assigned OWNER.
    """

    def __init__(self, db_path: str | Path = "data/rbac.db") -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS role_assignments (
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'member',
                assigned_by TEXT DEFAULT 'system',
                assigned_at REAL NOT NULL,
                PRIMARY KEY (tenant_id, user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_rbac_tenant
                ON role_assignments(tenant_id);
        """)
        self._conn.commit()

    # ── Role Management ──────────────────────────────────────────────

    def assign_role(
        self,
        tenant_id: str,
        user_id: str,
        role: Role,
        assigned_by: str = "system",
    ) -> None:
        """Assign a role to a user within a tenant."""
        self._conn.execute(
            """INSERT OR REPLACE INTO role_assignments
               (tenant_id, user_id, role, assigned_by, assigned_at)
               VALUES (?, ?, ?, ?, ?)""",
            (tenant_id, user_id, role.value, assigned_by, time.time()),
        )
        self._conn.commit()
        logger.info(
            "RBAC: assigned role=%s to user=%s in tenant=%s by=%s",
            role.value,
            user_id,
            tenant_id,
            assigned_by,
        )

    def get_role(self, tenant_id: str, user_id: str) -> Role:
        """Get the role of a user within a tenant. Returns VIEWER if unassigned."""
        row = self._conn.execute(
            "SELECT role FROM role_assignments WHERE tenant_id = ? AND user_id = ?",
            (tenant_id, user_id),
        ).fetchone()
        if row is None:
            return Role.VIEWER  # Default: read-only for unknown users
        try:
            return Role(row["role"])
        except ValueError:
            return Role.VIEWER

    def revoke_role(self, tenant_id: str, user_id: str) -> bool:
        """Remove a user's role assignment. Returns True if a row was deleted."""
        result = self._conn.execute(
            "DELETE FROM role_assignments WHERE tenant_id = ? AND user_id = ?",
            (tenant_id, user_id),
        )
        self._conn.commit()
        return result.rowcount > 0

    def list_users(self, tenant_id: str) -> list[dict[str, Any]]:
        """List all users with role assignments in a tenant."""
        rows = self._conn.execute(
            "SELECT user_id, role, assigned_by, assigned_at FROM role_assignments WHERE tenant_id = ?",
            (tenant_id,),
        ).fetchall()
        return [
            {
                "user_id": r["user_id"],
                "role": r["role"],
                "assigned_by": r["assigned_by"],
                "assigned_at": r["assigned_at"],
            }
            for r in rows
        ]

    # ── Permission Checking ──────────────────────────────────────────

    def check(
        self,
        tenant_id: str,
        user_id: str,
        permission: Permission,
    ) -> bool:
        """Check if a user has a specific permission in a tenant."""
        role = self.get_role(tenant_id, user_id)
        allowed_permissions = ROLE_PERMISSIONS.get(role, set())
        return permission in allowed_permissions

    def require(
        self,
        tenant_id: str,
        user_id: str,
        permission: Permission,
    ) -> None:
        """Raise PermissionError if the user lacks the required permission."""
        if not self.check(tenant_id, user_id, permission):
            role = self.get_role(tenant_id, user_id)
            raise PermissionError(
                f"User '{user_id}' with role '{role.value}' lacks permission "
                f"'{permission.value}' in tenant '{tenant_id}'"
            )

    def get_permissions(self, tenant_id: str, user_id: str) -> set[Permission]:
        """Get all permissions for a user in a tenant."""
        role = self.get_role(tenant_id, user_id)
        return ROLE_PERMISSIONS.get(role, set())

    # ── Cleanup ──────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()
