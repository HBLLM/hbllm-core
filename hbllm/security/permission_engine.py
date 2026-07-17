"""
Permission Engine — Mobile-style sandbox scoping for plugins and packages.

Enforces a capability-based permission model where plugins must declare
what they need, users can review and approve, and the engine denies
undeclared access at runtime.

Design principles (modeled after iOS/Android permissions):
    1. Plugins declare required permissions in their manifest.
    2. Users review and approve during installation.
    3. Permissions can be revoked at any time.
    4. Undeclared access is denied by default.
    5. Audit trail logs all permission checks.

Permission scopes mirror ``hbllm.hbpkg.manifest.PermissionScope``.

Architecture::

    Plugin attempts action (e.g., shell command)
        ↓
    PermissionEngine.check("my-plugin", "shell")
        ↓
    Policy lookup: granted? denied? ask-user?
        ↓
    Result: Allow / Deny / Prompt

Usage::

    from hbllm.security.permission_engine import PermissionEngine

    engine = PermissionEngine(data_dir="data")
    await engine.init_db()

    # Grant permissions during installation
    await engine.grant("devops-agent", ["shell", "internet", "read_files"])

    # Runtime check
    if await engine.check("devops-agent", "shell"):
        # Execute shell command
        ...

    # Revoke a permission
    await engine.revoke("devops-agent", "shell")

    # Audit: what does this plugin have access to?
    perms = await engine.list_grants("devops-agent")
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Grant Status
# ═══════════════════════════════════════════════════════════════════════════


class GrantStatus(StrEnum):
    """Status of a permission grant."""

    GRANTED = "granted"  # Explicitly allowed
    DENIED = "denied"  # Explicitly denied
    ASK = "ask"  # Prompt user each time
    UNSET = "unset"  # No policy — depends on default


class DefaultPolicy(StrEnum):
    """Default policy when no explicit grant exists."""

    DENY = "deny"  # Safe default — deny undeclared access
    ALLOW = "allow"  # Permissive — allow unless denied (dev mode)
    ASK = "ask"  # Interactive — prompt user


# ═══════════════════════════════════════════════════════════════════════════
# Permission Grant Record
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PermissionGrant:
    """A single permission grant or denial."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    principal: str = ""  # Plugin name or package name
    scope: str = ""  # Permission scope (e.g., "shell", "internet")
    status: GrantStatus = GrantStatus.GRANTED
    granted_by: str = "system"  # Who granted (user, system, auto)
    reason: str = ""
    granted_at: float = field(default_factory=time.time)
    expires_at: float = 0.0  # 0 = never
    tenant_id: str = "default"

    @property
    def is_expired(self) -> bool:
        if self.expires_at == 0:
            return False
        return time.time() > self.expires_at


# ═══════════════════════════════════════════════════════════════════════════
# Audit Entry
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PermissionAuditEntry:
    """An audit log entry for a permission check."""

    principal: str
    scope: str
    result: str  # "allowed", "denied"
    checked_at: float = field(default_factory=time.time)
    context: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Permission Engine
# ═══════════════════════════════════════════════════════════════════════════


class PermissionEngine:
    """Enforces capability-based permission scoping for plugins.

    Plugins declare required permissions; users approve/deny them.
    The engine checks every privileged action at runtime.

    Thread-safe for concurrent reads. Write operations (grant/revoke)
    serialize through SQLite transactions.
    """

    def __init__(
        self,
        data_dir: str = "data",
        default_policy: DefaultPolicy = DefaultPolicy.DENY,
    ) -> None:
        self._db_path = Path(data_dir) / "permissions.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._default_policy = default_policy

        # In-memory cache for fast lookups
        self._cache: dict[tuple[str, str], GrantStatus] = {}

        # Audit ring buffer (last 200 checks)
        self._audit: list[PermissionAuditEntry] = []

    async def init_db(self) -> None:
        """Create the permissions tables."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS permission_grants (
                    id TEXT PRIMARY KEY,
                    principal TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'granted',
                    granted_by TEXT DEFAULT 'system',
                    reason TEXT DEFAULT '',
                    granted_at REAL NOT NULL,
                    expires_at REAL DEFAULT 0,
                    tenant_id TEXT DEFAULT 'default',
                    UNIQUE(principal, scope, tenant_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_perm_principal
                ON permission_grants(principal, tenant_id)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS permission_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    principal TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    result TEXT NOT NULL,
                    checked_at REAL NOT NULL,
                    context TEXT DEFAULT ''
                )
            """)
            conn.commit()

        # Warm cache
        await self._warm_cache()
        logger.info("PermissionEngine initialized (policy=%s)", self._default_policy)

    # ── Core API ─────────────────────────────────────────────────────────

    async def check(
        self,
        principal: str,
        scope: str,
        *,
        tenant_id: str = "default",
        context: str = "",
    ) -> bool:
        """Check if a principal has a permission.

        Args:
            principal: Plugin or package name.
            scope: Permission scope to check.
            tenant_id: Tenant scope.
            context: Optional context for audit logging.

        Returns:
            True if the action is allowed.
        """
        # Check cache first
        cache_key = (principal, scope)
        cached = self._cache.get(cache_key)

        if cached is not None:
            allowed = cached == GrantStatus.GRANTED
        else:
            # Check DB
            grant = await self._get_grant(principal, scope, tenant_id)
            if grant is None:
                # No explicit grant — apply default policy
                allowed = self._default_policy == DefaultPolicy.ALLOW
            elif grant.is_expired:
                allowed = False
            else:
                allowed = grant.status == GrantStatus.GRANTED
                self._cache[cache_key] = grant.status

        # Audit
        result = "allowed" if allowed else "denied"
        self._record_audit(principal, scope, result, context)

        if not allowed:
            logger.debug(
                "Permission DENIED: %s → %s (context=%s)",
                principal,
                scope,
                context,
            )

        return allowed

    def is_allowed(self, principal: str, scope: str) -> bool:
        """Synchronous check (uses cache only).

        For use in hot paths where async overhead is unacceptable.
        """
        cache_key = (principal, scope)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached == GrantStatus.GRANTED
        return self._default_policy == DefaultPolicy.ALLOW

    # ── Grant / Revoke ───────────────────────────────────────────────────

    async def grant(
        self,
        principal: str,
        scopes: list[str],
        *,
        granted_by: str = "user",
        reason: str = "",
        tenant_id: str = "default",
        ttl_seconds: float = 0,
    ) -> int:
        """Grant permissions to a principal.

        Args:
            principal: Plugin or package name.
            scopes: List of permission scopes to grant.
            granted_by: Who is granting (user, system, auto).
            reason: Why the grant was made.
            tenant_id: Tenant scope.
            ttl_seconds: Time-to-live (0 = forever).

        Returns:
            Number of permissions granted.
        """
        now = time.time()
        expires = now + ttl_seconds if ttl_seconds > 0 else 0

        count = 0
        with sqlite3.connect(str(self._db_path)) as conn:
            for scope in scopes:
                grant_id = str(uuid.uuid4())[:12]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO permission_grants
                    (id, principal, scope, status, granted_by, reason,
                     granted_at, expires_at, tenant_id)
                    VALUES (?, ?, ?, 'granted', ?, ?, ?, ?, ?)
                    """,
                    (grant_id, principal, scope, granted_by, reason, now, expires, tenant_id),
                )
                self._cache[(principal, scope)] = GrantStatus.GRANTED
                count += 1
            conn.commit()

        logger.info(
            "Granted %d permissions to '%s': %s",
            count,
            principal,
            scopes,
        )
        return count

    async def deny(
        self,
        principal: str,
        scopes: list[str],
        *,
        reason: str = "",
        tenant_id: str = "default",
    ) -> int:
        """Explicitly deny permissions to a principal."""
        now = time.time()
        count = 0

        with sqlite3.connect(str(self._db_path)) as conn:
            for scope in scopes:
                grant_id = str(uuid.uuid4())[:12]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO permission_grants
                    (id, principal, scope, status, granted_by, reason,
                     granted_at, expires_at, tenant_id)
                    VALUES (?, ?, ?, 'denied', 'user', ?, ?, 0, ?)
                    """,
                    (grant_id, principal, scope, reason, now, tenant_id),
                )
                self._cache[(principal, scope)] = GrantStatus.DENIED
                count += 1
            conn.commit()

        logger.info("Denied %d permissions for '%s': %s", count, principal, scopes)
        return count

    async def revoke(
        self,
        principal: str,
        scope: str,
        *,
        tenant_id: str = "default",
    ) -> bool:
        """Revoke a single permission grant.

        Returns True if the grant was found and removed.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                """
                DELETE FROM permission_grants
                WHERE principal = ? AND scope = ? AND tenant_id = ?
                """,
                (principal, scope, tenant_id),
            )
            conn.commit()
            removed = cursor.rowcount > 0

        self._cache.pop((principal, scope), None)

        if removed:
            logger.info("Revoked '%s' permission from '%s'", scope, principal)
        return removed

    async def revoke_all(self, principal: str, *, tenant_id: str = "default") -> int:
        """Revoke all permissions for a principal."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM permission_grants WHERE principal = ? AND tenant_id = ?",
                (principal, tenant_id),
            )
            conn.commit()
            count = cursor.rowcount

        # Clear cache
        keys_to_remove = [k for k in self._cache if k[0] == principal]
        for k in keys_to_remove:
            del self._cache[k]

        logger.info("Revoked all %d permissions from '%s'", count, principal)
        return count

    # ── Introspection ────────────────────────────────────────────────────

    async def list_grants(
        self,
        principal: str,
        *,
        tenant_id: str = "default",
    ) -> list[PermissionGrant]:
        """List all permission grants for a principal."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM permission_grants
                WHERE principal = ? AND tenant_id = ?
                ORDER BY scope
                """,
                (principal, tenant_id),
            ).fetchall()

        return [
            PermissionGrant(
                id=row["id"],
                principal=row["principal"],
                scope=row["scope"],
                status=GrantStatus(row["status"]),
                granted_by=row["granted_by"],
                reason=row["reason"],
                granted_at=row["granted_at"],
                expires_at=row["expires_at"],
                tenant_id=row["tenant_id"],
            )
            for row in rows
        ]

    async def list_all_principals(self, *, tenant_id: str = "default") -> list[str]:
        """List all principals with any grants."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT DISTINCT principal FROM permission_grants WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def get_audit_log(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent permission check audit entries."""
        entries = self._audit[-limit:]
        return [
            {
                "principal": e.principal,
                "scope": e.scope,
                "result": e.result,
                "checked_at": e.checked_at,
                "context": e.context,
            }
            for e in entries
        ]

    # ── Helpers ──────────────────────────────────────────────────────────

    async def _get_grant(
        self, principal: str, scope: str, tenant_id: str
    ) -> PermissionGrant | None:
        """Look up a specific grant from the database."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM permission_grants
                WHERE principal = ? AND scope = ? AND tenant_id = ?
                """,
                (principal, scope, tenant_id),
            ).fetchone()

        if row is None:
            return None

        return PermissionGrant(
            id=row["id"],
            principal=row["principal"],
            scope=row["scope"],
            status=GrantStatus(row["status"]),
            granted_by=row["granted_by"],
            reason=row["reason"],
            granted_at=row["granted_at"],
            expires_at=row["expires_at"],
            tenant_id=row["tenant_id"],
        )

    async def _warm_cache(self) -> None:
        """Load all grants into memory cache."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute("SELECT principal, scope, status FROM permission_grants").fetchall()

        for principal, scope, status in rows:
            self._cache[(principal, scope)] = GrantStatus(status)

        logger.debug("Warmed permission cache with %d entries", len(self._cache))

    def _record_audit(self, principal: str, scope: str, result: str, context: str) -> None:
        """Record an audit entry (in-memory ring buffer)."""
        self._audit.append(
            PermissionAuditEntry(
                principal=principal,
                scope=scope,
                result=result,
                context=context,
            )
        )
        # Keep last 200
        if len(self._audit) > 200:
            self._audit = self._audit[-200:]
