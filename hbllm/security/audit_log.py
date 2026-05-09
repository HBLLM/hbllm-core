"""
Security Audit Log — Unified audit trail for all security-sensitive operations.

Captures every significant action for SOC2/GDPR compliance:
  - Authentication (login, failed, logout, token refresh)
  - Data mutations (create, update, delete)
  - Admin operations (tenant management, policy changes)
  - Access events (who accessed what data, when)
  - Edge device events (connect, disconnect, capability registration)

Every entry is immutable once written (append-only log).
Identity-aware: captures the full (tenant_id, user_id, device_id) triplet.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AuditAction(str, Enum):
    """Standard audit actions."""

    # Auth
    AUTH_LOGIN = "auth.login"
    AUTH_FAILED = "auth.failed"
    AUTH_LOGOUT = "auth.logout"
    AUTH_REFRESH = "auth.refresh"
    AUTH_KEY_GENERATED = "auth.key_generated"
    AUTH_KEY_REVOKED = "auth.key_revoked"

    # WebSocket / Edge
    AUTH_WS_CONNECT = "auth.ws_connect"
    AUTH_WS_DISCONNECT = "auth.ws_disconnect"
    EDGE_CAPABILITY_REGISTERED = "edge.capability_registered"

    # Tenant
    TENANT_CREATED = "tenant.created"
    TENANT_UPDATED = "tenant.updated"
    TENANT_DEACTIVATED = "tenant.deactivated"
    TENANT_DATA_PURGED = "tenant.data_purged"

    # Data
    DATA_ACCESSED = "data.accessed"
    DATA_CREATED = "data.created"
    DATA_UPDATED = "data.updated"
    DATA_DELETED = "data.deleted"
    DATA_EXPORTED = "data.exported"

    # Policy
    POLICY_CREATED = "policy.created"
    POLICY_UPDATED = "policy.updated"
    POLICY_DELETED = "policy.deleted"

    # Chat
    CHAT_MESSAGE = "chat.message"
    CHAT_CONVERSATION_CREATED = "chat.conversation_created"
    CHAT_CONVERSATION_DELETED = "chat.conversation_deleted"

    # Tools
    TOOL_EXECUTED = "tool.executed"
    TOOL_FAILED = "tool.failed"

    # Admin
    ADMIN_ACTION = "admin.action"
    ADMIN_CONFIG_CHANGED = "admin.config_changed"

    # Webhook
    WEBHOOK_REGISTERED = "webhook.registered"
    WEBHOOK_DELIVERED = "webhook.delivered"
    WEBHOOK_FAILED = "webhook.failed"


@dataclass
class AuditEntry:
    """A single audit log entry with full identity context."""

    id: str
    timestamp: float
    tenant_id: str
    user_id: str = ""
    device_id: str = ""
    actor: str = "system"  # user ID, API key hash, or "system"
    action: str = ""
    resource: str = ""  # e.g., "conversation:abc123", "tenant:acme"
    ip_address: str = ""
    user_agent: str = ""
    severity: str = "info"
    details: dict[str, Any] = field(default_factory=dict)
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "severity": self.severity,
            "details": self.details,
            "success": self.success,
        }


class AuditLog:
    """
    Append-only security audit log with full identity context.

    Thread-safe SQLite store with indexed queries by tenant, user, action,
    time range, and severity.
    """

    def __init__(self, db_path: str = "data/audit.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                tenant_id TEXT NOT NULL,
                user_id TEXT DEFAULT '',
                device_id TEXT DEFAULT '',
                actor TEXT NOT NULL DEFAULT 'system',
                action TEXT NOT NULL,
                resource TEXT NOT NULL DEFAULT '',
                ip_address TEXT DEFAULT '',
                user_agent TEXT DEFAULT '',
                severity TEXT DEFAULT 'info',
                details TEXT DEFAULT '{}',
                success INTEGER DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_audit_tenant
                ON audit_log(tenant_id, timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_audit_user
                ON audit_log(tenant_id, user_id, timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_audit_action
                ON audit_log(action, timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_audit_severity
                ON audit_log(severity, timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_audit_time
                ON audit_log(timestamp DESC);
        """)
        self._conn.commit()

    # ─── Write ───────────────────────────────────────────────────────

    def log(
        self,
        action: str | AuditAction,
        tenant_id: str = "system",
        user_id: str = "",
        device_id: str = "",
        actor: str = "system",
        resource: str = "",
        ip_address: str = "",
        user_agent: str = "",
        severity: str | AuditSeverity = AuditSeverity.INFO,
        details: dict[str, Any] | None = None,
        success: bool = True,
    ) -> AuditEntry:
        """Log an audit event with full identity context."""
        entry_id = str(uuid.uuid4())[:12]
        now = time.time()
        action_str = action.value if isinstance(action, AuditAction) else action
        severity_str = severity.value if isinstance(severity, AuditSeverity) else severity

        self._conn.execute(
            """INSERT INTO audit_log
               (id, timestamp, tenant_id, user_id, device_id, actor, action, resource,
                ip_address, user_agent, severity, details, success)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry_id,
                now,
                tenant_id,
                user_id,
                device_id,
                actor,
                action_str,
                resource,
                ip_address,
                user_agent,
                severity_str,
                json.dumps(details or {}),
                int(success),
            ),
        )
        self._conn.commit()

        entry = AuditEntry(
            id=entry_id,
            timestamp=now,
            tenant_id=tenant_id,
            user_id=user_id,
            device_id=device_id,
            actor=actor,
            action=action_str,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            severity=severity_str,
            details=details or {},
            success=success,
        )

        # Log critical events to Python logger too
        if severity_str == "critical":
            logger.warning(
                "AUDIT CRITICAL: %s by %s on %s (tenant=%s user=%s device=%s)",
                action_str,
                actor,
                resource,
                tenant_id,
                user_id,
                device_id,
            )

        return entry

    # ─── Query ───────────────────────────────────────────────────────

    def query(
        self,
        tenant_id: str | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        action: str | None = None,
        severity: str | None = None,
        actor: str | None = None,
        since: float | None = None,
        until: float | None = None,
        success: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query audit log with filters."""
        conditions = []
        params: list[Any] = []

        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if device_id:
            conditions.append("device_id = ?")
            params.append(device_id)
        if action:
            conditions.append("action = ?")
            params.append(action)
        if severity:
            conditions.append("severity = ?")
            params.append(severity)
        if actor:
            conditions.append("actor = ?")
            params.append(actor)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp <= ?")
            params.append(until)
        if success is not None:
            conditions.append("success = ?")
            params.append(int(success))

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM audit_log WHERE {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "tenant_id": r["tenant_id"],
                "user_id": r["user_id"],
                "device_id": r["device_id"],
                "actor": r["actor"],
                "action": r["action"],
                "resource": r["resource"],
                "ip_address": r["ip_address"],
                "severity": r["severity"],
                "details": json.loads(r["details"]),
                "success": bool(r["success"]),
            }
            for r in rows
        ]

    def count(
        self,
        tenant_id: str | None = None,
        action: str | None = None,
        since: float | None = None,
    ) -> int:
        """Count audit entries matching filters."""
        conditions = []
        params: list[Any] = []

        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        if action:
            conditions.append("action = ?")
            params.append(action)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)

        where = " AND ".join(conditions) if conditions else "1=1"
        row = self._conn.execute(
            f"SELECT COUNT(*) as c FROM audit_log WHERE {where}",
            params,
        ).fetchone()
        return int(row["c"])

    def failed_logins(
        self,
        tenant_id: str | None = None,
        hours: int = 24,
    ) -> int:
        """Count failed login attempts in the last N hours."""
        since = time.time() - (hours * 3600)
        conditions = ["action = 'auth.failed'", "timestamp >= ?"]
        params: list[Any] = [since]

        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)

        where = " AND ".join(conditions)
        row = self._conn.execute(
            f"SELECT COUNT(*) as c FROM audit_log WHERE {where}",
            params,
        ).fetchone()
        return int(row["c"])

    # ─── Export & Retention ──────────────────────────────────────────

    def export_json(
        self,
        tenant_id: str,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        """Export all audit entries for a tenant (compliance/GDPR)."""
        return self.query(tenant_id=tenant_id, since=since, limit=100000)

    def purge_old_entries(self, older_than_days: int = 365) -> int:
        """Delete audit entries older than N days (default 1 year)."""
        cutoff = time.time() - (older_than_days * 86400)
        result = self._conn.execute(
            "DELETE FROM audit_log WHERE timestamp < ?",
            (cutoff,),
        )
        self._conn.commit()
        count = result.rowcount
        if count > 0:
            logger.info("Purged %d audit entries older than %d days", count, older_than_days)
        return count

    def stats(self) -> dict[str, Any]:
        """System-wide audit stats."""
        total = self._conn.execute("SELECT COUNT(*) as c FROM audit_log").fetchone()["c"]
        by_severity = {}
        for sev in ("info", "warning", "critical"):
            c = self._conn.execute(
                "SELECT COUNT(*) as c FROM audit_log WHERE severity = ?",
                (sev,),
            ).fetchone()["c"]
            by_severity[sev] = c

        recent_critical = self.query(severity="critical", limit=5)
        return {
            "total_entries": total,
            "by_severity": by_severity,
            "recent_critical": recent_critical,
        }

    def close(self) -> None:
        self._conn.close()
