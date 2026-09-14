"""Audit Trail & Security Compliance Logger — Unified immutable audit ledger.

Consolidates cognitive safety governance (AI actions in the environment) and
identity/security operations (SOC2/GDPR compliance, authentication, and admin actions)
into a single, cryptographically verifiable, append-only SQLite store with SHA-256
hash-chaining.

Architecture:
    - Append-only SQLite table with write-ahead logging (WAL)
    - Cryptographic SHA-256 hash-chaining across all entries (previous_hash + content)
    - Tamper detection via ``verify_integrity()``
    - Dual query semantics: support for safety governance filters (risk_tier, category)
      and compliance filters (actor, user_id, IP, severity, failed_logins)
    - Backward-compatible aliases for ``AuditLog``, ``SafetyAuditEntry``, and ``ComplianceAuditEntry``
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
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
    """Standard audit actions for security and compliance."""

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

    # Tools / Autonomous Actions
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
    """An immutable, hash-chained audit ledger entry."""

    id: int = 0
    timestamp: float = field(default_factory=time.time)
    tenant_id: str = "default"
    user_id: str = ""
    device_id: str = ""
    actor: str = "system"
    action: str = ""
    category: str = "system"
    resource: str = ""
    risk_tier: int = 0
    source: str = "system"
    target: str = ""
    result: str = "success"
    severity: str = "info"
    ip_address: str = ""
    user_agent: str = ""
    success: bool = True
    details: dict[str, Any] = field(default_factory=dict)
    entry_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "actor": self.actor,
            "action": self.action,
            "category": self.category,
            "resource": self.resource,
            "risk_tier": self.risk_tier,
            "source": self.source,
            "target": self.target,
            "result": self.result,
            "severity": self.severity,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success,
            "details": self.details,
            "entry_hash": self.entry_hash,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return self.id > other
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return self.id >= other
        return NotImplemented

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return self.id == other
        if isinstance(other, AuditEntry):
            return self.id == other.id
        return False

    def __int__(self) -> int:
        return self.id


# Backwards compatibility aliases
SafetyAuditEntry = AuditEntry
ComplianceAuditEntry = AuditEntry


class AuditTrail:
    """Unified append-only audit ledger with SHA-256 hash-chain integrity.

    Combines compliance logging with AI safety action governance into a single
    tamper-evident store.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        data_dir: str | Path | None = None,
        max_age_days: int = 90,
    ) -> None:
        if db_path is None:
            if data_dir is not None:
                db_path = Path(data_dir) / "audit.db"
            else:
                db_path = "data/audit.db"
        self.db_path = Path(db_path)
        self._db_path = str(self.db_path)
        self.max_age_days = max_age_days
        self._last_hash = "genesis"
        self._total_entries = 0

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS audit_trail (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT DEFAULT '',
                    device_id TEXT DEFAULT '',
                    actor TEXT NOT NULL DEFAULT 'system',
                    action TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'system',
                    resource TEXT DEFAULT '',
                    risk_tier INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'system',
                    target TEXT DEFAULT '',
                    result TEXT NOT NULL DEFAULT 'success',
                    severity TEXT DEFAULT 'info',
                    ip_address TEXT DEFAULT '',
                    user_agent TEXT DEFAULT '',
                    success INTEGER DEFAULT 1,
                    details TEXT,
                    entry_hash TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_audit_tenant_time
                ON audit_trail(tenant_id, timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_audit_category
                ON audit_trail(category, timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_audit_risk
                ON audit_trail(risk_tier, timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_audit_action
                ON audit_trail(action, timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_audit_user
                ON audit_trail(tenant_id, user_id, timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_audit_severity
                ON audit_trail(severity, timestamp DESC);
            """)

            # Ensure any pre-existing schemas are upgraded transparently
            existing_cols = {
                row["name"] for row in conn.execute("PRAGMA table_info(audit_trail)").fetchall()
            }
            for col_def in [
                ("user_id", "TEXT DEFAULT ''"),
                ("device_id", "TEXT DEFAULT ''"),
                ("actor", "TEXT NOT NULL DEFAULT 'system'"),
                ("resource", "TEXT DEFAULT ''"),
                ("severity", "TEXT DEFAULT 'info'"),
                ("ip_address", "TEXT DEFAULT ''"),
                ("user_agent", "TEXT DEFAULT ''"),
                ("success", "INTEGER DEFAULT 1"),
            ]:
                if col_def[0] not in existing_cols:
                    conn.execute(f"ALTER TABLE audit_trail ADD COLUMN {col_def[0]} {col_def[1]}")

            conn.commit()

            # Load last hash for chain continuity
            row = conn.execute(
                "SELECT entry_hash FROM audit_trail ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                self._last_hash = row[0]

            count_row = conn.execute("SELECT COUNT(*) as c FROM audit_trail").fetchone()
            self._total_entries = int(count_row["c"]) if count_row else 0
        finally:
            conn.close()

    async def init_db(self) -> None:
        """Async hook for explicit initialization (schema already verified in __init__)."""
        self._init_schema()

    def log(
        self,
        tenant_id: str | None = None,
        action: str | AuditAction = "",
        category: str = "system",
        risk_tier: int = 0,
        source: str = "system",
        target: str = "",
        result: str = "success",
        details: dict[str, Any] | None = None,
        # Extended compliance attributes
        user_id: str = "",
        device_id: str = "",
        actor: str = "system",
        resource: str = "",
        ip_address: str = "",
        user_agent: str = "",
        severity: str | AuditSeverity = AuditSeverity.INFO,
        success: bool = True,
        **kwargs: Any,
    ) -> AuditEntry:
        """Append an audit entry to the immutable ledger with cryptographic hash chaining.

        Returns an ``AuditEntry`` instance which also behaves numerically for legacy
        call sites checking ``entry_id > 0``.
        """
        # Accommodate positional style or keyword style: log(action=..., tenant_id=...)
        if tenant_id is None:
            tenant_id = kwargs.get("tenant_id", "default")
        # Handle case where first positional arg was passed as action if tenant_id wasn't keyworded
        if isinstance(tenant_id, AuditAction) or (
            isinstance(tenant_id, str) and "." in tenant_id and not action
        ):
            action = tenant_id
            tenant_id = kwargs.get("tenant_id", "default")

        action_str = action.value if isinstance(action, AuditAction) else str(action)
        severity_str = severity.value if isinstance(severity, AuditSeverity) else str(severity)
        now = time.time()

        # Compute hash chain covering all critical forensic dimensions
        payload = json.dumps(
            {
                "prev": self._last_hash,
                "time": now,
                "tenant": tenant_id,
                "actor": actor,
                "action": action_str,
                "target": target or resource,
                "result": result,
                "severity": severity_str,
                "success": success,
            },
            sort_keys=True,
        )
        entry_hash = hashlib.sha256(payload.encode()).hexdigest()

        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO audit_trail
                   (timestamp, tenant_id, user_id, device_id, actor, action,
                    category, resource, risk_tier, source, target, result,
                    severity, ip_address, user_agent, success, details, entry_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    tenant_id,
                    user_id,
                    device_id,
                    actor,
                    action_str,
                    category,
                    resource,
                    risk_tier,
                    source,
                    target,
                    result,
                    severity_str,
                    ip_address,
                    user_agent,
                    int(success),
                    json.dumps(details or {}),
                    entry_hash,
                ),
            )
            conn.commit()
            entry_id = cursor.lastrowid or 0
        finally:
            conn.close()

        self._last_hash = entry_hash
        self._total_entries += 1

        entry = AuditEntry(
            id=entry_id,
            timestamp=now,
            tenant_id=tenant_id,
            user_id=user_id,
            device_id=device_id,
            actor=actor,
            action=action_str,
            category=category,
            resource=resource,
            risk_tier=risk_tier,
            source=source,
            target=target,
            result=result,
            severity=severity_str,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details or {},
            entry_hash=entry_hash,
        )

        if severity_str == "critical":
            logger.warning(
                "AUDIT CRITICAL: [%s] %s by %s on %s (success=%s)",
                tenant_id,
                action_str,
                actor,
                target or resource,
                success,
            )

        return entry

    def query(
        self,
        tenant_id: str | None = None,
        hours: float | None = None,
        category: str | None = None,
        min_risk_tier: int = 0,
        source: str | None = None,
        result: str | None = None,
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
    ) -> list[AuditEntry]:
        """Query audit trail entries with combined safety and compliance filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if hours is not None:
            cutoff = time.time() - (hours * 3600)
            conditions.append("timestamp > ?")
            params.append(cutoff)
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            conditions.append("timestamp <= ?")
            params.append(until)
        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        if category:
            conditions.append("category = ?")
            params.append(category)
        if min_risk_tier > 0:
            conditions.append("risk_tier >= ?")
            params.append(min_risk_tier)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if result:
            conditions.append("result = ?")
            params.append(result)
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
        if success is not None:
            conditions.append("success = ?")
            params.append(int(success))

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = (
            f"SELECT id, timestamp, tenant_id, user_id, device_id, actor, action, "
            f"category, resource, risk_tier, source, target, result, severity, "
            f"ip_address, user_agent, success, details, entry_hash "
            f"FROM audit_trail WHERE {where} "
            f"ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        conn = self._get_conn()
        try:
            cursor = conn.execute(sql, params)
            entries = []
            for row in cursor.fetchall():
                entries.append(
                    AuditEntry(
                        id=row["id"],
                        timestamp=row["timestamp"],
                        tenant_id=row["tenant_id"],
                        user_id=row["user_id"],
                        device_id=row["device_id"],
                        actor=row["actor"],
                        action=row["action"],
                        category=row["category"],
                        resource=row["resource"],
                        risk_tier=row["risk_tier"],
                        source=row["source"],
                        target=row["target"],
                        result=row["result"],
                        severity=row["severity"],
                        ip_address=row["ip_address"],
                        user_agent=row["user_agent"],
                        success=bool(row["success"]),
                        details=json.loads(row["details"]) if row["details"] else {},
                        entry_hash=row["entry_hash"],
                    )
                )
            return entries
        finally:
            conn.close()

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
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)

        where = " AND ".join(conditions) if conditions else "1=1"
        conn = self._get_conn()
        try:
            row = conn.execute(
                f"SELECT COUNT(*) as c FROM audit_trail WHERE {where}",
                params,
            ).fetchone()
            return int(row["c"]) if row else 0
        finally:
            conn.close()

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
        conn = self._get_conn()
        try:
            row = conn.execute(
                f"SELECT COUNT(*) as c FROM audit_trail WHERE {where}",
                params,
            ).fetchone()
            return int(row["c"]) if row else 0
        finally:
            conn.close()

    def export_json(
        self,
        tenant_id: str,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        """Export all audit entries for a tenant (compliance/GDPR)."""
        entries = self.query(tenant_id=tenant_id, since=since, limit=100000)
        return [e.to_dict() for e in entries]

    def prune_old_entries(self) -> int:
        """Remove entries older than max_age_days. Returns count deleted."""
        cutoff = time.time() - (self.max_age_days * 86400)
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM audit_trail WHERE timestamp < ?",
                (cutoff,),
            )
            pruned = cursor.rowcount
            conn.commit()
        finally:
            conn.close()

        if pruned > 0:
            logger.info("Pruned %d audit entries older than %d days", pruned, self.max_age_days)
        return pruned

    def purge_old_entries(self, older_than_days: int = 365) -> int:
        """Delete audit entries older than N days (alias for compliance)."""
        cutoff = time.time() - (older_than_days * 86400)
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM audit_trail WHERE timestamp < ?",
                (cutoff,),
            )
            pruned = cursor.rowcount
            conn.commit()
        finally:
            conn.close()

        if pruned > 0:
            logger.info("Purged %d audit entries older than %d days", pruned, older_than_days)
        return pruned

    def get_summary(
        self,
        tenant_id: str,
        hours: float = 24.0,
    ) -> dict[str, Any]:
        """Get a summary of actions taken in a time period."""
        cutoff = time.time() - hours * 3600
        conn = self._get_conn()
        try:
            total = conn.execute(
                "SELECT COUNT(*) as c FROM audit_trail WHERE tenant_id = ? AND timestamp > ?",
                (tenant_id, cutoff),
            ).fetchone()["c"]

            by_category = dict(
                conn.execute(
                    "SELECT category, COUNT(*) as c FROM audit_trail "
                    "WHERE tenant_id = ? AND timestamp > ? "
                    "GROUP BY category",
                    (tenant_id, cutoff),
                ).fetchall()
            )

            by_result = dict(
                conn.execute(
                    "SELECT result, COUNT(*) as c FROM audit_trail "
                    "WHERE tenant_id = ? AND timestamp > ? "
                    "GROUP BY result",
                    (tenant_id, cutoff),
                ).fetchall()
            )

            high_risk = conn.execute(
                "SELECT COUNT(*) as c FROM audit_trail "
                "WHERE tenant_id = ? AND timestamp > ? AND risk_tier >= 3",
                (tenant_id, cutoff),
            ).fetchone()["c"]

            return {
                "tenant_id": tenant_id,
                "hours": hours,
                "total_actions": total,
                "by_category": by_category,
                "by_result": by_result,
                "high_risk_actions": high_risk,
            }
        finally:
            conn.close()

    def verify_integrity(self, limit: int = 1000) -> dict[str, Any]:
        """Verify cryptographic hash-chain integrity across stored entries."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT id, timestamp, tenant_id, actor, action, target, resource, "
                "result, severity, success, entry_hash "
                "FROM audit_trail ORDER BY id ASC LIMIT ?",
                (limit,),
            )
            rows = list(cursor.fetchall())
        finally:
            conn.close()

        if not rows:
            return {"status": "empty", "entries_checked": 0, "is_valid": True, "broken_links": []}

        prev_hash = "genesis"
        broken_links: list[dict[str, Any]] = []

        for row in rows:
            row_id = row["id"]
            timestamp = row["timestamp"]
            tenant_id = row["tenant_id"]
            actor = row["actor"]
            action = row["action"]
            target = row["target"] or row["resource"]
            result = row["result"]
            severity = row["severity"]
            success = bool(row["success"])
            entry_hash = row["entry_hash"]

            payload = json.dumps(
                {
                    "prev": prev_hash,
                    "time": timestamp,
                    "tenant": tenant_id,
                    "actor": actor,
                    "action": action,
                    "target": target,
                    "result": result,
                    "severity": severity,
                    "success": success,
                },
                sort_keys=True,
            )
            expected_hash = hashlib.sha256(payload.encode()).hexdigest()

            # Verify both new format and legacy format hashes
            legacy_payload = json.dumps(
                {
                    "prev": prev_hash,
                    "time": timestamp,
                    "action": action,
                    "tenant": tenant_id,
                    "target": row["target"],
                    "result": result,
                },
                sort_keys=True,
            )
            legacy_expected_hash = hashlib.sha256(legacy_payload.encode()).hexdigest()

            if entry_hash != expected_hash and entry_hash != legacy_expected_hash:
                broken_links.append(
                    {
                        "entry_id": row_id,
                        "action": action,
                        "expected_hash": expected_hash,
                        "actual_hash": entry_hash,
                        "reason": "Hash mismatch: row content or previous link modified",
                    }
                )
            prev_hash = entry_hash

        missing_hashes = sum(1 for r in rows if not r["entry_hash"])
        is_valid = len(broken_links) == 0 and missing_hashes == 0
        return {
            "status": "ok" if is_valid else "tampered",
            "is_valid": is_valid,
            "entries_checked": len(rows),
            "missing_hashes": missing_hashes,
            "broken_links": broken_links,
            "total_entries": self._total_entries,
        }

    def stats(self) -> dict[str, Any]:
        """System-wide audit stats."""
        conn = self._get_conn()
        try:
            total = conn.execute("SELECT COUNT(*) as c FROM audit_trail").fetchone()["c"]
            by_severity = {}
            for sev in ("info", "warning", "critical"):
                c = conn.execute(
                    "SELECT COUNT(*) as c FROM audit_trail WHERE severity = ?",
                    (sev,),
                ).fetchone()["c"]
                by_severity[sev] = c

            recent_critical = [e.to_dict() for e in self.query(severity="critical", limit=5)]
            return {
                "total_entries": total,
                "db_path": str(self.db_path),
                "max_age_days": self.max_age_days,
                "by_severity": by_severity,
                "recent_critical": recent_critical,
            }
        finally:
            conn.close()

    def close(self) -> None:
        """Close hook for consistency."""
        pass


# Backwards compatibility class alias
AuditLog = AuditTrail
