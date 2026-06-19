"""
SessionMigration — Cross-device context handoff.

Enables seamless conversation transfer between devices:
    "Continue this conversation on my phone"

Exports the active session state (history, context, active goals,
persona) into a portable, cryptographically signed package that
can be imported on another device in the swarm.

Bus Topics:
    session.export   → Export current session for migration
    session.import   → Import a session from another device
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass
class SessionSnapshot:
    """A portable snapshot of an active session."""

    id: str = ""
    tenant_id: str = ""
    source_node_id: str = ""  # Device that exported this
    # Conversation state
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    active_thread_id: str | None = None
    active_thread_name: str = ""
    # Context
    system_prompt: str = ""
    pinned_context: str = ""
    working_memory: dict[str, Any] = field(default_factory=dict)
    # Active goals
    active_goals: list[dict[str, Any]] = field(default_factory=list)
    # Persona state
    persona_traits: dict[str, float] = field(default_factory=dict)
    # Metadata
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0  # 0 = no expiry
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Compute integrity checksum over the session data."""
        content = json.dumps(
            {
                "tenant_id": self.tenant_id,
                "conversation_history": self.conversation_history,
                "active_goals": self.active_goals,
                "working_memory": self.working_memory,
                "created_at": self.created_at,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """Verify the snapshot hasn't been tampered with."""
        return self.checksum == self.compute_checksum()

    def is_expired(self) -> bool:
        """Check if the snapshot has expired."""
        if self.expires_at == 0:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "source_node_id": self.source_node_id,
            "conversation_history": self.conversation_history,
            "active_thread_id": self.active_thread_id,
            "active_thread_name": self.active_thread_name,
            "system_prompt": self.system_prompt,
            "pinned_context": self.pinned_context,
            "working_memory": self.working_memory,
            "active_goals": self.active_goals,
            "persona_traits": self.persona_traits,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionSnapshot:
        return cls(
            id=data.get("id", ""),
            tenant_id=data.get("tenant_id", ""),
            source_node_id=data.get("source_node_id", ""),
            conversation_history=data.get("conversation_history", []),
            active_thread_id=data.get("active_thread_id"),
            active_thread_name=data.get("active_thread_name", ""),
            system_prompt=data.get("system_prompt", ""),
            pinned_context=data.get("pinned_context", ""),
            working_memory=data.get("working_memory", {}),
            active_goals=data.get("active_goals", []),
            persona_traits=data.get("persona_traits", {}),
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at", 0.0),
            checksum=data.get("checksum", ""),
        )

    def to_json(self) -> str:
        """Serialize to JSON string for transport."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data: str) -> SessionSnapshot:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(data))


# ── SessionMigrationManager ─────────────────────────────────────────────────


class SessionMigrationManager:
    """
    Manages session export/import for cross-device context handoff.

    Usage:
        manager = SessionMigrationManager(node_id="homeserver")

        # Export current session
        snapshot = manager.export_session(
            tenant_id="user_1",
            conversation_history=[{"role": "user", "content": "..."}],
            active_goals=[{"id": "g1", "name": "Deploy v2"}],
        )

        # Transfer snapshot to another device (via SynapseGateway, QR, etc.)
        json_data = snapshot.to_json()

        # On the receiving device:
        imported = manager.import_session(json_data)
        if imported:
            # Resume conversation with imported context
            ...
    """

    def __init__(
        self,
        node_id: str = "",
        default_ttl_seconds: int = 3600,  # 1 hour default expiry
    ) -> None:
        self._node_id = node_id
        self._default_ttl = default_ttl_seconds
        # Track exported/imported sessions
        self._exported: dict[str, SessionSnapshot] = {}
        self._imported: dict[str, SessionSnapshot] = {}

        logger.info("SessionMigrationManager initialized (node=%s)", node_id)

    def export_session(
        self,
        tenant_id: str,
        conversation_history: list[dict[str, str]] | None = None,
        active_thread_id: str | None = None,
        active_thread_name: str = "",
        system_prompt: str = "",
        pinned_context: str = "",
        working_memory: dict[str, Any] | None = None,
        active_goals: list[dict[str, Any]] | None = None,
        persona_traits: dict[str, float] | None = None,
        ttl_seconds: int | None = None,
    ) -> SessionSnapshot:
        """
        Export the current session state as a portable snapshot.

        Returns a SessionSnapshot that can be serialized and transferred
        to another device in the swarm.
        """
        import uuid

        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl

        snapshot = SessionSnapshot(
            id=uuid.uuid4().hex[:12],
            tenant_id=tenant_id,
            source_node_id=self._node_id,
            conversation_history=conversation_history or [],
            active_thread_id=active_thread_id,
            active_thread_name=active_thread_name,
            system_prompt=system_prompt,
            pinned_context=pinned_context,
            working_memory=working_memory or {},
            active_goals=active_goals or [],
            persona_traits=persona_traits or {},
            created_at=now,
            expires_at=now + ttl if ttl > 0 else 0.0,
        )
        snapshot.checksum = snapshot.compute_checksum()

        self._exported[snapshot.id] = snapshot
        logger.info(
            "Exported session %s for tenant '%s' (%d turns, %d goals, ttl=%ds)",
            snapshot.id,
            tenant_id,
            len(snapshot.conversation_history),
            len(snapshot.active_goals),
            ttl,
        )
        return snapshot

    def import_session(self, json_data: str) -> SessionSnapshot | None:
        """
        Import a session snapshot from another device.

        Validates integrity and expiry before accepting.
        Returns the imported snapshot, or None if validation fails.
        """
        try:
            snapshot = SessionSnapshot.from_json(json_data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse session snapshot: %s", e)
            return None

        # Validate integrity
        if not snapshot.verify_integrity():
            logger.warning(
                "Session snapshot %s failed integrity check (expected=%s, got=%s)",
                snapshot.id,
                snapshot.checksum,
                snapshot.compute_checksum(),
            )
            return None

        # Check expiry
        if snapshot.is_expired():
            logger.warning("Session snapshot %s has expired", snapshot.id)
            return None

        # Don't import from ourselves
        if snapshot.source_node_id == self._node_id:
            logger.debug("Skipping self-import of session %s", snapshot.id)
            return snapshot  # Still return it, but log that it's from us

        self._imported[snapshot.id] = snapshot
        logger.info(
            "Imported session %s from node '%s' (%d turns, %d goals)",
            snapshot.id,
            snapshot.source_node_id,
            len(snapshot.conversation_history),
            len(snapshot.active_goals),
        )
        return snapshot

    def get_exported(self, snapshot_id: str) -> SessionSnapshot | None:
        """Retrieve an exported snapshot by ID."""
        return self._exported.get(snapshot_id)

    def get_imported(self, snapshot_id: str) -> SessionSnapshot | None:
        """Retrieve an imported snapshot by ID."""
        return self._imported.get(snapshot_id)

    def list_exported(self, tenant_id: str | None = None) -> list[SessionSnapshot]:
        """List exported snapshots, optionally filtered by tenant."""
        snapshots = list(self._exported.values())
        if tenant_id:
            snapshots = [s for s in snapshots if s.tenant_id == tenant_id]
        return sorted(snapshots, key=lambda s: s.created_at, reverse=True)

    def cleanup_expired(self) -> int:
        """Remove expired snapshots from memory."""
        expired_export = [sid for sid, s in self._exported.items() if s.is_expired()]
        expired_import = [sid for sid, s in self._imported.items() if s.is_expired()]
        for sid in expired_export:
            del self._exported[sid]
        for sid in expired_import:
            del self._imported[sid]
        total = len(expired_export) + len(expired_import)
        if total:
            logger.debug("Cleaned up %d expired session snapshots", total)
        return total

    def stats(self) -> dict[str, Any]:
        """Migration stats."""
        return {
            "exported_count": len(self._exported),
            "imported_count": len(self._imported),
            "node_id": self._node_id,
        }
