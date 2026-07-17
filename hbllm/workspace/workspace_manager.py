"""
Workspace Manager — Isolated cognitive contexts for multi-domain work.

Users juggle multiple domains (work projects, personal research,
side businesses). The Workspace Manager provides hierarchical isolation
so that memories, goals, skills, and conversation history stay
separated and context-appropriate.

Hierarchy::

    Workspace (e.g., "Startup", "Research", "Personal")
        ↓
    Projects (auto-detected via ProjectGraph or manual)
        ↓
    Documents / Files
        ↓
    Memories (scoped episodic + semantic)
        ↓
    Goals (scoped DAG tasks)
        ↓
    Skills (workspace-specific procedures)
        ↓
    Chats (conversation sessions)

CLI commands::

    hbllm workspace create Startup
    hbllm workspace switch Startup
    hbllm workspace list
    hbllm workspace archive Research
    hbllm workspace delete old-project

Architecture::

    ConversationSession.workspace_id
        ↓
    WorkspaceManager resolves scope
        ↓
    Memory / ProjectGraph / SkillRegistry filter by workspace

This integrates naturally with multi-tenant isolation: workspaces
are scoped within a tenant.
"""

from __future__ import annotations

import json
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
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════


class WorkspaceState(StrEnum):
    """Lifecycle state of a workspace."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class WorkspaceInfo:
    """Metadata about a workspace."""

    id: str
    tenant_id: str
    name: str
    description: str = ""
    state: WorkspaceState = WorkspaceState.ACTIVE
    icon: str = "📁"  # Emoji or icon identifier

    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)

    # Statistics
    project_count: int = 0
    memory_count: int = 0
    chat_count: int = 0

    # Configuration
    default_model: str | None = None  # Override brain profile model
    custom_system_prompt: str | None = None  # Workspace-specific instructions
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "description": self.description,
            "state": self.state.value,
            "icon": self.icon,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "project_count": self.project_count,
            "memory_count": self.memory_count,
            "chat_count": self.chat_count,
            "default_model": self.default_model,
            "custom_system_prompt": self.custom_system_prompt,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspaceInfo:
        return cls(
            id=data["id"],
            tenant_id=data["tenant_id"],
            name=data["name"],
            description=data.get("description", ""),
            state=WorkspaceState(data.get("state", "active")),
            icon=data.get("icon", "📁"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            last_accessed_at=data.get("last_accessed_at", time.time()),
            project_count=data.get("project_count", 0),
            memory_count=data.get("memory_count", 0),
            chat_count=data.get("chat_count", 0),
            default_model=data.get("default_model"),
            custom_system_prompt=data.get("custom_system_prompt"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Workspace Manager
# ═══════════════════════════════════════════════════════════════════════════


class WorkspaceManager:
    """Manages isolated workspace contexts for multi-domain work.

    Each workspace scopes:
      - Conversation sessions
      - Memory queries (episodic, semantic, procedural)
      - Project graphs and goals
      - Skills and procedures
      - Chat history

    The manager persists workspace metadata in SQLite and provides
    workspace resolution for the Gateway and cognitive nodes.

    Usage::

        manager = WorkspaceManager(data_dir="data")
        await manager.init_db()

        # Create a workspace
        ws = await manager.create("default", "Startup", description="My startup project")

        # Switch active workspace
        await manager.set_active("default", ws.id)

        # Get active workspace for a tenant
        active = await manager.get_active("default")
    """

    def __init__(self, data_dir: str = "data") -> None:
        self._db_path = Path(data_dir) / "workspaces.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory active workspace per tenant
        self._active_workspace: dict[str, str] = {}  # tenant_id → workspace_id

    async def init_db(self) -> None:
        """Create the workspaces table if it doesn't exist."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'active',
                    icon TEXT DEFAULT '📁',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    project_count INTEGER DEFAULT 0,
                    memory_count INTEGER DEFAULT 0,
                    chat_count INTEGER DEFAULT 0,
                    default_model TEXT,
                    custom_system_prompt TEXT,
                    tags_json TEXT DEFAULT '[]',
                    metadata_json TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workspaces_tenant
                ON workspaces(tenant_id, state)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_workspaces (
                    tenant_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
                )
            """)
            conn.commit()

        # Load active workspaces into memory
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute("SELECT tenant_id, workspace_id FROM active_workspaces").fetchall()
            for tenant_id, workspace_id in rows:
                self._active_workspace[tenant_id] = workspace_id

        logger.info("WorkspaceManager initialized (db=%s)", self._db_path)

    # ── CRUD ─────────────────────────────────────────────────────────────

    async def create(
        self,
        tenant_id: str,
        name: str,
        *,
        description: str = "",
        icon: str = "📁",
        default_model: str | None = None,
        custom_system_prompt: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceInfo:
        """Create a new workspace.

        Args:
            tenant_id: Owning tenant.
            name: Human-readable workspace name.
            description: Optional description.
            icon: Emoji or icon identifier.
            default_model: Override model for this workspace.
            custom_system_prompt: Workspace-specific system instructions.
            tags: Organizational tags.
            metadata: Additional metadata.

        Returns:
            The created WorkspaceInfo.

        Raises:
            ValueError: If a workspace with this name already exists for the tenant.
        """
        # Check for duplicate names
        existing = await self.find_by_name(tenant_id, name)
        if existing:
            raise ValueError(f"Workspace '{name}' already exists for tenant '{tenant_id}'")

        now = time.time()
        ws = WorkspaceInfo(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            name=name,
            description=description,
            icon=icon,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
            default_model=default_model,
            custom_system_prompt=custom_system_prompt,
            tags=tags or [],
            metadata=metadata or {},
        )

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                INSERT INTO workspaces
                (id, tenant_id, name, description, state, icon,
                 created_at, updated_at, last_accessed_at,
                 default_model, custom_system_prompt, tags_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ws.id,
                    ws.tenant_id,
                    ws.name,
                    ws.description,
                    ws.state.value,
                    ws.icon,
                    ws.created_at,
                    ws.updated_at,
                    ws.last_accessed_at,
                    ws.default_model,
                    ws.custom_system_prompt,
                    json.dumps(ws.tags),
                    json.dumps(ws.metadata),
                ),
            )
            conn.commit()

        # Auto-set as active if this is the first workspace for the tenant
        if tenant_id not in self._active_workspace:
            await self.set_active(tenant_id, ws.id)

        logger.info("Created workspace '%s' (%s) for tenant '%s'", name, ws.id[:8], tenant_id)
        return ws

    async def get(self, workspace_id: str) -> WorkspaceInfo | None:
        """Get a workspace by ID."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()

        if row is None:
            return None
        return self._row_to_info(row)

    async def find_by_name(self, tenant_id: str, name: str) -> WorkspaceInfo | None:
        """Find a workspace by tenant and name."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM workspaces WHERE tenant_id = ? AND name = ? AND state != 'deleted'",
                (tenant_id, name),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_info(row)

    async def list_workspaces(
        self,
        tenant_id: str,
        *,
        include_archived: bool = False,
    ) -> list[WorkspaceInfo]:
        """List all workspaces for a tenant.

        Args:
            tenant_id: Tenant to list workspaces for.
            include_archived: Include archived workspaces.

        Returns:
            List of WorkspaceInfo ordered by last_accessed_at descending.
        """
        states = ["active"]
        if include_archived:
            states.append("archived")

        placeholders = ",".join("?" for _ in states)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM workspaces
                WHERE tenant_id = ? AND state IN ({placeholders})
                ORDER BY last_accessed_at DESC
                """,
                (tenant_id, *states),
            ).fetchall()

        return [self._row_to_info(row) for row in rows]

    async def update(
        self,
        workspace_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        icon: str | None = None,
        default_model: str | None = None,
        custom_system_prompt: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceInfo | None:
        """Update workspace fields.

        Only provided (non-None) fields are updated.

        Returns:
            Updated WorkspaceInfo, or None if not found.
        """
        ws = await self.get(workspace_id)
        if ws is None:
            return None

        if name is not None:
            ws.name = name
        if description is not None:
            ws.description = description
        if icon is not None:
            ws.icon = icon
        if default_model is not None:
            ws.default_model = default_model
        if custom_system_prompt is not None:
            ws.custom_system_prompt = custom_system_prompt
        if tags is not None:
            ws.tags = tags
        if metadata is not None:
            ws.metadata = metadata

        ws.updated_at = time.time()

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                UPDATE workspaces SET
                    name = ?, description = ?, icon = ?,
                    default_model = ?, custom_system_prompt = ?,
                    tags_json = ?, metadata_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    ws.name,
                    ws.description,
                    ws.icon,
                    ws.default_model,
                    ws.custom_system_prompt,
                    json.dumps(ws.tags),
                    json.dumps(ws.metadata),
                    ws.updated_at,
                    workspace_id,
                ),
            )
            conn.commit()

        return ws

    async def archive(self, workspace_id: str) -> bool:
        """Archive a workspace (soft delete)."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                "UPDATE workspaces SET state = 'archived', updated_at = ? WHERE id = ?",
                (time.time(), workspace_id),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info("Archived workspace %s", workspace_id[:8])
                return True
        return False

    async def delete(self, workspace_id: str) -> bool:
        """Permanently mark a workspace as deleted."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                "UPDATE workspaces SET state = 'deleted', updated_at = ? WHERE id = ?",
                (time.time(), workspace_id),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info("Deleted workspace %s", workspace_id[:8])
                return True
        return False

    # ── Active Workspace ─────────────────────────────────────────────────

    async def set_active(self, tenant_id: str, workspace_id: str) -> None:
        """Set the active workspace for a tenant.

        Args:
            tenant_id: Tenant ID.
            workspace_id: Workspace to activate.

        Raises:
            ValueError: If the workspace doesn't exist.
        """
        ws = await self.get(workspace_id)
        if ws is None:
            raise ValueError(f"Workspace '{workspace_id}' not found")

        self._active_workspace[tenant_id] = workspace_id

        # Update last_accessed_at
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE workspaces SET last_accessed_at = ? WHERE id = ?",
                (time.time(), workspace_id),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO active_workspaces (tenant_id, workspace_id)
                VALUES (?, ?)
                """,
                (tenant_id, workspace_id),
            )
            conn.commit()

        logger.info(
            "Switched active workspace for tenant '%s' to '%s' (%s)",
            tenant_id,
            ws.name,
            workspace_id[:8],
        )

    async def get_active(self, tenant_id: str) -> WorkspaceInfo | None:
        """Get the active workspace for a tenant.

        Returns None if no workspace is active. The Gateway uses this
        to inject workspace_id into ConversationSession.
        """
        workspace_id = self._active_workspace.get(tenant_id)
        if workspace_id is None:
            return None
        return await self.get(workspace_id)

    async def get_active_id(self, tenant_id: str) -> str:
        """Get the active workspace ID, defaulting to 'default'."""
        return self._active_workspace.get(tenant_id, "default")

    # ── Statistics ───────────────────────────────────────────────────────

    async def increment_stat(
        self,
        workspace_id: str,
        field: str,
        amount: int = 1,
    ) -> None:
        """Increment a workspace statistic counter.

        Args:
            workspace_id: Workspace to update.
            field: One of 'project_count', 'memory_count', 'chat_count'.
            amount: Amount to increment by.
        """
        if field not in ("project_count", "memory_count", "chat_count"):
            raise ValueError(f"Invalid stat field: {field}")

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                f"UPDATE workspaces SET {field} = {field} + ? WHERE id = ?",
                (amount, workspace_id),
            )
            conn.commit()

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_info(row: Any) -> WorkspaceInfo:
        """Convert a sqlite3.Row to WorkspaceInfo."""
        return WorkspaceInfo(
            id=row["id"],
            tenant_id=row["tenant_id"],
            name=row["name"],
            description=row["description"] or "",
            state=WorkspaceState(row["state"]),
            icon=row["icon"] or "📁",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"],
            project_count=row["project_count"],
            memory_count=row["memory_count"],
            chat_count=row["chat_count"],
            default_model=row["default_model"],
            custom_system_prompt=row["custom_system_prompt"],
            tags=json.loads(row["tags_json"] or "[]"),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )
