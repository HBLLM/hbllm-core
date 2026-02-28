"""
Identity Core Node — persistent per-tenant persona profiles.

Each tenant can have a distinct personality, system prompt, goals,
and behavioral constraints. The IdentityNode publishes this context
to the Workspace blackboard so all reasoning is persona-aware.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class IdentityProfile:
    """A tenant's identity profile."""

    __slots__ = ("tenant_id", "persona_name", "system_prompt", "goals",
                 "constraints", "personality_traits", "created_at", "updated_at")

    def __init__(
        self,
        tenant_id: str,
        persona_name: str = "default",
        system_prompt: str = "",
        goals: list[str] | None = None,
        constraints: list[str] | None = None,
        personality_traits: dict[str, Any] | None = None,
        created_at: str = "",
        updated_at: str = "",
    ):
        self.tenant_id = tenant_id
        self.persona_name = persona_name
        self.system_prompt = system_prompt
        self.goals = goals or []
        self.constraints = constraints or []
        self.personality_traits = personality_traits or {}
        now = datetime.now(timezone.utc).isoformat()
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "persona_name": self.persona_name,
            "system_prompt": self.system_prompt,
            "goals": self.goals,
            "constraints": self.constraints,
            "personality_traits": self.personality_traits,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_context_string(self) -> str:
        """Produce a compact context string for injection into prompts."""
        parts = []
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")
        if self.goals:
            parts.append(f"Goals: {', '.join(self.goals)}")
        if self.constraints:
            parts.append(f"Constraints: {', '.join(self.constraints)}")
        if self.personality_traits:
            traits = ", ".join(f"{k}={v}" for k, v in self.personality_traits.items())
            parts.append(f"Traits: {traits}")
        return " | ".join(parts) if parts else ""


class IdentityStore:
    """SQLite-backed identity profile persistence."""

    def __init__(self, db_path: str | Path = "identity.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS identities (
                    tenant_id TEXT PRIMARY KEY,
                    persona_name TEXT NOT NULL DEFAULT 'default',
                    system_prompt TEXT DEFAULT '',
                    goals_json TEXT DEFAULT '[]',
                    constraints_json TEXT DEFAULT '[]',
                    traits_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def upsert(self, profile: IdentityProfile) -> None:
        """Insert or update a tenant's identity profile."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO identities 
                    (tenant_id, persona_name, system_prompt, goals_json, 
                     constraints_json, traits_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id) DO UPDATE SET
                    persona_name = excluded.persona_name,
                    system_prompt = excluded.system_prompt,
                    goals_json = excluded.goals_json,
                    constraints_json = excluded.constraints_json,
                    traits_json = excluded.traits_json,
                    updated_at = excluded.updated_at
            """, (
                profile.tenant_id, profile.persona_name, profile.system_prompt,
                json.dumps(profile.goals), json.dumps(profile.constraints),
                json.dumps(profile.personality_traits), profile.created_at, now,
            ))

    def get(self, tenant_id: str) -> IdentityProfile | None:
        """Retrieve a tenant's identity profile."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM identities WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()

            if row is None:
                return None

            return IdentityProfile(
                tenant_id=row["tenant_id"],
                persona_name=row["persona_name"],
                system_prompt=row["system_prompt"],
                goals=json.loads(row["goals_json"]),
                constraints=json.loads(row["constraints_json"]),
                personality_traits=json.loads(row["traits_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def delete(self, tenant_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM identities WHERE tenant_id = ?", (tenant_id,),
            )
            return cursor.rowcount > 0


class IdentityNode(Node):
    """
    Service node for per-tenant identity/persona management.
    
    Subscribes to:
        identity.query — return the identity profile for a tenant
        identity.update — create or update a tenant's identity
    
    Publishes:
        workspace.identity — identity context for workspace reasoning
    """

    def __init__(self, node_id: str, db_path: str | Path = "identity.db"):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.MEMORY,
            capabilities=["identity_management"],
        )
        self.store = IdentityStore(db_path)

    async def on_start(self) -> None:
        logger.info("Starting IdentityNode")
        await self.bus.subscribe("identity.query", self.handle_query)
        await self.bus.subscribe("identity.update", self.handle_update)

    async def on_stop(self) -> None:
        logger.info("Stopping IdentityNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def handle_query(self, message: Message) -> Message | None:
        """Return the identity profile for the requesting tenant."""
        try:
            tenant_id = message.tenant_id or message.payload.get("tenant_id", "default")
            profile = self.store.get(tenant_id)

            if profile is None:
                return message.create_response({
                    "found": False,
                    "profile": None,
                    "context_string": "",
                })

            return message.create_response({
                "found": True,
                "profile": profile.to_dict(),
                "context_string": profile.to_context_string(),
            })

        except Exception as e:
            logger.error("Identity query failed: %s", e)
            return message.create_error(str(e))

    async def handle_update(self, message: Message) -> Message | None:
        """Create or update a tenant's identity profile."""
        try:
            payload = message.payload
            tenant_id = message.tenant_id or payload.get("tenant_id", "default")

            profile = IdentityProfile(
                tenant_id=tenant_id,
                persona_name=payload.get("persona_name", "default"),
                system_prompt=payload.get("system_prompt", ""),
                goals=payload.get("goals", []),
                constraints=payload.get("constraints", []),
                personality_traits=payload.get("personality_traits", {}),
            )

            self.store.upsert(profile)
            logger.info("Updated identity for tenant '%s'", tenant_id)

            # Publish identity context to workspace
            await self.publish("workspace.identity", Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=tenant_id,
                topic="workspace.identity",
                payload={
                    "tenant_id": tenant_id,
                    "context_string": profile.to_context_string(),
                    "profile": profile.to_dict(),
                },
            ))

            return message.create_response({"status": "updated", "tenant_id": tenant_id})

        except Exception as e:
            logger.error("Identity update failed: %s", e)
            return message.create_error(str(e))
