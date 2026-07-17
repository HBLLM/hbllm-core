"""
Identity State — Persistent cognitive identity across sessions and restarts.

The Identity State is what makes HBLLM feel like the *same* agent across
conversations. It preserves:
    - Agent-level continuity (total sessions, creation date, agent ID)
    - Cross-session narrative memory (what happened in past sessions)
    - Unified identity context for system prompt enrichment

This module is a **facade** that composes existing subsystems rather
than duplicating them:

    PersonaEngine      → Personality traits (formality, humor, empathy)
    UserModelEngine    → Per-user preferences, expertise, trust
    SelfModel          → Domain capabilities, strengths, weaknesses
    RelationshipMemory → Social graph, person mentions

This module ONLY owns:
    1. CoreIdentity — agent-level persistent stats (agent_id, total_sessions)
    2. SessionNarrativeStore — cross-session summaries

Architecture::

    Session ends
        ↓
    IdentityStateManager.end_session(summary=...)
        ↓
    SQLite: identity_state.db (CoreIdentity + SessionNarratives)
        ↓
    Next session starts
        ↓
    IdentityStateManager.begin_session()
        ↓
    get_identity_context() composes:
        PersonaEngine.to_system_prompt_fragment()
        + SelfModel.get_strengths() / get_weaknesses()
        + SessionNarrativeStore (last 3 sessions)
        + UserModelEngine.get_model() (expertise, focus)

Usage::

    from hbllm.brain.self_model.identity import IdentityStateManager

    identity = IdentityStateManager(data_dir="data")
    await identity.init_db()

    # Restore identity at boot
    state = await identity.restore()

    # Begin a new session
    await identity.begin_session()

    # End session with summary
    await identity.end_session(
        user_id="dumith",
        tenant_id="default",
        summary="Discussed HBLLM plugin system architecture.",
        topics=["architecture", "plugins"],
        message_count=24,
    )

    # Get unified identity context for system prompt
    context = await identity.get_identity_context(
        user_id="dumith",
        persona_engine=persona_engine,
        self_model=self_model,
        user_model_engine=user_model_engine,
    )
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Core Identity (agent-level — NOT per-user)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CoreIdentity:
    """The agent's persistent core identity.

    This is purely agent-level metadata that survives restarts.
    Persona traits live in PersonaEngine; user preferences live in
    UserModelEngine. This only tracks what neither of those covers.
    """

    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = "HBLLM"
    version: str = "1.0.0"

    # Accumulated experience counters
    total_sessions: int = 0
    total_interactions: int = 0
    total_users_served: int = 0
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "total_sessions": self.total_sessions,
            "total_interactions": self.total_interactions,
            "total_users_served": self.total_users_served,
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Session Narrative (cross-session continuity)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SessionNarrative:
    """Cross-session narrative continuity entry.

    Stores a compact summary of what happened in a session so the agent
    can reference past conversations ("Last time we discussed...").
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    user_id: str = ""
    tenant_id: str = "default"
    summary: str = ""
    topics: list[str] = field(default_factory=list)
    sentiment: float = 0.5
    message_count: int = 0
    created_at: float = field(default_factory=time.time)
    workspace_id: str = "default"


# ═══════════════════════════════════════════════════════════════════════════
# Identity State Manager (Facade)
# ═══════════════════════════════════════════════════════════════════════════


class IdentityStateManager:
    """Manages persistent cognitive identity across sessions.

    Owns only:
      1. CoreIdentity — agent-level stats (total_sessions, agent_id)
      2. SessionNarrativeStore — cross-session summaries

    For everything else, it delegates to existing subsystems:
      - PersonaEngine → personality traits, style modulation
      - UserModelEngine → per-user preferences, expertise, trust
      - SelfModel → domain capabilities, strengths/weaknesses
      - RelationshipMemory → social graph, person trust

    This avoids duplicating the well-wired subsystems that already
    exist in ``hbllm.brain.social.*`` and ``hbllm.brain.self_model.*``.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self._db_path = Path(data_dir) / "identity_state.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._identity: CoreIdentity | None = None
        self._known_users: set[str] = set()

    async def init_db(self) -> None:
        """Create identity tables."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS core_identity (
                    agent_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT DEFAULT '1.0.0',
                    total_sessions INTEGER DEFAULT 0,
                    total_interactions INTEGER DEFAULT 0,
                    total_users_served INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_active_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_narratives (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    tenant_id TEXT DEFAULT 'default',
                    summary TEXT NOT NULL,
                    topics TEXT DEFAULT '[]',
                    sentiment REAL DEFAULT 0.5,
                    message_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    workspace_id TEXT DEFAULT 'default'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_narratives_user
                ON session_narratives(user_id, tenant_id, created_at DESC)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS known_users (
                    user_id TEXT PRIMARY KEY,
                    first_seen_at REAL NOT NULL
                )
            """)
            conn.commit()

        # Load known users for dedup
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute("SELECT user_id FROM known_users").fetchall()
            self._known_users = {r[0] for r in rows}

        logger.info("IdentityStateManager initialized (db=%s)", self._db_path)

    # ── Core Identity Lifecycle ──────────────────────────────────────────

    async def restore(self) -> CoreIdentity:
        """Restore identity state from persistence, or create a new one."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM core_identity ORDER BY created_at ASC LIMIT 1"
            ).fetchone()

        if row:
            self._identity = CoreIdentity(
                agent_id=row["agent_id"],
                name=row["name"],
                version=row["version"],
                total_sessions=row["total_sessions"],
                total_interactions=row["total_interactions"],
                total_users_served=row["total_users_served"],
                created_at=row["created_at"],
                last_active_at=row["last_active_at"],
            )
            logger.info(
                "Restored identity '%s' (sessions=%d, interactions=%d, users=%d)",
                self._identity.name,
                self._identity.total_sessions,
                self._identity.total_interactions,
                self._identity.total_users_served,
            )
        else:
            self._identity = CoreIdentity()
            await self._persist_identity()
            logger.info(
                "Created new identity '%s' (id=%s)", self._identity.name, self._identity.agent_id
            )

        return self._identity

    @property
    def identity(self) -> CoreIdentity:
        """Get the current identity (call restore() first)."""
        if self._identity is None:
            self._identity = CoreIdentity()
        return self._identity

    async def persist(self) -> None:
        """Persist current identity state."""
        if self._identity:
            self._identity.last_active_at = time.time()
            await self._persist_identity()

    # ── Session Lifecycle ────────────────────────────────────────────────

    async def begin_session(self) -> None:
        """Called when a new session starts."""
        self.identity.total_sessions += 1
        await self.persist()
        logger.debug("Session #%d started", self.identity.total_sessions)

    async def end_session(
        self,
        user_id: str,
        *,
        tenant_id: str = "default",
        summary: str = "",
        topics: list[str] | None = None,
        sentiment: float = 0.5,
        message_count: int = 0,
        workspace_id: str = "default",
    ) -> None:
        """Called when a session ends. Records narrative and updates counters."""
        self.identity.total_interactions += message_count

        # Track new users
        if user_id not in self._known_users:
            self._known_users.add(user_id)
            self.identity.total_users_served += 1
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO known_users (user_id, first_seen_at) VALUES (?, ?)",
                    (user_id, time.time()),
                )
                conn.commit()

        # Store session narrative if a summary was provided
        if summary:
            narrative = SessionNarrative(
                user_id=user_id,
                tenant_id=tenant_id,
                summary=summary,
                topics=topics or [],
                sentiment=sentiment,
                message_count=message_count,
                workspace_id=workspace_id,
            )
            await self._persist_narrative(narrative)

        await self.persist()

    # ── Session Narratives ───────────────────────────────────────────────

    async def get_recent_narratives(
        self,
        user_id: str,
        tenant_id: str = "default",
        limit: int = 5,
    ) -> list[SessionNarrative]:
        """Get recent session narratives for a user."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM session_narratives
                WHERE user_id = ? AND tenant_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (user_id, tenant_id, limit),
            ).fetchall()

        return [
            SessionNarrative(
                id=row["id"],
                user_id=row["user_id"],
                tenant_id=row["tenant_id"],
                summary=row["summary"],
                topics=json.loads(row["topics"] or "[]"),
                sentiment=row["sentiment"],
                message_count=row["message_count"],
                created_at=row["created_at"],
                workspace_id=row["workspace_id"],
            )
            for row in rows
        ]

    # ── Unified Identity Context (Facade) ────────────────────────────────

    async def get_identity_context(
        self,
        user_id: str | None = None,
        tenant_id: str = "default",
        *,
        persona_engine: Any = None,
        self_model: Any = None,
        user_model_engine: Any = None,
    ) -> str:
        """Generate unified identity context for system prompt enrichment.

        Composes data from multiple existing subsystems into a single
        context block that can be injected into the LLM system prompt.

        Sources:
            1. CoreIdentity — agent-level stats (this module)
            2. PersonaEngine — style instructions (existing)
            3. SelfModel — strengths/weaknesses (existing)
            4. UserModelEngine — user expertise/preferences (existing)
            5. SessionNarratives — past session summaries (this module)

        Args:
            user_id: Current user ID.
            tenant_id: Current tenant ID.
            persona_engine: Optional PersonaEngine instance.
            self_model: Optional SelfModel instance.
            user_model_engine: Optional UserModelEngine instance.

        Returns:
            A compact text block for system prompt injection.
        """
        identity = self.identity
        lines: list[str] = []

        # ── 1. Agent identity ────────────────────────────────────────
        lines.append(f"You are {identity.name}.")
        if identity.total_sessions > 0:
            lines.append(
                f"You have completed {identity.total_sessions} sessions "
                f"with {identity.total_users_served} users "
                f"({identity.total_interactions} total interactions)."
            )

        # ── 2. Self-model: strengths/weaknesses ──────────────────────
        if self_model:
            try:
                strengths = self_model.get_strengths()
                if strengths:
                    lines.append(f"Your strengths: {', '.join(strengths[:5])}.")
                weaknesses = self_model.get_weaknesses()
                if weaknesses:
                    lines.append(f"Areas to improve: {', '.join(weaknesses[:3])}.")
            except Exception:
                pass

        # ── 3. Persona style ─────────────────────────────────────────
        if persona_engine:
            try:
                style = persona_engine.get_modulated_prompt(tenant_id)
                if style:
                    lines.append(f"Style: {style}")
            except Exception:
                pass

        # ── 4. User context ──────────────────────────────────────────
        if user_id and user_model_engine:
            try:
                model = user_model_engine.get_model(tenant_id)
                if model:
                    lines.append("")
                    name = model.display_name or user_id
                    lines.append(f"Current user: {name}")

                    # Expertise domains
                    if model.expertise:
                        top = sorted(
                            model.expertise.values(),
                            key=lambda e: e.level.value,
                            reverse=True,
                        )[:5]
                        domains = [f"{e.domain} ({e.level.value:.0%})" for e in top]
                        lines.append(f"Their expertise: {', '.join(domains)}.")

                    # Current focus
                    if model.current_focus.value and model.current_focus.confidence > 0.3:
                        lines.append(f"Current focus: {model.current_focus.value}")

                    # Stress
                    if model.stress_level > 0.6:
                        lines.append("They seem under pressure — be concise and supportive.")
            except Exception:
                pass

        # ── 5. Cross-session narrative ────────────────────────────────
        if user_id:
            try:
                narratives = await self.get_recent_narratives(user_id, tenant_id, limit=3)
                if narratives:
                    lines.append("")
                    lines.append("Recent sessions:")
                    for n in narratives:
                        topic_str = f" [{', '.join(n.topics[:3])}]" if n.topics else ""
                        lines.append(f"  - {n.summary}{topic_str}")
            except Exception:
                pass

        return "\n".join(lines)

    # ── Internal Persistence ─────────────────────────────────────────────

    async def _persist_identity(self) -> None:
        """Write core identity to database."""
        identity = self.identity
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO core_identity
                (agent_id, name, version,
                 total_sessions, total_interactions, total_users_served,
                 created_at, last_active_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identity.agent_id,
                    identity.name,
                    identity.version,
                    identity.total_sessions,
                    identity.total_interactions,
                    identity.total_users_served,
                    identity.created_at,
                    identity.last_active_at,
                ),
            )
            conn.commit()

    async def _persist_narrative(self, narrative: SessionNarrative) -> None:
        """Write session narrative to database."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                INSERT INTO session_narratives
                (id, user_id, tenant_id, summary, topics, sentiment,
                 message_count, created_at, workspace_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    narrative.id,
                    narrative.user_id,
                    narrative.tenant_id,
                    narrative.summary,
                    json.dumps(narrative.topics),
                    narrative.sentiment,
                    narrative.message_count,
                    narrative.created_at,
                    narrative.workspace_id,
                ),
            )
            conn.commit()
        logger.debug(
            "Stored session narrative for user '%s': %s",
            narrative.user_id,
            narrative.summary[:60],
        )
