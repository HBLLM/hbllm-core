"""
ConversationThread — Named, resumable conversation threads.

Unlike flat chat history, threads allow users to maintain multiple
concurrent lines of discussion that can be resumed by name:

    "Continue the server analysis from yesterday"
    "Switch to the deployment planning thread"

Each thread has its own context window, turn history, and metadata.
Threads persist across sessions and can be searched by name or topic.

Bus Topics:
    thread.create    → Create a new named thread
    thread.resume    → Resume an existing thread by name/id
    thread.archive   → Archive a thread (keep data, remove from active)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass
class ThreadTurn:
    """A single turn (message) within a conversation thread."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThreadTurn:
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationThread:
    """A named, resumable conversation thread."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tenant_id: str = ""
    name: str = ""  # User-facing name, e.g. "Server Analysis"
    topic: str = ""  # Auto-detected or user-set topic tag
    turns: list[ThreadTurn] = field(default_factory=list)
    # Context metadata
    system_prompt: str = ""  # Thread-specific system context
    pinned_context: str = ""  # Persistent context carried across all turns
    # Lifecycle
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    archived: bool = False
    archived_at: float | None = None
    # Stats
    total_turns: int = 0

    @property
    def last_message(self) -> ThreadTurn | None:
        return self.turns[-1] if self.turns else None

    @property
    def summary_line(self) -> str:
        """One-line summary for thread listing."""
        last = self.last_message
        preview = ""
        if last:
            preview = last.content[:60] + ("..." if len(last.content) > 60 else "")
        return f"{self.name}: {preview}" if preview else self.name

    def add_turn(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> ThreadTurn:
        """Add a new turn to the thread."""
        turn = ThreadTurn(role=role, content=content, metadata=metadata or {})
        self.turns.append(turn)
        self.total_turns += 1
        self.updated_at = time.time()
        return turn

    def get_context_window(self, max_turns: int = 20) -> list[dict[str, str]]:
        """Get recent turns formatted for LLM context."""
        recent = self.turns[-max_turns:]
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.pinned_context:
            messages.append({"role": "system", "content": f"Thread context: {self.pinned_context}"})
        for turn in recent:
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "topic": self.topic,
            "turns": [t.to_dict() for t in self.turns],
            "system_prompt": self.system_prompt,
            "pinned_context": self.pinned_context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archived": self.archived,
            "archived_at": self.archived_at,
            "total_turns": self.total_turns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationThread:
        thread = cls(
            id=data["id"],
            tenant_id=data.get("tenant_id", ""),
            name=data.get("name", ""),
            topic=data.get("topic", ""),
            system_prompt=data.get("system_prompt", ""),
            pinned_context=data.get("pinned_context", ""),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            archived=data.get("archived", False),
            archived_at=data.get("archived_at"),
            total_turns=data.get("total_turns", 0),
        )
        thread.turns = [ThreadTurn.from_dict(t) for t in data.get("turns", [])]
        return thread


# ── Thread Manager ───────────────────────────────────────────────────────────


class ThreadManager:
    """
    Manages conversation threads with persistence.

    Usage:
        manager = ThreadManager(storage_dir="data/threads")

        # Create a new thread
        thread = manager.create("user_1", name="Server Analysis")

        # Add turns
        thread.add_turn("user", "What's the CPU usage pattern?")
        thread.add_turn("assistant", "Looking at the last 24h...")
        manager.save(thread)

        # Resume later
        thread = manager.get_by_name("user_1", "Server Analysis")
        context = thread.get_context_window()

        # List active threads
        threads = manager.list_threads("user_1")
    """

    def __init__(self, storage_dir: str | Path = "data/threads") -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        # In-memory index: tenant_id → {thread_id → ConversationThread}
        self._threads: dict[str, dict[str, ConversationThread]] = {}
        self._load_all()

        logger.info(
            "ThreadManager initialized with %d threads from %s",
            sum(len(v) for v in self._threads.values()),
            self._storage_dir,
        )

    def _tenant_dir(self, tenant_id: str) -> Path:
        d = self._storage_dir / tenant_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_all(self) -> None:
        """Load all threads from disk."""
        for tenant_dir in self._storage_dir.iterdir():
            if not tenant_dir.is_dir():
                continue
            tenant_id = tenant_dir.name
            self._threads[tenant_id] = {}
            for path in tenant_dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    thread = ConversationThread.from_dict(data)
                    self._threads[tenant_id][thread.id] = thread
                except Exception as e:
                    logger.warning("Failed to load thread from %s: %s", path, e)

    def save(self, thread: ConversationThread) -> None:
        """Persist a thread to disk."""
        path = self._tenant_dir(thread.tenant_id) / f"{thread.id}.json"
        try:
            path.write_text(json.dumps(thread.to_dict(), indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save thread %s: %s", thread.id, e)

    def create(
        self,
        tenant_id: str,
        name: str,
        topic: str = "",
        system_prompt: str = "",
        pinned_context: str = "",
    ) -> ConversationThread:
        """Create a new conversation thread."""
        thread = ConversationThread(
            tenant_id=tenant_id,
            name=name,
            topic=topic,
            system_prompt=system_prompt,
            pinned_context=pinned_context,
        )
        if tenant_id not in self._threads:
            self._threads[tenant_id] = {}
        self._threads[tenant_id][thread.id] = thread
        self.save(thread)
        logger.info("Created thread '%s' (id=%s) for tenant '%s'", name, thread.id, tenant_id)
        return thread

    def get(self, tenant_id: str, thread_id: str) -> ConversationThread | None:
        """Get a thread by ID."""
        return self._threads.get(tenant_id, {}).get(thread_id)

    def get_by_name(self, tenant_id: str, name: str) -> ConversationThread | None:
        """Get the most recent thread matching a name (case-insensitive)."""
        name_lower = name.lower()
        matches = [
            t
            for t in self._threads.get(tenant_id, {}).values()
            if name_lower in t.name.lower() and not t.archived
        ]
        if not matches:
            return None
        return max(matches, key=lambda t: t.updated_at)

    def list_threads(
        self,
        tenant_id: str,
        include_archived: bool = False,
        limit: int = 50,
    ) -> list[ConversationThread]:
        """List threads for a tenant, most recently updated first."""
        threads = list(self._threads.get(tenant_id, {}).values())
        if not include_archived:
            threads = [t for t in threads if not t.archived]
        threads.sort(key=lambda t: t.updated_at, reverse=True)
        return threads[:limit]

    def archive(self, tenant_id: str, thread_id: str) -> bool:
        """Archive a thread (keep data, remove from active list)."""
        thread = self.get(tenant_id, thread_id)
        if thread:
            thread.archived = True
            thread.archived_at = time.time()
            self.save(thread)
            logger.info("Archived thread '%s' for tenant '%s'", thread.name, tenant_id)
            return True
        return False

    def delete(self, tenant_id: str, thread_id: str) -> bool:
        """Permanently delete a thread."""
        threads = self._threads.get(tenant_id, {})
        if thread_id in threads:
            thread = threads.pop(thread_id)
            path = self._tenant_dir(tenant_id) / f"{thread_id}.json"
            if path.exists():
                path.unlink()
            logger.info("Deleted thread '%s' for tenant '%s'", thread.name, tenant_id)
            return True
        return False

    def search(self, tenant_id: str, query: str, limit: int = 10) -> list[ConversationThread]:
        """Search threads by name, topic, or content."""
        query_lower = query.lower()
        results = []
        for thread in self._threads.get(tenant_id, {}).values():
            score = 0
            if query_lower in thread.name.lower():
                score += 3
            if query_lower in thread.topic.lower():
                score += 2
            # Check recent turns
            for turn in thread.turns[-10:]:
                if query_lower in turn.content.lower():
                    score += 1
                    break
            if score > 0:
                results.append((score, thread))
        results.sort(key=lambda x: (x[0], x[1].updated_at), reverse=True)
        return [t for _, t in results[:limit]]

    def stats(self, tenant_id: str) -> dict[str, Any]:
        """Stats for a tenant's threads."""
        threads = list(self._threads.get(tenant_id, {}).values())
        active = [t for t in threads if not t.archived]
        return {
            "total_threads": len(threads),
            "active_threads": len(active),
            "archived_threads": len(threads) - len(active),
            "total_turns": sum(t.total_turns for t in threads),
        }
