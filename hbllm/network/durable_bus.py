"""
Durable Message Bus — wraps any MessageBus with persistence.

Provides:
  - At-least-once delivery guarantee via SQLite journal
  - Dead letter queue for failed messages
  - Configurable retry with exponential backoff
  - Message deduplication

Wraps InProcessBus or RedisBus transparently.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

from hbllm.network.bus import MessageBus, InProcessBus
from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


class MessageStatus(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD = "dead"  # Max retries exceeded


@dataclass
class DurableMessage:
    """A message with durability metadata."""
    id: str
    topic: str
    payload_json: str
    status: MessageStatus = MessageStatus.PENDING
    attempts: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    next_retry_at: float = 0.0
    error: str = ""


class DurableBus(MessageBus):
    """
    Wraps any MessageBus with SQLite-backed persistence.

    Messages are journaled before publish and marked delivered on
    successful handler execution. Failed messages are retried with
    exponential backoff. Messages exceeding max_retries go to the
    dead letter queue.

    Usage:
        inner_bus = InProcessBus()
        bus = DurableBus(inner_bus, db_path="messages.db")
        await bus.start()
    """

    def __init__(
        self,
        inner: MessageBus | None = None,
        db_path: str = "hbllm_messages.db",
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        self._inner = inner or InProcessBus()
        self._db_path = db_path
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._conn: sqlite3.Connection | None = None
        self._retry_task: asyncio.Task | None = None
        self._running = False
        self._seen_ids: set[str] = set()

    async def start(self) -> None:
        """Initialize DB and start inner bus + retry loop."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                created_at REAL NOT NULL,
                next_retry_at REAL DEFAULT 0,
                error TEXT DEFAULT ''
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON messages(status)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_retry ON messages(next_retry_at)
        """)
        self._conn.commit()

        await self._inner.start()
        self._running = True
        self._retry_task = asyncio.create_task(self._retry_loop())
        logger.info("DurableBus started (db=%s, max_retries=%d)", self._db_path, self._max_retries)

    async def stop(self) -> None:
        """Stop retry loop, inner bus, and close DB."""
        self._running = False
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass

        await self._inner.stop()

        if self._conn:
            self._conn.close()
            self._conn = None

        logger.info("DurableBus stopped")

    async def publish(self, topic: str, message: Message) -> None:
        """Journal message then publish to inner bus."""
        msg_id = message.id or str(uuid.uuid4())

        # Deduplication
        if msg_id in self._seen_ids:
            return
        self._seen_ids.add(msg_id)
        # Keep seen set bounded
        if len(self._seen_ids) > 10000:
            self._seen_ids = set(list(self._seen_ids)[-5000:])

        # Journal
        self._journal(msg_id, topic, message)

        # Publish
        try:
            await self._inner.publish(topic, message)
            self._mark_delivered(msg_id)
        except Exception as e:
            self._mark_failed(msg_id, str(e))
            logger.warning("Publish failed for %s, will retry: %s", msg_id, e)

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Message], Awaitable[Message | None]],
    ) -> str:
        """Subscribe via inner bus."""
        return await self._inner.subscribe(topic, handler)

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe via inner bus."""
        await self._inner.unsubscribe(subscription_id)

    async def request(
        self,
        topic: str,
        message: Message,
        timeout: float = 10.0,
    ) -> Message:
        """Request/reply via inner bus (not journaled — synchronous)."""
        return await self._inner.request(topic, message, timeout)

    # ─── Persistence ──────────────────────────────────────────────────────

    def _journal(self, msg_id: str, topic: str, message: Message) -> None:
        """Write message to SQLite journal."""
        if not self._conn:
            return

        try:
            payload = json.dumps({
                "type": message.type.value if hasattr(message.type, 'value') else str(message.type),
                "source_node_id": message.source_node_id,
                "topic": topic,
                "tenant_id": message.tenant_id,
                "session_id": message.session_id,
                "correlation_id": message.correlation_id,
                "payload": message.payload,
            })

            self._conn.execute(
                """INSERT OR IGNORE INTO messages
                   (id, topic, payload_json, status, max_retries, created_at)
                   VALUES (?, ?, ?, 'pending', ?, ?)""",
                (msg_id, topic, payload, self._max_retries, time.time()),
            )
            self._conn.commit()
        except Exception as e:
            logger.error("Journal write failed: %s", e)

    def _mark_delivered(self, msg_id: str) -> None:
        """Mark message as successfully delivered."""
        if not self._conn:
            return
        self._conn.execute(
            "UPDATE messages SET status='delivered' WHERE id=?", (msg_id,)
        )
        self._conn.commit()

    def _mark_failed(self, msg_id: str, error: str) -> None:
        """Mark message as failed with retry scheduling."""
        if not self._conn:
            return

        row = self._conn.execute(
            "SELECT attempts, max_retries FROM messages WHERE id=?", (msg_id,)
        ).fetchone()

        if not row:
            return

        attempts, max_retries = row
        attempts += 1

        if attempts >= max_retries:
            self._conn.execute(
                "UPDATE messages SET status='dead', attempts=?, error=? WHERE id=?",
                (attempts, error, msg_id),
            )
        else:
            delay = min(self._base_delay * (2 ** attempts), self._max_delay)
            next_retry = time.time() + delay
            self._conn.execute(
                "UPDATE messages SET status='failed', attempts=?, next_retry_at=?, error=? WHERE id=?",
                (attempts, next_retry, error, msg_id),
            )
        self._conn.commit()

    # ─── Retry Loop ───────────────────────────────────────────────────────

    async def _retry_loop(self) -> None:
        """Periodically retry failed messages."""
        while self._running:
            try:
                await asyncio.sleep(2.0)
                if not self._conn:
                    continue

                now = time.time()
                rows = self._conn.execute(
                    """SELECT id, topic, payload_json FROM messages
                       WHERE status='failed' AND next_retry_at <= ?
                       LIMIT 10""",
                    (now,),
                ).fetchall()

                for msg_id, topic, payload_json in rows:
                    try:
                        data = json.loads(payload_json)
                        message = Message(
                            id=msg_id,
                            type=MessageType(data.get("type", "event")),
                            source_node_id=data.get("source_node_id", "retry"),
                            topic=topic,
                            tenant_id=data.get("tenant_id", "default"),
                            session_id=data.get("session_id", "default"),
                            correlation_id=data.get("correlation_id"),
                            payload=data.get("payload", {}),
                        )

                        await self._inner.publish(topic, message)
                        self._mark_delivered(msg_id)
                        logger.info("Retry succeeded for message %s", msg_id)

                    except Exception as e:
                        self._mark_failed(msg_id, str(e))
                        logger.warning("Retry failed for %s: %s", msg_id, e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Retry loop error: %s", e)

    # ─── Dead Letter Queue ────────────────────────────────────────────────

    def get_dead_letters(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve messages from the dead letter queue."""
        if not self._conn:
            return []

        rows = self._conn.execute(
            """SELECT id, topic, payload_json, attempts, created_at, error
               FROM messages WHERE status='dead'
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()

        return [
            {
                "id": r[0], "topic": r[1], "payload": json.loads(r[2]),
                "attempts": r[3], "created_at": r[4], "error": r[5],
            }
            for r in rows
        ]

    def dead_letter_count(self) -> int:
        """Count of dead letter messages."""
        if not self._conn:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM messages WHERE status='dead'").fetchone()
        return row[0] if row else 0

    def pending_count(self) -> int:
        """Count of pending/failed (retryable) messages."""
        if not self._conn:
            return 0
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE status IN ('pending', 'failed')"
        ).fetchone()
        return row[0] if row else 0

    def stats(self) -> dict[str, Any]:
        """Get bus durability stats."""
        if not self._conn:
            return {"status": "not_initialized"}

        counts = {}
        for status in ("pending", "delivered", "failed", "dead"):
            row = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE status=?", (status,)
            ).fetchone()
            counts[status] = row[0] if row else 0

        return {
            "db_path": self._db_path,
            "max_retries": self._max_retries,
            "inner_bus": type(self._inner).__name__,
            **counts,
        }
