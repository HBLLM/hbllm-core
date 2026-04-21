import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from typing import Any

from hbllm.network.messages import FeedbackPayload, Message, MessageType, SystemImprovePayload
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class MetaReasoningNode(Node):
    """
    AGI Layer: Meta-Reasoning Supervisor.

    Monitors system-wide user feedback. If it detects a systemic weakness
    in a specific domain (high negative feedback volume), it orchestrates a
    self-improvement offline loop by dumping failed interactions to a reflection
    dataset and signaling the admin/system bus to trigger heavy offline fine-tuning.
    """

    def __init__(
        self,
        node_id: str,
        weakness_threshold: int = 3,
        cooldown_seconds: float = 300.0,
    ) -> None:
        super().__init__(node_id=node_id, node_type=NodeType.ROUTER)

        # Configurable threshold and cooldown
        self.weakness_threshold = weakness_threshold
        self.cooldown_seconds = cooldown_seconds

        # In-memory buffer (fast access) + persistence
        self.negative_feedback_buffer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._last_reflection_time: dict[str, float] = {}

        self.reflection_dir = "workspace/reflection"
        os.makedirs(self.reflection_dir, exist_ok=True)

        # SQLite persistence for feedback buffer
        self._db_path = os.path.join(self.reflection_dir, "meta_feedback.db")
        self._init_db()
        self._load_from_db()

    def _init_db(self) -> None:
        """Initialize SQLite schema for persistent feedback storage."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS negative_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    instruction TEXT,
                    response TEXT,
                    created_at REAL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nf_domain ON negative_feedback(domain)")

    def _load_from_db(self) -> None:
        """Restore in-memory buffer from SQLite on startup."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT domain, instruction, response FROM negative_feedback"
            ).fetchall()
            for row in rows:
                self.negative_feedback_buffer[row["domain"]].append(
                    {
                        "instruction": row["instruction"],
                        "response": row["response"],
                        "rejected": True,
                        "domain": row["domain"],
                    }
                )

    async def on_start(self) -> None:
        """Subscribe to feedback and salience broadcasts."""
        logger.info("Starting MetaReasoningNode '%s' Supervisor", self.node_id)
        await self.bus.subscribe("system.feedback", self.handle_message)
        await self.bus.subscribe("system.salience", self.handle_salience)

    async def on_stop(self) -> None:
        """Clean up."""
        logger.info("Stopping MetaReasoningNode")

    async def handle_salience(self, message: Message) -> None:
        """Handle high-salience events for reflection."""
        if message.type != MessageType.SALIENCE_SCORE:
            return

        payload = message.payload
        if payload.get("is_priority"):
            logger.info(
                "MetaReasoningNode detected high-salience event. Triggering priority reflection."
            )
            # Map salience to the reflection engine (Node M)
            await self._trigger_reflection(
                domain="high_salience",
                reason=f"High saliency score detected: {payload.get('score')}",
                content=str(payload.get("content", "")),
            )

    async def handle_message(self, message: Message) -> Message | None:
        """Process incoming feedback silently to monitor health."""
        if message.type != MessageType.FEEDBACK:
            return None

        try:
            payload = FeedbackPayload(**message.payload)
        except Exception:
            return None  # Ignore invalid

        domain = payload.module_id or "general"
        rating = payload.rating

        if rating == -1:
            logger.warning("MetaReasoningNode detected negative feedback for domain '%s'", domain)

            # Store the interaction context if available
            if payload.prompt and payload.response:
                sample = {
                    "instruction": payload.prompt,
                    "response": payload.response,
                    "rejected": True,
                    "domain": domain,
                }
                self.negative_feedback_buffer[domain].append(sample)

                # Persist to SQLite
                try:
                    with sqlite3.connect(self._db_path) as conn:
                        conn.execute(
                            "INSERT INTO negative_feedback (domain, instruction, response, created_at) VALUES (?, ?, ?, ?)",
                            (domain, payload.prompt, payload.response, time.time()),
                        )
                except Exception as e:
                    logger.warning("Failed to persist feedback to SQLite: %s", e)

                # Check if this crosses the systemic weakness threshold
                if len(self.negative_feedback_buffer[domain]) >= self.weakness_threshold:
                    # Enforce cooldown to prevent rapid-fire reflection
                    last = self._last_reflection_time.get(domain, 0.0)
                    if time.time() - last < self.cooldown_seconds:
                        logger.info(
                            "MetaReasoningNode: cooldown active for domain '%s', skipping reflection",
                            domain,
                        )
                    else:
                        await self._trigger_reflection(domain)

        return None

    async def _trigger_reflection(
        self, domain: str, reason: str | None = None, content: str | None = None
    ) -> None:
        """Creates a reflection dataset and triggers the self-improvement loop."""
        logger.critical("--- REFLECTION INITIATED FOR DOMAIN '%s' ---", domain.upper())
        logger.info("MetaReasoningNode is initiating a self-improvement loop.")

        # 1. Dump dataset to disk
        filename = f"reflection_{domain}_{uuid.uuid4().hex[:8]}.jsonl"
        filepath = os.path.join(self.reflection_dir, filename)

        dataset = self.negative_feedback_buffer[domain]
        if not dataset and content:
            dataset = [{"content": content, "domain": domain}]

        effective_reason = (
            reason or f"Accumulated {self.weakness_threshold} negative feedback events recently."
        )

        try:
            # Thread file IO
            def _write() -> None:
                with open(filepath, "w") as f:
                    for item in dataset:
                        f.write(json.dumps(item) + "\n")

            await asyncio.to_thread(_write)
            logger.info("Saved reflection dataset to %s", filepath)

        except Exception as e:
            logger.error("Failed to dump reflection dataset: %s", e)
            return

        # 2. Fire the improvement signal over the bus
        improve_msg = Message(
            type=MessageType.SYSTEM_IMPROVE,
            source_node_id=self.node_id,
            target_node_id="",
            topic="system.improve",
            payload=SystemImprovePayload(
                domain=domain, reasoning=effective_reason, dataset_path=filepath
            ).model_dump(),
        )
        await self.bus.publish("system.improve", improve_msg)

        # 3. Clear the buffer (both in-memory and on disk)
        self.negative_feedback_buffer[domain] = []
        self._last_reflection_time[domain] = time.time()
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("DELETE FROM negative_feedback WHERE domain = ?", (domain,))
        except Exception as e:
            logger.warning("Failed to clear SQLite feedback for domain '%s': %s", domain, e)

    def stats(self) -> dict[str, Any]:
        """Return meta-reasoning statistics."""
        return {
            "buffered_domains": {k: len(v) for k, v in self.negative_feedback_buffer.items() if v},
            "weakness_threshold": self.weakness_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "last_reflections": dict(self._last_reflection_time),
        }
