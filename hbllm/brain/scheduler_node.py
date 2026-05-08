"""Generalized event-driven task scheduler backed by SQLite.

Manages a local SQLite database of scheduled tasks and emits them onto the
MessageBus when they become due. Supports one-off and recurring cron-style
triggers via human-readable interval expressions (e.g. 'every 5 minutes').
"""

import asyncio
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeInfo, NodeType

logger = logging.getLogger(__name__)

# ── Cron interval parsing ────────────────────────────────────────────────────
_INTERVAL_RE = re.compile(
    r"^every\s+(\d+)\s*(s|sec|seconds?|m|min|minutes?|h|hr|hours?|d|days?)$",
    re.IGNORECASE,
)
_UNIT_SECONDS = {
    "s": 1,
    "sec": 1,
    "second": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hr": 3600,
    "hour": 3600,
    "hours": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
}


def parse_interval_seconds(expr: str) -> float | None:
    """Parse a human-readable interval expression into seconds.

    Supports: 'every 5 minutes', 'every 30s', 'every 1 hour', etc.
    Returns None if the expression cannot be parsed.
    """
    match = _INTERVAL_RE.match(expr.strip())
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2).lower()
    multiplier = _UNIT_SECONDS.get(unit)
    if multiplier is None:
        return None
    return float(value * multiplier)


class SchedulerNode(Node):
    """
    Generalized event-driven scheduler.
    Manages a local SQLite database of scheduled tasks and emits them to the
    MessageBus when they are due.
    """

    def __init__(self, node_id: str, data_dir: str, tick_interval: float = 5.0):
        super().__init__(node_id, node_type=NodeType.CORE)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "scheduler.db"
        self.tick_interval = tick_interval
        self._loop_task: asyncio.Task[None] | None = None
        self._running = False

        # Telemetry counters
        self._tasks_scheduled = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._tasks_rescheduled = 0

        self._init_db()

    @property
    def info(self) -> NodeInfo:
        return NodeInfo(
            node_id=self.node_id,
            node_type=NodeType.WORKER,  # Acts as a background worker
            description="Proactive event scheduler and autonomous task runner.",
        )

    # ── Sync SQLite helpers (safe to call via asyncio.to_thread) ──────────

    def _db_get_due_tasks(self, now: float) -> list[dict[str, Any]]:
        """Return all pending tasks whose trigger_time <= now."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM scheduled_tasks WHERE status = 'pending' AND trigger_time <= ?",
                (now,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def _db_mark_processing(self, task_id: str) -> None:
        """Transition a task to the 'processing' state."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE scheduled_tasks SET status = 'processing' WHERE task_id = ?",
                (task_id,),
            )

    def _db_resolve_task(self, task: dict[str, Any], success: bool, now: float) -> str:
        """Persist the post-execution state of a task.

        Returns an outcome string: 'rescheduled', 'completed', 'retrying', or 'failed'.
        """
        with sqlite3.connect(self.db_path) as conn:
            if success:
                if task["cron_expression"]:
                    interval = parse_interval_seconds(task["cron_expression"])
                    if interval and interval > 0:
                        conn.execute(
                            "UPDATE scheduled_tasks SET status = 'pending', trigger_time = ? WHERE task_id = ?",
                            (now + interval, task["task_id"]),
                        )
                        return "rescheduled"
                    else:
                        logger.warning(
                            "Could not parse cron expression '%s' for task %s; marking completed",
                            task["cron_expression"],
                            task["task_id"],
                        )
                conn.execute(
                    "UPDATE scheduled_tasks SET status = 'completed' WHERE task_id = ?",
                    (task["task_id"],),
                )
                return "completed"
            else:
                if task["retry_policy"] == "retry":
                    conn.execute(
                        "UPDATE scheduled_tasks SET status = 'pending', trigger_time = ? WHERE task_id = ?",
                        (now + 60.0, task["task_id"]),
                    )
                    return "retrying"
                conn.execute(
                    "UPDATE scheduled_tasks SET status = 'failed' WHERE task_id = ?",
                    (task["task_id"],),
                )
                return "failed"

    # ── Schema init ──────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Initialize the SQLite schema (called synchronously at construction time)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    task_id TEXT PRIMARY KEY,
                    tenant_id TEXT,
                    trigger_time REAL,
                    cron_expression TEXT,
                    route_topic TEXT,
                    payload TEXT,
                    retry_policy TEXT,
                    status TEXT,
                    created_at REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trigger_time ON scheduled_tasks(trigger_time)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON scheduled_tasks(status)")

    async def start(self, bus: Any) -> None:
        """Start the node and the background ticking loop."""
        await super().start(bus)

        # Subscribe to scheduler commands
        await self.bus.subscribe("system.scheduler.schedule", self._handle_schedule_command)
        await self.bus.subscribe("system.scheduler.cancel", self._handle_cancel_command)

        self._running = True
        self._loop_task = asyncio.create_task(self._tick_loop())
        logger.info("SchedulerNode %s started. DB: %s", self.node_id, self.db_path)

    async def on_start(self) -> None:
        """Lifecycle hook."""
        pass

    async def on_stop(self) -> None:
        """Lifecycle hook."""
        pass

    async def handle_message(self, message: Message) -> Message | None:
        """Default handler for generic messages. Unused as we route commands specifically."""
        return None

    async def _handle_schedule_command(self, msg: Message) -> None:
        """Handle incoming schedule commands."""
        # Unpack payload into schedule_event
        p = msg.payload
        self.schedule_event(
            task_id=p.get("task_id", f"task_{time.time()}"),
            tenant_id=p.get("tenant_id", "default"),
            trigger_time=p.get("trigger_time", 0.0),
            route_topic=p.get("route_topic", "cognitive.inference"),
            payload=p.get("payload", {}),
            cron_expression=p.get("cron_expression"),
            retry_policy=p.get("retry_policy", "fire_and_forget"),
        )

    async def _handle_cancel_command(self, msg: Message) -> None:
        """Handle incoming cancel commands."""
        task_id = msg.payload.get("task_id")
        if task_id:
            self.cancel_task(task_id)

    async def stop(self) -> None:
        """Stop the background loop and node."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        await super().stop()

    def schedule_event(
        self,
        task_id: str,
        tenant_id: str,
        trigger_time: float,
        route_topic: str,
        payload: dict[str, Any],
        cron_expression: str | None = None,
        retry_policy: str = "fire_and_forget",
    ) -> None:
        """Schedule a new generalized event."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scheduled_tasks
                (task_id, tenant_id, trigger_time, cron_expression, route_topic, payload, retry_policy, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    tenant_id,
                    trigger_time,
                    cron_expression,
                    route_topic,
                    json.dumps(payload),
                    retry_policy,
                    "pending",
                    time.time(),
                ),
            )
        self._tasks_scheduled += 1
        logger.debug("Scheduled task %s for %s on topic %s", task_id, trigger_time, route_topic)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task. Returns True if a task was modified."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE scheduled_tasks SET status = 'cancelled' WHERE task_id = ? AND status = 'pending'",
                (task_id,),
            )
            return cursor.rowcount > 0

    def stats(self) -> dict[str, Any]:
        """Return scheduler telemetry."""
        pending = 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM scheduled_tasks WHERE status = 'pending'"
                ).fetchone()
                pending = row[0] if row else 0
        except Exception:
            pass
        return {
            "tasks_scheduled": self._tasks_scheduled,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "tasks_rescheduled": self._tasks_rescheduled,
            "pending_queue_depth": pending,
        }

    async def _tick_loop(self) -> None:
        """Background loop to poll for due tasks and publish them."""
        while self._running:
            try:
                await self._process_due_tasks()
            except Exception as e:
                logger.error("Error in SchedulerNode tick loop: %s", e)

            await asyncio.sleep(self.tick_interval)

    async def _process_due_tasks(self) -> None:
        """Query tasks past their trigger_time and dispatch them.

        All SQLite operations are offloaded to a thread pool via
        ``asyncio.to_thread`` to avoid blocking the event loop.
        """
        now = time.time()
        due_tasks = await asyncio.to_thread(self._db_get_due_tasks, now)
        if not due_tasks:
            return

        for task in due_tasks:
            await asyncio.to_thread(self._db_mark_processing, task["task_id"])

            success = await self._execute_task(task)

            outcome = await asyncio.to_thread(self._db_resolve_task, task, success, now)
            if outcome == "rescheduled":
                self._tasks_rescheduled += 1
                logger.debug("Rescheduled recurring task %s", task["task_id"])
            elif outcome == "completed":
                self._tasks_completed += 1
            elif outcome == "failed":
                self._tasks_failed += 1

    async def _execute_task(self, task: dict[str, Any]) -> bool:
        """Publish the scheduled task to the MessageBus."""
        try:
            payload = json.loads(task["payload"])
            msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic=task["route_topic"],
                payload=payload,
            )
            if self.bus:
                await self.bus.publish(task["route_topic"], msg)
            return True
        except Exception as e:
            logger.error("Failed to execute task %s: %s", task["task_id"], e)
            return False
