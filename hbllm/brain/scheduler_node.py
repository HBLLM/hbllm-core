import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeInfo, NodeType

logger = logging.getLogger(__name__)


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
        self._init_db()

    @property
    def info(self) -> NodeInfo:
        return NodeInfo(
            node_id=self.node_id,
            node_type=NodeType.WORKER,  # Acts as a background worker
            description="Proactive event scheduler and autonomous task runner.",
        )

    def _init_db(self) -> None:
        """Initialize the SQLite schema."""
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trigger_time ON scheduled_tasks(trigger_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON scheduled_tasks(status)")

    async def start(self, bus: Any) -> None:
        """Start the node and the background ticking loop."""
        await super().start(bus)

        # Subscribe to scheduler commands
        await self.bus.subscribe("system.scheduler.schedule", self._handle_schedule_command)
        await self.bus.subscribe("system.scheduler.cancel", self._handle_cancel_command)

        self._loop_task = asyncio.create_task(self._tick_loop())
        logger.info(f"SchedulerNode {self.node_id} started. DB: {self.db_path}")

    async def on_start(self) -> None:
        """Lifecycle hook."""
        pass

    async def on_stop(self) -> None:
        """Lifecycle hook."""
        pass

    async def handle_message(self, message: Message) -> None:
        """Default handler for generic messages. Unused as we route commands specifically."""
        pass

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
        logger.debug(f"Scheduled task {task_id} for {trigger_time} on topic {route_topic}")

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task. Returns True if a task was modified."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE scheduled_tasks SET status = 'cancelled' WHERE task_id = ? AND status = 'pending'",
                (task_id,),
            )
            return cursor.rowcount > 0

    async def _tick_loop(self) -> None:
        """Background loop to poll for due tasks and publish them."""
        while True:
            try:
                await self._process_due_tasks()
            except Exception as e:
                logger.error(f"Error in SchedulerNode tick loop: {e}")

            await asyncio.sleep(self.tick_interval)

    async def _process_due_tasks(self) -> None:
        """Query tasks that are past their trigger_time and pending."""
        now = time.time()
        due_tasks = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM scheduled_tasks WHERE status = 'pending' AND trigger_time <= ?",
                (now,),
            )
            due_tasks = [dict(row) for row in cursor.fetchall()]

        if not due_tasks:
            return

        for task in due_tasks:
            # Mark processing
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE scheduled_tasks SET status = 'processing' WHERE task_id = ?",
                    (task["task_id"],),
                )

            success = await self._execute_task(task)

            # Post-execution resolution
            with sqlite3.connect(self.db_path) as conn:
                if success:
                    # If it's a cron expression, reschedule. Otherwise complete.
                    if task["cron_expression"]:
                        # Basic future calculation for next trigger (e.g., simplistic +interval for now)
                        # We'll rely on the actual caller to reset specific complex windows or implement croniter logic later.
                        pass # To be handled properly in cron processing
                    conn.execute(
                        "UPDATE scheduled_tasks SET status = 'completed' WHERE task_id = ?",
                        (task["task_id"],),
                    )
                else:
                    if task["retry_policy"] == "retry":
                        # Simplistic backoff: retry in 60s
                        new_trigger = now + 60.0
                        conn.execute(
                            "UPDATE scheduled_tasks SET status = 'pending', trigger_time = ? WHERE task_id = ?",
                            (new_trigger, task["task_id"]),
                        )
                    else:
                        conn.execute(
                            "UPDATE scheduled_tasks SET status = 'failed' WHERE task_id = ?",
                            (task["task_id"],),
                        )

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
            logger.error(f"Failed to execute task {task['task_id']}: {e}")
            return False
