import asyncio
import json
import sqlite3
import time

import pytest

from hbllm.brain.scheduler_node import SchedulerNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

pytestmark = pytest.mark.asyncio


class TestSchedulerNode:
    @pytest.fixture
    async def bus(self):
        bus = InProcessBus()
        await bus.start()
        yield bus
        await bus.stop()

    @pytest.fixture
    async def scheduler(self, bus, tmp_path):
        node = SchedulerNode(
            node_id="test_scheduler",
            data_dir=str(tmp_path),
            tick_interval=0.1,
        )
        await node.start(bus)
        yield node
        await node.stop()

    async def test_schedule_event_schema(self, scheduler):
        """Test scheduling an event directly records it in the DB."""
        payload = {"directive": "move", "velocity": 1.5}
        scheduler.schedule_event(
            task_id="t1",
            tenant_id="default",
            trigger_time=time.time() + 10.0,
            route_topic="robot.move",
            payload=payload,
        )

        with sqlite3.connect(scheduler.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM scheduled_tasks WHERE task_id = 't1'")
            row = cursor.fetchone()

        assert row is not None
        assert row["task_id"] == "t1"
        assert row["status"] == "pending"
        assert row["route_topic"] == "robot.move"
        assert json.loads(row["payload"]) == payload

    async def test_cancel_task(self, scheduler):
        """Test canceling an existing pending task."""
        scheduler.schedule_event(
            task_id="t2",
            tenant_id="default",
            trigger_time=time.time() + 10.0,
            route_topic="robot.move",
            payload={},
        )

        assert scheduler.cancel_task("t2") is True

        with sqlite3.connect(scheduler.db_path) as conn:
            cursor = conn.execute("SELECT status FROM scheduled_tasks WHERE task_id = 't2'")
            status = cursor.fetchone()[0]

        assert status == "cancelled"

    async def test_scheduler_emits_event_on_due_time(self, scheduler, bus):
        """Test the tick_loop triggers and publishes events when due."""
        received = []

        async def _capture(msg: Message):
            received.append(msg)

        await bus.subscribe("agent.think", _capture)

        # Schedule tasks in the past so it triggers immediately
        trigger_time = time.time() - 10.0
        scheduler.schedule_event(
            task_id="t3",
            tenant_id="default",
            trigger_time=trigger_time,
            route_topic="agent.think",
            payload={"prompt": "Hello"},
        )

        # Wait for the tick_loop interval to poll DB and publish
        await asyncio.sleep(0.3)

        assert len(received) == 1
        assert received[0].topic == "agent.think"
        assert received[0].payload == {"prompt": "Hello"}

        # Verify DB status is 'completed'
        with sqlite3.connect(scheduler.db_path) as conn:
            cursor = conn.execute("SELECT status FROM scheduled_tasks WHERE task_id = 't3'")
            status = cursor.fetchone()[0]
        assert status == "completed"

    async def test_retry_policy_on_publish_failure(self, scheduler):
        """Test scheduler retry logic when publishing fails."""
        # Force the execute_task to fail by not having a bus or mocking failure
        scheduler._bus = None  # simulate bus dropout

        trigger_time = time.time() - 10.0
        scheduler.schedule_event(
            task_id="t4",
            tenant_id="default",
            trigger_time=trigger_time,
            route_topic="test",
            payload={},
            retry_policy="retry",
        )

        await asyncio.sleep(0.3)

        # Should have updated status back to pending, and shifted the trigger time
        with sqlite3.connect(scheduler.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT status, trigger_time FROM scheduled_tasks WHERE task_id = 't4'"
            )
            row = cursor.fetchone()

        assert row["status"] == "pending"
        # It should have shifted +60s from the execution time (now)
        assert row["trigger_time"] > trigger_time + 10.0

    async def test_command_handling_over_bus(self, scheduler, bus):
        """Test tools can schedule and cancel tasks by publishing commands to the bus."""
        # 1. Send schedule command
        schedule_msg = Message(
            type=MessageType.EVENT,
            source_node_id="tool",
            topic="system.scheduler.schedule",
            payload={
                "task_id": "bus_t1",
                "trigger_time": time.time() + 10.0,
                "route_topic": "dummy",
                "payload": {"data": 123},
            },
        )
        await bus.publish("system.scheduler.schedule", schedule_msg)
        await asyncio.sleep(0.1)

        with sqlite3.connect(scheduler.db_path) as conn:
            cursor = conn.execute("SELECT status FROM scheduled_tasks WHERE task_id = 'bus_t1'")
            row = cursor.fetchone()
        assert row is not None
        assert row[0] == "pending"

        # 2. Send cancel command
        cancel_msg = Message(
            type=MessageType.EVENT,
            source_node_id="tool",
            topic="system.scheduler.cancel",
            payload={"task_id": "bus_t1"},
        )
        await bus.publish("system.scheduler.cancel", cancel_msg)
        await asyncio.sleep(0.1)

        with sqlite3.connect(scheduler.db_path) as conn:
            cursor = conn.execute("SELECT status FROM scheduled_tasks WHERE task_id = 'bus_t1'")
            row = cursor.fetchone()
        assert row[0] == "cancelled"
